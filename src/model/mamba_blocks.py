"""MambaOFR's Dynamic Mamba Restoration Block (DMRB), rebuilt from its code
(``VP_code/models/mambair_arch.py``, MambaIR-based) and paper (CVPR 2025, sec. 3.3).

    VSSBlock:  x = s1 * x + SS2D(LN(x))                  vision Mamba block, 4-direction scan
               x = s2 * x + CAB(LN(x))                   degradation-aware prompt block:
                   CAB = DynConv3x3 -> GELU -> DynConv3x3 -> channel attention
               DynConv mixes N kernels per frame with weights computed from the
               globally pooled feature - the "degradation prompt" that adapts the
               block to the degradation of the current frame.
    ResidualGroup:   VSSBlocks + conv3x3, residual.
    MambaRestorer:   MambaIR in its denoising configuration, as MambaOFR applies
                     it to the propagated feature of every frame:
                     conv_first -> LN -> groups -> LN -> conv + skip -> conv_last + skip.

Differences from the previous REVIVID blocks, which produced flat "watercolour"
regions: LayerNorm per pixel instead of GroupNorm (spatial statistics + FiLM is
AdaIN, known for blob artefacts), MambaIR's dt initialisation (dt in [1e-3, 0.1]
keeps the long-range memory of the scan; the default Linear init gave dt ~ 0.5-1,
i.e. a short, almost local scan) and the 3x3 convolutional CAB instead of a
1x1 MLP (local texture).

Requires ``mamba_ssm`` with the fused CUDA selective-scan kernel. Install with:
    pip install mamba-ssm
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


def selective_scan(u, delta, A, B, C, D, delta_bias):
    return selective_scan_fn(u, delta, A, B, C, D, z=None, delta_bias=delta_bias, delta_softplus=True)


class DropPath(nn.Module):
    """Stochastic depth per sample (MambaOFR inherits MambaIR's drop_path_rate 0.1)."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.dim() - 1)).bernoulli_(keep)
        return x * mask / keep


class SS2D(nn.Module):
    """2-D selective scan (VMamba / MambaIR): four directional S6 scans of one
    feature map, all in a single kernel call. Input / output (B, H, W, C)."""

    K = 4

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: float = 1.2,
        d_conv: int = 3,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
    ):
        super().__init__()
        self.d_state = d_state
        self.d_inner = d_inner = int(expand * d_model)
        self.dt_rank = dt_rank = math.ceil(d_model / 16)
        K = self.K

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv2d = nn.Conv2d(d_inner, d_inner, d_conv, padding=(d_conv - 1) // 2, groups=d_inner)
        self.act = nn.SiLU()

        # x_proj keeps nn.Linear's default init; dt_proj Mamba's.
        bound = 1.0 / math.sqrt(d_inner)
        self.x_proj_weight = nn.Parameter(torch.empty(K, dt_rank + 2 * d_state, d_inner).uniform_(-bound, bound))
        std = dt_rank**-0.5
        self.dt_projs_weight = nn.Parameter(torch.empty(K, d_inner, dt_rank).uniform_(-std, std))
        dt = torch.exp(
            torch.rand(K, d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        self.dt_projs_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(K * d_inner, 1)
        self.A_logs = nn.Parameter(torch.log(A))  # (K * d_inner, d_state), S4D-real
        self.Ds = nn.Parameter(torch.ones(K * d_inner))

        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def _scan(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, D, H, W) -> sum of the four directional scans, (B, D, L)."""
        b, d, h, w = x.shape
        l, K = h * w, self.K
        hw = torch.stack([x.reshape(b, d, l), x.transpose(2, 3).reshape(b, d, l)], dim=1)
        xs = torch.cat([hw, torch.flip(hw, dims=[-1])], dim=1)  # (b, K, d, l)

        x_dbl = torch.einsum("bkdl,kcd->bkcl", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("bkrl,kdr->bkdl", dts, self.dt_projs_weight)
        y = selective_scan(
            xs.float().reshape(b, -1, l),
            dts.contiguous().float().reshape(b, -1, l),
            -torch.exp(self.A_logs.float()),
            Bs.float().contiguous(),
            Cs.float().contiguous(),
            self.Ds.float(),
            self.dt_projs_bias.float().reshape(-1),
        ).view(b, K, d, l)

        inv = torch.flip(y[:, 2:4], dims=[-1])
        wh = y[:, 1].view(b, d, w, h).transpose(2, 3).reshape(b, d, l)
        inv_wh = inv[:, 1].view(b, d, w, h).transpose(2, 3).reshape(b, d, l)
        return y[:, 0] + inv[:, 0] + wh + inv_wh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, _ = x.shape
        x, z = self.in_proj(x).chunk(2, dim=-1)
        x = self.act(self.conv2d(x.permute(0, 3, 1, 2).contiguous()))
        y = self._scan(x).transpose(1, 2).reshape(b, h, w, -1)
        y = self.out_norm(y.to(z.dtype)) * F.silu(z)
        return self.out_proj(y)


class DynamicConv2d(nn.Module):
    """Degradation-aware prompt convolution (MambaOFR's ``DynamicConvolution``,
    paper eq. 5-7): ``num_kernels`` kernels (the prompt) mixed per frame with
    softmax weights computed from the globally pooled input feature."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, num_kernels: int = 8, extend: int = 2):
        super().__init__()
        hidden = max(1, in_ch * extend)
        self.to_scores = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(in_ch, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden),
        )
        self.prompt = nn.Parameter(torch.rand(1, num_kernels, hidden))
        self.weight = nn.Parameter(torch.empty(num_kernels, out_ch, in_ch, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(num_kernels, out_ch))
        for k in range(num_kernels):
            nn.init.kaiming_uniform_(self.weight[k], a=math.sqrt(5))
        bound = 1.0 / math.sqrt(in_ch * kernel_size * kernel_size)
        nn.init.uniform_(self.bias, -bound, bound)
        self.padding = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        alpha = torch.softmax((self.to_scores(x).unsqueeze(1) * self.prompt).sum(-1), dim=-1)  # (b, N)
        weight = torch.einsum("bn,noihw->boihw", alpha, self.weight).reshape(-1, c, *self.weight.shape[-2:])
        bias = (alpha @ self.bias).reshape(-1)
        out = F.conv2d(x.reshape(1, b * c, h, w), weight, bias, padding=self.padding, groups=b)
        return out.view(b, -1, h, w)


class ChannelAttention(nn.Module):
    def __init__(self, ch: int, squeeze: int = 30):
        super().__init__()
        mid = max(1, ch // squeeze)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, mid, 1), nn.ReLU(inplace=True),
            nn.Conv2d(mid, ch, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class CAB(nn.Module):
    """Degradation-aware prompt block: two dynamic 3x3 convolutions + channel attention."""

    def __init__(self, ch: int, num_kernels: int = 8, compress: int = 3, squeeze: int = 30):
        super().__init__()
        mid = max(1, ch // compress)
        self.net = nn.Sequential(
            DynamicConv2d(ch, mid, 3, num_kernels),
            nn.GELU(),
            DynamicConv2d(mid, ch, 3, num_kernels),
            ChannelAttention(ch, squeeze),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VSSBlock(nn.Module):
    """(B, H, W, C) -> (B, H, W, C)."""

    def __init__(self, dim: int, d_state: int = 16, expand: float = 1.2, num_kernels: int = 8, drop_path: float = 0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.ss2d = SS2D(dim, d_state=d_state, expand=expand)
        self.drop_path = DropPath(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(dim))
        self.ln_2 = nn.LayerNorm(dim)
        self.cab = CAB(dim, num_kernels)
        self.skip_scale2 = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.skip_scale + self.drop_path(self.ss2d(self.ln_1(x)))
        y = self.cab(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1)
        return x * self.skip_scale2 + y


class ResidualGroup(nn.Module):
    """Residual state-space group: VSSBlocks + conv3x3 with a skip. (B, C, H, W)."""

    def __init__(self, dim, depth, d_state, expand, num_kernels, drop_paths, use_checkpoint=True):
        super().__init__()
        self.blocks = nn.ModuleList(
            [VSSBlock(dim, d_state, expand, num_kernels, drop_paths[i]) for i in range(depth)]
        )
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.permute(0, 2, 3, 1).contiguous()
        for blk in self.blocks:
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                h = ckpt.checkpoint(blk, h, use_reentrant=False)
            else:
                h = blk(h)
        return self.conv(h.permute(0, 3, 1, 2).contiguous()) + x


class MambaRestorer(nn.Module):
    """MambaIR in its denoising configuration on a ``in_ch``-channel feature,
    as MambaOFR runs it on the propagated state of every frame (one per
    propagation direction): (B, in_ch, H, W) -> (B, in_ch, H, W), residual."""

    def __init__(
        self,
        in_ch: int,
        embed_dim: int = 64,
        depths=(2, 2, 2),
        d_state: int = 16,
        expand: float = 1.2,
        num_kernels: int = 8,
        drop_path: float = 0.1,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.conv_first = nn.Conv2d(in_ch, embed_dim, 3, 1, 1)
        self.norm_in = nn.LayerNorm(embed_dim)
        dpr = torch.linspace(0, drop_path, sum(depths)).tolist()
        self.groups = nn.ModuleList()
        start = 0
        for depth in depths:
            self.groups.append(
                ResidualGroup(embed_dim, depth, d_state, expand, num_kernels, dpr[start : start + depth], use_checkpoint)
            )
            start += depth
        self.norm_out = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.conv_last = nn.Conv2d(embed_dim, in_ch, 3, 1, 1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        # MambaIR: every nn.Linear (projections, prompt MLPs) trunc-normal 0.02.
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @staticmethod
    def _ln(norm: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
        return norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = self.conv_first(x)
        h = self._ln(self.norm_in, first)
        for group in self.groups:
            h = group(h)
        h = self._ln(self.norm_out, h)
        return x + self.conv_last(self.conv_after_body(h) + first)
