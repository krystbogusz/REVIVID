"""2-D selective state-space (Mamba) feature blocks.

This is a genuine Mamba implementation (S6 selective scan), scanned along four
spatial directions (left->right, right->left, top->bottom, bottom->top) as in
VMamba / MambaIR - not a convolutional surrogate.

Requires ``mamba_ssm`` with the fused CUDA kernel. Install with:
    pip install mamba-ssm
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.ops.selective_scan_interface import (
    selective_scan_fn as _selective_scan_cuda,
)


def _selective_scan(u, delta, A, B, C, D, delta_bias):
    return _selective_scan_cuda(
        u, delta, A, B, C, D, z=None, delta_bias=delta_bias, delta_softplus=True
    )


class SS2D(nn.Module):
    """2-D selective scan module (four directional scans)."""

    K = 4

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 3,
        dt_rank="auto",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv2d = nn.Conv2d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv // 2,
            groups=self.d_inner,
        )
        self.act = nn.SiLU()

        self.x_proj = nn.ModuleList(
            [
                nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
                for _ in range(self.K)
            ]
        )
        self.dt_proj = nn.ModuleList(
            [nn.Linear(self.dt_rank, self.d_inner, bias=True) for _ in range(self.K)]
        )

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_logs = nn.Parameter(
            torch.log(A).unsqueeze(0).repeat(self.K, 1, 1).contiguous()
        )
        self.Ds = nn.Parameter(torch.ones(self.K, self.d_inner))

        self.out_norm = nn.LayerNorm(self.d_inner, eps=1e-4)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _scan_sequences(self, x: torch.Tensor) -> torch.Tensor:
        """x: (b, d, h, w) -> directional sequences (b, K, d, L)."""
        b, d, h, w = x.shape
        l = h * w
        hor = x.reshape(b, d, l)
        ver = x.transpose(2, 3).reshape(b, d, l)
        stacked = torch.stack([hor, ver], dim=1)
        flipped = torch.flip(stacked, dims=[-1])
        return torch.cat([stacked, flipped], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        l = h * w

        xz = self.in_proj(x.permute(0, 2, 3, 1))
        x_in, z = xz.chunk(2, dim=-1)
        x_in = x_in.permute(0, 3, 1, 2).contiguous()
        x_in = self.act(self.conv2d(x_in))

        xs = self._scan_sequences(x_in)

        out = 0
        for k in range(self.K):
            u = xs[:, k]
            x_dbl = self.x_proj[k](u.transpose(1, 2))
            dt, Bk, Ck = torch.split(
                x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
            )
            dt = self.dt_proj[k](dt).transpose(1, 2).contiguous()
            Bk = Bk.transpose(1, 2).contiguous()
            Ck = Ck.transpose(1, 2).contiguous()
            A = -torch.exp(self.A_logs[k].float())
            y = _selective_scan(
                u, dt, A, Bk, Ck, self.Ds[k].float(), self.dt_proj[k].bias.float()
            )

            if k in (1, 3):
                y = torch.flip(y, dims=[-1])
            if k in (2, 3):
                y = (
                    y.reshape(b, self.d_inner, w, h)
                    .transpose(2, 3)
                    .reshape(b, self.d_inner, l)
                )
            out = out + y

        out = out.transpose(1, 2)
        out = self.out_norm(out)
        out = out * F.silu(z.reshape(b, l, self.d_inner))
        out = self.out_proj(out)
        return out.transpose(1, 2).reshape(b, c, h, w)


class SSBlock(nn.Module):
    """Residual SS2D block with a feed-forward mixer."""

    def __init__(
        self, dim: int, d_state: int = 16, expand: int = 2, mlp_ratio: float = 2.0
    ):
        super().__init__()

        self.norm1 = nn.GroupNorm(min(8, dim), dim, eps=1e-4)
        self.ss2d = SS2D(dim, d_state=d_state, expand=expand)
        self.norm2 = nn.GroupNorm(min(8, dim), dim, eps=1e-4)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, 1), nn.GELU(), nn.Conv2d(hidden, dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ss2d(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MambaFeatureBlocks(nn.Module):
    """Stack of SSBlocks with channel adaptation, applied as a residual group."""

    def __init__(
        self,
        in_chans: int,
        embed_dim: int = 64,
        depth: int = 6,
        d_state: int = 16,
        expand: int = 2,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        self.blocks = nn.ModuleList(
            [SSBlock(embed_dim, d_state=d_state, expand=expand) for _ in range(depth)]
        )
        self.conv_out = nn.Conv2d(embed_dim, in_chans, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        h = self.conv_in(x)
        for blk in self.blocks:
            h = blk(h)
        return identity + self.conv_out(h)


def build_feature_blocks(
    in_chans: int,
    embed_dim: int = 64,
    depth: int = 6,
    d_state: int = 16,
    expand: int = 2,
) -> nn.Module:
    return MambaFeatureBlocks(
        in_chans, embed_dim=embed_dim, depth=depth, d_state=d_state, expand=expand
    )
