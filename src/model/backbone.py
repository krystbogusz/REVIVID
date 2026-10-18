"""Restoration + 2x SR backbone built on MambaOFR (CVPR 2025), prepared for the
inpainting of persistent holes (burned-in regions that stay for the whole clip).

Per clip ``lrs`` (N, T, 3, h, w) in [-1, 1], as MambaOFR's ``RNN_Swin_4.Video_Backbone``:

1. RAFT (large) optical flow between neighbouring LR frames.
2. Bidirectional first-order recurrent propagation; per frame and direction:
     ``proj``                    conv3x3 of the current frame (latent feature)
     ``FlowGuidedMaskAlignment`` the previous state, aligned by a deformable
                                 conv whose offsets = flow + a residual predicted
                                 from the warped state, the latent feature, the
                                 flow and the warped previous mask (paper eq. 9-10)
     ``MaskGatedAggregation``    soft mask M = MG(aligned, latent, residual
                                 indicator, warped previous mask); state =
                                 M * aligned + (1 - M) * latent (eq. 11-12) - the
                                 mask keeps scratches / dirt of single frames
                                 from being propagated
     ``MambaRestorer``           MambaIR with the degradation-aware prompt blocks
                                 (``mamba_blocks``), one per direction
3. Reconstruction: fuse both directions, residual blocks at LR, PixelShuffle x
   ``sr_scale``, conv, + bilinear upsampled input, tanh (MambaOFR's global
   residual; MambaOFR has no SR, its "upsamplers" are x1).

Inpainting (phase 2, ``use_holes``; off in phase 1, where the model is exactly
the above): ``HoleDetector`` finds the holes (nobody tells the model where they
are), ``push_pull_fill`` pre-fills them so the flow is not pinned to 0,
``FlowCompletion`` re-estimates the flow inside them, the mask gate is floored
at the hole probability (the state is carried over the hole, content seen in
other frames survives until it is used) and a validity map tracks how much of
the state comes from real observations - conditioning of the refiner.

Outputs (HR = h * sr_scale unless noted):
    * ``coarse``      (N, T, 3, H, W) restored frames
    * ``cond``        (N, T, cond_dim, H, W) features for the refiner
    * ``hole_logits`` (N, T, 1, h, w) hole detector logits at LR (-20 = no hole
                      when ``use_holes`` is off)
    * ``validity``    (N, T, 1, H, W) share of the state from real observations
    * ``flow_fwd``    (N, T-1, 2, h, w) flow at LR, ``[:, k]`` maps frame k+1
                      onto frame k
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

from .blocks import ResidualBlockNoBN, make_layer
from .flow import build_flow_estimator, flow_warp
from .mamba_blocks import MambaRestorer


def build_upsampler(in_ch: int, scale: int) -> nn.Module:
    """Sub-pixel (PixelShuffle) upsampler. ``scale`` must be a power of two."""
    if scale <= 1:
        return nn.Identity()
    if scale & (scale - 1) != 0:
        raise ValueError(f"sr_scale must be a power of two, got {scale}")
    layers = []
    s = scale
    while s > 1:
        layers += [
            nn.Conv2d(in_ch, in_ch * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, inplace=True),
        ]
        s //= 2
    return nn.Sequential(*layers)


def rgb_to_luma(x: torch.Tensor) -> torch.Tensor:
    """Luma channel from an RGB tensor in [-1, 1] (or [0, 1]). Keeps the range."""
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _lrelu() -> nn.Module:
    return nn.LeakyReLU(0.1, inplace=True)


@torch.no_grad()
def push_pull_fill(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Fill pixels with ``valid == 0`` from their surroundings, for any hole size.

    Push: average the valid pixels down a pyramid (normalised by the valid
    weight) until every pixel has support. Pull: upsample back, keeping the
    valid pixels. ``x`` (B, C, H, W), ``valid`` (B, 1, H, W) in [0, 1].
    """
    h, w = x.shape[-2:]
    if min(h, w) <= 2:
        wsum = valid.sum(dim=(-2, -1), keepdim=True)
        mean = (x * valid).sum(dim=(-2, -1), keepdim=True) / wsum.clamp(min=1e-6)
        return valid * x + (1 - valid) * mean
    xs = F.avg_pool2d(x * valid, 2, ceil_mode=True)
    vs = F.avg_pool2d(valid, 2, ceil_mode=True)
    coarse = push_pull_fill(xs / vs.clamp(min=1e-6), (vs > 0).to(x.dtype))
    up = F.interpolate(coarse, size=(h, w), mode="bilinear", align_corners=False)
    return valid * x + (1 - valid) * up


class HoleDetector(nn.Module):
    """Per-pixel persistent-hole logits at LR.

    A burned-in hole is constant in time and sits at the bottom of the range,
    while dark content moves or at least carries grain. The clip's temporal
    min / max / std of luma expose exactly that, so they are fed next to the
    frame. Trained against the true hole mask (BCE).
    """

    def __init__(self, ch: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, ch, 3, 1, 1), _lrelu(),
            nn.Conv2d(ch, ch, 3, 1, 2, dilation=2), _lrelu(),
            nn.Conv2d(ch, ch, 3, 1, 4, dilation=4), _lrelu(),
            nn.Conv2d(ch, ch, 3, 1, 8, dilation=8), _lrelu(),
            nn.Conv2d(ch, 1, 3, 1, 1),
        )
        # Start out predicting "no hole" (p ~ 0.02), so the untrained detector
        # does not make the rest of the network treat everything as a hole.
        nn.init.constant_(self.net[-1].bias, -4.0)

    def forward(self, lrs: torch.Tensor) -> torch.Tensor:
        n, t, c, h, w = lrs.shape
        y = rgb_to_luma(lrs.reshape(n * t, c, h, w)).view(n, t, 1, h, w)
        stats = torch.cat([y.amin(1), y.amax(1), y.std(1, correction=0)], dim=1)
        x = torch.cat([lrs, stats.unsqueeze(1).expand(-1, t, -1, -1, -1)], dim=2)
        return self.net(x.reshape(n * t, c + 3, h, w)).view(n, t, 1, h, w)


class FlowCompletion(nn.Module):
    """Re-estimate the optical flow inside holes from the motion around them.

    The flow inside the (soft) hole mask is hidden from the network, which has
    to infer it from the surrounding motion and both frames; outside the mask
    the RAFT flow passes through unchanged. Zero-initialised output, so it
    starts as the identity. Trained end-to-end through the restoration losses.
    """

    def __init__(self, ch: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2 + 1 + 6, ch, 3, 1, 1), _lrelu(),
            nn.Conv2d(ch, ch, 3, 1, 2, dilation=2), _lrelu(),
            nn.Conv2d(ch, ch, 3, 1, 4, dilation=4), _lrelu(),
            nn.Conv2d(ch, ch, 3, 1, 8, dilation=8), _lrelu(),
            nn.Conv2d(ch, ch, 3, 1, 16, dilation=16), _lrelu(),
            nn.Conv2d(ch, 2, 3, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, flow, mask, frame_a, frame_b):
        x = torch.cat([flow * (1 - mask), mask, frame_a, frame_b], dim=1)
        return flow + mask * self.net(x)


class FlowGuidedMaskAlignment(nn.Module):
    """Flow-guided mask deformable alignment (MambaOFR, eq. 9-10).

    Offsets = optical flow + a residual bounded by ``max_residue`` px, predicted
    from the flow-warped state, the current latent feature, the flow, the warped
    previous mask (where the previous frame did not trust its own propagation,
    e.g. on a scratch) and the hole map. DCNv2 modulation lets the conv drop
    untrusted samples. MambaOFR's code passed the flow of the first clip of the
    batch, vertically flipped and with (dx, dy) swapped, as the offset base; here
    every clip gets its own flow as (dy, dx), which is what torchvision expects.
    """

    def __init__(self, ch: int, groups: int = 4, max_residue: float = 10.0):
        super().__init__()
        if ch % groups:
            raise ValueError(f"dcn_groups={groups} must divide num_feat={ch}")
        self.max_residue = max_residue
        self.conv_offset = nn.Sequential(
            nn.Conv2d(2 * ch + 2 + 1 + 1, ch, 3, 1, 1), _lrelu(),
            nn.Conv2d(ch, ch, 3, 1, 1), _lrelu(),
            nn.Conv2d(ch, ch, 3, 1, 1), _lrelu(),
            nn.Conv2d(ch, 27 * groups, 3, 1, 1),
        )
        # Zero residual + modulation 0.5 at start: a plain flow-guided warp.
        nn.init.zeros_(self.conv_offset[-1].weight)
        nn.init.zeros_(self.conv_offset[-1].bias)
        self.weight = nn.Parameter(torch.empty(ch, ch, 3, 3))
        self.bias = nn.Parameter(torch.zeros(ch))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, feat, warped, curr, flow, mask_w, hole):
        """``feat`` the unwarped previous state, ``warped`` it warped by ``flow``
        (N, 2, h, w) as (dx, dy), ``mask_w`` the warped previous mask."""
        out = self.conv_offset(torch.cat([warped, curr, flow, mask_w, hole], dim=1))
        o1, o2, modulation = torch.chunk(out, 3, dim=1)
        offset = self.max_residue * torch.tanh(torch.cat([o1, o2], dim=1))
        offset = offset + flow.flip(1).repeat(1, offset.size(1) // 2, 1, 1)
        return deform_conv2d(feat, offset, self.weight, self.bias, padding=1, mask=torch.sigmoid(modulation))


class MaskGatedAggregation(nn.Module):
    """MambaOFR's mask generator and fusion (``Gated_Aggregation``, eq. 11-12).

    M = MG(aligned, latent, residual indicator, warped previous mask, hole), the
    weight of the propagated state against the current frame's latent feature.
    The residual indicator |warp(previous frame) - current frame| flags pixels
    the motion does not explain: a scratch or a speck of dirt of one frame.
    Inside a hole M is floored at the hole probability (state carried over).
    ``validity`` = how much of the state comes from real observations: ``1 -
    hole`` for a fresh pixel, the propagated validity for a carried one.
    """

    def __init__(self, ch: int):
        super().__init__()
        c_in = 2 * ch + 3
        self.mask_gen = nn.Sequential(
            nn.Conv2d(c_in, c_in, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c_in, ch // 2, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch // 2, 1, 3, 1, 1), nn.Sigmoid(),
        )

    def forward(self, aligned, curr, res_ind, mask_w, hole, v_prop):
        g = self.mask_gen(torch.cat([aligned, curr, res_ind, mask_w, hole], dim=1))
        g = torch.maximum(g, hole)
        fused = g * aligned + (1 - g) * curr
        validity = g * v_prop + (1 - g) * (1 - hole)
        return fused, g, validity


_DIRS = ("bwd", "fwd")  # "forward" would clash with nn.Module.forward


class ConditioningBackbone(nn.Module):
    def __init__(
        self,
        num_feat: int = 32,
        embed_dim: int = 64,
        depths: Sequence[int] = (2, 2, 2),
        d_state: int = 16,
        ssm_expand: float = 1.2,
        dyn_kernels: int = 8,
        drop_path: float = 0.1,
        recon_feat: int = 64,
        recon_blocks: int = 2,
        cond_dim: int = 64,
        sr_scale: int = 2,
        dcn_groups: int = 4,
        hole_threshold: float = 0.9,
        raft: str = "large",
        use_holes: bool = False,
    ):
        super().__init__()
        self.hole_threshold = hole_threshold
        # Phase 2 switch (not saved in the state dict, set from ModelConfig).
        self.use_holes = use_holes
        c = num_feat

        self.hole_detector = HoleDetector()
        self.flow_net = build_flow_estimator(raft)
        self.flow_completion = FlowCompletion()

        self.proj = nn.ModuleDict({d: nn.Conv2d(3, c, 3, 1, 1) for d in _DIRS})
        self.align = nn.ModuleDict({d: FlowGuidedMaskAlignment(c, dcn_groups) for d in _DIRS})
        self.agg = nn.ModuleDict({d: MaskGatedAggregation(c) for d in _DIRS})
        self.restorer = nn.ModuleDict(
            {d: MambaRestorer(c, embed_dim, tuple(depths), d_state, ssm_expand, dyn_kernels, drop_path) for d in _DIRS}
        )

        # Reconstruction: both states + their validity maps -> SR frame.
        self.fuse = nn.Conv2d(2 * c + 2, recon_feat, 3, 1, 1)
        self.recon = make_layer(ResidualBlockNoBN, recon_blocks, num_feat=recon_feat)
        self.upsampler = build_upsampler(recon_feat, sr_scale)
        self.conv_hr = nn.Conv2d(recon_feat, recon_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(recon_feat, 3, 3, 1, 1)
        self.cond_head = nn.Conv2d(recon_feat, cond_dim, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def _comp_flow(self, frames: torch.Tensor, m: torch.Tensor):
        """(forward_flow, backward_flow), each (N, T-1, 2, h, w).

        ``forward_flow[:, k]`` maps frame k+1 onto frame k (warps the state of
        frame k into frame k+1); ``backward_flow[:, k]`` maps frame k onto k+1.
        With ``use_holes`` the flow inside the holes is completed.
        """
        n, t, c, h, w = frames.size()
        if t < 2:
            empty = frames.new_zeros(n, 0, 2, h, w)
            return empty, empty
        a = frames[:, 1:].reshape(-1, c, h, w)
        b = frames[:, :-1].reshape(-1, c, h, w)
        fwd, bwd = self.flow_net(a, b), self.flow_net(b, a)
        if self.use_holes:
            pair_m = torch.maximum(m[:, 1:], m[:, :-1]).reshape(-1, 1, h, w)
            fwd = self.flow_completion(fwd, pair_m, a, b)
            bwd = self.flow_completion(bwd, pair_m, b, a)
        return fwd.view(n, t - 1, 2, h, w), bwd.view(n, t - 1, 2, h, w)

    def _step(
        self, d: str, frame: torch.Tensor, hole: torch.Tensor, prev: Optional[Tuple[torch.Tensor, ...]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One propagation step in direction ``d``; ``prev = (state, mask,
        validity, frame, flow)`` of the previous frame in propagation order
        (``None`` at the start). Returns (state, mask, validity)."""
        curr = self.proj[d](frame)
        if prev is None:
            feat, mask, validity = curr, torch.ones_like(hole), 1.0 - hole
        else:
            p_feat, p_mask, p_val, p_frame, flow = prev
            grid = flow.permute(0, 2, 3, 1)
            warped = flow_warp(p_feat, grid)
            mask_w = flow_warp(p_mask, grid, padding_mode="zeros")
            v_w = flow_warp(p_val, grid, padding_mode="zeros")
            res_ind = torch.abs(rgb_to_luma(flow_warp(p_frame, grid)) - rgb_to_luma(frame))
            aligned = self.align[d](p_feat, warped, curr, flow, mask_w, hole)
            feat, mask, validity = self.agg[d](aligned, curr, res_ind, mask_w, hole, v_w)
        return self.restorer[d](feat), mask, validity

    def forward(self, lrs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            lrs: (N, T, 3, H_lr, W_lr) LQ input.

        Returns ``coarse``, ``cond``, ``validity`` at the SR output resolution
        and ``hole_logits``, ``flow_fwd`` at the input (LR) resolution.
        """
        n, t, c, h, w = lrs.size()

        if self.use_holes:
            hole_logits = self.hole_detector(lrs)
            m = torch.sigmoid(hole_logits).detach()
            hard = (m > self.hole_threshold).to(lrs.dtype)
            filled = push_pull_fill(
                lrs.reshape(n * t, c, h, w), 1.0 - hard.reshape(n * t, 1, h, w)
            ).view(n, t, c, h, w)
        else:
            hole_logits = lrs.new_full((n, t, 1, h, w), -20.0)
            m = lrs.new_zeros(n, t, 1, h, w)
            filled = lrs

        fwd_flow, bwd_flow = self._comp_flow(filled, m)

        # ---- backward propagation (t-1 -> 0), states kept for the fusion
        back = [None] * t
        for i in range(t - 1, -1, -1):
            prev = (*back[i + 1], filled[:, i + 1], bwd_flow[:, i]) if i < t - 1 else None
            back[i] = self._step("bwd", filled[:, i], m[:, i], prev)

        # ---- forward propagation (0 -> t-1), reconstructing each frame on the fly
        coarse_out, cond_out, val_out = [], [], []
        state = None
        for i in range(t):
            prev = (*state, filled[:, i - 1], fwd_flow[:, i - 1]) if i > 0 else None
            state = self._step("fwd", filled[:, i], m[:, i], prev)
            (b_feat, _, b_val), (f_feat, _, f_val) = back[i], state

            x = self.lrelu(self.fuse(torch.cat([b_feat, f_feat, b_val, f_val], dim=1)))
            x = self.upsampler(self.recon(x))
            x = self.lrelu(self.conv_hr(x))
            hr_size = x.shape[-2:]
            base = F.interpolate(filled[:, i], size=hr_size, mode="bilinear", align_corners=False)
            coarse_out.append(torch.tanh(self.conv_last(x) + base))
            cond_out.append(self.cond_head(x))
            val_out.append(
                F.interpolate(torch.maximum(b_val, f_val), size=hr_size, mode="bilinear", align_corners=False)
            )

        return {
            "coarse": torch.stack(coarse_out, dim=1),
            "cond": torch.stack(cond_out, dim=1),
            "hole_logits": hole_logits,
            "validity": torch.stack(val_out, dim=1),
            "flow_fwd": fwd_flow,
        }
