"""Hole-aware bidirectional recurrent conditioning backbone.

Handles restoration + 2x SR of old film and prepares the inpainting of
persistent holes (burned-in regions that stay in place for the whole clip).

Per clip ``lrs`` of shape (N, T, 3, h, w) in [-1, 1]:

1. ``HoleDetector``      — per-pixel hole logits at LR, learned from the frame
                           and the clip's temporal statistics. Nothing tells the
                           model where the holes are; the true mask is only its
                           training target.
2. ``push_pull_fill``    — holes are pre-filled from their surroundings, so the
                           flat black patch does not pin the optical flow to 0.
3. RAFT (large) + ``FlowCompletion`` — flow on the pre-filled frames, then re-estimated
                           inside holes from the motion around them. With the
                           right motion, the content hidden under a static hole
                           at frame t is fetched from frames where it was visible.
4. ``DegradationEncoder``— self-learned degradation code per frame (from a
                           3-frame window, no labels); it modulates every SS2D
                           block (FiLM).
5. Bidirectional SECOND-ORDER propagation (states of t±1 and t±2):
     ``SecondOrderAlignment`` — flow-guided DCNv2 with a modulation mask driven
                                by validity and holes;
     ``ValidityAggregation``  — gate between the aligned state and the current
                                frame; inside holes the state is carried over,
                                and a validity map tracks how much of it comes
                                from real observations;
     SS2D blocks (Mamba), FiLM-modulated by the degradation code.
6. Fusion, PixelShuffle x sr_scale, HR blocks, heads.

Outputs (HR = h * sr_scale unless noted):
    * ``coarse``      — (N, T, 3, H, W) first-pass restored frames.
    * ``cond``        — (N, T, cond_dim, H, W) conditioning for the refiner.
    * ``hole_logits`` — (N, T, 1, h, w) hole detector logits at LR.
    * ``validity``    — (N, T, 1, H, W) how much of each pixel's state comes
                        from real (non-hole) observations, in [0, 1].
    * ``flow_fwd``    — (N, T-1, 2, h, w) completed flow at LR, ``[:, k]`` maps
                        frame k+1 onto frame k (used to move the refiner's noise
                        with the scene).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
from torchvision.ops import deform_conv2d

from .blocks import ResidualBlockNoBN, make_layer
from .flow import build_flow_estimator, flow_warp
from .mamba_blocks import MambaFeatureBlocks


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


class DegradationEncoder(nn.Module):
    """Self-learned degradation code per frame (no labels, no supervision).

    Sees the frame with its two neighbours: scratches, dirt and flicker change
    from frame to frame while content moves smoothly, and the encoder is free
    to pick that up. The code modulates the SS2D blocks (FiLM).
    """

    def __init__(self, out_dim: int = 64, ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(9, ch // 2, 3, 2, 1), _lrelu(),
            nn.Conv2d(ch // 2, ch, 3, 2, 1), _lrelu(),
            nn.Conv2d(ch, ch, 3, 2, 1), _lrelu(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, out_dim), _lrelu(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, lrs: torch.Tensor) -> torch.Tensor:
        n, t, c, h, w = lrs.shape
        prev = torch.cat([lrs[:, :1], lrs[:, :-1]], dim=1)
        nxt = torch.cat([lrs[:, 1:], lrs[:, -1:]], dim=1)
        x = torch.cat([prev, lrs, nxt], dim=2).reshape(n * t, 3 * c, h, w)
        return self.net(x).view(n, t, -1)


class SecondOrderAlignment(nn.Module):
    """Flow-guided modulated deformable alignment of the t±1 and t±2 states.

    Sampling offsets = optical flow + a learned residual bounded by
    ``max_residue`` px, predicted from the flow-warped states, the current
    frame feature, both flows, the validity of both states and the hole mask.
    The DCNv2 modulation mask lets the module drop samples that come from holes
    or other untrusted places (MambaOFR's flow-guided DCN runs without one).
    """

    def __init__(self, ch: int, groups: int = 8, max_residue: float = 10.0):
        super().__init__()
        if (2 * ch) % groups:
            raise ValueError(f"dcn_groups={groups} must divide 2*num_feat={2 * ch}")
        self.max_residue = max_residue
        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * ch + 2 + 2 + 2 + 1, ch, 3, 1, 1), _lrelu(),
            nn.Conv2d(ch, ch, 3, 1, 1), _lrelu(),
            nn.Conv2d(ch, ch, 3, 1, 1), _lrelu(),
            nn.Conv2d(ch, 27 * groups, 3, 1, 1),
        )
        # Zero offsets residual + modulation 0.5 at start: pure flow-guided warp.
        nn.init.zeros_(self.conv_offset[-1].weight)
        nn.init.zeros_(self.conv_offset[-1].bias)
        self.weight = nn.Parameter(torch.empty(ch, 2 * ch, 3, 3))
        self.bias = nn.Parameter(torch.zeros(ch))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, h1, h2, w1, w2, curr, f1, f2, v1, v2, m):
        """``h*`` unwarped states, ``w*`` their flow-warped versions, ``f*``
        flows (N, 2, h, w) as (dx, dy), ``v*`` warped validity, ``m`` hole mask."""
        out = self.conv_offset(torch.cat([w1, w2, curr, f1, f2, v1, v2, m], dim=1))
        o1, o2, modulation = torch.chunk(out, 3, dim=1)
        offset = self.max_residue * torch.tanh(torch.cat([o1, o2], dim=1))
        off1, off2 = torch.chunk(offset, 2, dim=1)
        # torchvision wants (dy, dx) per sampling point; the flows are (dx, dy).
        off1 = off1 + f1.flip(1).repeat(1, off1.size(1) // 2, 1, 1)
        off2 = off2 + f2.flip(1).repeat(1, off2.size(1) // 2, 1, 1)
        return deform_conv2d(
            torch.cat([h1, h2], dim=1),
            torch.cat([off1, off2], dim=1),
            self.weight,
            self.bias,
            padding=1,
            mask=torch.sigmoid(modulation),
        )


class ValidityAggregation(nn.Module):
    """Fuse the aligned state with the current frame's feature, tracking validity.

    The gate sees the residual indicator (how inconsistent the warped neighbour
    is with the current frame), the propagated validity and the hole mask.
    Inside a hole the gate is floored at the hole probability, so the state is
    carried over and content seen in earlier frames survives until it is used.
    ``validity`` = how much of the state comes from real observations:
    ``1 - hole`` for a fresh pixel, the propagated validity for a carried one.
    """

    def __init__(self, ch: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(2 * ch + 3, ch, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, aligned, curr, res_ind, v_prop, m):
        g = self.gate(torch.cat([aligned, curr, res_ind, v_prop, m], dim=1))
        g = torch.maximum(g, m)
        fused = g * aligned + (1 - g) * curr
        validity = g * v_prop + (1 - g) * (1 - m)
        return fused, validity


_DIRS = ("bwd", "fwd")  # "forward" would clash with nn.Module.forward


class ConditioningBackbone(nn.Module):
    def __init__(
        self,
        num_feat: int = 32,
        num_block: int = 6,
        cond_dim: int = 64,
        embed_dim: int = 64,
        d_state: int = 16,
        ssm_expand: int = 2,
        sr_scale: int = 2,
        deg_dim: int = 64,
        dcn_groups: int = 8,
        hole_threshold: float = 0.9,
        raft: str = "large",
    ):
        super().__init__()
        self.hole_threshold = hole_threshold
        c = num_feat

        self.hole_detector = HoleDetector()
        self.flow_net = build_flow_estimator(raft)
        self.flow_completion = FlowCompletion()
        self.degradation_encoder = DegradationEncoder(deg_dim)

        self.proj = nn.ModuleDict({d: nn.Conv2d(3, c, 3, 1, 1) for d in _DIRS})
        self.align = nn.ModuleDict({d: SecondOrderAlignment(c, dcn_groups) for d in _DIRS})
        self.agg = nn.ModuleDict({d: ValidityAggregation(c) for d in _DIRS})
        self.blocks = nn.ModuleDict(
            {d: MambaFeatureBlocks(c, embed_dim, num_block, d_state, ssm_expand, deg_dim) for d in _DIRS}
        )

        # backward + forward state, plus their validity maps
        self.fuse = nn.Conv2d(2 * c + 2, 2 * c, 3, 1, 1)
        self.trunk = make_layer(ResidualBlockNoBN, 4, num_feat=2 * c)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

        self.upsampler = build_upsampler(2 * c, sr_scale)
        # HR-space reconstruction: refine features after PixelShuffle so the
        # coarse head is not a single conv away from the sub-pixel output.
        self.hr_blocks = make_layer(ResidualBlockNoBN, 2, num_feat=2 * c)

        self.coarse_head = nn.Sequential(
            nn.Conv2d(2 * c, 2 * c, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(2 * c, 3, 3, 1, 1),
        )
        self.cond_head = nn.Conv2d(2 * c, cond_dim, 3, 1, 1)

    def _comp_flow(self, frames: torch.Tensor, m: torch.Tensor):
        """(forward_flow, backward_flow), each (N, T-1, 2, h, w), holes completed.

        ``forward_flow[:, k]`` maps frame k+1 onto frame k (warps the state of
        frame k into frame k+1); ``backward_flow[:, k]`` maps frame k onto k+1.
        """
        n, t, c, h, w = frames.size()
        if t < 2:
            empty = frames.new_zeros(n, 0, 2, h, w)
            return empty, empty
        a = frames[:, 1:].reshape(-1, c, h, w)
        b = frames[:, :-1].reshape(-1, c, h, w)
        pair_m = torch.maximum(m[:, 1:], m[:, :-1]).reshape(-1, 1, h, w)
        fwd = self.flow_completion(self.flow_net(a, b), pair_m, a, b)
        bwd = self.flow_completion(self.flow_net(b, a), pair_m, b, a)
        return fwd.view(n, t - 1, 2, h, w), bwd.view(n, t - 1, 2, h, w)

    def _step(
        self,
        d: str,
        frame: torch.Tensor,
        m_i: torch.Tensor,
        z_i: torch.Tensor,
        s1: Optional[Tuple[torch.Tensor, ...]],
        s2: Optional[Tuple[torch.Tensor, ...]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One propagation step in direction ``d``.

        ``s1 = (state, validity, frame, flow)`` of the previous frame in the
        propagation order, ``s2 = (state, validity, flow)`` of the one before
        it (flow already composed to reach the current frame); ``None`` at the
        start of the sequence.
        """
        curr = self.proj[d](frame)
        if s1 is None:
            feat, validity = curr, 1.0 - m_i
        else:
            h1, val1, frame1, f1 = s1
            g1 = f1.permute(0, 2, 3, 1)
            w1 = flow_warp(h1, g1)
            v1 = flow_warp(val1, g1, padding_mode="zeros")
            if s2 is None:
                h2 = torch.zeros_like(h1)
                w2 = h2
                v2 = torch.zeros_like(v1)
                f2 = torch.zeros_like(f1)
            else:
                h2, val2, f2 = s2
                g2 = f2.permute(0, 2, 3, 1)
                w2 = flow_warp(h2, g2)
                v2 = flow_warp(val2, g2, padding_mode="zeros")
            aligned = self.align[d](h1, h2, w1, w2, curr, f1, f2, v1, v2, m_i)
            res_ind = torch.abs(rgb_to_luma(flow_warp(frame1, g1)) - rgb_to_luma(frame))
            feat, validity = self.agg[d](aligned, curr, res_ind, torch.maximum(v1, v2), m_i)
        feat = ckpt.checkpoint(self.blocks[d], feat, z_i, use_reentrant=False)
        return feat, validity

    def forward(self, lrs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            lrs: (N, T, 3, H_lr, W_lr) LQ input.

        Returns ``coarse``, ``cond``, ``validity`` at the SR output resolution
        and ``hole_logits``, ``flow_fwd`` at the input (LR) resolution.
        """
        n, t, c, h, w = lrs.size()

        hole_logits = self.hole_detector(lrs)
        m = torch.sigmoid(hole_logits).detach()
        hard = (m > self.hole_threshold).to(lrs.dtype)
        filled = push_pull_fill(
            lrs.reshape(n * t, c, h, w), 1.0 - hard.reshape(n * t, 1, h, w)
        ).view(n, t, c, h, w)

        z = self.degradation_encoder(lrs)
        fwd_flow, bwd_flow = self._comp_flow(filled, m)

        # ---- backward propagation (t-1 -> 0), states kept for the fusion
        back_feats = [None] * t
        back_vals = [None] * t
        for i in range(t - 1, -1, -1):
            s1 = s2 = None
            if i < t - 1:
                f1 = bwd_flow[:, i]
                s1 = (back_feats[i + 1], back_vals[i + 1], filled[:, i + 1], f1)
                if i < t - 2:
                    f2 = f1 + flow_warp(bwd_flow[:, i + 1], f1.permute(0, 2, 3, 1))
                    s2 = (back_feats[i + 2], back_vals[i + 2], f2)
            back_feats[i], back_vals[i] = self._step(
                "bwd", filled[:, i], m[:, i], z[:, i], s1, s2
            )

        # ---- forward propagation (0 -> t-1), reconstructing each frame on the fly
        coarse_out, cond_out, val_out = [], [], []
        p1 = p2 = None  # (state, validity) of frames i-1 and i-2
        for i in range(t):
            s1 = s2 = None
            if i > 0:
                f1 = fwd_flow[:, i - 1]
                s1 = (p1[0], p1[1], filled[:, i - 1], f1)
                if i > 1:
                    f2 = f1 + flow_warp(fwd_flow[:, i - 2], f1.permute(0, 2, 3, 1))
                    s2 = (p2[0], p2[1], f2)
            feat, val = self._step("fwd", filled[:, i], m[:, i], z[:, i], s1, s2)
            p2, p1 = p1, (feat, val)

            fused = self.fuse(torch.cat([back_feats[i], feat, back_vals[i], val], dim=1))
            fused = self.trunk(self.lrelu(fused))
            fused_hr = self.upsampler(fused)
            fused_hr = ckpt.checkpoint(self.hr_blocks, fused_hr, use_reentrant=False)
            hr_size = fused_hr.shape[-2:]

            base = F.interpolate(filled[:, i], size=hr_size, mode="bilinear", align_corners=False)
            coarse_out.append(torch.tanh(self.coarse_head(fused_hr) + base))
            cond_out.append(self.cond_head(fused_hr))
            val_out.append(
                F.interpolate(
                    torch.maximum(back_vals[i], val), size=hr_size, mode="bilinear", align_corners=False
                )
            )

        return {
            "coarse": torch.stack(coarse_out, dim=1),
            "cond": torch.stack(cond_out, dim=1),
            "hole_logits": hole_logits,
            "validity": torch.stack(val_out, dim=1),
            "flow_fwd": fwd_flow,
        }
