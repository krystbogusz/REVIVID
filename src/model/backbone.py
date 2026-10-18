"""Restoration + 2x SR backbone: MambaOFR (CVPR 2025) one to one plus 2x SR,
prepared for the inpainting of persistent holes (burned-in regions that stay
for the whole clip).

The network is MambaOFR's ``RNN_Swin_4.Video_Backbone`` - same modules, same
forward, run on the LR input - with the SR its own tail is built for: that tail
comes from BasicVSR, with the PixelShuffle upsampler ``up1`` and the global
residual ``img_up`` set to scale 1; here they are set to ``sr_scale``.

Per clip ``lrs`` (N, T, 3, h, w) in [-1, 1]:

1. RAFT optical flow between neighbouring LR frames (MambaOFR: raft-sintel,
   24 iterations, frozen).
2. Bidirectional first-order recurrent propagation; per frame and direction:
     ``proj``      conv3x3 of the current frame (latent feature)
     ``align``     flow-guided deformable alignment of the previous state
                   (MambaOFR's ``DCNv2PackFlowGuided``)
     ``agg``       mask generator M from (aligned, latent, residual indicator,
                   warped previous M); state = M * aligned + (1 - M) * latent -
                   keeps scratches / dirt of single frames from propagating
     ``restorer``  MambaIR with the degradation-aware prompt blocks
3. Reconstruction (MambaOFR's tail): fuse both directions, ``up1`` conv +
   PixelShuffle x ``sr_scale``, two convs, + the bilinear-upsampled input, tanh.

Inpainting (phase 2, ``use_holes``; off in phase 1, where the model is exactly
the above) adds no weights to MambaOFR's modules: ``HoleDetector`` finds the
holes at LR (nobody tells the model where they are), ``push_pull_fill``
pre-fills them so the flow is not pinned to 0, ``FlowCompletion`` re-estimates
the flow inside them, the mask M is floored at the hole probability (the state
is carried over the hole, content seen in other frames survives until it is
used) and a validity map tracks how much of the state comes from real
observations - conditioning of the refiner.

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

from .flow import build_flow_estimator, flow_warp
from .mamba_blocks import MambaRestorer


def _lrelu() -> nn.Module:
    return nn.LeakyReLU(0.1, inplace=True)


def _warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """MambaOFR's ``flow_warp``: bilinear, zeros outside the frame."""
    return flow_warp(x, flow.permute(0, 2, 3, 1), padding_mode="zeros")


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


def rgb_to_luma(x: torch.Tensor) -> torch.Tensor:
    """Luma channel from an RGB tensor in [-1, 1] (or [0, 1]). Keeps the range."""
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


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


class FlowGuidedDCN(nn.Module):
    """MambaOFR's ``DCNv2PackFlowGuided`` (deformable groups 1, no modulation).

    Offsets = the flow + a residual (<= ``max_residue`` px) predicted from the
    flow-warped state, the latent feature and the flow. MambaOFR's code added
    the flow of the first clip of the batch, vertically flipped, in (dx, dy)
    order to every clip; here each clip gets its own flow as the (dy, dx) that
    torchvision's deform_conv2d reads.
    """

    def __init__(self, ch: int = 16, max_residue: float = 10.0):
        super().__init__()
        self.max_residue = max_residue
        self.conv_offset = nn.Sequential(
            nn.Conv2d(2 * ch + 2, ch, 3, 1, 1), _lrelu(),
            nn.Conv2d(ch, ch, 3, 1, 1), _lrelu(),
            nn.Conv2d(ch, ch, 3, 1, 1), _lrelu(),
            nn.Conv2d(ch, 18, 3, 1, 1),
        )
        nn.init.zeros_(self.conv_offset[-1].weight)
        nn.init.zeros_(self.conv_offset[-1].bias)
        self.weight = nn.Parameter(torch.empty(ch, ch, 3, 3))
        self.bias = nn.Parameter(torch.zeros(ch))
        nn.init.uniform_(self.weight, -1.0 / math.sqrt(ch * 9), 1.0 / math.sqrt(ch * 9))

    def forward(self, feat, warped, curr, flow):
        """``feat`` the unwarped previous state, ``warped`` it warped by ``flow``."""
        offset = self.max_residue * torch.tanh(self.conv_offset(torch.cat([warped, curr, flow], dim=1)))
        offset = offset + flow.flip(1).repeat(1, offset.size(1) // 2, 1, 1)
        return deform_conv2d(feat, offset, self.weight, self.bias, padding=1)


class MaskGenerator(nn.Module):
    """MambaOFR's ``Gated_Aggregation`` mask: M from (state, latent, residual
    indicator |warp(previous frame) - current frame|, warped previous M)."""

    def __init__(self, ch: int = 16):
        super().__init__()
        c_in = 2 * ch + 2
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_in, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c_in, ch // 2, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch // 2, 1, 3, 1, 1), nn.Sigmoid(),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, state, curr, res_ind, prev_mask):
        return self.net(torch.cat([state, curr, res_ind, prev_mask], dim=1))


_DIRS = ("bwd", "fwd")  # "forward" would clash with nn.Module.forward


class ConditioningBackbone(nn.Module):
    def __init__(
        self,
        num_feat: int = 16,
        embed_dim: int = 64,
        depths: Sequence[int] = (2, 2, 2),
        d_state: int = 16,
        ssm_expand: float = 1.2,
        dyn_kernels: int = 8,
        drop_path: float = 0.1,
        cond_dim: int = 64,
        sr_scale: int = 2,
        hole_threshold: float = 0.9,
        raft: str = "sintel",
        raft_iters: int = 24,
        use_holes: bool = False,
    ):
        super().__init__()
        self.sr_scale = sr_scale
        self.hole_threshold = hole_threshold
        # Phase 2 switch (not saved in the state dict, set from ModelConfig).
        self.use_holes = use_holes
        c = num_feat

        self.hole_detector = HoleDetector()
        self.flow_net = build_flow_estimator(raft, raft_iters)
        self.flow_completion = FlowCompletion()

        self.proj = nn.ModuleDict({d: nn.Conv2d(3, c, 3, 1, 1) for d in _DIRS})
        self.align = nn.ModuleDict({d: FlowGuidedDCN(c) for d in _DIRS})
        self.agg = nn.ModuleDict({d: MaskGenerator(c) for d in _DIRS})
        self.restorer = nn.ModuleDict(
            {d: MambaRestorer(c, embed_dim, tuple(depths), d_state, ssm_expand, dyn_kernels, drop_path) for d in _DIRS}
        )
        for d in _DIRS:
            nn.init.kaiming_normal_(self.proj[d].weight)

        self.concate = nn.Conv2d(2 * c, c, 3, 1, 1)
        # MambaOFR's PSUpsample(c, c, scale): conv to c * scale^2 + PixelShuffle.
        self.up1 = nn.Sequential(nn.Conv2d(c, c * sr_scale**2, 3, 1, 1), nn.PixelShuffle(sr_scale))
        self.up2 = nn.Conv2d(c, c, 3, 1, 1)
        self.conv_hr = nn.Conv2d(c, c, 3, 1, 1)
        self.conv_last = nn.Conv2d(c, 3, 3, 1, 1)
        self.cond_head = nn.Conv2d(c, cond_dim, 3, 1, 1)
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
        """One propagation step of MambaOFR in direction ``d``; ``prev = (state,
        mask, validity, frame, flow)`` of the previous frame in propagation
        order (``None`` at the start). Returns (state, mask, validity)."""
        curr = self.proj[d](frame)
        if prev is None:
            # MambaOFR's head: the state is the latent feature; the mask is
            # computed from a zero state, zero indicator and an all-ones mask.
            zeros = torch.zeros_like(curr)
            mask = self.agg[d](zeros, curr, torch.zeros_like(hole), torch.ones_like(hole))
            feat, validity = curr, 1.0 - hole
        else:
            p_feat, p_mask, p_val, p_frame, flow = prev
            warped = _warp(p_feat, flow)
            res_ind = torch.abs(_warp(p_frame, flow)[:, :1] - frame[:, :1])
            mask_w = _warp(p_mask, flow)
            aligned = self.align[d](p_feat, warped, curr, flow)
            mask = torch.maximum(self.agg[d](aligned, curr, res_ind, mask_w), hole)
            feat = mask * aligned + (1 - mask) * curr
            validity = mask * _warp(p_val, flow) + (1 - mask) * (1 - hole)
        return self.restorer[d](feat), mask, validity

    def forward(self, lrs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            lrs: (N, T, 3, h, w) LQ input.

        Returns ``coarse``, ``cond``, ``validity`` at the output resolution and
        ``hole_logits``, ``flow_fwd`` at the input (LR) resolution.
        """
        n, t, c, h, w = lrs.size()

        if self.use_holes:
            hole_logits = self.hole_detector(lrs)
            m = torch.sigmoid(hole_logits).detach()
            x = push_pull_fill(
                lrs.reshape(n * t, c, h, w), 1.0 - (m > self.hole_threshold).to(lrs.dtype).reshape(n * t, 1, h, w)
            ).view(n, t, c, h, w)
        else:
            hole_logits = lrs.new_full((n, t, 1, h, w), -20.0)
            m = lrs.new_zeros(n, t, 1, h, w)
            x = lrs

        fwd_flow, bwd_flow = self._comp_flow(x, m)

        # ---- backward propagation (t-1 -> 0), states kept for the fusion
        back = [None] * t
        for i in range(t - 1, -1, -1):
            prev = (*back[i + 1], x[:, i + 1], bwd_flow[:, i]) if i < t - 1 else None
            back[i] = self._step("bwd", x[:, i], m[:, i], prev)

        # ---- forward propagation (0 -> t-1), reconstructing each frame on the fly
        coarse_out, cond_out, val_out = [], [], []
        state = None
        for i in range(t):
            prev = (*state, x[:, i - 1], fwd_flow[:, i - 1]) if i > 0 else None
            state = self._step("fwd", x[:, i], m[:, i], prev)
            (b_feat, _, b_val), (f_feat, _, f_val) = back[i], state

            y = self.lrelu(self.concate(torch.cat([b_feat, f_feat], dim=1)))
            y = self.lrelu(self.up1(y))
            y = self.lrelu(self.up2(y))
            y = self.lrelu(self.conv_hr(y))
            hr = y.shape[-2:]
            base = F.interpolate(x[:, i], size=hr, mode="bilinear", align_corners=False)  # MambaOFR's img_up
            coarse_out.append(torch.tanh(self.conv_last(y) + base))
            cond_out.append(self.cond_head(y))
            val_out.append(F.interpolate(torch.maximum(b_val, f_val), size=hr, mode="bilinear", align_corners=False))

        return {
            "coarse": torch.stack(coarse_out, dim=1),
            "cond": torch.stack(cond_out, dim=1),
            "hole_logits": hole_logits,
            "validity": torch.stack(val_out, dim=1),
            "flow_fwd": fwd_flow,
        }
