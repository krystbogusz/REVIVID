"""Optical-flow estimation (torchvision RAFT) and warping.

The backbone's RAFT starts frozen and is fine-tuned on degraded frames after a
warmup (``training.raft_unfreeze_iter``) at a reduced learning rate; the
trainer's GT-flow RAFT (flicker loss / metric) stays frozen.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def flow_warp(
    x: torch.Tensor,
    flow: torch.Tensor,
    interp_mode: str = "bilinear",
    padding_mode: str = "border",
    align_corners: bool = True,
) -> torch.Tensor:
    """Warp ``x`` (n, c, h, w) according to ``flow`` (n, h, w, 2) [dx, dy] in pixels."""
    n, _, h, w = x.size()
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h, device=x.device, dtype=x.dtype),
        torch.arange(0, w, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    grid = torch.stack((grid_x, grid_y), dim=2)[None].expand(n, -1, -1, -1)
    vgrid = grid + flow
    vgrid_x = 2.0 * vgrid[..., 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[..., 1] / max(h - 1, 1) - 1.0
    return F.grid_sample(
        x,
        torch.stack((vgrid_x, vgrid_y), dim=3),
        mode=interp_mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


class RAFTFlow(nn.Module):
    """torchvision RAFT (``large`` or ``small``, pretrained) returning
    flow_{a->b} as (n, 2, h, w) in pixels; inputs in [-1, 1].

    Frozen by default; :meth:`set_trainable` enables fine-tuning. Batch-norm
    statistics (RAFT large's context encoder) always stay frozen — the small
    training batches would corrupt the pretrained ones (as in RAFT's own
    fine-tuning, ``freeze_bn``).
    """

    MIN_SIZE = 128  # RAFT works at >= 128 px; smaller inputs are upsampled

    def __init__(self, variant: str = "large"):
        super().__init__()
        from torchvision.models import optical_flow as of

        if variant == "large":
            self.raft = of.raft_large(weights=of.Raft_Large_Weights.DEFAULT)
        elif variant == "small":
            self.raft = of.raft_small(weights=of.Raft_Small_Weights.DEFAULT)
        else:
            raise ValueError(f"unknown RAFT variant {variant!r} (large | small)")
        self.set_trainable(False)
        self.eval()

    def set_trainable(self, flag: bool = True) -> None:
        self._trainable = bool(flag)
        self.raft.requires_grad_(self._trainable)

    def train(self, mode: bool = True):
        super().train(mode)
        for m in self.raft.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
        return self

    def forward(self, frame_a: torch.Tensor, frame_b: torch.Tensor) -> torch.Tensor:
        h, w = frame_a.shape[-2:]
        H = max(self.MIN_SIZE, math.ceil(h / 8) * 8)
        W = max(self.MIN_SIZE, math.ceil(w / 8) * 8)
        a, b = frame_a, frame_b
        if (H, W) != (h, w):
            a = F.interpolate(a, size=(H, W), mode="bilinear", align_corners=False)
            b = F.interpolate(b, size=(H, W), mode="bilinear", align_corners=False)

        with torch.set_grad_enabled(self._trainable and torch.is_grad_enabled()):
            flow = self.raft(a.contiguous(), b.contiguous())[-1]

        if (H, W) != (h, w):
            flow = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False)
            flow = torch.stack([flow[:, 0] * (w / W), flow[:, 1] * (h / H)], dim=1)
        return flow


def build_flow_estimator(variant: str = "large") -> RAFTFlow:
    return RAFTFlow(variant)
