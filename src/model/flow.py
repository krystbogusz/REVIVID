"""Optical-flow estimation (torchvision RAFT, frozen) and warping.

The backbone's RAFT is frozen as in MambaOFR (its flow runs under no_grad);
the trainer's GT-flow RAFT (flicker metric) is frozen too.
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
    """Frozen torchvision RAFT returning flow_{a->b} as (n, 2, h, w) in pixels;
    inputs in [-1, 1]. Variants: ``large`` (C_T_SKHT_V2), ``sintel`` (large
    with C_T_SKHT_V1 = the original ``raft-sintel.pth`` MambaOFR uses),
    ``small``. ``iters`` = refinement iterations (MambaOFR: 24, torchvision
    default 12).

    Always in eval mode: the batch-norm statistics of RAFT large's context
    encoder stay the pretrained ones.
    """

    MIN_SIZE = 128  # RAFT works at >= 128 px; smaller inputs are upsampled

    def __init__(self, variant: str = "large", iters: int = 12):
        super().__init__()
        from torchvision.models import optical_flow as of

        if variant == "large":
            self.raft = of.raft_large(weights=of.Raft_Large_Weights.DEFAULT)
        elif variant == "sintel":
            self.raft = of.raft_large(weights=of.Raft_Large_Weights.C_T_SKHT_V1)
        elif variant == "small":
            self.raft = of.raft_small(weights=of.Raft_Small_Weights.DEFAULT)
        else:
            raise ValueError(f"unknown RAFT variant {variant!r} (large | sintel | small)")
        self.iters = int(iters)
        self.raft.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, frame_a: torch.Tensor, frame_b: torch.Tensor) -> torch.Tensor:
        h, w = frame_a.shape[-2:]
        H = max(self.MIN_SIZE, math.ceil(h / 8) * 8)
        W = max(self.MIN_SIZE, math.ceil(w / 8) * 8)
        a, b = frame_a, frame_b
        if (H, W) != (h, w):
            a = F.interpolate(a, size=(H, W), mode="bilinear", align_corners=False)
            b = F.interpolate(b, size=(H, W), mode="bilinear", align_corners=False)

        flow = self.raft(a.contiguous(), b.contiguous(), num_flow_updates=self.iters)[-1]

        if (H, W) != (h, w):
            flow = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False)
            flow = torch.stack([flow[:, 0] * (w / W), flow[:, 1] * (h / H)], dim=1)
        return flow


def build_flow_estimator(variant: str = "large", iters: int = 12) -> RAFTFlow:
    return RAFTFlow(variant, iters)
