"""Optical-flow estimation and warping.

REVIVID uses a single, real optical-flow estimator: torchvision's RAFT
(``raft_small``) with pretrained weights. The estimator is frozen by default
(``finetune_flow: false``) and run under ``no_grad`` for speed / memory, mirroring
how the reference MambaOFR keeps its flow network fixed.
"""

from __future__ import annotations

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
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    return F.grid_sample(
        x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners
    )


class RAFTFlow(nn.Module):
    """torchvision RAFT wrapper returning flow_{a->b} as (n, 2, h, w)."""

    def __init__(self, finetune: bool = False):
        super().__init__()
        from torchvision.models.optical_flow import raft_small

        try:
            from torchvision.models.optical_flow import Raft_Small_Weights

            self.raft = raft_small(weights=Raft_Small_Weights.DEFAULT)
        except Exception:  # offline: build with random weights
            self.raft = raft_small(weights=None)

        self.finetune = finetune
        if not finetune:
            for p in self.raft.parameters():
                p.requires_grad_(False)

    # RAFT downsamples by 8 and the correlation pyramid needs feature maps >= 16,
    # so inputs must be at least 128 px and a multiple of 8 on each side.
    _MIN_SIZE = 128

    def _work_size(self, h: int, w: int):
        import math

        H = max(self._MIN_SIZE, math.ceil(h / 8) * 8)
        W = max(self._MIN_SIZE, math.ceil(w / 8) * 8)
        return H, W

    def forward(self, frame_a: torch.Tensor, frame_b: torch.Tensor) -> torch.Tensor:
        h, w = frame_a.shape[-2:]
        H, W = self._work_size(h, w)
        a, b = frame_a, frame_b
        if (H, W) != (h, w):
            a = F.interpolate(a, size=(H, W), mode="bilinear", align_corners=False)
            b = F.interpolate(b, size=(H, W), mode="bilinear", align_corners=False)

        ctx = torch.enable_grad() if (self.finetune and self.training) else torch.no_grad()
        with ctx:
            flow = self.raft(a.contiguous(), b.contiguous())[-1]  # (n, 2, H, W)

        if (H, W) != (h, w):
            flow = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False)
            flow = flow.clone()
            flow[:, 0] *= w / W
            flow[:, 1] *= h / H
        return flow


def build_flow_estimator(finetune: bool = False) -> nn.Module:
    return RAFTFlow(finetune=finetune)
