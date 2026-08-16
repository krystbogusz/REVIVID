"""Optical-flow estimation and warping.

REVIVID uses a single, real optical-flow estimator: torchvision's RAFT
(``raft_small``) with pretrained weights. The estimator starts frozen and is
unfrozen by the trainer after a warmup (MambaOFR recipe: fine-tune the flow on
degraded frames at a reduced learning rate once the rest of the network has
stabilised).
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
        x,
        vgrid_scaled,
        mode=interp_mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


class RAFTFlow(nn.Module):
    """torchvision RAFT wrapper returning flow_{a->b} as (n, 2, h, w).

    Frozen by default; the trainer calls :meth:`set_trainable` after the
    warmup phase to fine-tune the flow at a reduced learning rate.
    """

    def __init__(self):
        super().__init__()
        from torchvision.models.optical_flow import raft_small

        try:
            from torchvision.models.optical_flow import Raft_Small_Weights

            self.raft = raft_small(weights=Raft_Small_Weights.DEFAULT)
        except Exception:
            self.raft = raft_small(weights=None)

        self._trainable = False
        for p in self.raft.parameters():
            p.requires_grad_(False)
        self.eval()

    def set_trainable(self, flag: bool = True) -> None:
        """Enable/disable fine-tuning of the RAFT weights."""
        self._trainable = bool(flag)
        for p in self.raft.parameters():
            p.requires_grad_(self._trainable)

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

        grad_ok = self._trainable and torch.is_grad_enabled()
        with torch.set_grad_enabled(grad_ok):
            flow = self.raft(a.contiguous(), b.contiguous())[-1]

        if (H, W) != (h, w):
            flow = F.interpolate(
                flow, size=(h, w), mode="bilinear", align_corners=False
            )
            flow = flow.clone()
            flow[:, 0] *= w / W
            flow[:, 1] *= h / H
        return flow


def build_flow_estimator() -> nn.Module:
    return RAFTFlow()
