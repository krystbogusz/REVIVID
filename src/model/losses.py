"""Losses of REVIVID training — only what the realism objective needs.

Generator (coarse output = the final image everywhere outside holes):
    * ``CharbonnierLoss``         — fidelity to GT (robust L1); keeps content right.
    * ``VGGFeatures.perceptual``  — MambaOFR's VGG19 feature distance; keeps texture/structure right.
    * ``HingeGANLoss.generator``  — realism: the projected patch discriminator must
                                    not tell the output from real footage.
    * ``TemporalConsistencyLoss`` — no flicker: frame-to-frame change must follow
                                    the GT's own change along the GT motion.
Hole detector:
    * ``HoleDetectionLoss``       — BCE against the true hole mask.
Refiner: its v-prediction loss lives in ``GaussianDiffusion.training_loss``.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flow import flow_warp


class CharbonnierLoss(nn.Module):
    """Robust L1; ``weight`` re-weights pixels (weighted mean)."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        loss = torch.sqrt((pred - target) ** 2 + self.eps * self.eps)
        w = weight.expand_as(loss)
        return (loss * w).sum() / w.sum().clamp(min=1.0)


class VGGFeatures(nn.Module):
    """Frozen VGG19 of images in [-1, 1]; one forward per step feeds both users:

    * ``"perc"`` relu1_1 … relu5_1 — MambaOFR's perceptual loss (``VGGLoss_torch``:
      same layers, weights ``PERC_WEIGHTS``, the [-1, 1] input fed as is);
    * ``"disc"`` relu1_2, relu2_2, relu3_4, relu4_4 — the projected discriminator.
    """

    PERC_LAYERS = (1, 6, 11, 20, 29)
    PERC_WEIGHTS = (1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0)
    DISC_LAYERS = (3, 8, 17, 26)
    CHANNELS = (64, 128, 256, 512)  # of the "disc" maps

    def __init__(self):
        super().__init__()
        from torchvision import models

        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.layers = vgg[: max(self.PERC_LAYERS + self.DISC_LAYERS) + 1]
        for m in self.layers:
            if isinstance(m, nn.ReLU):
                m.inplace = False  # a tapped map must not be overwritten by the next ReLU
        self.requires_grad_(False)
        self.eval()

    def forward(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        x = x.float()
        feats = {"perc": [], "disc": []}
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.PERC_LAYERS:
                feats["perc"].append(x)
            if i in self.DISC_LAYERS:
                feats["disc"].append(x)
        return feats

    @classmethod
    def perceptual(cls, fake: List[torch.Tensor], real: List[torch.Tensor]) -> torch.Tensor:
        return sum(w * F.l1_loss(a, b.detach()) for w, a, b in zip(cls.PERC_WEIGHTS, fake, real))


class HingeGANLoss:
    """Hinge adversarial loss over a list of patch-logit maps (one per scale)."""

    @staticmethod
    def discriminator(real: List[torch.Tensor], fake: List[torch.Tensor]) -> torch.Tensor:
        return sum(F.relu(1.0 - r).mean() + F.relu(1.0 + f).mean() for r, f in zip(real, fake)) / len(real)

    @staticmethod
    def generator(fake: List[torch.Tensor]) -> torch.Tensor:
        return sum(-f.mean() for f in fake) / len(fake)


class TemporalConsistencyLoss(nn.Module):
    """Penalises flicker against the GT's own motion.

    With ``W`` = warp of frame t-1 onto frame t by the GT flow:
    ``|(out_t - W out_{t-1}) - (gt_t - W gt_{t-1})|`` — the restoration error may
    not change from frame to frame where the scene does not. Pixels whose GT
    change the motion does not explain (occlusions, flow errors) are masked out.
    """

    def __init__(self, occlusion_threshold: float = 0.05):
        super().__init__()
        self.occlusion_threshold = occlusion_threshold

    def forward(self, out: torch.Tensor, gt: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """``out``, ``gt``: (N, T, C, H, W); ``flow``: (N, T-1, 2, H, W),
        ``flow[:, k]`` maps frame k+1 onto frame k."""
        n, t, c, h, w = out.shape
        grid = flow.reshape(-1, 2, h, w).permute(0, 2, 3, 1)
        d_out = out[:, 1:].reshape(-1, c, h, w) - flow_warp(out[:, :-1].reshape(-1, c, h, w), grid)
        d_gt = gt[:, 1:].reshape(-1, c, h, w) - flow_warp(gt[:, :-1].reshape(-1, c, h, w), grid)
        mask = (d_gt.abs().mean(1, keepdim=True) < self.occlusion_threshold).to(out.dtype)
        return ((d_out - d_gt).abs() * mask).sum() / (mask.sum() * c).clamp(min=1.0)


class HoleDetectionLoss(nn.Module):
    """BCE of the hole detector; holes are a small pixel minority, so the hole
    class is up-weighted by ``pos_weight``."""

    def __init__(self, pos_weight: float = 10.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight)
