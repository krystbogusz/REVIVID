"""Losses of REVIVID training.

Generator (MambaOFR's loss; phase 2 adds the detector term):
    * ``CharbonnierLoss``         — fidelity to GT (robust L1); pixels can be re-weighted
                                    (phase 2: true-hole pixels weigh more).
    * ``VGGFeatures.perceptual``  — MambaOFR's VGG19 feature distance.
    * ``HingeGANLoss.generator``  — realism against MambaOFR's 3-D clip discriminator.
    * ``HoleDetectionLoss``       — BCE of the hole detector against the true mask.
Validation:
    * ``TemporalConsistencyLoss`` — the flicker metric.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flow import flow_warp


class CharbonnierLoss(nn.Module):
    """Robust L1; ``weight`` re-weights pixels (weighted mean)."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        loss = torch.sqrt((pred - target) ** 2 + self.eps * self.eps)
        if weight is None:
            return loss.mean()
        w = weight.expand_as(loss)
        return (loss * w).sum() / w.sum().clamp(min=1.0)


class VGGFeatures(nn.Module):
    """Frozen VGG19 of images in [-1, 1]: relu1_1 … relu5_1, MambaOFR's
    perceptual loss (``VGGLoss_torch``: same layers, weights ``WEIGHTS``,
    the [-1, 1] input fed as is)."""

    LAYERS = (1, 6, 11, 20, 29)
    WEIGHTS = (1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0)

    def __init__(self):
        super().__init__()
        from torchvision import models

        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.layers = vgg[: max(self.LAYERS) + 1]
        for m in self.layers:
            if isinstance(m, nn.ReLU):
                m.inplace = False  # a tapped map must not be overwritten by the next ReLU
        self.requires_grad_(False)
        self.eval()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = x.float()
        feats = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.LAYERS:
                feats.append(x)
        return feats

    @classmethod
    def perceptual(cls, fake: List[torch.Tensor], real: List[torch.Tensor]) -> torch.Tensor:
        return sum(w * F.l1_loss(a, b.detach()) for w, a, b in zip(cls.WEIGHTS, fake, real))


class HingeGANLoss:
    """MambaOFR's hinge adversarial loss on patch-logit maps."""

    @staticmethod
    def discriminator(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        return 0.5 * (F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean())

    @staticmethod
    def generator(fake: torch.Tensor) -> torch.Tensor:
        return -fake.mean()


class TemporalConsistencyLoss(nn.Module):
    """Flicker against the GT's own motion.

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
