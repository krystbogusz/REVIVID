"""Loss functions for REVIVID training (pure diffusion model, no GAN).

* ``charbonnier_loss`` - robust L1 used for the coarse restoration.
* ``VGGPerceptualLoss`` - VGG19 feature loss (torchvision weights downloaded on
  first use; falls back to an untrained network offline).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.sqrt((pred - target) ** 2 + eps * eps).mean()


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss over a few VGG19 feature maps. Inputs are in [-1, 1]."""

    def __init__(self, layers=(2, 7, 16, 25), resize: bool = False):
        super().__init__()
        from torchvision import models

        try:
            weights = models.VGG19_Weights.IMAGENET1K_V1
            vgg = models.vgg19(weights=weights).features
        except Exception:
            vgg = models.vgg19(weights=None).features  # offline fallback

        self.layers = set(layers)
        self.slices = nn.ModuleList()
        prev = 0
        max_layer = max(self.layers)
        block = []
        modules = list(vgg.children())[: max_layer + 1]
        for idx, module in enumerate(modules):
            block.append(module)
            if idx in self.layers:
                self.slices.append(nn.Sequential(*block))
                block = []
        for p in self.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.resize = resize
        self.eval()

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1.0) / 2.0  # [-1,1] -> [0,1]
        x = (x - self.mean) / self.std
        if self.resize:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return x

    @torch.autocast("cuda", enabled=False)
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Always run VGG in float32 — deep layers (e.g. layer 25) can overflow
        # fp16 range under AMP, producing Inf activations and NaN gradients.
        pred = self._prep(pred.float())
        # GT never needs a gradient – detach before the frozen VGG forward
        # to prevent PyTorch building a redundant backward graph.
        target = self._prep(target.detach().float())
        loss = pred.new_zeros(())
        x, y = pred, target
        for slc in self.slices:
            x, y = slc(x), slc(y)
            loss = loss + F.l1_loss(x, y)
        return loss
