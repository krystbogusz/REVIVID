"""Loss functions for REVIVID training (pure diffusion model, no GAN).

* ``CharbonnierLoss`` - robust L1 used for the coarse restoration.
* ``VGGPerceptualLoss`` - VGG19 feature loss.
* ``HoleDetectionLoss`` - BCE for the persistent-hole detector.
* ``DiffusionLoss`` - wrapper for V-prediction diffusion step.
* ``FocalFrequencyLoss`` - L1 distance in spectral/frequency domain to combat oversmoothing.
* ``MaskedReconstructionLoss`` - targeted Charbonnier for missing VFI frames.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((pred - target) ** 2 + self.eps * self.eps).mean()


class HoleDetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, target_mask)


class MaskedReconstructionLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, frame_mask: torch.Tensor
    ) -> torch.Tensor:
        n, t, c, h, w = pred.shape
        pred_f = pred.reshape(n * t, c, h, w)
        target_f = target.reshape(n * t, c, h, w)
        mask_f = frame_mask.reshape(-1)

        missing = ~mask_f
        if not missing.any():
            return pred.new_zeros(())

        pred_missing = pred_f[missing]
        target_missing = target_f[missing]

        return torch.sqrt(
            (pred_missing - target_missing) ** 2 + self.eps * self.eps
        ).mean()


class DiffusionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        diffusion_obj,
        refine_unet,
        residual_target: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        loss, _ = diffusion_obj.training_losses(
            refine_unet,
            residual_target,
            model_kwargs={"cond": cond},
        )
        return loss


class FocalFrequencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.autocast("cuda", enabled=False)
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_fft = torch.fft.rfft2(pred.float(), norm="ortho")
        target_fft = torch.fft.rfft2(target.float(), norm="ortho")

        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)

        return F.l1_loss(pred_amp, target_amp)


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss over a few VGG19 feature maps. Inputs are in [-1, 1]."""

    def __init__(self, layers=(2, 7, 16, 25), resize: bool = False):
        super().__init__()
        from torchvision import models

        try:
            weights = models.VGG19_Weights.IMAGENET1K_V1
            vgg = models.vgg19(weights=weights).features
        except Exception:
            vgg = models.vgg19(weights=None).features

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
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        self.resize = resize
        self.eval()

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1.0) / 2.0
        x = (x - self.mean) / self.std
        if self.resize:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return x

    @torch.autocast("cuda", enabled=False)
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        pred = self._prep(pred.float())

        target = self._prep(target.detach().float())
        loss = pred.new_zeros(())
        x, y = pred, target
        for slc in self.slices:
            x, y = slc(x), slc(y)
            loss = loss + F.l1_loss(x, y)
        return loss
