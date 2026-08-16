"""Loss functions for REVIVID training (pure diffusion model, no GAN).

Every loss is its own ``nn.Module`` class — there are no free-standing helper
functions:

* ``CharbonnierLoss`` - robust L1 used for the coarse restoration.
* ``VGGPerceptualLoss`` - VGG19 feature loss.
* ``HoleDetectionLoss`` - BCE for the persistent-hole detector.
* ``DiffusionLoss`` - wrapper for V-prediction diffusion step.
* ``FocalFrequencyLoss`` - L1 distance in spectral/frequency domain to combat oversmoothing.
* ``GradientLoss`` - L1 on Sobel gradients; penalises soft/blurry edges directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Robust L1. ``weight`` optionally re-weights pixels (e.g. boost holes);
    the result is a weighted mean so the loss scale stays comparable."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = torch.sqrt((pred - target) ** 2 + self.eps * self.eps)
        if weight is None:
            return loss.mean()
        w = weight.expand_as(loss)
        return (loss * w).sum() / w.sum().clamp(min=1.0)


class HoleDetectionLoss(nn.Module):
    """BCE for the persistent-hole detector.

    Holes cover only a small fraction of pixels, so plain BCE is dominated by
    the background and the detector ends up under-firing. ``pos_weight``
    up-weights the positive (hole) class to compensate.
    """

    def __init__(self, pos_weight: float = 10.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))

    def forward(self, logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, target_mask, pos_weight=self.pos_weight
        )


class DiffusionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        diffusion_obj,
        refine_unet,
        residual_target: torch.Tensor,
        cond: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss, info = diffusion_obj.training_losses(
            refine_unet,
            residual_target,
            model_kwargs={"cond": cond},
            loss_mask=loss_mask,
        )
        return loss, info


class FocalFrequencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_fft = torch.fft.rfft2(pred.float(), norm="ortho")
        target_fft = torch.fft.rfft2(target.float(), norm="ortho")

        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)

        return F.l1_loss(pred_amp, target_amp)


class GradientLoss(nn.Module):
    """L1 on Sobel image gradients.

    Plain pixel losses (Charbonnier / MSE) tolerate blur — a soft edge is only
    "slightly wrong" per pixel. Comparing spatial gradients instead makes blur
    expensive: a blurry edge has much weaker gradient magnitude than a sharp
    one, so this loss directly pushes the output towards crisp edges.
    """

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", kx.t().contiguous().view(1, 1, 3, 3))

    def _grad(self, x: torch.Tensor):
        c = x.shape[1]
        kx = self.kx.expand(c, 1, 3, 3).contiguous()
        ky = self.ky.expand(c, 1, 3, 3).contiguous()
        gx = F.conv2d(x, kx, padding=1, groups=c)
        gy = F.conv2d(x, ky, padding=1, groups=c)
        return gx, gy

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pgx, pgy = self._grad(pred)
        tgx, tgy = self._grad(target)
        return F.l1_loss(pgx, tgx) + F.l1_loss(pgy, tgy)


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

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        pred = self._prep(pred.float())

        target = self._prep(target.detach().float())
        loss = pred.new_zeros(())
        x, y = pred, target
        for slc in self.slices:
            x, y = slc(x), slc(y)
            loss = loss + F.l1_loss(x, y)
        return loss
