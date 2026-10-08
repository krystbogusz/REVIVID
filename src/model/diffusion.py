"""Gaussian diffusion of the refiner: cosine schedule, v-prediction, DDIM.

The diffusion runs directly in residual space (gt - coarse, normalised), no
latent VAE. The denoiser is supplied externally as
``model_fn(x_t, t, **model_kwargs) -> v_pred``.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn


def cosine_beta_schedule(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule of Nichol & Dhariwal (2021)."""
    x = torch.linspace(0, num_timesteps, num_timesteps + 1, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-8, 0.999)


def _extract(arr: torch.Tensor, t: torch.Tensor, shape) -> torch.Tensor:
    """Schedule values at timesteps ``t``, broadcast to ``shape``."""
    out = arr.to(device=t.device)[t].float()
    while out.dim() < len(shape):
        out = out[..., None]
    return out.expand(shape)


class GaussianDiffusion(nn.Module):
    def __init__(self, num_timesteps: int = 1000, min_snr_gamma: float = 5.0):
        super().__init__()
        self.num_timesteps = int(num_timesteps)
        # Min-SNR-gamma loss weighting (Hang et al. 2023); 0 disables it.
        self.min_snr_gamma = float(min_snr_gamma)

        alphas_cumprod = torch.cumprod(1.0 - cosine_beta_schedule(self.num_timesteps), dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).float())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod).float())

    def training_loss(
        self,
        model_fn: Callable[..., torch.Tensor],
        x_start: torch.Tensor,
        t: torch.Tensor,
        loss_mask: torch.Tensor,
        **model_kwargs,
    ) -> torch.Tensor:
        """v-prediction MSE inside ``loss_mask``.

        Each sample's loss is the mask-weighted mean, Min-SNR weighted, and the
        batch average runs over the samples whose mask is non-empty — frames
        without anything to learn do not dilute it.
        """
        noise = torch.randn_like(x_start)
        sqrt_acp = _extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_1m = _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        x_t = sqrt_acp * x_start + sqrt_1m * noise
        v_target = sqrt_acp * noise - sqrt_1m * x_start
        v_pred = model_fn(x_t, t, **model_kwargs)

        mask = loss_mask.expand_as(v_pred)
        mask_sum = mask.flatten(1).sum(1)
        per_sample = ((v_pred - v_target) ** 2 * mask).flatten(1).sum(1) / mask_sum.clamp(min=1.0)
        if self.min_snr_gamma > 0:
            acp = self.alphas_cumprod[t]
            snr = acp / (1.0 - acp).clamp(min=1e-8)
            per_sample = per_sample * snr.clamp(max=self.min_snr_gamma) / (snr + 1.0)
        return per_sample.sum() / (mask_sum > 0).sum().clamp(min=1)

    @torch.no_grad()
    def ddim_sample(
        self,
        model_fn: Callable[..., torch.Tensor],
        x_init: torch.Tensor,
        num_steps: int,
        **model_kwargs,
    ) -> torch.Tensor:
        """Deterministic DDIM (eta = 0) from pure noise ``x_init``; returns the
        clean sample.

        Trailing timestep spacing (Lin et al. 2024): the first step is t = T-1,
        where the training input really is (almost) pure noise — e.g. 999, 874,
        ..., 124 for 8 of 1000 steps. Leading spacing (0, 125, ..., 875) would
        start at t = 875, where training inputs still hold ~19 % signal under
        the cosine schedule, so pure noise there is out of distribution.
        """
        num_steps = max(1, min(num_steps, self.num_timesteps))
        stride = self.num_timesteps / num_steps
        ts = [int(round(self.num_timesteps - i * stride)) - 1 for i in range(num_steps)]

        x = x_init
        for i, t_cur in enumerate(ts):
            t = torch.full((x.shape[0],), t_cur, device=x.device, dtype=torch.long)
            v = model_fn(x, t, **model_kwargs)
            sqrt_acp = _extract(self.sqrt_alphas_cumprod, t, x.shape)
            sqrt_1m = _extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
            x0 = sqrt_acp * x - sqrt_1m * v
            if i == len(ts) - 1:
                return x0
            eps = sqrt_1m * x + sqrt_acp * v
            acp_next = self.alphas_cumprod[ts[i + 1]]
            x = acp_next.sqrt() * x0 + (1.0 - acp_next).sqrt() * eps
        return x
