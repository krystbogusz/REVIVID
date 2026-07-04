"""Gaussian diffusion utilities for REVIVID.

A compact, dependency-free implementation that supports:
    * a cosine noise schedule,
    * the v-prediction parameterization (stable, used for both the residual
      refinement and the generative inpainting heads),
    * deterministic DDIM sampling with an arbitrary number of inference steps.

The diffusion operates directly in image / residual space (no latent VAE), so
the whole pipeline stays self contained and runnable on CPU as well as CUDA.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn


def cosine_beta_schedule(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule proposed in Nichol & Dhariwal (2021)."""
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 1e-8, 0.999)


def _extract(arr: torch.Tensor, t: torch.Tensor, broadcast_shape) -> torch.Tensor:
    """Gather schedule values at timesteps ``t`` and broadcast to ``broadcast_shape``."""
    out = arr.to(device=t.device)[t].float()
    while out.dim() < len(broadcast_shape):
        out = out[..., None]
    return out.expand(broadcast_shape)


class GaussianDiffusion(nn.Module):
    """v-prediction Gaussian diffusion with DDIM sampling.

    The denoiser is supplied externally as ``model_fn(x_t, t, **cond) -> v_pred``.
    """

    def __init__(self, num_timesteps: int = 1000, schedule: str = "cosine"):
        super().__init__()
        self.num_timesteps = int(num_timesteps)

        if schedule == "cosine":
            betas = cosine_beta_schedule(self.num_timesteps)
        elif schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, self.num_timesteps, dtype=torch.float64)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).float())
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod).float()
        )

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_acp = _extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_1m = _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_acp * x_start + sqrt_1m * noise

    def get_v_target(
        self, x_start: torch.Tensor, noise: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """v = sqrt(acp) * noise - sqrt(1-acp) * x_start."""
        sqrt_acp = _extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_1m = _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_acp * noise - sqrt_1m * x_start

    def predict_start_from_v(
        self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        sqrt_acp = _extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_1m = _extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return sqrt_acp * x_t - sqrt_1m * v

    def predict_noise_from_v(
        self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        sqrt_acp = _extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_1m = _extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return sqrt_1m * x_t + sqrt_acp * v

    def training_losses(
        self,
        model_fn: Callable[..., torch.Tensor],
        x_start: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        model_kwargs: Optional[dict] = None,
        noise: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ):
        """Return (v_loss, info) for a v-prediction objective.

        ``loss_mask`` optionally restricts the loss to certain spatial regions
        (used by the inpainting head to focus on the holes).
        """
        model_kwargs = model_kwargs or {}
        b = x_start.shape[0]
        if t is None:
            t = torch.randint(0, self.num_timesteps, (b,), device=x_start.device).long()
        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = self.q_sample(x_start, t, noise=noise)
        v_target = self.get_v_target(x_start, noise, t)
        v_pred = model_fn(x_t, t, **model_kwargs)

        if loss_mask is not None:
            diff = (v_pred - v_target) ** 2 * loss_mask
            denom = loss_mask.sum().clamp(min=1.0)
            v_loss = diff.sum() / denom
        else:
            v_loss = ((v_pred - v_target) ** 2).mean()

        x0_pred = self.predict_start_from_v(x_t, t, v_pred)
        return v_loss, {"x0_pred": x0_pred, "x_t": x_t, "t": t, "v_pred": v_pred}

    def _ddim_timesteps(self, num_steps: int) -> torch.Tensor:
        num_steps = max(1, min(num_steps, self.num_timesteps))
        step = self.num_timesteps / num_steps
        ts = (torch.arange(num_steps) * step).round().long()
        ts = ts.clamp(max=self.num_timesteps - 1)
        return torch.flip(ts, dims=[0])

    @torch.no_grad()
    def ddim_sample(
        self,
        model_fn: Callable[..., torch.Tensor],
        shape,
        num_steps: int,
        model_kwargs: Optional[dict] = None,
        device: Optional[torch.device] = None,
        x_init: Optional[torch.Tensor] = None,
        eta: float = 0.0,
        return_x0_only: bool = True,
    ):
        model_kwargs = model_kwargs or {}
        device = device or self.betas.device
        x = torch.randn(shape, device=device) if x_init is None else x_init

        timesteps = self._ddim_timesteps(num_steps).to(device)
        for i, t_cur in enumerate(timesteps):
            t_batch = torch.full(
                (shape[0],), int(t_cur), device=device, dtype=torch.long
            )
            v_pred = model_fn(x, t_batch, **model_kwargs)
            x0 = self.predict_start_from_v(x, t_batch, v_pred)
            eps = self.predict_noise_from_v(x, t_batch, v_pred)

            if i == len(timesteps) - 1:
                x = x0
                break

            t_next = timesteps[i + 1]
            acp_next = self.alphas_cumprod[t_next]
            sqrt_acp_next = acp_next.sqrt()
            sqrt_1m_next = (1.0 - acp_next).sqrt()

            if eta > 0:
                acp_cur = self.alphas_cumprod[t_cur]
                sigma = (
                    eta
                    * torch.sqrt((1 - acp_next) / (1 - acp_cur))
                    * torch.sqrt(1 - acp_cur / acp_next)
                )
                noise = torch.randn_like(x)
                x = (
                    sqrt_acp_next * x0
                    + torch.sqrt((sqrt_1m_next**2 - sigma**2).clamp(min=0)) * eps
                    + sigma * noise
                )
            else:
                x = sqrt_acp_next * x0 + sqrt_1m_next * eps

        return x if return_x0_only else (x, x0)
