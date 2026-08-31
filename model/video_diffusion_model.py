"""DiffMambaOFR — diffusion restoration network.

One diffusion head handles two tasks:

    Restoration + SR : degraded LR frames    → clean HR frames
    Inpainting       : persistent spatial holes → hallucinated content

Pipeline (per clip ``lq`` of shape (N, T, 3, h, w), values in [-1, 1]):

    backbone(lq) → coarse (N,T,3,H,W), cond (N,T,C,H,W),
                   hole_logits (N,T,1,H,W)
    hole_mask = sigmoid(hole_logits) > threshold

    cond_refine = cat[coarse_t, coarse_{t-1}, coarse_{t+1}, hole_mask, cond]
    DDIM(refine_unet, cond_refine) → residual → refined = coarse + residual

The refiner is conditioned on the previous/next coarse frames (temporal
context) and DDIM starts from the SAME noise for every frame of a clip, both
of which suppress frame-to-frame flicker in the refined detail.

``compute_losses`` is called by the trainer; ``restore`` runs DDIM at inference.

Note on AttnBlock: the refine_unet is built with ``attn_levels=()`` (no spatial
self-attention) so that inference on arbitrary resolutions stays memory-bounded.
Global context is already provided by the Mamba backbone through ``cond``.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ConditioningBackbone
from .config import ModelConfig
from .diffusion import GaussianDiffusion
from .losses import CharbonnierLoss, DiffusionLoss, HoleDetectionLoss
from .unet import ConditionalUNet


def _flatten_time(x: torch.Tensor):
    n, t = x.shape[:2]
    return x.reshape(n * t, *x.shape[2:]), (n, t)


def _unflatten_time(x: torch.Tensor, nt) -> torch.Tensor:
    n, t = nt
    return x.reshape(n, t, *x.shape[1:])


class Video_Backbone(nn.Module):
    """Diffusion restoration model (restoration + SR + hole inpainting)."""

    def __init__(self, config: Optional[ModelConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = ModelConfig(
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k in ModelConfig.__dataclass_fields__
                }
            )
        self.cfg = config

        self.backbone = ConditioningBackbone(
            num_feat=config.num_feat,
            num_block=config.num_block,
            cond_dim=config.cond_dim,
            embed_dim=config.embed_dim,
            d_state=config.d_state,
            ssm_expand=config.ssm_expand,
            sr_scale=config.sr_scale,
        )

        self.diffusion = GaussianDiffusion(
            config.num_timesteps,
            schedule=config.schedule,
            min_snr_gamma=config.min_snr_gamma,
        )

        # coarse_t + coarse_{t-1} + coarse_{t+1} + hole_mask + backbone features
        cond_ch = 3 * 3 + 1 + config.cond_dim

        self.refine_unet = ConditionalUNet(
            in_channels=3,
            cond_channels=cond_ch,
            out_channels=3,
            base_channels=config.refiner_base,
            channel_mult=config.channel_mult,
            num_res_blocks=config.num_res_blocks,
            attn_levels=(),
            use_checkpoint=True,
        )

        # Running estimate of std(gt - coarse). The diffusion works on the
        # residual divided by this, so the target always lands in the noise
        # schedule's native ~N(0, 1) range no matter how good `coarse` gets.
        # Registered as a buffer so it rides along in state_dict() -> it is
        # saved with every checkpoint and tracked by ModelEMA for free.
        self.register_buffer(
            "residual_std", torch.tensor(float(config.residual_std_init))
        )
        self.register_buffer("residual_std_steps", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def update_residual_std(self, residual: torch.Tensor) -> torch.Tensor:
        """Update the residual scale from a training batch, return it.

        For the first `residual_std_warmup` iterations the batch std is used
        directly instead of the EMA. Early on the coarse branch is random, so
        the residual is far larger than its steady state; an EMA seeded at
        `residual_std_init` would lag badly behind and mis-scale the target in
        the opposite direction. The batch std is computed over millions of
        pixels, so it is a low-variance estimate and safe to use raw.
        """
        cfg = self.cfg
        self.residual_std_steps += 1
        batch_std = residual.detach().float().std()
        if torch.isfinite(batch_std) and batch_std > 0:
            if self.residual_std_steps <= cfg.residual_std_warmup:
                self.residual_std.copy_(batch_std)
            else:
                m = float(cfg.residual_std_momentum)
                self.residual_std.mul_(m).add_(batch_std, alpha=1.0 - m)
            self.residual_std.clamp_(min=float(cfg.residual_std_min))
        return self.residual_std

    def _build_cond(
        self,
        coarse: torch.Tensor,
        hole_mask_f: torch.Tensor,
        cond_f: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate all conditioning signals for the refine_unet.

        ``coarse`` is the UNFLATTENED (N, T, 3, H, W) coarse output — the
        previous/next frames are appended as temporal context (edge frames
        repeat themselves).
        """
        prev = torch.cat([coarse[:, :1], coarse[:, :-1]], dim=1)
        nxt = torch.cat([coarse[:, 1:], coarse[:, -1:]], dim=1)
        coarse_f, _ = _flatten_time(coarse)
        prev_f, _ = _flatten_time(prev)
        nxt_f, _ = _flatten_time(nxt)
        return torch.cat([coarse_f, prev_f, nxt_f, hole_mask_f, cond_f], dim=1)

    def forward(self, lq: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        Returns unweighted predictions and building blocks for the loss functions.
        """
        out = self.backbone(lq)
        coarse, cond, hole_logits = out["coarse"], out["cond"], out["hole_logits"]

        coarse_f, nt = _flatten_time(coarse)
        cond_f, _ = _flatten_time(cond)
        logits_f, _ = _flatten_time(hole_logits)

        lq_f, _ = _flatten_time(lq)
        lq_hr = (
            F.interpolate(lq_f, size=coarse_f.shape[-2:], mode="nearest")
            if lq_f.shape[-2:] != coarse_f.shape[-2:]
            else lq_f
        )
        hole_mask_f = (lq_hr.mean(dim=1, keepdim=True) < -0.95).float()

        refine_cond = self._build_cond(coarse.detach(), hole_mask_f, cond_f)

        return {
            "coarse": coarse,
            "coarse_f": coarse_f,
            "hole_logits_f": logits_f,
            "hole_mask_f": hole_mask_f,
            "refine_cond": refine_cond,
        }

    @torch.no_grad()
    def restore(
        self,
        lq: torch.Tensor,
        refine_steps: Optional[int] = None,
        return_coarse: bool = False,
    ) -> torch.Tensor:
        """Run DDIM restoration on a clip.

        Args:
            lq:         (N, T, 3, h, w) LQ input.
            refine_steps: DDIM steps (default: cfg.refine_steps).
            return_coarse: also return the backbone's coarse output (before
                        diffusion refinement) for diagnostics.

        Returns:
            (N, T, 3, H, W) restored HR clip in [-1, 1];
            with ``return_coarse=True`` a tuple ``(refined, coarse)``.
        """
        refine_steps = refine_steps or self.cfg.refine_steps

        out = self.backbone(lq)
        coarse, cond, hole_logits = out["coarse"], out["cond"], out["hole_logits"]

        coarse_f, nt = _flatten_time(coarse)
        cond_f, _ = _flatten_time(cond)
        logits_f, _ = _flatten_time(hole_logits)

        device = coarse_f.device
        shape = coarse_f.shape

        hole_mask_f = (torch.sigmoid(logits_f) > self.cfg.hole_threshold).float()

        lq_f, _ = _flatten_time(lq)
        lq_hr = (
            F.interpolate(lq_f, size=coarse_f.shape[-2:], mode="nearest")
            if lq_f.shape[-2:] != coarse_f.shape[-2:]
            else lq_f
        )

        fill_holes = (lq_hr.mean(dim=1, keepdim=True) < -0.95).float()
        hole_mask_f = torch.maximum(hole_mask_f, fill_holes)

        refine_cond = self._build_cond(coarse, hole_mask_f, cond_f)

        # Share the initial DDIM noise across all frames of a clip: with the
        # deterministic (eta=0) sampler this makes the hallucinated detail
        # consistent between frames instead of flickering.
        n, t = nt
        noise = torch.randn((n, 1, *shape[1:]), device=device)
        noise = noise.expand(n, t, *shape[1:]).reshape(shape)

        residual = self.diffusion.ddim_sample(
            self.refine_unet,
            shape,
            refine_steps,
            model_kwargs={"cond": refine_cond},
            device=device,
            x_init=noise,
        )
        # The diffusion operates on the residual normalised by residual_std
        # (see trainer) — undo that normalisation before compositing.
        residual = residual * self.residual_std
        refined = torch.clamp(coarse_f + residual, -1.0, 1.0)
        if return_coarse:
            return _unflatten_time(refined, nt), _unflatten_time(coarse_f, nt)
        return _unflatten_time(refined, nt)


def build_model(config: Optional[ModelConfig] = None, **kwargs) -> Video_Backbone:
    return Video_Backbone(config=config, **kwargs)


def _selftest_losses(
    net: "Video_Backbone",
    lq: torch.Tensor,
    gt: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Mirror the trainer's loss wiring for a quick forward/backward smoke test."""
    out = net(lq)

    n, t, c, hr_h, hr_w = gt.shape
    gt_f = gt.reshape(n * t, c, hr_h, hr_w)
    residual = gt_f - out["coarse_f"]
    residual_target = (residual / net.update_residual_std(residual)).detach()

    loss_pix = CharbonnierLoss()(out["coarse"], gt)
    loss_detect = HoleDetectionLoss()(out["hole_logits_f"], out["hole_mask_f"])
    loss_v, _ = DiffusionLoss()(
        net.diffusion, net.refine_unet, residual_target, out["refine_cond"]
    )
    return {"pix": loss_pix, "detect": loss_detect, "v": loss_v}


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = ModelConfig(
        num_timesteps=50,
        refine_steps=2,
        num_block=1,
        embed_dim=32,
        d_state=8,
    )
    net = Video_Backbone(cfg)
    n, t, h, w = 1, 4, 32, 32
    hr = h * cfg.sr_scale

    lq = torch.randn(n, t, 3, h, w).clamp(-1, 1)
    gt = torch.randn(n, t, 3, hr, hr).clamp(-1, 1)
    losses = _selftest_losses(net, lq, gt)
    total = losses["pix"] + losses["detect"] + losses["v"]
    total.backward()
    print(
        "restoration losses:",
        {k: float(v.detach()) for k, v in losses.items()},
    )

    with torch.no_grad():
        y = net.restore(lq)
    print("restore output:", tuple(y.shape))
