"""DiffMambaOFR — Unified Masked Frame Prediction network.

One diffusion head handles all three tasks:

    Restoration  : observed (degraded) frames → clean HR frames
    VFI          : masked (missing) frames    → interpolated HR frames
    Inpainting   : persistent spatial holes   → hallucinated content

Pipeline (per clip ``lq`` of shape (N, T, 3, h, w), values in [-1, 1]):

    frame_mask (N, T) bool  — True = observed, False = masked for VFI
    lq[:, mask==False] = 0  — zeros at VFI positions

    backbone(lq, frame_mask) → coarse (N,T,3,H,W), cond (N,T,C,H,W),
                                hole_logits (N,T,1,H,W)
    hole_mask = sigmoid(hole_logits) > threshold

    cond_refine = cat[coarse, hole_mask, cond, frame_mask_emb]
    DDIM(refine_unet, cond_refine) → residual → refined = coarse + residual

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
    """Unified Masked Frame Prediction model (restoration + SR + VFI)."""

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
            config.num_timesteps, schedule=config.schedule
        )

        self.mask_embed = nn.Embedding(2, config.mask_embed_dim)

        cond_ch = 3 + 1 + config.cond_dim + config.mask_embed_dim

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

    def _build_cond(
        self,
        coarse_f: torch.Tensor,
        hole_mask_f: torch.Tensor,
        cond_f: torch.Tensor,
        frame_mask_f: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate all conditioning signals for the refine_unet."""
        h, w = coarse_f.shape[-2:]

        mask_emb = self.mask_embed(frame_mask_f.long())
        mask_emb = mask_emb[:, :, None, None].expand(-1, -1, h, w)
        return torch.cat([coarse_f, hole_mask_f, cond_f, mask_emb], dim=1)

    def forward(
        self,
        lq: torch.Tensor,
        frame_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        Returns unweighted predictions and building blocks for the loss functions.
        """
        if frame_mask is None:
            frame_mask = lq.new_ones(lq.shape[:2], dtype=torch.bool)

        out = self.backbone(lq, frame_mask=frame_mask)
        coarse, cond, hole_logits = out["coarse"], out["cond"], out["hole_logits"]

        coarse_f, nt = _flatten_time(coarse)
        cond_f, _ = _flatten_time(cond)
        logits_f, _ = _flatten_time(hole_logits)

        frame_mask_f = frame_mask.reshape(-1)

        lq_f, _ = _flatten_time(lq)
        lq_hr = (
            F.interpolate(lq_f, size=coarse_f.shape[-2:], mode="nearest")
            if lq_f.shape[-2:] != coarse_f.shape[-2:]
            else lq_f
        )
        hole_mask_f = (lq_hr.mean(dim=1, keepdim=True) < -0.95).float()

        refine_cond = self._build_cond(
            coarse_f.detach(), hole_mask_f, cond_f, frame_mask_f
        )

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
        frame_mask: Optional[torch.Tensor] = None,
        refine_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Run DDIM restoration / interpolation on a clip.

        Args:
            lq:         (N, T, 3, h, w) LQ input; zeros at VFI positions.
            frame_mask: (N, T) bool — True = observed, False = masked.
                        None → all frames observed (pure restoration mode).
            refine_steps: DDIM steps (default: cfg.refine_steps).

        Returns:
            (N, T, 3, H, W) restored/interpolated HR clip in [-1, 1].
        """
        refine_steps = refine_steps or self.cfg.refine_steps

        if frame_mask is None:
            frame_mask = lq.new_ones(lq.shape[:2], dtype=torch.bool)

        out = self.backbone(lq, frame_mask=frame_mask)
        coarse, cond, hole_logits = out["coarse"], out["cond"], out["hole_logits"]

        coarse_f, nt = _flatten_time(coarse)
        cond_f, _ = _flatten_time(cond)
        logits_f, _ = _flatten_time(hole_logits)
        frame_mask_f = frame_mask.reshape(-1)

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

        refine_cond = self._build_cond(coarse_f, hole_mask_f, cond_f, frame_mask_f)
        residual = self.diffusion.ddim_sample(
            self.refine_unet,
            shape,
            refine_steps,
            model_kwargs={"cond": refine_cond},
            device=device,
        )
        refined = torch.clamp(coarse_f + residual, -1.0, 1.0)
        return _unflatten_time(refined, nt)


def build_model(config: Optional[ModelConfig] = None, **kwargs) -> Video_Backbone:
    return Video_Backbone(config=config, **kwargs)


def _selftest_losses(
    net: "Video_Backbone",
    lq: torch.Tensor,
    gt: torch.Tensor,
    frame_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Mirror the trainer's loss wiring for a quick forward/backward smoke test."""
    out = net(lq, frame_mask)

    n, t, c, hr_h, hr_w = gt.shape
    gt_f = gt.reshape(n * t, c, hr_h, hr_w)
    residual_target = (gt_f - out["coarse_f"]).detach()

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

    mask = torch.ones(n, t, dtype=torch.bool)
    mask[0, 2] = False
    lq_vfi = lq.clone()
    lq_vfi[:, 2] = 0.0

    net.zero_grad()
    losses_vfi = _selftest_losses(net, lq_vfi, gt, frame_mask=mask)
    total_vfi = losses_vfi["pix"] + losses_vfi["detect"] + losses_vfi["v"]
    total_vfi.backward()
    print(
        "VFI losses:",
        {k: float(v.detach()) for k, v in losses_vfi.items()},
    )

    with torch.no_grad():
        y = net.restore(lq_vfi, frame_mask=mask)
    print("restore output:", tuple(y.shape))
