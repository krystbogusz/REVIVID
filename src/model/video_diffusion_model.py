"""REVIVID — restoration + 2x SR + inpainting of persistent holes.

Per clip ``lq`` (N, T, 3, h, w) in [-1, 1] — the ONLY input:

    backbone(lq) → coarse (N,T,3,H,W), cond, hole_logits (LR), validity, flow_fwd
    hole = sigmoid(hole_logits) at HR                (the model's own detection)
    G    = hole > threshold, dilated + feathered     (where the refiner may act)
    DDIM(refine_unet | coarse_{t-1,t,t+1}, hole, validity, cond) → residual
    refined = coarse + G * residual

Outside holes the output IS the coarse restoration. A clip in which the
detector finds no hole skips DDIM altogether.

Temporal consistency of the invented content (flicker gives a repair away
first): the UNet attends across the frames of the window, and the DDIM
starting noise moves with the scene (``warped_noise``).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ConditioningBackbone
from .config import ModelConfig
from .diffusion import GaussianDiffusion
from .flow import flow_warp
from .unet import ConditionalUNet


def _flat(x: torch.Tensor) -> torch.Tensor:
    """(N, T, ...) → (N*T, ...)."""
    return x.reshape(-1, *x.shape[2:])


class Video_Backbone(nn.Module):
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.cfg = cfg = config or ModelConfig()

        self.backbone = ConditioningBackbone(
            num_feat=cfg.num_feat,
            num_block=cfg.num_block,
            cond_dim=cfg.cond_dim,
            embed_dim=cfg.embed_dim,
            d_state=cfg.d_state,
            ssm_expand=cfg.ssm_expand,
            sr_scale=cfg.sr_scale,
            deg_dim=cfg.deg_dim,
            dcn_groups=cfg.dcn_groups,
            hole_threshold=cfg.hole_threshold,
        )
        self.diffusion = GaussianDiffusion(cfg.num_timesteps, cfg.min_snr_gamma)
        self.refine_unet = ConditionalUNet(
            in_channels=3,
            # coarse_t, coarse_{t-1}, coarse_{t+1}, hole, validity, backbone features
            cond_channels=3 * 3 + 1 + 1 + cfg.cond_dim,
            out_channels=3,
            base_channels=cfg.refiner_base,
            channel_mult=cfg.channel_mult,
            num_res_blocks=cfg.num_res_blocks,
            temporal_levels=tuple(cfg.refiner_temporal_levels),
        )
        # Running std(gt - coarse) inside the holes; a buffer, so it is saved
        # with the checkpoints and tracked by the EMA.
        self.register_buffer("residual_std", torch.tensor(float(cfg.residual_std_init)))
        self.register_buffer("residual_std_steps", torch.tensor(0, dtype=torch.long))

    # ------------------------------------------------------------------ holes

    def generation_mask(self, hole: torch.Tensor) -> torch.Tensor:
        """Where the refiner may change the image, (B, 1, H, W) in [0, 1]:
        ``hole`` (probability or {0, 1} mask, HR) thresholded, dilated by
        ``refine_mask_dilate`` px and feathered over half of that."""
        mask = (hole > self.cfg.hole_threshold).to(hole.dtype)
        k = int(self.cfg.refine_mask_dilate)
        if k > 0:
            mask = F.max_pool2d(mask, 2 * k + 1, 1, k)
            r = k // 2
            mask = F.avg_pool2d(mask, 2 * r + 1, 1, r, count_include_pad=False)
        return mask

    @torch.no_grad()
    def update_residual_std(self, residual: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Update the running std of ``residual`` inside ``mask`` and return it.
        The first ``residual_std_warmup`` updates take the batch std directly
        (the early coarse output is random, an EMA would lag far behind)."""
        cfg = self.cfg
        values = residual.detach().float()[mask.expand_as(residual) > 0]
        batch_std = values.std() if values.numel() > 1 else values.new_tensor(float("nan"))
        if torch.isfinite(batch_std) and batch_std > 0:
            self.residual_std_steps += 1
            if self.residual_std_steps <= cfg.residual_std_warmup:
                self.residual_std.copy_(batch_std)
            else:
                m = float(cfg.residual_std_momentum)
                self.residual_std.mul_(m).add_(batch_std, alpha=1.0 - m)
            self.residual_std.clamp_(min=float(cfg.residual_std_min))
        return self.residual_std

    # --------------------------------------------------------------- forward

    def forward(self, lq: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Backbone + the refiner's conditioning (built from the model's OWN
        hole detection; the true mask is only ever a training target)."""
        out = self.backbone(lq)
        coarse = out["coarse"]
        coarse_f = _flat(coarse)
        hole_logits_f = _flat(out["hole_logits"])
        hole_f = F.interpolate(
            torch.sigmoid(hole_logits_f.detach()), size=coarse_f.shape[-2:],
            mode="bilinear", align_corners=False,
        )
        c = coarse.detach()
        prev = torch.cat([c[:, :1], c[:, :-1]], dim=1)
        nxt = torch.cat([c[:, 1:], c[:, -1:]], dim=1)
        refine_cond = torch.cat(
            [_flat(c), _flat(prev), _flat(nxt), hole_f,
             _flat(out["validity"]).detach(), _flat(out["cond"])],
            dim=1,
        )
        return {
            "coarse": coarse,
            "coarse_f": coarse_f,
            "hole_logits_f": hole_logits_f,
            "hole_f": hole_f,
            "refine_cond": refine_cond,
            "flow_fwd": out["flow_fwd"],
        }

    # ------------------------------------------------------------- inference

    @staticmethod
    def warped_noise(flow_fwd: torch.Tensor, n: int, t: int, chw, device) -> torch.Tensor:
        """DDIM starting noise (N*T, C, H, W) that moves with the scene.

        Frame 0 draws fresh N(0, 1) noise; frame i takes frame i-1's noise
        warped by ``flow_fwd[:, i-1]`` (LR, maps frame i onto i-1, rescaled to
        HR). Nearest-neighbour sampling copies values instead of averaging them,
        so the noise stays exactly unit Gaussian; pixels coming from outside the
        frame get fresh noise.
        """
        c, h, w = chw
        noise = torch.empty(n, t, c, h, w, device=device)
        noise[:, 0] = torch.randn(n, c, h, w, device=device)
        for i in range(1, t):
            f = flow_fwd[:, i - 1]
            fh, fw = f.shape[-2:]
            if (fh, fw) != (h, w):
                f = F.interpolate(f, size=(h, w), mode="bilinear", align_corners=False)
                f = torch.stack([f[:, 0] * (w / fw), f[:, 1] * (h / fh)], dim=1)
            grid = f.permute(0, 2, 3, 1)
            prev = noise[:, i - 1]
            warped = flow_warp(prev, grid, interp_mode="nearest", padding_mode="zeros")
            inside = flow_warp(torch.ones_like(prev[:, :1]), grid, interp_mode="nearest", padding_mode="zeros")
            noise[:, i] = warped * inside + torch.randn_like(prev) * (1 - inside)
        return noise.reshape(n * t, c, h, w)

    @torch.no_grad()
    def restore_full(self, lq: torch.Tensor, refine_steps: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Full inference on a clip; every output is (N, T, ...):
        ``refined`` (the result), ``coarse``, ``hole`` (detected hole
        probability) and ``generation_mask`` (where the refiner acted)."""
        n, t = lq.shape[:2]
        out = self.forward(lq)
        coarse_f = out["coarse_f"]
        gen = self.generation_mask(out["hole_f"])

        refined = coarse_f
        if bool((gen > 0).any()):
            noise = self.warped_noise(out["flow_fwd"], n, t, coarse_f.shape[1:], coarse_f.device)
            residual = self.diffusion.ddim_sample(
                self.refine_unet, noise, refine_steps or self.cfg.refine_steps,
                cond=out["refine_cond"], num_frames=t,
            )
            refined = coarse_f + gen * residual * self.residual_std

        unflat = lambda x: x.reshape(n, t, *x.shape[1:])  # noqa: E731
        return {
            "refined": unflat(refined.clamp(-1.0, 1.0)),
            "coarse": out["coarse"],
            "hole": unflat(out["hole_f"]),
            "generation_mask": unflat(gen),
        }

    @torch.no_grad()
    def restore(self, lq: torch.Tensor, refine_steps: Optional[int] = None) -> torch.Tensor:
        """(N, T, 3, h, w) LQ → (N, T, 3, H, W) restored clip in [-1, 1]."""
        return self.restore_full(lq, refine_steps)["refined"]
