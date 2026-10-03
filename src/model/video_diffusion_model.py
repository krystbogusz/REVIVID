"""REVIVID v4 — restoration + 2x SR + inpainting of persistent holes.

Pipeline (per clip ``lq`` of shape (N, T, 3, h, w), values in [-1, 1]):

    backbone(lq) → coarse (N,T,3,H,W), cond (N,T,C,H,W),
                   hole_logits (N,T,1,h,w), validity (N,T,1,H,W)
    hole = sigmoid(hole_logits) upsampled to HR      (the model's OWN detection)

    cond_refine = cat[coarse_t, coarse_{t-1}, coarse_{t+1}, hole, validity, cond]
    DDIM(refine_unet, cond_refine) → residual
    refined = coarse + G * residual,  G = dilated, feathered hole mask

The refiner only acts inside holes (G = 0 everywhere else), so outside them the
output IS the coarse restoration and the refiner cannot cost any PSNR there.
A clip in which the detector finds no hole skips DDIM altogether. Inside holes
``validity`` tells the refiner which pixels the propagation already recovered
from other frames and which ones it has to hallucinate.

The refiner is conditioned on the previous/next coarse frames and DDIM starts
from the SAME noise for every frame of a clip, both of which keep the
hallucinated content consistent from frame to frame.

``forward`` returns the tensors the trainer builds the losses from; ``restore``
runs the full inference path (detection, DDIM inside holes, compositing).
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
    """Restoration + 2x SR + hole inpainting (refiner restricted to holes)."""

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
            deg_dim=config.deg_dim,
            dcn_groups=config.dcn_groups,
            hole_threshold=config.hole_threshold,
        )

        self.diffusion = GaussianDiffusion(
            config.num_timesteps,
            schedule=config.schedule,
            min_snr_gamma=config.min_snr_gamma,
        )

        # coarse_t + coarse_{t-1} + coarse_{t+1} + hole + validity + backbone features
        cond_ch = 3 * 3 + 1 + 1 + config.cond_dim

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

        # Running estimate of std(gt - coarse) inside the holes. The diffusion
        # works on the residual divided by this, so the target always lands in
        # the noise schedule's native ~N(0, 1) range no matter how good
        # `coarse` gets. Registered as a buffer so it rides along in
        # state_dict() -> saved with every checkpoint and tracked by ModelEMA.
        self.register_buffer(
            "residual_std", torch.tensor(float(config.residual_std_init))
        )
        self.register_buffer("residual_std_steps", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def update_residual_std(
        self, residual: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Update the residual scale from a training batch, return it.

        Only the pixels the refiner works on (``mask > 0``) count — inside a
        hole the residual is far larger than in the rest of the frame. Without
        any such pixel the estimate is left unchanged.

        For the first `residual_std_warmup` updates the batch std is used
        directly instead of the EMA. Early on the coarse branch is random, so
        the residual is far larger than its steady state; an EMA seeded at
        `residual_std_init` would lag badly behind and mis-scale the target.
        """
        cfg = self.cfg
        values = residual.detach().float()
        if mask is not None:
            sel = mask.expand_as(values) > 0
            if not bool(sel.any()):
                return self.residual_std
            values = values[sel]
        batch_std = values.std()
        if torch.isfinite(batch_std) and batch_std > 0:
            self.residual_std_steps += 1
            if self.residual_std_steps <= cfg.residual_std_warmup:
                self.residual_std.copy_(batch_std)
            else:
                m = float(cfg.residual_std_momentum)
                self.residual_std.mul_(m).add_(batch_std, alpha=1.0 - m)
            self.residual_std.clamp_(min=float(cfg.residual_std_min))
        return self.residual_std

    @staticmethod
    def hole_probability(hole_logits_f: torch.Tensor, size) -> torch.Tensor:
        """(N*T, 1, h, w) detector logits → (N*T, 1, H, W) hole probability."""
        return F.interpolate(
            torch.sigmoid(hole_logits_f), size=size, mode="bilinear", align_corners=False
        )

    def generation_mask(self, hole: torch.Tensor) -> torch.Tensor:
        """Where the refiner may change the image, (N, 1, H, W) in [0, 1].

        ``hole`` (a probability or a {0, 1} mask at HR) is thresholded,
        dilated by ``refine_mask_dilate`` px and feathered over half of that,
        so the original hole stays fully covered and the seam is soft.
        """
        mask = (hole > self.cfg.hole_threshold).to(hole.dtype)
        k = int(self.cfg.refine_mask_dilate)
        if k > 0:
            mask = F.max_pool2d(mask, 2 * k + 1, 1, k)
            r = k // 2
            mask = F.avg_pool2d(mask, 2 * r + 1, 1, r, count_include_pad=False)
        return mask

    def _build_cond(
        self,
        coarse: torch.Tensor,
        hole_f: torch.Tensor,
        validity_f: torch.Tensor,
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
        return torch.cat([coarse_f, prev_f, nxt_f, hole_f, validity_f, cond_f], dim=1)

    def forward(self, lq: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for training: the tensors the losses are built from.

        The refiner is conditioned on the model's OWN hole detection — the true
        mask is only ever a training target (see the trainer).
        """
        out = self.backbone(lq)
        coarse = out["coarse"]

        coarse_f, _ = _flatten_time(coarse)
        cond_f, _ = _flatten_time(out["cond"])
        validity_f, _ = _flatten_time(out["validity"])
        logits_f, _ = _flatten_time(out["hole_logits"])

        hole_f = self.hole_probability(logits_f.detach(), coarse_f.shape[-2:])
        refine_cond = self._build_cond(
            coarse.detach(), hole_f, validity_f.detach(), cond_f
        )

        return {
            "coarse": coarse,
            "coarse_f": coarse_f,
            "hole_logits_f": logits_f,
            "hole_f": hole_f,
            "refine_cond": refine_cond,
        }

    @torch.no_grad()
    def restore_full(
        self, lq: torch.Tensor, refine_steps: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Full inference on a clip; returns every intermediate as (N, T, ...).

        Keys: ``refined`` (output), ``coarse``, ``hole`` (detected hole
        probability) and ``generation_mask`` (where the refiner acted).
        """
        refine_steps = refine_steps or self.cfg.refine_steps

        out = self.forward(lq)
        coarse_f = out["coarse_f"]
        n, t = lq.shape[:2]
        shape = coarse_f.shape

        gen = self.generation_mask(out["hole_f"])
        if bool((gen > 0).any()):
            # Share the initial DDIM noise across all frames of a clip: with the
            # deterministic (eta=0) sampler this keeps the hallucinated content
            # consistent between frames instead of flickering.
            noise = torch.randn((n, 1, *shape[1:]), device=coarse_f.device)
            noise = noise.expand(n, t, *shape[1:]).reshape(shape)
            residual = self.diffusion.ddim_sample(
                self.refine_unet,
                shape,
                refine_steps,
                model_kwargs={"cond": out["refine_cond"]},
                device=coarse_f.device,
                x_init=noise,
            )
            # The diffusion works on the residual normalised by residual_std.
            refined = torch.clamp(coarse_f + gen * residual * self.residual_std, -1.0, 1.0)
        else:
            refined = coarse_f.clamp(-1.0, 1.0)

        nt = (n, t)
        return {
            "refined": _unflatten_time(refined, nt),
            "coarse": out["coarse"],
            "hole": _unflatten_time(out["hole_f"], nt),
            "generation_mask": _unflatten_time(gen, nt),
        }

    @torch.no_grad()
    def restore(
        self,
        lq: torch.Tensor,
        refine_steps: Optional[int] = None,
        return_coarse: bool = False,
    ):
        """Restore a clip: (N, T, 3, h, w) LQ → (N, T, 3, H, W) in [-1, 1].

        With ``return_coarse=True`` returns ``(refined, coarse)``.
        """
        r = self.restore_full(lq, refine_steps)
        if return_coarse:
            return r["refined"], r["coarse"]
        return r["refined"]


def build_model(config: Optional[ModelConfig] = None, **kwargs) -> Video_Backbone:
    return Video_Backbone(config=config, **kwargs)


def _selftest_losses(
    net: "Video_Backbone",
    lq: torch.Tensor,
    gt: torch.Tensor,
    hole_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Mirror the trainer's loss wiring for a quick forward/backward smoke test."""
    out = net(lq)

    n, t, c, hr_h, hr_w = gt.shape
    gt_f = gt.reshape(n * t, c, hr_h, hr_w)
    hole_lr_f = hole_mask.reshape(n * t, 1, *hole_mask.shape[-2:])
    hole_hr_f = F.interpolate(hole_lr_f, size=(hr_h, hr_w), mode="nearest")
    gen = net.generation_mask(hole_hr_f)

    residual = gt_f - out["coarse_f"]
    residual_target = (residual / net.update_residual_std(residual, gen)).detach()

    loss_pix = CharbonnierLoss()(out["coarse"], gt)
    loss_detect = HoleDetectionLoss()(out["hole_logits_f"], hole_lr_f)
    loss_v, _ = DiffusionLoss()(
        net.diffusion, net.refine_unet, residual_target, out["refine_cond"], loss_mask=gen
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
    hole = torch.zeros(n, t, 1, h, w)
    hole[..., 8:16, 8:20] = 1.0
    lq = lq.masked_fill(hole.expand_as(lq) > 0, -1.0)
    losses = _selftest_losses(net, lq, gt, hole)
    total = losses["pix"] + losses["detect"] + losses["v"]
    total.backward()
    print(
        "restoration losses:",
        {k: float(v.detach()) for k, v in losses.items()},
    )

    with torch.no_grad():
        y = net.restore(lq)
    print("restore output:", tuple(y.shape))
