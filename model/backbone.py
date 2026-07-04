"""Degradation-aware bidirectional recurrent conditioning backbone.

Unified Masked Frame Prediction (MFP) variant — handles three tasks:

1. **Restoration** — all T frames are observed (degraded LR); backbone propagates
   features bidirectionally and produces a coarse HR restoration.
2. **VFI** — some frames carry ``frame_mask=False`` (entire frame is missing).
   The backbone propagates through those positions using its SSM state without
   anchoring to an observed frame. RAFT receives linearly-interpolated synthetic
   frames at masked positions so that optical flow stays physically meaningful.
3. **Spatial inpainting** — persistent holes are baked into the LQ video by the
   DatasetCreator (fill value −1.0). The backbone sees those pixels and the
   ``hole_head`` produces per-pixel logits used as explicit conditioning for the
   diffusion UNet.

Outputs (all at HR resolution ``H = h*sr_scale, W = w*sr_scale``):
    * ``coarse``      — (N, T, 3, H, W) first-pass restored/interpolated frames.
    * ``cond``        — (N, T, cond_dim, H, W) backbone conditioning features.
    * ``hole_logits`` — (N, T, 1, H, W) persistent-hole detector logits.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.checkpoint as ckpt

from .blocks import ResidualBlockNoBN, make_layer
from .flow import build_flow_estimator, flow_warp
from .mamba_blocks import build_feature_blocks


def build_upsampler(in_ch: int, scale: int) -> nn.Module:
    """Sub-pixel (PixelShuffle) upsampler. ``scale`` must be a power of two."""
    if scale <= 1:
        return nn.Identity()
    if scale & (scale - 1) != 0:
        raise ValueError(f"sr_scale must be a power of two, got {scale}")
    layers = []
    s = scale
    while s > 1:
        layers += [
            nn.Conv2d(in_ch, in_ch * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, inplace=True),
        ]
        s //= 2
    return nn.Sequential(*layers)


def rgb_to_luma(x: torch.Tensor) -> torch.Tensor:
    """Luma channel from an RGB tensor in [-1, 1] (or [0, 1]). Keeps the range."""
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


class GatedAggregation(nn.Module):
    """Fuse the warped hidden state with the current-frame feature.

    The gate is conditioned on a *residual indicator* (how inconsistent the
    warped neighbour is with the current frame) and a propagated confidence
    mask, so that unreliable, heavily-degraded regions rely more on the freshly
    extracted feature than on the (possibly corrupt) propagated state.
    """

    def __init__(self, num_feat: int):
        super().__init__()
        self.proj_curr = nn.Conv2d(3, num_feat, 3, 1, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(num_feat * 2 + 2, num_feat, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_feat, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden, curr_lr, residual_indicator, pre_mask, is_head: bool):
        feat_curr = self.proj_curr(curr_lr)
        if is_head:
            return feat_curr, torch.ones_like(residual_indicator)
        x = torch.cat([hidden, feat_curr, residual_indicator, pre_mask], dim=1)
        gate = self.gate(x)
        fused = gate * hidden + (1 - gate) * feat_curr
        return fused, gate


class ConditioningBackbone(nn.Module):
    def __init__(
        self,
        num_feat: int = 16,
        num_block: int = 6,
        cond_dim: int = 64,
        embed_dim: int = 64,
        d_state: int = 16,
        ssm_expand: int = 2,
        sr_scale: int = 1,
    ):
        super().__init__()
        self.num_feat = num_feat
        self.cond_dim = cond_dim
        self.sr_scale = sr_scale

        self.flow_net = build_flow_estimator()

        self.backward_agg = GatedAggregation(num_feat)
        self.forward_agg = GatedAggregation(num_feat)
        blk_kwargs = dict(
            embed_dim=embed_dim, depth=num_block, d_state=d_state, expand=ssm_expand
        )
        self.backward_blocks = build_feature_blocks(num_feat, **blk_kwargs)
        self.forward_blocks = build_feature_blocks(num_feat, **blk_kwargs)

        self.fuse = nn.Conv2d(num_feat * 2, num_feat * 2, 3, 1, 1)
        self.trunk = make_layer(ResidualBlockNoBN, 2, num_feat=num_feat * 2)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

        self.upsampler = build_upsampler(num_feat * 2, sr_scale)

        self.coarse_head = nn.Sequential(
            nn.Conv2d(num_feat * 2, num_feat * 2, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(num_feat * 2, 3, 3, 1, 1),
        )
        self.cond_head = nn.Conv2d(num_feat * 2, cond_dim, 3, 1, 1)

        self.hole_head = nn.Sequential(
            nn.Conv2d(num_feat * 2 + 1, num_feat, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(num_feat, 1, 3, 1, 1),
        )

    def _make_effective_frames(
        self, lrs: torch.Tensor, frame_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return a version of ``lrs`` where masked positions are filled with
        a linear interpolation of their nearest observed neighbours.

        RAFT always receives physically meaningful frames instead of zeros,
        producing sensible optical flow even around masked positions.

        Args:
            lrs:        (N, T, C, H, W) — original LQ frames (zeros at masked).
            frame_mask: (N, T) bool     — True = observed, False = masked.

        Returns:
            lrs_eff: (N, T, C, H, W) — lrs with masked positions interpolated.
        """
        n, t = frame_mask.shape

        obs = frame_mask[0].tolist()

        if all(obs):
            return lrs

        lrs_eff = lrs.clone()
        for i in range(t):
            if obs[i]:
                continue

            prev_obs = next((j for j in range(i - 1, -1, -1) if obs[j]), None)

            next_obs = next((j for j in range(i + 1, t) if obs[j]), None)

            if prev_obs is not None and next_obs is not None:
                alpha = (i - prev_obs) / (next_obs - prev_obs)
                lrs_eff[:, i] = (1.0 - alpha) * lrs[:, prev_obs] + alpha * lrs[
                    :, next_obs
                ]
            elif prev_obs is not None:
                lrs_eff[:, i] = lrs[:, prev_obs]
            elif next_obs is not None:
                lrs_eff[:, i] = lrs[:, next_obs]

        return lrs_eff

    def _comp_flow(self, lrs_eff: torch.Tensor):
        """Return (forward_flow, backward_flow), each (N, T-1, 2, H, W).

        ``lrs_eff`` must already have masked positions replaced with their
        interpolated neighbours (see ``_make_effective_frames``).
        """
        n, t, c, h, w = lrs_eff.size()
        a = lrs_eff[:, 1:].reshape(-1, c, h, w)
        b = lrs_eff[:, :-1].reshape(-1, c, h, w)
        fwd = self.flow_net(a, b).view(n, t - 1, 2, h, w)
        bwd = self.flow_net(b, a).view(n, t - 1, 2, h, w)
        return fwd, bwd

    def forward(
        self,
        lrs: torch.Tensor,
        frame_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            lrs:        (N, T, 3, H_lr, W_lr)  LQ input; zeros at masked positions.
            frame_mask: (N, T) bool             True = observed, False = masked.
                        None → all frames observed (pure restoration mode).

        Returns dict with keys ``coarse``, ``cond``, ``hole_logits`` — all at
        the SR output resolution (H_lr * sr_scale, W_lr * sr_scale).
        """
        n, t, c, h, w = lrs.size()

        if frame_mask is None:
            frame_mask = lrs.new_ones((n, t), dtype=torch.bool)

        obs: list[bool] = frame_mask[0].tolist()

        lrs_eff = self._make_effective_frames(lrs, frame_mask)
        fwd_flow, bwd_flow = self._comp_flow(lrs_eff)

        zero_feat = lrs.new_zeros(n, self.num_feat, h, w)
        zero_ind = lrs.new_zeros(n, 1, h, w)

        back_feats = [None] * t
        residual_acc = [zero_ind] * t
        feat_prop = zero_feat
        pre_mask = torch.ones_like(zero_ind)

        for i in range(t - 1, -1, -1):
            if i < t - 1:
                flow = bwd_flow[:, i].permute(0, 2, 3, 1)
                feat_prop = flow_warp(feat_prop, flow)
                pre_mask = flow_warp(pre_mask, flow)

                warped_pix = flow_warp(lrs_eff[:, i + 1], flow)
                res_ind = torch.abs(
                    rgb_to_luma(warped_pix) - rgb_to_luma(lrs_eff[:, i])
                )

                if obs[i]:

                    feat_prop, gate = self.backward_agg(
                        feat_prop, lrs[:, i], res_ind, pre_mask, is_head=False
                    )
                    pre_mask = gate

            else:

                res_ind = zero_ind
                if obs[i]:
                    feat_prop, _ = self.backward_agg(
                        None, lrs[:, i], res_ind, pre_mask, is_head=True
                    )
                else:
                    feat_prop = zero_feat

            feat_prop = ckpt.checkpoint(
                self.backward_blocks, feat_prop, use_reentrant=False
            )
            back_feats[i] = feat_prop
            residual_acc[i] = res_ind

        coarse_out, cond_out, hole_out = [], [], []
        feat_prop = zero_feat
        pre_mask = torch.ones_like(zero_ind)

        for i in range(t):
            if i > 0:
                flow = fwd_flow[:, i - 1].permute(0, 2, 3, 1)
                feat_prop = flow_warp(feat_prop, flow)
                pre_mask = flow_warp(pre_mask, flow)

                warped_pix = flow_warp(lrs_eff[:, i - 1], flow)
                res_ind = torch.abs(
                    rgb_to_luma(warped_pix) - rgb_to_luma(lrs_eff[:, i])
                )

                if obs[i]:
                    feat_prop, gate = self.forward_agg(
                        feat_prop, lrs[:, i], res_ind, pre_mask, is_head=False
                    )
                    pre_mask = gate

            else:
                res_ind = zero_ind
                if obs[i]:
                    feat_prop, _ = self.forward_agg(
                        None, lrs[:, i], res_ind, pre_mask, is_head=True
                    )
                else:
                    feat_prop = zero_feat

            feat_prop = ckpt.checkpoint(
                self.forward_blocks, feat_prop, use_reentrant=False
            )

            fused = self.lrelu(self.fuse(torch.cat([back_feats[i], feat_prop], dim=1)))
            fused = self.trunk(fused)
            fused_hr = self.upsampler(fused)
            hr_size = fused_hr.shape[-2:]

            if self.sr_scale > 1:
                curr_hr = F.interpolate(
                    lrs_eff[:, i], size=hr_size, mode="bilinear", align_corners=False
                )
            else:
                curr_hr = lrs_eff[:, i]

            coarse_res = self.coarse_head(fused_hr)
            if obs[i]:

                coarse = torch.tanh(coarse_res + curr_hr)
            else:

                coarse = torch.tanh(coarse_res)

            cond = self.cond_head(fused_hr)

            evidence = torch.maximum(residual_acc[i], res_ind)
            if self.sr_scale > 1:
                evidence = F.interpolate(
                    evidence, size=hr_size, mode="bilinear", align_corners=False
                )
            hole_logit = self.hole_head(torch.cat([fused_hr, evidence], dim=1))

            coarse_out.append(coarse)
            cond_out.append(cond)
            hole_out.append(hole_logit)

        return {
            "coarse": torch.stack(coarse_out, dim=1),
            "cond": torch.stack(cond_out, dim=1),
            "hole_logits": torch.stack(hole_out, dim=1),
        }
