"""Discriminators of the adversarial loss (``training.gan.type``).

``TemporalPatchDiscriminator`` ("temporal", phase 1 default) is MambaOFR's
``discriminator_v2``: spectral-normalised 3-D convolutions on the raw clip, so
it judges spatio-temporal patches - texture AND flicker - with a hinge loss
(MambaOFR's G weight 0.01).

``ProjectedPatchDiscriminator`` ("projected") judges FROZEN pretrained VGG19 features — the very maps
``VGGFeatures`` computes in the same pass as the perceptual loss —
at four scales, each with a small trainable patch head (the "projected GAN"
idea):

* Pretrained features make the adversarial game far more data-efficient and
  stable; REDS train is only 240 clips.
* Every head outputs a map of logits, one per image patch, so the critic is
  local: a smeared scratch, a seam around a filled hole or a plastic-looking
  texture patch is judged where it is, not averaged into one score per clip.
* The four scales cover grain and fine texture (relu1_2) up to object-level
  structure (relu4_4).

Flicker is not this module's job — ``TemporalConsistencyLoss`` handles it.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

from .losses import VGGFeatures


class TemporalPatchDiscriminator(nn.Module):
    """MambaOFR's spatio-temporal patch critic: (N, T, 3, H, W) clip in [-1, 1]
    -> (N, T, C, H / 32, W / 32) logits. Five 3-D convs, kernel (3, 5, 5),
    stride (1, 2, 2), spectral norm on all but the last."""

    def __init__(self, in_channels: int = 3, nf: int = 64):
        super().__init__()

        def conv(c_in, c_out, padding, sn=True):
            m = nn.Conv3d(c_in, c_out, (3, 5, 5), (1, 2, 2), padding, bias=not sn)
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            return spectral_norm(m) if sn else m

        self.net = nn.Sequential(
            conv(in_channels, nf, 1), nn.LeakyReLU(0.2, inplace=True),
            conv(nf, 2 * nf, (1, 2, 2)), nn.LeakyReLU(0.2, inplace=True),
            conv(2 * nf, 2 * nf, (1, 2, 2)), nn.LeakyReLU(0.2, inplace=True),
            conv(2 * nf, 2 * nf, (1, 2, 2)), nn.LeakyReLU(0.2, inplace=True),
            conv(2 * nf, 2 * nf, (1, 2, 2), sn=False),
        )

    def forward(self, clip: torch.Tensor) -> List[torch.Tensor]:
        """One logit map in a list, like the projected discriminator's scales."""
        return [self.net(clip.transpose(1, 2)).transpose(1, 2)]


class ProjectedPatchDiscriminator(nn.Module):
    def __init__(
        self,
        feat_channels: Sequence[int] = VGGFeatures.CHANNELS,
        width: int = 64,
    ):
        super().__init__()
        heads = []
        for i, c in enumerate(feat_channels):
            # The full-resolution scale is strided once to keep memory in check.
            first = (
                nn.Conv2d(c, width, 4, 2, 1) if i == 0 else nn.Conv2d(c, width, 1)
            )
            heads.append(
                nn.Sequential(
                    spectral_norm(first),
                    nn.LeakyReLU(0.2, inplace=True),
                    spectral_norm(nn.Conv2d(width, width, 3, 1, 1)),
                    nn.LeakyReLU(0.2, inplace=True),
                    spectral_norm(nn.Conv2d(width, 1, 3, 1, 1)),
                )
            )
        self.heads = nn.ModuleList(heads)

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        """VGG feature maps → one patch-logit map per scale."""
        return [head(f) for head, f in zip(self.heads, feats)]
