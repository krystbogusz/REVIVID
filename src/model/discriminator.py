"""Projected patch discriminator for the adversarial loss on the whole frame.

This is NOT MambaOFR's discriminator (a 3-D conv net on raw pixels, judging the
whole clip). It judges FROZEN pretrained VGG19 features — the very maps
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
