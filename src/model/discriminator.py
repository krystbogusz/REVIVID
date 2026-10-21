"""Discriminator of the adversarial loss: MambaOFR's ``discriminator_v2``.

Spectral-normalised 3-D convolutions on the raw clip, so it judges
spatio-temporal patches - texture AND flicker - with a hinge loss (the
T-PatchGAN critic video inpainting uses as well).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalPatchDiscriminator(nn.Module):
    """(N, T, 3, H, W) clip in [-1, 1] -> (N, T, C, H / 32, W / 32) logits.
    Five 3-D convs, kernel (3, 5, 5), stride (1, 2, 2), spectral norm on all
    but the last."""

    def __init__(self, in_channels: int = 3, nf: int = 64):
        super().__init__()

        def conv(c_in, c_out, padding, sn=True):
            m = nn.Conv3d(c_in, c_out, (3, 5, 5), (1, 2, 2), padding, bias=not sn)
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            return nn.utils.spectral_norm(m) if sn else m

        self.conv = nn.Sequential(
            conv(in_channels, nf, 1), nn.LeakyReLU(0.2, inplace=True),
            conv(nf, 2 * nf, (1, 2, 2)), nn.LeakyReLU(0.2, inplace=True),
            conv(2 * nf, 2 * nf, (1, 2, 2)), nn.LeakyReLU(0.2, inplace=True),
            conv(2 * nf, 2 * nf, (1, 2, 2)), nn.LeakyReLU(0.2, inplace=True),
            conv(2 * nf, 2 * nf, (1, 2, 2), sn=False),
        )

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        return self.conv(clip.transpose(1, 2)).transpose(1, 2)
