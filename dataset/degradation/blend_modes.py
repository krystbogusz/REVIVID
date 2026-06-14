"""Alpha-compositing blend modes (addition, subtract, multiply).

Operates on RGBA tensors in [0, 255]. Used by the degradation pipeline to
composite film-grain textures over video frames.
"""

from __future__ import annotations

import torch


def _compose_alpha(img_in, img_layer, opacity):
    comp_alpha = torch.min(img_in[:, :, 3], img_layer[:, :, 3]) * opacity
    new_alpha = img_in[:, :, 3] + (1.0 - img_in[:, :, 3]) * comp_alpha

    ratio = torch.where(new_alpha == 0, torch.zeros_like(comp_alpha), comp_alpha / new_alpha)
    return ratio


def addition(img_rgba, texture_rgba, opacity):
    img_in = img_rgba / 255.0
    img_layer = texture_rgba.clone()
    img_layer[:, :, :3] = 1.0 - (img_layer[:, :, :3] / 255.0)

    ratio = _compose_alpha(img_in, img_layer, opacity)
    comp = img_in[:, :, :3] + img_layer[:, :, :3]

    ratio_rs = ratio.unsqueeze(-1).expand(-1, -1, 3)
    img_out = torch.clamp(comp * ratio_rs + img_in[:, :, :3] * (1.0 - ratio_rs), 0.0, 1.0)
    return img_out * 255.0


def subtract(img_rgba, texture_rgba, opacity):
    img_in = img_rgba / 255.0
    img_layer = texture_rgba.clone()
    img_layer[:, :, :3] = 1.0 - (img_layer[:, :, :3] / 255.0)

    ratio = _compose_alpha(img_in, img_layer, opacity)
    comp = img_in[:, :, :3] - img_layer[:, :, :3]

    ratio_rs = ratio.unsqueeze(-1).expand(-1, -1, 3)
    img_out = torch.clamp(comp * ratio_rs + img_in[:, :, :3] * (1.0 - ratio_rs), 0.0, 1.0)
    return img_out * 255.0


def multiply(img_rgba, texture_rgba, opacity):
    img_in = img_rgba / 255.0
    img_layer = texture_rgba / 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)
    comp = torch.clamp(img_layer[:, :, :3] * img_in[:, :, :3], 0.0, 1.0)

    ratio_rs = ratio.unsqueeze(-1).expand(-1, -1, 3)
    img_out = comp * ratio_rs + img_in[:, :, :3] * (1.0 - ratio_rs)
    return img_out * 255.0