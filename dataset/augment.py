"""Spatial augmentations for on-the-fly training (MambaOFR-style)."""

from __future__ import annotations

import random

import cv2
import numpy as np


def augment_frames(
    imgs: list[np.ndarray],
    hflip: bool = True,
    rotation: bool = True,
) -> list[np.ndarray]:
    """Apply the same hflip / vflip / 90° rotation to every frame in *imgs*."""
    if not imgs:
        return imgs

    do_hflip = hflip and random.random() < 0.5
    do_vflip = rotation and random.random() < 0.5
    do_rot90 = rotation and random.random() < 0.5

    if not (do_hflip or do_vflip or do_rot90):
        return imgs

    out: list[np.ndarray] = []
    for img in imgs:
        aug = img
        if do_hflip:
            aug = cv2.flip(aug, 1)
        if do_vflip:
            aug = cv2.flip(aug, 0)
        if do_rot90:
            aug = aug.transpose(1, 0, 2)
        out.append(aug)
    return out
