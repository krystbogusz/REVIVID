"""MambaOFR's texture blend modes (``Data_Degradation/blend_modes``: addition,
subtract, multiply) for its opaque layers, where the composite ratio is the
opacity. ``img`` and ``layer`` are greyscale float arrays in [0, 1]; the result
is in [0, 1].
"""

import numpy as np


def addition(img, layer, opacity):
    comp = img + (1.0 - layer)
    return np.clip(comp * opacity + img * (1.0 - opacity), 0.0, 1.0)


def subtract(img, layer, opacity):
    comp = img - (1.0 - layer)
    return np.clip(comp * opacity + img * (1.0 - opacity), 0.0, 1.0)


def multiply(img, layer, opacity):
    comp = np.clip(layer * img, 0.0, 1.0)
    return comp * opacity + img * (1.0 - opacity)
