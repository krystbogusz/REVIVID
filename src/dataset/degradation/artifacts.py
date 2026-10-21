"""MambaOFR's per-frame degradations (``VP_code/data/Data_Degradation/util.py``,
the ops of ``degradation_v3`` and ``color_jitter``), one to one.

Every op takes and returns a greyscale PIL image ("L"): as in MambaOFR, the
frame is quantised to uint8 (truncated) between the steps. MambaOFR runs them
on three identical channels; one channel gives the same values.
"""

from __future__ import annotations

import random
from io import BytesIO

import numpy as np
import scipy.stats as ss
from PIL import Image
from scipy import ndimage

RESAMPLE = (Image.BICUBIC, Image.BILINEAR, Image.LANCZOS)


def random_scaling(img, w, h):
    """Resize to (w, h) with a random filter (``random_scaling``)."""
    return img.resize((w, h), random.choice(RESAMPLE))


def apply_downsampling(img, params):
    """Rescale by the clip's factor (``downsampling_artifact_v3_fixed``): up
    with p 0.2, down with p 0.7, else unchanged."""
    w, h = img.size
    if params["rnum"] > 0.8:
        sf = params["up_scale"]
    elif params["rnum"] < 0.7:
        sf = params["down_scale"]
    else:
        sf = 1.0
    return random_scaling(img, int(sf * w), int(sf * h))


def gm_blur_kernel(mean, cov, size=15):
    center = size / 2.0 + 0.5
    c = np.arange(size) - center + 1
    cx, cy = np.meshgrid(c, c)
    k = ss.multivariate_normal.pdf(np.stack([cx, cy], -1), mean=mean, cov=cov)
    return k / np.sum(k)


def anisotropic_gaussian(ksize=15, theta=np.pi, l1=6, l2=6):
    v = np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array([1.0, 0.0]))
    V = np.array([[v[0], v[1]], [v[1], -v[0]]])
    D = np.array([[l1, 0], [0, l2]])
    sigma = np.dot(np.dot(V, D), np.linalg.inv(V))
    return gm_blur_kernel(mean=[0, 0], cov=sigma, size=ksize)


def fspecial_gaussian(hsize, sigma):
    siz = (hsize - 1.0) / 2.0
    x, y = np.meshgrid(np.arange(-siz, siz + 1), np.arange(-siz, siz + 1))
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(float).eps * h.max()] = 0
    if h.sum() != 0:
        h = h / h.sum()
    return h


def _jitter(value, spread):
    return float(np.clip(value + (random.random() - 0.5) / spread, 1e-8, 1 - 1e-8))


def apply_blur(img, params):
    """The clip's anisotropic or isotropic Gaussian, jittered per frame (``add_blur_fixed``)."""
    wd2 = 4.0 + 4
    wd = 2.0 + 0.2 * 4
    ksize = 2 * params["shape_value"] + 3
    if params["type_value"] < 0.5:
        l1 = wd2 * _jitter(params["l1_value"], 10.0)
        l2 = wd2 * _jitter(params["l2_value"], 10.0)
        k = anisotropic_gaussian(ksize=ksize, theta=_jitter(params["angle_value"], 5.0) * np.pi, l1=l1, l2=l2)
    else:
        k = fspecial_gaussian(ksize, wd * _jitter(params["l1_value"], 10.0))
    x = ndimage.convolve(np.asarray(img, np.float32) / 255.0, k, mode="mirror")
    return Image.fromarray(np.clip(x * 255, 0, 255).astype(np.uint8))


def apply_noise(img, std):
    """Gaussian or speckle noise, the clip's std jittered per frame
    (``gaussian_noise_artifact_v2`` / ``speckle_noise_artifact_v2``)."""
    speckle = random.choice([1, 2]) == 2
    x = np.asarray(img, np.float32) / 255.0
    std = np.clip(std + random.uniform(-0.5, 0.5) / 255.0, 5.0 / 255.0, 10.0 / 255.0)
    gauss = np.random.normal(loc=0.0, scale=std, size=x.shape)
    noisy = x + gauss * x if speckle else x + gauss
    noisy = np.clip(noisy, 0, 1).astype(np.float32)
    return Image.fromarray(np.clip(noisy * 255, 0, 255).astype(np.uint8))


def apply_jpeg(img, quality):
    """JPEG round trip at the clip's quality +- 15 (``jpeg_artifact_v2``)."""
    q = int(np.clip(quality + random.randint(-15, 15), 40, 100))
    with BytesIO() as f:
        img.save(f, format="JPEG", quality=q)
        f.seek(0)
        return Image.open(f).convert("L")


def apply_color_jitter(img):
    """MambaOFR's ``A.ColorJitter(brightness=[0.8, 1.2], contrast=[0.9, 1.0],
    saturation=1, hue=0, always_apply=True)`` on a grey frame, as in its
    albumentations 1.4.14: applied to every frame, brightness and contrast in
    random order (saturation and hue are identities here). Implemented here
    because albumentations 2 drops ``always_apply`` and would jitter only half
    of the frames."""
    x = np.asarray(img)
    brightness = random.uniform(0.8, 1.2)
    contrast = random.uniform(0.9, 1.0)
    for op in random.sample(("brightness", "contrast"), 2):
        if op == "brightness":
            lut = np.arange(256) * brightness
        else:
            lut = np.arange(256) * contrast + x.mean() * (1 - contrast)
        x = np.clip(lut, 0, 255).astype(np.uint8)[x]
    return Image.fromarray(x)
