"""Image / video quality metrics for REVIVID.

Tensors are expected in ``[-1, 1]`` (the network's working range). Helpers
convert to ``[0, 255]`` internally. Implementations are dependency-light
(numpy + torch) so they run without scikit-image.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


def _to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip((img + 1.0) / 2.0 * 255.0, 0, 255).round()


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    a, b = _to_uint8(img1).astype(np.float64), _to_uint8(img2).astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def _ssim_single(a: np.ndarray, b: np.ndarray) -> float:
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    a, b = a.astype(np.float64), b.astype(np.float64)
    import cv2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(a, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(b, -1, window)[5:-5, 5:-5]
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    sigma1 = cv2.filter2D(a**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2 = cv2.filter2D(b**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(a * b, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1 + sigma2 + c2)
    )
    return float(ssim_map.mean())


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    a, b = _to_uint8(img1), _to_uint8(img2)
    if a.ndim == 2:
        return _ssim_single(a, b)
    return float(
        np.mean([_ssim_single(a[..., c], b[..., c]) for c in range(a.shape[-1])])
    )


@torch.no_grad()
def evaluate_clip(pred: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    """``pred`` / ``gt`` are (t, 3, h, w) in [-1, 1]. Returns mean PSNR / SSIM."""
    pred = pred.detach().cpu().float()
    gt = gt.detach().cpu().float()
    psnrs, ssims = [], []
    for p, g in zip(pred, gt):
        p_np = p.permute(1, 2, 0).numpy()
        g_np = g.permute(1, 2, 0).numpy()
        psnrs.append(calculate_psnr(p_np, g_np))
        ssims.append(calculate_ssim(p_np, g_np))
    finite = [v for v in psnrs if np.isfinite(v)]
    return {
        "psnr": float(np.mean(finite)) if finite else float("inf"),
        "ssim": float(np.mean(ssims)),
    }
