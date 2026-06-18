"""Low-level image degradation primitives for REVIVID dataset creation.

All functions operate on single frames (torch tensors in [0, 1] or [0, 255]
depending on convention — see each function's docstring). They are called by
``pipeline.process_video_frames`` which handles batching and normalisation.

Conventions that match MambaOFR (degradation_video_list_5):
  - ``random_scaling`` supports bilinear / bicubic / lanczos (OpenCV for lanczos).
  - ``apply_jpeg_artifact`` receives and returns a **greyscale uint8 ndarray** (H, W),
    matching MambaOFR's PIL.convert("L") → BytesIO JPEG path.
"""

from __future__ import annotations

import random

import cv2
import numpy as np
import scipy.stats as ss
import torch
import torch.nn.functional as F
import torchvision.transforms as T

def random_scaling(img_tensor, target_w, target_h):
    """Resize a (1, C, H, W) tensor to (target_h, target_w).

    Randomly selects bilinear, bicubic, or lanczos — matching MambaOFR's
    random_scaling which draws from the same three methods.
    Lanczos is not natively supported by torch.nn.functional.interpolate, so
    we drop to OpenCV for that mode and convert back to tensor.
    """
    mode_choice = random.randint(0, 2)  # 0=bilinear, 1=bicubic, 2=lanczos
    if mode_choice == 2:  # lanczos — use OpenCV
        import numpy as np
        # img_tensor shape: (1, C, H, W), values in [0, 1]
        arr = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, C)
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        resized = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        if resized.ndim == 2:  # greyscale edge case
            resized = resized[:, :, None]
        resized_tensor = torch.from_numpy(resized).float().to(img_tensor.device) / 255.0
        return resized_tensor.permute(2, 0, 1).unsqueeze(0)
    torch_mode = 'bilinear' if mode_choice == 0 else 'bicubic'
    return F.interpolate(img_tensor, size=(target_h, target_w), mode=torch_mode, align_corners=False)

def apply_downsampling(img_tensor, params):
    _, _, img_h, img_w = img_tensor.shape
    rand_num = params['rnum']

    if rand_num > 0.8:
        scale_factor = params['up_scale']
    elif rand_num < 0.7:
        scale_factor = params['down_scale']
    else:
        scale_factor = 1.0

    new_w = int(scale_factor * img_w)
    new_h = int(scale_factor * img_h)

    if scale_factor != 1.0:
        return random_scaling(img_tensor, new_w, new_h)
    return img_tensor

def apply_jpeg_artifact(img_gray: np.ndarray, quality: int) -> np.ndarray:
    """Apply JPEG compression artefacts to a **greyscale uint8** frame (H, W).

    Matches MambaOFR's ``jpeg_artifact_v2`` which encodes a PIL grayscale image
    via BytesIO and decodes back — both paths yield equivalent blocking/ringing
    artefacts on a single-channel signal.

    Args:
        img_gray: uint8 ndarray of shape (H, W) — greyscale frame.
        quality:  base JPEG quality [40, 100].

    Returns:
        uint8 ndarray (H, W) after JPEG encode/decode.
    """
    quality_variance = random.randint(-15, 15)
    new_quality = int(np.clip(quality + quality_variance, 40, 100))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), new_quality]
    success, encoded_img = cv2.imencode('.jpg', img_gray, encode_param)
    if success:
        return cv2.imdecode(encoded_img, cv2.IMREAD_GRAYSCALE)
    return img_gray

def gm_blur_kernel(mean, cov, size=15):
    center = size / 2.0 + 0.5
    kernel = np.zeros([size, size])
    for y in range(size):
        for x in range(size):
            cy = y - center + 1
            cx = x - center + 1
            kernel[y, x] = ss.multivariate_normal.pdf([cx, cy], mean=mean, cov=cov)
    return kernel / np.sum(kernel)

def anisotropic_gaussian_kernel(ksize=15, theta=np.pi, l1=6, l2=6):
    v = np.dot(
        np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]),
        np.array([1., 0.])
    )
    v_matrix = np.array([[v[0], v[1]], [v[1], -v[0]]])
    d_matrix = np.array([[l1, 0], [0, l2]])
    sigma = np.dot(np.dot(v_matrix, d_matrix), np.linalg.inv(v_matrix))
    return gm_blur_kernel(mean=[0, 0], cov=sigma, size=ksize)

def fspecial_gaussian(hsize, sigma):
    siz = [(hsize - 1.0) / 2.0, (hsize - 1.0) / 2.0]
    x, y = np.meshgrid(np.arange(-siz[1], siz[1] + 1), np.arange(-siz[0], siz[0] + 1))
    arg = -(x * x + y * y) / (2 * sigma * sigma)
    h = np.exp(arg)
    h[h < np.finfo(float).eps * h.max()] = 0
    sum_h = h.sum()
    if sum_h != 0:
        h = h / sum_h
    return h

def apply_blur(img_tensor, params):
    device = img_tensor.device
    wd2 = 4.0 + 4
    wd = 2.0 + 0.2 * 4

    if params['type_value'] < 0.5:
        l1 = wd2 * float(np.clip(params['l1_value'] + (random.random() - 0.5) / 10., 1e-8, 1 - 1e-8))
        l2 = wd2 * float(np.clip(params['l2_value'] + (random.random() - 0.5) / 10., 1e-8, 1 - 1e-8))
        theta = float(np.clip(params['angle_value'] + (random.random() - 0.5) / 5., 1e-8, 1 - 1e-8)) * np.pi
        kernel = anisotropic_gaussian_kernel(ksize=2 * params['shape_value'] + 3, theta=theta, l1=l1, l2=l2)
    else:
        sigma = wd * float(np.clip(params['l1_value'] + (random.random() - 0.5) / 10., 1e-8, 1 - 1e-8))
        kernel = fspecial_gaussian(2 * params['shape_value'] + 3, sigma)

    kernel_tensor = torch.tensor(kernel, dtype=torch.float32, device=device)
    kernel_tensor = kernel_tensor.view(1, 1, kernel_tensor.shape[0], kernel_tensor.shape[1])
    kernel_tensor = kernel_tensor.repeat(img_tensor.shape[1], 1, 1, 1)

    padding = kernel_tensor.shape[-1] // 2
    blurred = F.conv2d(img_tensor, kernel_tensor, padding=padding, groups=img_tensor.shape[1])
    return blurred

def apply_color_jitter(img_tensor):
    jitter = T.ColorJitter(brightness=[0.8, 1.2], contrast=[0.9, 1.0], saturation=[1.0, 1.0], hue=0.0)
    if random.random() < 0.5:
        return jitter(img_tensor)
    return img_tensor

def apply_noise(img_tensor, std, noise_type="gaussian"):
    noise = torch.randn_like(img_tensor) * std

    if noise_type == "gaussian":
        noisy_img = img_tensor + noise
    else:
        noisy_img = img_tensor + noise * img_tensor

    return torch.clamp(noisy_img, 0.0, 1.0)