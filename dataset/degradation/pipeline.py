"""Degradation pipeline for REVIVID dataset creation.

Applies a randomised sequence of classical film-degradation operations
(blur, noise, JPEG compression, resampling, texture overlay, holes) to a
list of BGR frames and returns the degraded frames at the requested output size.
"""

from __future__ import annotations

import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .blend_modes import addition, subtract, multiply
from .textures import (
    generate_texture,
    generate_moving_line_texture,
    generate_persistent_hole_mask,
)
from .artifacts import (
    apply_color_jitter,
    apply_blur,
    apply_jpeg_artifact,
    apply_downsampling,
    random_scaling,
    apply_noise,
)


def _add_alpha_channel(tensor_rgb: torch.Tensor) -> torch.Tensor:
    alpha = torch.ones(
        (tensor_rgb.shape[0], tensor_rgb.shape[1], 1),
        device=tensor_rgb.device, dtype=tensor_rgb.dtype,
    ) * 255.0
    return torch.cat((tensor_rgb, alpha), dim=2)


# Per-degree degradation parameters (blur strength, noise level, JPEG quality).
_DEG_PARAMS = [
    {
        "shape_value": lambda: random.randint(2, 5),
        "noise_std":   lambda: random.uniform(5.0 / 255., 6.0 / 255.),
        "jpeg_quality": lambda: random.randint(80, 100),
        "up_scale":    lambda: random.uniform(1, 1.5),
        "down_scale":  lambda: random.uniform(0.5, 1),
    },
    {
        "shape_value": lambda: random.randint(5, 8),
        "noise_std":   lambda: random.uniform(6.0 / 255., 8.0 / 255.),
        "jpeg_quality": lambda: random.randint(60, 80),
        "up_scale":    lambda: random.uniform(1, 2),
        "down_scale":  lambda: random.uniform(0.25, 1),
    },
    {
        "shape_value": lambda: random.randint(8, 11),
        "noise_std":   lambda: random.uniform(8.0 / 255., 10.0 / 255.),
        "jpeg_quality": lambda: random.randint(40, 60),
        "up_scale":    lambda: random.uniform(1, 2),
        "down_scale":  lambda: random.uniform(0.125, 1),
    },
]

_HOLE_PROB = {0: 0.01, 1: 0.05, 2: 0.10}


def process_video_frames(
    frame_list_cv2: list,
    texture_cache,
    degree: int = 1,
    downscale_factor: int = 4,
    device: torch.device | None = None,
    bake_holes: bool = True,
    out_size: tuple | None = None,
) -> list:
    """Degrade frames at their native resolution and resize them at the very end.

    All degradations run on the original resolution. The final size is either an
    explicit ``out_size=(height, width)`` (takes precedence) or the native size
    divided by ``downscale_factor``.
    """
    if not frame_list_cv2:
        return []

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    first_frame = frame_list_cv2[0]
    original_h, original_w = first_frame.shape[:2]

    dist_sequence = list(np.random.permutation(["blur", "noise", "jpeg", "downsample"]))

    # Sample per-degree params with small per-frame jitter applied in the loop.
    _p = _DEG_PARAMS[degree]
    deg_params = {
        "type_value":  random.random(),
        "l1_value":    random.random(),
        "l2_value":    random.random(),
        "angle_value": random.random(),
        "shape_value": _p["shape_value"](),
        "noise_std":   _p["noise_std"](),
        "jpeg_quality": _p["jpeg_quality"](),
        "rnum":        np.random.rand(),
        "up_scale":    _p["up_scale"](),
        "down_scale":  _p["down_scale"](),
    }

    use_heavy_damage = bake_holes and (random.random() < _HOLE_PROB.get(degree, 0.05))
    use_moving_line = random.random() < 0.2
    moving_line_mode = random.randint(0, 2)
    last_moving_texture = None

    batch_hole_mask_tensor = None
    if use_heavy_damage:
        hole_mask_cv2 = generate_persistent_hole_mask(original_h, original_w)
        batch_hole_mask_tensor = (
            torch.from_numpy(hole_mask_cv2).to(device).float() / 255.0
        ).unsqueeze(-1)

    all_texture_keys = texture_cache.get_all_keys()
    moving_line_keys = texture_cache.get_moving_line_keys()
    available_keys = moving_line_keys if use_moving_line else all_texture_keys

    degraded_frames = []

    for frame_cv2 in frame_list_cv2:
        # Convert to greyscale and back — simulates the tonal look of old film.
        current_frame = cv2.cvtColor(
            cv2.cvtColor(frame_cv2.copy(), cv2.COLOR_BGR2GRAY),
            cv2.COLOR_GRAY2BGR,
        )

        # Apply JPEG compression in BGR space (before tensor conversion).
        if "jpeg" in dist_sequence:
            current_frame = apply_jpeg_artifact(current_frame, deg_params["jpeg_quality"])

        frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        frame_tensor = (
            torch.from_numpy(frame_rgb).float().permute(2, 0, 1).unsqueeze(0).to(device)
            / 255.0
        )

        # Blur and resampling distortions.
        for dist_type in dist_sequence:
            if dist_type == "downsample":
                frame_tensor = apply_downsampling(frame_tensor, deg_params)
            elif dist_type == "blur":
                frame_tensor = apply_blur(frame_tensor, deg_params)

        # Restore to original resolution if resampling changed it.
        if frame_tensor.shape[2:] != (original_h, original_w):
            frame_tensor = random_scaling(frame_tensor, original_w, original_h)

        frame_tensor = frame_tensor.squeeze(0).permute(1, 2, 0) * 255.0

        # --- Texture overlay ---
        selected_key = random.choice(available_keys)
        texture_img, folder_name = texture_cache.get_texture(selected_key)
        blend_mode = 0 if folder_name == "011" else random.randint(0, 2)

        if not use_moving_line:
            processed_texture = generate_texture(texture_img, folder_name, original_h, original_w)
        else:
            processed_texture, last_moving_texture = generate_moving_line_texture(
                texture_img, last_moving_texture, original_h, original_w
            )

        texture_rgb = cv2.cvtColor(processed_texture, cv2.COLOR_GRAY2RGB)
        texture_tensor = torch.from_numpy(texture_rgb).float().to(device)

        frame_rgba = _add_alpha_channel(frame_tensor)
        texture_rgba = _add_alpha_channel(texture_tensor)
        opacity = random.uniform(0.6, 1.0)

        effective_blend = blend_mode if not use_moving_line else moving_line_mode
        if effective_blend == 0:
            frame_tensor = addition(frame_rgba, texture_rgba, opacity)
        elif effective_blend == 1:
            frame_tensor = subtract(frame_rgba, texture_rgba, opacity)
        else:
            frame_tensor = multiply(frame_rgba, texture_rgba, opacity)

        # --- Persistent spatial holes ---
        if batch_hole_mask_tensor is not None:
            frame_tensor = frame_tensor * (1.0 - batch_hole_mask_tensor)

        # --- Gaussian / speckle noise ---
        noise_type = "gaussian" if random.choice([1, 2]) == 1 else "speckle"
        frame_tensor = frame_tensor / 255.0
        std_variance = random.uniform(-0.5, 0.5)
        new_std = float(np.clip(
            deg_params["noise_std"] + std_variance / 255.0,
            5.0 / 255.0, 10.0 / 255.0,
        ))
        frame_tensor = apply_noise(frame_tensor, new_std, noise_type)
        frame_tensor = frame_tensor * 255.0

        # --- Colour jitter ---
        frame_tensor = frame_tensor.permute(2, 0, 1)
        frame_tensor = apply_color_jitter(frame_tensor / 255.0) * 255.0

        # --- Final resize ---
        if out_size is not None:
            target_h, target_w = int(out_size[0]), int(out_size[1])
        elif downscale_factor > 1:
            target_h = original_h // downscale_factor
            target_w = original_w // downscale_factor
        else:
            target_h, target_w = None, None

        if target_h is not None and (target_h, target_w) != (original_h, original_w):
            frame_tensor = F.interpolate(
                frame_tensor.unsqueeze(0),
                size=(target_h, target_w),
                mode="bilinear", align_corners=False,
            ).squeeze(0)

        current_frame = frame_tensor.permute(1, 2, 0).byte().cpu().numpy()
        degraded_frames.append(cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR))

    return degraded_frames