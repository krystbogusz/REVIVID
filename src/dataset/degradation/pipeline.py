"""Degradation pipeline for REVIVID dataset creation.

Applies a randomised sequence of classical film-degradation operations
(blur, noise, JPEG compression, resampling, texture overlay, holes) to a list
of BGR frames and returns the degraded frames at the requested output size.

Order of operations per frame — the one MambaOFR actually trains with
(``degradation_video_list_4`` → ``degradation_v3``), on a single grey channel:
    1. BGR → greyscale
    2. Textures (static or moving-line, random blend mode), at native resolution
    3. Blur
    4. Resampling (down / up / none) — the frame STAYS at the resampled size
    5. Gaussian / speckle noise  (at the resampled size)
    6. JPEG on greyscale uint8   (at the resampled size)
    7. Resize back to native resolution
    8. Color jitter (50 % probability)
    9. Final resize to LR resolution (LR = GT / sr_scale), replicate to 3 channels
   10. Persistent holes: the frames are split into windows of ``hole_window``
       and each window gets, with probability ``hole_prob``, one hole mask
       shared by all its frames.

Do not follow ``degradation_video_list_5`` (MambaOFR's ``degradation.py``):
it passes ``distortion_probability = [1, 1, 1, 1]`` and ``degradation_v3``
tests ``p < 1.0``, so blur, noise and JPEG never run there. Its per-degree
parameter ranges are still the ones used in ``_DEG_PARAMS`` below.
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
    alpha = (
        torch.ones(
            (tensor_rgb.shape[0], tensor_rgb.shape[1], 1),
            device=tensor_rgb.device,
            dtype=tensor_rgb.dtype,
        )
        * 255.0
    )
    return torch.cat((tensor_rgb, alpha), dim=2)


_DEG_PARAMS = [
    {
        "shape_value": lambda: random.randint(2, 5),
        "noise_std": lambda: random.uniform(5.0 / 255.0, 6.0 / 255.0),
        "jpeg_quality": lambda: random.randint(80, 100),
        "up_scale": lambda: random.uniform(1, 1.5),
        "down_scale": lambda: random.uniform(0.5, 1),
    },
    {
        "shape_value": lambda: random.randint(5, 8),
        "noise_std": lambda: random.uniform(6.0 / 255.0, 8.0 / 255.0),
        "jpeg_quality": lambda: random.randint(60, 80),
        "up_scale": lambda: random.uniform(1, 2),
        "down_scale": lambda: random.uniform(0.25, 1),
    },
    {
        "shape_value": lambda: random.randint(8, 11),
        "noise_std": lambda: random.uniform(8.0 / 255.0, 10.0 / 255.0),
        "jpeg_quality": lambda: random.randint(40, 60),
        "up_scale": lambda: random.uniform(1, 2),
        "down_scale": lambda: random.uniform(0.125, 1),
    },
]


_DEGREE_WEIGHTS = [0.30, 0.30, 0.40]


def sample_degree() -> int:
    """Sample a degradation degree (0, 1, 2) with weights 30 / 30 / 40 %."""
    return random.choices([0, 1, 2], weights=_DEGREE_WEIGHTS, k=1)[0]


def apply_holes_to_window(
    frames_bgr: list,
    hole_prob: float,
) -> tuple[list, np.ndarray | None]:
    """Optionally overlay a persistent spatial hole mask on every frame of a window.

    With probability ``hole_prob`` a single hole mask is generated for the
    window's native resolution and applied to every frame (consistent damage
    across the whole window, as if the film strip was torn).  Hole pixels are
    set to 0 in uint8 space, which becomes -1.0 after [-1, 1] normalisation.
    The model is NOT told where they are — it has to detect them; the returned
    mask is only a training target.

    Args:
        frames_bgr: list of BGR uint8 ndarrays, all same spatial size.
        hole_prob:  probability in [0, 1] that holes are applied to this window.

    Returns:
        ``(frames, mask)``: the frames (copies with holes burned in, or the
        input list untouched) and the (H, W) uint8 {0, 1} hole mask, or ``None``
        when this window got no holes.
    """
    if not frames_bgr or hole_prob <= 0.0 or random.random() >= hole_prob:
        return frames_bgr, None

    h, w = frames_bgr[0].shape[:2]
    mask_bool = generate_persistent_hole_mask(h, w) > 127

    result = []
    for frame in frames_bgr:
        f = frame.copy()
        f[mask_bool] = 0
        result.append(f)
    return result, mask_bool.astype(np.uint8)


def process_video_frames(
    frame_list_cv2: list,
    texture_cache,
    degree: int = 1,
    downscale_factor: int = 4,
    device: torch.device | None = None,
    out_size: tuple | None = None,
    hole_prob: float = 0.0,
    hole_window: int | None = None,
    return_masks: bool = False,
) -> list | tuple[list, list]:
    """Degrade frames at their native resolution and resize them at the very end.

    All degradations run on the original resolution. The final size is either an
    explicit ``out_size=(height, width)`` (takes precedence) or the native size
    divided by ``downscale_factor``. Holes are burned in last, at the output
    size, per window of ``hole_window`` frames (default: all frames are one window).

    With ``return_masks=True`` it returns ``(frames, masks)``, where ``masks``
    holds one (h, w) uint8 {0, 1} hole mask per output frame — the training
    target for the hole detector. Without it, only the frames (as before).
    """
    if not frame_list_cv2:
        return []

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    first_frame = frame_list_cv2[0]
    original_h, original_w = first_frame.shape[:2]

    _p = _DEG_PARAMS[degree]
    deg_params = {
        "type_value": random.random(),
        "l1_value": random.random(),
        "l2_value": random.random(),
        "angle_value": random.random(),
        "shape_value": _p["shape_value"](),
        "noise_std": _p["noise_std"](),
        "jpeg_quality": _p["jpeg_quality"](),
        "rnum": np.random.rand(),
        "up_scale": _p["up_scale"](),
        "down_scale": _p["down_scale"](),
    }

    use_moving_line = random.random() < 0.2
    moving_line_mode = random.randint(0, 2)
    last_moving_texture = None

    all_texture_keys = texture_cache.get_all_keys()
    moving_line_keys = texture_cache.get_moving_line_keys()
    available_keys = moving_line_keys if use_moving_line else all_texture_keys

    degraded_frames = []

    for frame_cv2 in frame_list_cv2:

        gray_frame = cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2GRAY)
        frame_tensor = (
            torch.from_numpy(cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB))
            .float()
            .to(device)
        )

        # 1. Textures, on the clean frame at native resolution — so the
        #    scratches go through blur / resampling / noise / JPEG like the content.
        selected_key = random.choice(available_keys)
        texture_img, folder_name = texture_cache.get_texture(selected_key)
        blend_mode = 0 if folder_name == "011" else random.randint(0, 2)

        if not use_moving_line:
            processed_texture = generate_texture(
                texture_img, folder_name, original_h, original_w
            )
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
            blended = addition(frame_rgba, texture_rgba, opacity)
        elif effective_blend == 1:
            blended = subtract(frame_rgba, texture_rgba, opacity)
        else:
            blended = multiply(frame_rgba, texture_rgba, opacity)

        # Frame and texture are both grey replicated to RGB, so the blend has
        # three identical channels. Continue on one (1, 1, H, W) channel: the
        # noise must be drawn once per pixel, as on MambaOFR's PIL "L" image —
        # per-channel noise would be averaged down ~33% by the final luma.
        frame_tensor = blended[:, :, :1].permute(2, 0, 1).unsqueeze(0) / 255.0

        # 2. Blur, 3. resample. The frame stays at the resampled size, so
        #    noise and JPEG below act at that scale and get resized with it.
        frame_tensor = apply_blur(frame_tensor, deg_params)
        frame_tensor = apply_downsampling(frame_tensor, deg_params)

        # 4. Noise
        noise_type = "gaussian" if random.choice([1, 2]) == 1 else "speckle"
        std_variance = random.uniform(-0.5, 0.5)
        new_std = float(
            np.clip(
                deg_params["noise_std"] + std_variance / 255.0,
                5.0 / 255.0,
                10.0 / 255.0,
            )
        )
        frame_tensor = apply_noise(frame_tensor, new_std, noise_type)

        # 5. JPEG (greyscale uint8)
        small = (frame_tensor[0, 0] * 255.0).round().byte().cpu().numpy()
        small = apply_jpeg_artifact(small, deg_params["jpeg_quality"])
        frame_tensor = torch.from_numpy(small).float().to(device)[None, None] / 255.0

        # 6. Back to native resolution
        if frame_tensor.shape[2:] != (original_h, original_w):
            frame_tensor = random_scaling(frame_tensor, original_w, original_h)

        # 7. Color jitter, then back to 3 identical channels
        frame_tensor = apply_color_jitter(frame_tensor.squeeze(0).clamp(0.0, 1.0))
        frame_tensor = frame_tensor.repeat(3, 1, 1) * 255.0

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
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        current_frame = (
            frame_tensor.clamp(0.0, 255.0).round().permute(1, 2, 0).byte().cpu().numpy()
        )
        degraded_frames.append(cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR))

    window = hole_window or len(degraded_frames)
    with_holes, masks = [], []
    for start in range(0, len(degraded_frames), window):
        chunk, mask = apply_holes_to_window(degraded_frames[start : start + window], hole_prob)
        with_holes.extend(chunk)
        if mask is None:
            mask = np.zeros(chunk[0].shape[:2], np.uint8)
        masks.extend([mask] * len(chunk))
    return (with_holes, masks) if return_masks else with_holes
