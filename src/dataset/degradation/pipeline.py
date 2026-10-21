"""Degradation pipeline for REVIVID datasets: MambaOFR's training degradation
(``degradation_video_list_4`` in VP_code/data/Data_Degradation/util.py) one to
one, with REVIVID's SR downscale before it and burned holes after it.

Per clip MambaOFR draws one set of parameters (``sample_clip_params``) and
whether it gets moving lines; per frame, on one grey channel:
    0. SR: the clean frame resized (bicubic) to the output size, GT / sr_scale.
       MambaOFR's steps then run at that size, so the input is degraded as
       MambaOFR's training crops are; with sr_scale 1 nothing changes.
    1. Texture, opacity 0.6-1, blended by addition / subtract / multiply:
       80 % of the clips a new texture on every frame, from any folder;
       20 % moving lines - one 001 template over the clip, shifted 5-15 px
       per frame, one blend mode
    2. Blur      the clip's anisotropic / isotropic Gaussian, 7-25 px
    3. Resample  the clip's factor: down 0.125-1 (p 0.7), up 1-2 (p 0.2), else none
    4. Noise     Gaussian or speckle, std 5-10 / 255
    5. JPEG      the clip's quality 40-100, +-15 per frame
    6. Resize back to the output size (random bicubic / bilinear / lanczos)
    7. ColorJitter: brightness 0.8-1.2, contrast 0.9-1.0
Per window of ``hole_window`` frames:
    8. Burned holes, with probability ``hole_prob`` (0 = a dataset without holes)

MambaOFR's GT is USM-sharpened; the training loader does that (``training.gt_usm``).
"""

from __future__ import annotations

import random

import cv2
import numpy as np
from PIL import Image

from .artifacts import (
    apply_blur,
    apply_color_jitter,
    apply_downsampling,
    apply_jpeg,
    apply_noise,
    random_scaling,
)
from .blend_modes import addition, subtract, multiply
from .textures import (
    generate_persistent_hole_mask,
    moving_line_texture_generator,
    texture_generator,
)

MOVING_LINE_PROB = 0.2
OPACITY = (0.6, 1.0)


def sample_clip_params() -> dict:
    """MambaOFR's per-clip ``distortion_degree``: blur kernel, noise, JPEG, resampling."""
    return {
        "type_value": random.random(),
        "l1_value": random.random(),
        "l2_value": random.random(),
        "angle_value": random.random(),
        "shape_value": random.randint(2, 11),
        "noise_std": random.uniform(5.0 / 255.0, 10.0 / 255.0),
        "jpeg_quality": random.randint(40, 100),
        "rnum": np.random.rand(),
        "up_scale": random.uniform(1, 2),
        "down_scale": random.uniform(0.5 / 4, 1),
    }


def _blend(img: Image.Image, texture: Image.Image, mode: int, opacity: float) -> Image.Image:
    """Texture over a frame; ``mode`` 0 / 1 / 2 = addition / subtract / multiply."""
    x = np.asarray(img, np.float64) / 255.0
    layer = np.asarray(texture, np.float64) / 255.0
    out = (addition, subtract, multiply)[mode](x, layer, opacity)
    return Image.fromarray(np.uint8(out * 255.0))


def apply_holes_to_window(frames: list, hole_prob: float) -> tuple[list, np.ndarray | None]:
    """Optionally burn one persistent hole mask into every frame of a window.

    With probability ``hole_prob`` a single hole mask is generated for the
    window's resolution and applied to every frame (consistent damage across
    the whole window, as if the film strip was torn). Hole pixels are set to 0
    in uint8 space, which becomes -1.0 after [-1, 1] normalisation. The model
    is NOT told where they are — it has to detect them; the returned mask is
    only a training target.

    Args:
        frames:    list of uint8 ndarrays, all the same spatial size.
        hole_prob: probability in [0, 1] that holes are applied to this window.

    Returns:
        ``(frames, mask)``: the frames (copies with holes burned in, or the
        input list untouched) and the (H, W) uint8 {0, 1} hole mask, or ``None``
        when this window got no holes.
    """
    if not frames or hole_prob <= 0.0 or random.random() >= hole_prob:
        return frames, None

    h, w = frames[0].shape[:2]
    mask_bool = generate_persistent_hole_mask(h, w) > 127

    result = []
    for frame in frames:
        f = frame.copy()
        f[mask_bool] = 0
        result.append(f)
    return result, mask_bool.astype(np.uint8)


def process_video_frames(
    frames_bgr: list,
    texture_cache,
    out_size: tuple | None = None,
    hole_prob: float = 0.0,
    hole_window: int | None = None,
    return_masks: bool = False,
) -> list | tuple[list, list]:
    """Degrade one clip of BGR uint8 frames; see the module docstring.

    ``out_size=(height, width)`` is the output (LR) size, by default the
    frames' own. ``hole_prob`` is the chance that a window of ``hole_window``
    frames (default: the whole clip) gets a burned hole.

    Returns the degraded frames, BGR uint8 with the grey picture in every
    channel; with ``return_masks=True`` ``(frames, masks)``, ``masks`` holding
    one (h, w) uint8 {0, 1} hole mask per frame - the training target of the
    hole detector.
    """
    if not frames_bgr:
        return ([], []) if return_masks else []

    h, w = out_size if out_size is not None else frames_bgr[0].shape[:2]
    h, w = int(h), int(w)

    params = sample_clip_params()
    moving_line = random.random() < MOVING_LINE_PROB
    if moving_line:
        template, _ = texture_cache.get_texture(random.choice(texture_cache.get_moving_line_keys()))
        line_template = Image.fromarray(template)
        line_mode = random.randint(0, 2)
        line = None
    keys = texture_cache.get_all_keys()

    frames = []
    for frame in frames_bgr:
        x = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        # 0. SR
        if x.size != (w, h):
            x = x.resize((w, h), Image.BICUBIC)

        # 1. texture
        if moving_line:
            line = moving_line_texture_generator(line_template, line, h, w)
            x = _blend(x, line, line_mode, random.uniform(*OPACITY))
        else:
            template, folder = texture_cache.get_texture(random.choice(keys))
            texture = texture_generator(Image.fromarray(template), folder, h, w)
            mode = random.randint(0, 2)
            if folder == "011":
                mode = 0
            x = _blend(x, texture, mode, random.uniform(*OPACITY))

        # 2.-6. MambaOFR's degradation_v3, in its order
        x = apply_blur(x, params)
        x = apply_downsampling(x, params)
        x = apply_noise(x, params["noise_std"])
        x = apply_jpeg(x, params["jpeg_quality"])
        x = random_scaling(x, w, h)

        # 7.
        frames.append(np.asarray(apply_color_jitter(x)))

    # 8. holes, per window
    window = hole_window or len(frames)
    out, masks = [], []
    for start in range(0, len(frames), window):
        chunk, mask = apply_holes_to_window(frames[start : start + window], hole_prob)
        out.extend(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) for f in chunk)
        if mask is None:
            mask = np.zeros((h, w), np.uint8)
        masks.extend([mask] * len(chunk))
    return (out, masks) if return_masks else out
