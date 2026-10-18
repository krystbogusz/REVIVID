"""Degradation pipeline for REVIVID dataset creation.

Turns clean BGR frames into what REVIVID meets in practice: old films as they
reach it, 360p uploads of scanned prints. Order of operations, following how
that footage came to be:

    Film (per film frame, at the GT resolution, on one grey channel):
        1. Tone of a faded print: lifted blacks, dimmed whites, gamma
        2. Textures: either scratches that stay over the clip and drift slowly
           (vertical, as on a film strip) or dust / dirt / grain that change
           every frame
        3. Exposure flicker: a per-frame gain
        4. Blur (MambaOFR's anisotropic / isotropic Gaussian kernels)
        5. Gaussian / speckle grain
    Transfer:
        6. Resampling (down / up / none) - the scan's resolution
        7. Resize to the output (LR) resolution, antialiased, often a slight
           Gaussian softness there (the transfer)
        8. Telecine: a film frame shown on several video frames
           (``frame_ids``) is the very same degraded picture
    Damage (``hole_prob``): persistent holes per window of ``hole_window``
    Upload:
        9. H.264 of the whole clip at the output resolution (``apply_video_codec``)

Calibration (``P``) is against data/raw/old_films: 216 five-second segments
of 53 films (mostly 480x360 H.264), measured as 7-frame 128 px windows at
native size - the unit the model trains on - against the same windows of REDS
train through this pipeline. Medians, real / this pipeline / MambaOFR's:
    frame-to-frame log brightness std (flicker)   0.007 / 0.010 / 0.150
    single-frame dust blobs per 10k px, p90       1.2 / 5.7 / 50  (REDS motion alone: 3.6)
    windows with repeated frames                   66 % / 66 % / 0 %
    block-edge / in-block gradient (8 px grid)     1.07 / 1.07 / 1.00
    noise in flat areas (Immerkaer, 0-255)         0.28 / 0.31 / 0.66
    high / low frequency power, log10              -2.96 / -2.97 / -3.01
    1st / 99th percentile grey level               23, 201 / 23, 197 / 17, 215
Not matched by degradation: REDS has much more motion than the films.

That is the realistic mode. Each clip draws its mode from ``MODE_PROBS``
(``training.degradation_modes`` overrides it):
    realistic  ``P`` as calibrated above
    heavy      ``P_HEAVY``: degradations past the sample's p90, so that films more
               damaged than these uploads (or less compressed, grain visible) are
               not out of distribution
    dirty      ``P_DIRTY``: MambaOFR's damage density - dust / dirt / blotches on
               (nearly) every film frame at MambaOFR's opacity, single-frame
               scratches from any texture folder, MambaOFR's grain. In the
               calibrated modes dust sits on a few % of the frames only (the
               uploads are mostly cleaned); trained on them alone the model
               learned to keep dirt as if it were content.

MambaOFR's ops are kept (blur kernels, resampling, noise, textures, blend
modes); its order and ColorJitter are not.
"""

from __future__ import annotations

import math
import random

import cv2
import numpy as np
import torch

from .blend_modes import addition, subtract, multiply
from .textures import (
    generate_texture,
    generate_moving_line_texture,
    generate_persistent_hole_mask,
)
from .artifacts import (
    apply_blur,
    apply_downsampling,
    apply_noise,
    apply_tone,
    apply_video_codec,
    random_scaling,
)

# Per-clip ranges; (a, b) is uniform unless noted.
P = {
    "tone_prob": 0.6,
    "black": (0.0, 0.08),          # added black level (0..1)
    "white": (0.85, 1.0),          # white level
    "gamma": (0.8, 1.25),          # log-uniform
    "flicker": (0.0003, 0.008),    # std of the per-frame log gain, log-uniform
    "tex_prob": (0.0, 0.2),        # chance a film frame carries dust / dirt / grain
    "tex_opacity": (0.25, 0.8),    # dust, dirt
    "tex_all_folders": False,      # per-frame textures also from the scratch folders
    "line_opacity": (0.5, 1.0),    # scratches
    "moving_line_prob": 0.4,       # scratches persisting over the clip
    "blur_prob": 0.6,
    "blur_strength": 0.8,          # scales MambaOFR's kernel spread (l1, l2)
    "lr_soft_prob": 0.6,           # transfer softness at the output resolution
    "lr_soft_sigma": (0.3, 0.7),   # Gaussian, output px
    "noise_std": (0.5 / 255.0, 2.5 / 255.0),
    "grain_size": (0.0, 0.0),      # px at the GT resolution; 0 = per-pixel noise
    "crf": (22, 36),               # x264 constant rate factor (higher = worse)
    "deblock": (0, 2),             # x264 loop-filter offset
    # Film frames per video frame: distinct / slow (animation on twos and
    # threes) / telecine-like, else. Never a still: no picture is held longer
    # than 3 video frames.
    "hold_none": 0.17,
    "hold_slow": 0.10,
    "hold_slow_rate": (0.34, 0.5),
    "hold_rate": (0.5, 0.97),
    "dust_with_scratches": False,  # dust on clips with persistent scratches too
}

# Heavy mode: a share of the clips is degraded past the calibrated sample's
# p90 (old_films are web uploads, many already cleaned): strong flicker, dense
# dust together with scratches, visible grain - also lightly compressed, as on
# a raw scan - harder blur and fading. Overrides of ``P``; the frame repeats
# stay as calibrated.
P_HEAVY = {
    "tone_prob": 1.0,
    "black": (0.05, 0.2),
    "white": (0.65, 0.9),
    "gamma": (0.6, 1.6),
    "flicker": (0.01, 0.05),
    "tex_prob": (0.4, 0.9),
    "tex_opacity": (0.5, 1.0),
    "line_opacity": (0.7, 1.0),
    "moving_line_prob": 0.7,
    "dust_with_scratches": True,
    "blur_prob": 0.8,
    "blur_strength": 1.0,
    "noise_std": (4.0 / 255.0, 12.0 / 255.0),
    "grain_size": (0.5, 1.2),      # clumped grain that survives the resize
    "lr_soft_prob": 0.3,
    "crf": (14, 30),               # grain visible needs a light encode (raw scan)
}

# Dirty mode: MambaOFR's degradation_video_list_4 damage (a texture on every
# frame, opacity 0.6-1, any folder; noise 5-10 / 255) on top of the calibrated
# transfer chain. Scratches drift slowly as in ``P``, not MambaOFR's 5-15 px jumps.
P_DIRTY = {
    "tex_prob": (0.8, 1.0),
    "tex_opacity": (0.6, 1.0),
    "tex_all_folders": True,
    "line_opacity": (0.6, 1.0),
    "moving_line_prob": 0.3,
    "dust_with_scratches": True,
    "noise_std": (2.0 / 255.0, 10.0 / 255.0),
    "flicker": (0.002, 0.03),
    "blur_prob": 0.7,
}

MODES = {"realistic": P, "heavy": {**P, **P_HEAVY}, "dirty": {**P, **P_DIRTY}}
MODE_PROBS = {"realistic": 0.4, "heavy": 0.2, "dirty": 0.4}


def sample_mode(mode_probs: dict | None = None) -> dict:
    """Parameters of one clip's degradation mode, drawn from ``mode_probs``
    (name -> weight, default ``MODE_PROBS``)."""
    probs = {k: float(v) for k, v in (mode_probs or MODE_PROBS).items() if float(v) > 0}
    unknown = set(probs) - set(MODES)
    if unknown or not probs:
        raise ValueError(f"degradation modes must be weights over {sorted(MODES)}, got {mode_probs}")
    name = random.choices(list(probs), weights=list(probs.values()))[0]
    return MODES[name]


# Texture folders of data/raw/noise_data (see their contents): vertical scratches
# (001 with the gate vignette), and dust, dirt, blotches and grain.
SCRATCH_FOLDERS = ("001", "002", "007", "009", "011", "012")
DUST_FOLDERS = ("003", "004", "005", "006", "008", "010")


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


def _blend(x: torch.Tensor, texture: np.ndarray, mode: int, opacity: float) -> torch.Tensor:
    """Blend a uint8 (H, W) texture onto a (1, 1, H, W) frame in [0, 1];
    ``mode`` 0 / 1 / 2 = addition / subtract / multiply."""
    frame_rgba = _add_alpha_channel((x[0, 0] * 255.0)[..., None].expand(-1, -1, 3))
    tex = torch.from_numpy(texture).float().to(x.device)[..., None].expand(-1, -1, 3)
    blended = (addition, subtract, multiply)[mode](frame_rgba, _add_alpha_channel(tex), opacity)
    return blended[:, :, :1].permute(2, 0, 1).unsqueeze(0) / 255.0


def _log_uniform(a: float, b: float) -> float:
    return math.exp(random.uniform(math.log(a), math.log(b)))


def _flatten(texture: np.ndarray) -> np.ndarray:
    """Lift a texture's background to white, keeping only its local marks.

    Several textures carry a vignette or a grey base (001, 008, 011). Drawn on
    single frames, that base changes the whole frame's brightness from frame to
    frame - with it the synthetic clips flickered 2x more than old_films.
    """
    t = texture.astype(np.float32)
    bg = cv2.GaussianBlur(t, (0, 0), max(t.shape) / 30.0)
    return np.clip(t + (255.0 - bg), 0, 255).astype(np.uint8)


def sample_frame_ids(n: int) -> list:
    """Which film frame each of ``n`` video frames shows (telecine / a low
    shooting rate): ``[0, 1, 1, 2, ...]``. In old_films each film frame is
    shown once (82 %) or twice (16 %)."""
    u = random.random()
    if u < P["hold_none"]:
        return list(range(n))
    if u < P["hold_none"] + P["hold_slow"]:
        rate = random.uniform(*P["hold_slow_rate"])
    else:
        rate = random.uniform(*P["hold_rate"])
    phase = random.random()
    return [int(k * rate + phase) - int(phase) for k in range(n)]


def apply_holes_to_window(
    frames_bgr: list,
    hole_prob: float,
) -> tuple[list, np.ndarray | None]:
    """Optionally overlay a persistent spatial hole mask on every frame of a window.

    With probability ``hole_prob`` a single hole mask is generated for the
    window's resolution and applied to every frame (consistent damage across
    the whole window, as if the film strip was torn). Hole pixels are set to 0
    in uint8 space, which becomes -1.0 after [-1, 1] normalisation. The model
    is NOT told where they are — it has to detect them; the returned mask is
    only a training target.

    Args:
        frames_bgr: list of uint8 ndarrays (greyscale or BGR), all same spatial size.
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
    downscale_factor: int = 4,
    device: torch.device | None = None,
    out_size: tuple | None = None,
    hole_prob: float = 0.0,
    hole_window: int | None = None,
    return_masks: bool = False,
    frame_ids: list | None = None,
    mode_probs: dict | None = None,
) -> list | tuple[list, list]:
    """Degrade a clip; see the module docstring for the order of operations.

    The output size is ``out_size=(height, width)`` (takes precedence) or the
    native size divided by ``downscale_factor``. ``frame_ids`` (one per frame,
    e.g. from ``sample_frame_ids``) marks frames showing the same film frame:
    they get the identical film / transfer degradation, as on a real telecine,
    and only the codec treats them as separate pictures. Default: all distinct.

    With ``return_masks=True`` it returns ``(frames, masks)``, where ``masks``
    holds one (h, w) uint8 {0, 1} hole mask per output frame — the training
    target for the hole detector. Without it, only the frames.
    ``mode_probs`` weights the degradation modes (default ``MODE_PROBS``).
    """
    if not frame_list_cv2:
        return ([], []) if return_masks else []

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if frame_ids is None:
        frame_ids = list(range(len(frame_list_cv2)))
    if len(frame_ids) != len(frame_list_cv2):
        raise ValueError("frame_ids needs one id per frame")

    original_h, original_w = frame_list_cv2[0].shape[:2]
    if out_size is not None:
        target_h, target_w = int(out_size[0]), int(out_size[1])
    elif downscale_factor > 1:
        target_h, target_w = original_h // downscale_factor, original_w // downscale_factor
    else:
        target_h, target_w = original_h, original_w

    # ---- per clip: degradation mode
    p = sample_mode(mode_probs)
    deg_params = {  # MambaOFR's degradation_video_list_4 ranges
        "type_value": random.random(),
        "l1_value": random.random(),
        "l2_value": random.random(),
        "angle_value": random.random(),
        "shape_value": random.randint(2, 11),
        "rnum": np.random.rand(),
        "up_scale": random.uniform(1, 2),
        "down_scale": random.uniform(0.5 / 4, 1),
    }
    deg_params["l1_value"] *= p["blur_strength"]
    deg_params["l2_value"] *= p["blur_strength"]
    lr_soft = random.uniform(*p["lr_soft_sigma"]) if random.random() < p["lr_soft_prob"] else 0.0
    if random.random() < p["tone_prob"]:
        tone = (random.uniform(*p["black"]), random.uniform(*p["white"]), _log_uniform(*p["gamma"]))
    else:
        tone = None
    flicker = _log_uniform(*p["flicker"])
    tex_prob = random.uniform(*p["tex_prob"])
    use_blur = random.random() < p["blur_prob"]
    noise_std = random.uniform(*p["noise_std"])
    grain_size = random.uniform(*p["grain_size"])
    crf = random.randint(*p["crf"])
    deblock = random.randint(*p["deblock"])

    # Scratches stay on the film for many frames, dust and grain change every
    # frame (a fresh set of scratches per frame flickered like exposure did).
    use_moving_line = random.random() < p["moving_line_prob"]
    use_dust = not use_moving_line or p["dust_with_scratches"]
    moving_line_mode = random.randint(0, 2)
    last_moving_texture = None
    all_keys = texture_cache.get_all_keys()  # another texture set: no folder split
    dust_keys = all_keys if p["tex_all_folders"] else (texture_cache.get_keys_in(DUST_FOLDERS) or all_keys)
    line_key = random.choice(texture_cache.get_keys_in(SCRATCH_FOLDERS) or all_keys)

    # ---- film + transfer, once per film frame
    film_frames = {}
    for frame_cv2, fid in zip(frame_list_cv2, frame_ids):
        if fid in film_frames:
            continue
        gray = cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2GRAY)
        x = torch.from_numpy(gray).float().to(device)[None, None] / 255.0

        # 1. tone of the print
        if tone is not None:
            x = apply_tone(x, *tone)

        # 2. textures on the film: persistent scratches, per-frame dust / dirt / grain
        if use_moving_line:
            texture_img, folder_name = texture_cache.get_texture(line_key)
            first = last_moving_texture is None
            processed_texture, last_moving_texture = generate_moving_line_texture(
                texture_img, last_moving_texture, original_h, original_w
            )
            if first and folder_name != "001":  # 001's gate vignette stays
                processed_texture = last_moving_texture = _flatten(processed_texture)
            x = _blend(x, processed_texture, moving_line_mode, random.uniform(*p["line_opacity"]))
        if use_dust and random.random() < tex_prob:
            texture_img, folder_name = texture_cache.get_texture(random.choice(dust_keys))
            processed_texture = _flatten(generate_texture(texture_img, folder_name, original_h, original_w))
            blend = 0 if folder_name == "011" else random.randint(0, 2)
            x = _blend(x, processed_texture, blend, random.uniform(*p["tex_opacity"]))

        # 3. exposure flicker
        x = x * math.exp(random.gauss(0.0, flicker))

        # 4. blur, 5. grain
        if use_blur:
            x = apply_blur(x, deg_params)
        noise_type = "gaussian" if random.random() < 0.5 else "speckle"
        x = apply_noise(x.clamp(0.0, 1.0), max(noise_std + random.uniform(-0.5, 0.5) / 255.0, 0.0), noise_type,
                        grain_size)

        # 6. scan resolution, 7. output resolution
        x = apply_downsampling(x, deg_params)
        if x.shape[-2:] != (target_h, target_w):
            x = random_scaling(x, target_w, target_h)
        lr = x[0, 0].clamp(0.0, 1.0).cpu().numpy()
        if lr_soft > 0:
            lr = cv2.GaussianBlur(lr, (0, 0), lr_soft)
        film_frames[fid] = np.clip(lr * 255.0 + 0.5, 0, 255).astype(np.uint8)

    # 8. telecine: repeated film frames are the same picture
    frames = [film_frames[fid] for fid in frame_ids]

    # holes on the film, per window, before the upload
    window = hole_window or len(frames)
    with_holes, masks = [], []
    for start in range(0, len(frames), window):
        chunk, mask = apply_holes_to_window(frames[start : start + window], hole_prob)
        with_holes.extend(chunk)
        if mask is None:
            mask = np.zeros(chunk[0].shape[:2], np.uint8)
        masks.extend([mask] * len(chunk))

    # 9. upload: H.264 of the clip at the output resolution
    coded = apply_video_codec(np.stack(with_holes), crf, deblock)
    out = [cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) for f in coded]
    return (out, masks) if return_masks else out
