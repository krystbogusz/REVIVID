"""Dataset creation pipeline for REVIVID.

Reads raw video/image sources, applies the degradation pipeline to the ENTIRE
clip, then divides the clip into non-overlapping windows of ``num_frame`` frames.
Holes and VFI masking are decided independently for each window so the model sees 
diverse augmentations within a single source video. Finally, the entire sequence
is written as a SINGLE paired MP4 clip (degraded LR, clean HR).

Pipeline per clip:
    1. Read all frames → resize to target GT resolution.
    2. Convert GT frames to greyscale (MambaOFR-style).
    3. Apply base degradations to the full clip via ``process_video_frames``
       (blur, noise, JPEG, texture overlay, color jitter — no holes/VFI).
    4. Chunk the clip into blocks of ``num_frame``. For each chunk:
         a. Randomly apply persistent holes (prob = ``hole_prob``).
         b. Randomly apply VFI masking (prob = ``vfi_prob``).
    5. Write all the chunked frames sequentially into one paired MP4 file.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from .degradation.cache import build_texture_mmap, get_texture_cache
from .degradation.pipeline import (
    process_video_frames,
    apply_holes_to_window,
    sample_degree,
)


def _to_grayscale_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR frame to greyscale (R=G=B), MambaOFR-style transfer_1 equivalent."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class DatasetCreator:
    def __init__(
        self,
        sr_scale: int = 4,
        num_frame: int = 7,
        vfi_prob: float = 0.5,
        vfi_mask_ratio: float = 0.3,
        hole_prob: float = 0.15,
    ):
        self.project_root = Path(__file__).parent.parent
        self.file_counter = 0
        self.video_exts = ('.mp4', '.mkv', '.avi', '.mov')
        self.img_exts = ('.png', '.jpg', '.jpeg')
        self.texture_dir = self.project_root / "data" / "raw" / "noise_data"
        self.fps = 24

        self.sr_scale = sr_scale
        self.num_frame = num_frame
        self.vfi_prob = vfi_prob
        self.vfi_mask_ratio = vfi_mask_ratio
        self.hole_prob = hole_prob

        self.size_multiple = 8  # keep H/W divisible by this (UNet + codec friendly)

    @classmethod
    def from_config(cls, config_path: Union[str, Path, None] = None) -> "DatasetCreator":
        """Build a DatasetCreator from ``config/REVIVID.yaml``.

        Reads ``model.sr_scale``, ``training.num_frame``,
        ``model.vfi_prob``, ``model.vfi_mask_ratio``, ``model.hole_prob``.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "REVIVID.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        model_cfg  = cfg.get("model", {}) or {}
        train_cfg  = cfg.get("training", {}) or {}

        sr_scale       = int(model_cfg.get("sr_scale", 4))
        num_frame      = int(train_cfg.get("num_frame", 7))
        vfi_prob       = float(model_cfg.get("vfi_prob", 0.5))
        vfi_mask_ratio = float(model_cfg.get("vfi_mask_ratio", 0.3))
        hole_prob      = float(model_cfg.get("hole_prob", 0.15))

        creator = cls(
            sr_scale=sr_scale,
            num_frame=num_frame,
            vfi_prob=vfi_prob,
            vfi_mask_ratio=vfi_mask_ratio,
            hole_prob=hole_prob,
        )
        print(
            f"[DatasetCreator] sr_scale={sr_scale}, num_frame={num_frame}, "
            f"vfi_prob={vfi_prob}, vfi_mask_ratio={vfi_mask_ratio}, hole_prob={hole_prob}"
        )
        return creator

    # ------------------------------------------------------------------ #
    # Texture setup
    # ------------------------------------------------------------------ #

    def ensure_texture_mmap(self, texture_dir=None) -> Path:
        """Build the shared mmap texture archive under ``data/training/noise_textures``."""
        source = texture_dir or self.texture_dir
        cache_dir = build_texture_mmap(source)
        print(f"[DatasetCreator] texture mmap ready: {cache_dir}")
        return cache_dir

    # ------------------------------------------------------------------ #
    # Input gathering
    # ------------------------------------------------------------------ #

    def _gather_inputs(self, source_paths: Union[str, List[str]]):
        if isinstance(source_paths, (str, Path)):
            source_paths = [source_paths]

        inputs = []
        for sp in source_paths:
            sp = Path(sp)
            if not sp.exists():
                continue

            if sp.is_file() and sp.suffix.lower() in self.video_exts:
                inputs.append(('video', sp))
                continue

            if sp.is_dir():
                for root, dirs, files in os.walk(sp):
                    root_path = Path(root)
                    images = sorted(
                        [f for f in files if Path(f).suffix.lower() in self.img_exts]
                    )
                    if len(images) > 0:
                        inputs.append(('frames', root_path, [root_path / img for img in images]))
                        dirs.clear()
                    else:
                        for f in files:
                            if Path(f).suffix.lower() in self.video_exts:
                                inputs.append(('video', root_path / f))

        return inputs

    # ------------------------------------------------------------------ #
    # Size helpers
    # ------------------------------------------------------------------ #

    def _target_size(self, height: int, width: int) -> Tuple[int, int]:
        """Round to nearest multiple of ``size_multiple``."""
        m = self.size_multiple
        th = max(m, int(round(height / m)) * m)
        tw = max(m, int(round(width / m)) * m)
        return th, tw

    def _iter_frames(self, item: tuple):
        """Yield BGR frames for a 'video' or 'frames' item."""
        item_type = item[0]
        if item_type == 'video':
            cap = cv2.VideoCapture(str(item[1]))
            if not cap.isOpened():
                return
            fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
            self._current_fps = fps if fps > 0 else self.fps
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
            cap.release()
        elif item_type == 'frames':
            self._current_fps = self.fps
            for img_p in item[2]:
                frame = cv2.imread(str(img_p))
                if frame is not None:
                    yield frame

    # ------------------------------------------------------------------ #
    # VFI mask helpers
    # ------------------------------------------------------------------ #

    def _sample_vfi_mask(self, window_size: int) -> Optional[List[bool]]:
        """Return a per-frame visibility list (True = visible) or None if no VFI.

        First and last frames are always visible (anchor frames).
        Internal frames are randomly masked up to ``vfi_mask_ratio``.
        """
        if self.vfi_prob <= 0.0 or random.random() >= self.vfi_prob:
            return None  # no VFI for this window

        n = window_size
        mask = [True] * n
        num_internal = n - 2
        if num_internal <= 0:
            return None

        num_to_mask = max(1, round(num_internal * self.vfi_mask_ratio))
        internal = list(range(1, n - 1))
        for idx in random.sample(internal, min(num_to_mask, len(internal))):
            mask[idx] = False
        return mask

    # ------------------------------------------------------------------ #
    # Core per-item degradation (new windowed approach)
    # ------------------------------------------------------------------ #

    def _degrade_item(
        self,
        item: tuple,
        degraded_dir: Path,
        gt_dir: Path,
        degree: int,
    ) -> int:
        """Degrade one source item and write it as ONE paired clip.

        Returns 1 if successful, 0 if skipped.
        """
        self._current_fps = self.fps

        # 1. Read ALL frames into memory and resize to GT target.
        all_gt_frames: List[np.ndarray] = []
        target_h = target_w = None
        deg_h = deg_w = None

        for frame in self._iter_frames(item):
            if target_h is None:
                h, w = frame.shape[:2]
                target_h, target_w = self._target_size(h, w)
                deg_h = target_h // self.sr_scale
                deg_w = target_w // self.sr_scale

            if frame.shape[:2] != (target_h, target_w):
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            # GT is greyscale (MambaOFR transfer_1 convention — no colour)
            all_gt_frames.append(_to_grayscale_bgr(frame))

        if not all_gt_frames:
            return 0

        total = len(all_gt_frames)
        fps = self._current_fps
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # 2. Degrade the FULL clip in one shot (no holes/VFI here).
        all_deg_frames = process_video_frames(
            all_gt_frames,
            self.texture_cache,
            degree=degree,
            device=self.device,
            out_size=(deg_h, deg_w),
        )

        # 3. Create writers for the ONE output file.
        filename = f"{self.file_counter:07d}.mp4"
        out_deg = degraded_dir / filename
        out_gt  = gt_dir / filename

        writer_gt  = cv2.VideoWriter(str(out_gt),  fourcc, fps, (target_w, target_h))
        writer_deg = cv2.VideoWriter(str(out_deg), fourcc, fps, (deg_w, deg_h))

        # 4. Iterate over the clip in non-overlapping chunks of num_frame.
        n = self.num_frame
        for start in range(0, total, n):
            gt_chunk  = all_gt_frames[start : start + n]
            deg_chunk = all_deg_frames[start : start + n]
            chunk_len = len(gt_chunk)

            # 4a. Random holes for this chunk
            deg_chunk = apply_holes_to_window(deg_chunk, self.hole_prob)

            # 4b. Random VFI for this chunk
            vfi_mask = self._sample_vfi_mask(chunk_len)
            if vfi_mask is not None:
                for fi, visible in enumerate(vfi_mask):
                    if not visible:
                        deg_chunk[fi] = np.zeros_like(deg_chunk[fi])

            # 4c. Write the frames of this chunk sequentially
            for gt_f, deg_f in zip(gt_chunk, deg_chunk):
                writer_gt.write(gt_f)
                writer_deg.write(deg_f)

        writer_gt.release()
        writer_deg.release()

        self.file_counter += 1
        return 1

    # ------------------------------------------------------------------ #
    # Public dataset creation methods
    # ------------------------------------------------------------------ #

    def _setup_dirs(self, dataset_mode: str) -> Tuple[Path, Path]:
        target_dir = self.project_root / "data" / "training" / dataset_mode
        degraded_dir = target_dir / "degraded"
        gt_dir = target_dir / "gt"
        degraded_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        return degraded_dir, gt_dir

    def _init_shared(self):
        self.ensure_texture_mmap()
        self.texture_cache = get_texture_cache(self.texture_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def create_dataset(
        self,
        dataset_mode: str,
        source_paths: Union[str, List[str]],
    ):
        if dataset_mode not in ["train", "valid", "test"]:
            raise ValueError("dataset_mode must be 'train', 'valid' or 'test'.")

        self.dataset_mode = dataset_mode
        self.file_counter = 0
        self._init_shared()

        degraded_dir, gt_dir = self._setup_dirs(dataset_mode)
        inputs = self._gather_inputs(source_paths)

        for item in tqdm(inputs, desc=f"Processing {dataset_mode} dataset"):
            degree = sample_degree()
            self._degrade_item(item, degraded_dir, gt_dir, degree)

    def create_test_dataset(self, source_paths: Union[str, List[str]]):
        """Tworzy zbiór testowy poprzez bezpośrednie kopiowanie plików wideo bez degradacji."""
        import shutil
        self.dataset_mode = "test"
        self.file_counter = 0

        target_dir = self.project_root / "data" / "training" / "test"
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        inputs = self._gather_inputs(source_paths)
        for item in tqdm(inputs, desc="Tworzenie datasetu testowego (tylko kopiowanie)"):
            item_type = item[0]
            if item_type == 'video':
                filename = f"{self.file_counter:07d}{item[1].suffix}"
                shutil.copy(str(item[1]), str(target_dir / filename))
                self.file_counter += 1
            else:
                print(f"[DatasetCreator] Pomijanie ścieżki z klatkami: {item[1]}. Oczekiwano wideo.")

    def create_reds_split_dataset(
        self,
        source_paths: Union[str, List[str]],
        train_ratio: float = 0.8,
        valid_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 2021,
    ):
        """Degrade all REDS clips with windowing, then assign to train/valid/test."""
        total_r = train_ratio + valid_ratio + test_ratio
        if abs(total_r - 1.0) > 1e-6:
            raise ValueError(f"split ratios must sum to 1.0, got {total_r}")

        self._init_shared()

        inputs = self._gather_inputs(source_paths)
        if not inputs:
            print("[DatasetCreator] no REDS inputs found, nothing to process.")
            return

        rng = random.Random(seed)
        rng.shuffle(inputs)

        n = len(inputs)
        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)

        splits = [
            ("train", inputs[:n_train]),
            ("valid", inputs[n_train:n_train + n_valid]),
            ("test",  inputs[n_train + n_valid:]),
        ]

        old_films_dir = self.project_root / "data" / "raw" / "old_films"
        if old_films_dir.exists():
            old_films_inputs = self._gather_inputs([str(old_films_dir)])
            if old_films_inputs:
                name, items = splits[2]
                splits[2] = (name, items + old_films_inputs)
                print(f"[DatasetCreator] Adding {len(old_films_inputs)} old_films clips to test split.")

        n_test = len(splits[2][1])
        print(
            f"[DatasetCreator] REDS split {n} clips → "
            f"train {n_train}, valid {n_valid}, test {n_test} (seed={seed})"
        )

        for dataset_mode, items in splits:
            if not items:
                print(f"[DatasetCreator] skipping empty split: {dataset_mode}")
                continue

            self.dataset_mode = dataset_mode
            self.file_counter = 0
            degraded_dir, gt_dir = self._setup_dirs(dataset_mode)

            for item in tqdm(items, desc=f"REDS → {dataset_mode}"):
                degree = sample_degree()
                self._degrade_item(item, degraded_dir, gt_dir, degree)