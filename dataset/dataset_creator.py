"""Dataset creation pipeline for REVIVID.

Reads raw video/image sources, applies the degradation pipeline, and writes
paired (degraded LR, clean HR) MP4 clips to ``data/training/{split}/``.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Union

import cv2
import torch
import yaml
from tqdm import tqdm

from .degradation.cache import build_texture_mmap, get_texture_cache
from .degradation.pipeline import process_video_frames


class DatasetCreator:
    def __init__(self, sr_scale=4):
        self.project_root = Path(__file__).parent.parent
        self.file_counter = 0
        self.video_exts = ('.mp4', '.mkv', '.avi', '.mov')
        self.img_exts = ('.png', '.jpg', '.jpeg')
        self.texture_dir = self.project_root / "data" / "raw" / "noise_data"
        self.batch_size = 24
        self.fps = 24

        # SR upscaling factor (GT = LR * sr_scale)
        self.sr_scale = sr_scale

        self.size_multiple = 8  # keep H/W divisible by this (UNet + codec friendly)

    @classmethod
    def from_config(cls, config_path: Union[str, Path, None] = None) -> "DatasetCreator":
        """Build a DatasetCreator from ``config/REVIVID.yaml``.

        Reads ``model.sr_scale``.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "REVIVID.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        model_cfg = cfg.get("model", {}) or {}
        sr_scale = int(model_cfg.get("sr_scale", 4))

        creator = cls(sr_scale=sr_scale)
        print(
            f"[DatasetCreator] "
            f"sr_scale={sr_scale} (LR = GT / {sr_scale})"
        )
        return creator

    def ensure_texture_mmap(self, texture_dir=None) -> Path:
        """Build the shared mmap texture archive under ``data/training/noise_textures``."""
        source = texture_dir or self.texture_dir
        cache_dir = build_texture_mmap(source)
        print(f"[DatasetCreator] texture mmap ready: {cache_dir}")
        return cache_dir

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
                        img_paths = [root_path / img for img in images]
                        inputs.append(('frames', root_path, img_paths))
                        dirs.clear()
                    else:
                        for f in files:
                            if Path(f).suffix.lower() in self.video_exts:
                                inputs.append(('video', root_path / f))

        return inputs

    def create_dataset(
        self,
        dataset_mode: str,
        source_paths: Union[str, List[str]],
        bake_holes: bool = None,
    ):
        if dataset_mode not in ["train", "valid", "test"]:
            raise ValueError("dataset_mode must be 'train', 'valid' or 'test'.")

        # Holes are baked into the degraded clips only for the test set so that
        # evaluation is reproducible. For train/valid they are injected on-the-fly
        # by the data loader, so the baked degradation must stay hole-free.
        if bake_holes is None:
            bake_holes = (dataset_mode == "test")

        self.dataset_mode = dataset_mode
        self.file_counter = 0

        target_dir = self.project_root / "data" / "training" / dataset_mode
        degraded_dir = target_dir / "degraded"
        gt_dir = target_dir / "gt"
        degraded_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

        self.ensure_texture_mmap()
        self.texture_cache = get_texture_cache(self.texture_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        inputs = self._gather_inputs(source_paths)

        for item in tqdm(inputs, desc=f"Processing {dataset_mode} dataset"):
            filename = f"{self.file_counter:07d}.mp4"
            out_path_degraded = degraded_dir / filename
            out_path_gt = gt_dir / filename

            current_degree = random.choices([0, 1, 2], weights=[0.1, 0.1, 0.8], k=1)[0]
            self._degrade_item(
                item, out_path_degraded, out_path_gt,
                degree=current_degree, bake_holes=bake_holes
            )

            self.file_counter += 1

    def create_test_dataset(self, source_paths: Union[str, List[str]]):
        """Tworzy zbiór testowy poprzez bezpośrednie kopiowanie plików wideo bez degradacji i podfolderów."""
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
                out_path = target_dir / filename
                shutil.copy(str(item[1]), str(out_path))
                self.file_counter += 1
            else:
                print(f"[DatasetCreator] Pomijanie ścieżki z klatkami zdjęciowymi: {item[1]}. Oczekiwano pełnego wideo.")

    def create_reds_split_dataset(
        self,
        source_paths: Union[str, List[str]],
        train_ratio: float = 0.8,
        valid_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 2021,
    ):
        """Degrade all REDS clips, then assign them to train/valid/test (8:1:1).

        ``source_paths`` should list both ``reds/train`` and ``reds/val`` (or any
        other folders under REDS). Other downloaded datasets are not touched.
        """
        total_r = train_ratio + valid_ratio + test_ratio
        if abs(total_r - 1.0) > 1e-6:
            raise ValueError(f"split ratios must sum to 1.0, got {total_r}")
        self.ensure_texture_mmap()
        self.texture_cache = get_texture_cache(self.texture_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        inputs = self._gather_inputs(source_paths)
        if not inputs:
            print("[DatasetCreator] no REDS inputs found, nothing to process.")
            return

        rng = random.Random(seed)
        rng.shuffle(inputs)

        n = len(inputs)
        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)
        n_test = n - n_train - n_valid

        splits = [
            ("train", inputs[:n_train], False),
            ("valid", inputs[n_train:n_train + n_valid], False),
            ("test", inputs[n_train + n_valid:], True),
        ]

        old_films_dir = self.project_root / "data" / "raw" / "old_films"
        if old_films_dir.exists():
            old_films_inputs = self._gather_inputs([str(old_films_dir)])
            if old_films_inputs:
                splits[2] = ("test", splits[2][1] + old_films_inputs, True)
                print(f"[DatasetCreator] Automatically adding {len(old_films_inputs)} old_films clips to the test split.")

        print(
            f"[DatasetCreator] REDS split {n} clips -> "
            f"train {n_train}, valid {n_valid}, test {n_test} (seed={seed})"
        )

        for dataset_mode, items, bake_holes in splits:
            if not items:
                print(f"[DatasetCreator] skipping empty split: {dataset_mode}")
                continue

            self.dataset_mode = dataset_mode
            self.file_counter = 0

            target_dir = self.project_root / "data" / "training" / dataset_mode
            degraded_dir = target_dir / "degraded"
            gt_dir = target_dir / "gt"
            degraded_dir.mkdir(parents=True, exist_ok=True)
            gt_dir.mkdir(parents=True, exist_ok=True)

            for item in tqdm(items, desc=f"REDS -> {dataset_mode}"):
                filename = f"{self.file_counter:07d}.mp4"
                current_degree = random.choices([0, 1, 2], weights=[0.1, 0.1, 0.8], k=1)[0]
                self._degrade_item(
                    item,
                    degraded_dir / filename,
                    gt_dir / filename,
                    degree=current_degree,
                    bake_holes=bake_holes,
                )
                self.file_counter += 1

    def _target_size(self, height: int, width: int):
        """Preserve original size, but ensure both sides are rounded to a multiple of ``self.size_multiple``."""
        m = self.size_multiple
        th = max(m, int(round(height / m)) * m)
        tw = max(m, int(round(width / m)) * m)
        return th, tw

    def _iter_frames(self, item: tuple):
        """Yield BGR frames for a 'video' or 'frames' item, plus the fps."""
        item_type = item[0]
        if item_type == 'video':
            cap = cv2.VideoCapture(str(item[1]))
            if not cap.isOpened():
                return
            fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
            if fps == 0:
                fps = self.fps
            self._current_fps = fps
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

    def _degrade_item(
        self, item: tuple, out_path_degraded: Path, out_path_gt: Path,
        degree: int, bake_holes: bool = False
    ):
        self._current_fps = self.fps
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        writer_gt = None
        writer_deg = None
        target_h = target_w = None
        deg_h = deg_w = None
        frame_batch = []

        def flush():
            nonlocal frame_batch
            if not frame_batch:
                return
            for f in frame_batch:
                resized = cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_AREA)
                writer_gt.write(resized)
            degraded_batch = process_video_frames(
                frame_batch, self.texture_cache, degree=degree,
                device=self.device, bake_holes=bake_holes,
                out_size=(deg_h, deg_w),
            )
            for df in degraded_batch:
                writer_deg.write(df)
            frame_batch = []

        for frame in self._iter_frames(item):
            if target_h is None:
                h, w = frame.shape[:2]
                target_h, target_w = self._target_size(h, w)
                deg_h, deg_w = target_h // self.sr_scale, target_w // self.sr_scale
                fps = self._current_fps
                writer_gt = cv2.VideoWriter(str(out_path_gt), fourcc, fps, (target_w, target_h))
                writer_deg = cv2.VideoWriter(str(out_path_degraded), fourcc, fps, (deg_w, deg_h))

            frame_batch.append(frame)
            if len(frame_batch) >= self.batch_size:
                flush()

        flush()

        if writer_gt is not None:
            writer_gt.release()
        if writer_deg is not None:
            writer_deg.release()