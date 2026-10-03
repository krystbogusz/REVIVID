"""Batches for training and evaluation, read from .mp4 files only.

Training:   ``<dir>/*.mp4`` clean clips. Each sample is a random window of
            ``num_frame`` frames and a random ``crop_size`` crop; the input is
            produced on the fly by the degradation pipeline (holes included).
            Batches are ``{"lq": (B, T, 3, h, w), "gt": (B, T, 3, H, W),
            "hole_mask": (B, T, 1, h, w)}``; lq/gt in [-1, 1] with H = h * sr_scale,
            GT greyscale replicated to 3 channels. ``hole_mask`` (1 = burned-in
            hole) is a training target only — the model never receives it.
Evaluation: ``<dir>/degraded/<name>.mp4`` + ``<dir>/gt/<name>.mp4``. Each item is
            one whole clip as frames, not a batch: ``{"name", "lq", "gt"}`` with
            lists of BGR uint8 frames exactly as stored in the files.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .augment import augment_frames
from .degradation.cache import get_texture_cache
from .degradation.pipeline import process_video_frames, sample_degree


def list_mp4(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".mp4")


def frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(n, 0)


def read_mp4(path: Path, start: int = 0, count: Optional[int] = None) -> List[np.ndarray]:
    """BGR frames ``[start, start + count)``; a short clip is padded with its last frame."""
    cap = cv2.VideoCapture(str(path))
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    while count is None or len(frames) < count:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise IOError(f"cannot read frames from {path}")
    if count is not None:
        frames += [frames[-1]] * (count - len(frames))
    return frames


def to_tensor(frames_bgr: List[np.ndarray], gray: bool = False) -> torch.Tensor:
    """BGR uint8 frames -> (T, 3, H, W) float in [-1, 1]."""
    out = []
    for f in frames_bgr:
        if gray:
            f = cv2.cvtColor(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)
        else:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        out.append(torch.from_numpy(f).permute(2, 0, 1))
    return torch.stack(out).float().div(127.5).sub(1.0)


class TrainDataset(Dataset):
    def __init__(
        self,
        clips_dir: str | Path,
        num_frame: int,
        sr_scale: int,
        crop_size: Optional[Sequence[int]],
        hole_prob: float,
        augment: bool = True,
        texture_cache_dir: str | Path | None = None,
    ):
        self.clips = list_mp4(clips_dir)
        if not self.clips:
            raise FileNotFoundError(f"no .mp4 clips in {clips_dir}")
        self.num_frame = num_frame
        self.sr = sr_scale
        self.hole_prob = hole_prob
        self.augment = augment
        self.texture_cache_dir = texture_cache_dir
        self.crop = None  # (h, w) in GT pixels
        if crop_size:
            w, h = int(crop_size[0]), int(crop_size[1])
            m = sr_scale * 4  # LQ stays a multiple of 4 for the refiner UNet
            if w % m or h % m:
                raise ValueError(f"crop_size {w}x{h} must be a multiple of sr_scale*4 = {m}")
            self.crop = (h, w)

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int):
        path = self.clips[index]
        start = random.randint(0, max(0, frame_count(path) - self.num_frame))
        frames = read_mp4(path, start, self.num_frame)

        sr = self.sr
        h, w = frames[0].shape[0] // sr * sr, frames[0].shape[1] // sr * sr
        ph, pw = (min(self.crop[0], h), min(self.crop[1], w)) if self.crop else (h, w)
        y = random.randint(0, h - ph) // sr * sr
        x = random.randint(0, w - pw) // sr * sr
        gt = [f[y : y + ph, x : x + pw] for f in frames]

        lq, masks = process_video_frames(
            gt,
            get_texture_cache(cache_dir=self.texture_cache_dir),
            degree=sample_degree(),
            device=torch.device("cpu"),
            out_size=(ph // sr, pw // sr),
            hole_prob=self.hole_prob,
            return_masks=True,
        )

        if self.augment:
            n = len(gt)
            both = augment_frames(gt + lq + masks, hflip=True, rotation=True, transpose=(ph == pw))
            gt, lq, masks = both[:n], both[n : 2 * n], both[2 * n :]

        return {
            "lq": to_tensor(lq),
            "gt": to_tensor(gt, gray=True),
            "hole_mask": torch.from_numpy(np.stack(masks)).float().unsqueeze(1),
        }


class EvalDataset(Dataset):
    def __init__(self, root: str | Path):
        root = Path(root)
        self.pairs = [(d, root / "gt" / d.name) for d in list_mp4(root / "degraded")]
        self.pairs = [(d, g) for d, g in self.pairs if g.is_file()]
        if not self.pairs:
            raise FileNotFoundError(f"no degraded/gt .mp4 pairs in {root}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        deg_path, gt_path = self.pairs[index]
        n = min(frame_count(deg_path), frame_count(gt_path))
        return {
            "name": deg_path.stem,
            "lq": read_mp4(deg_path, 0, n),
            "gt": read_mp4(gt_path, 0, n),
        }


def _seed_worker(worker_id: int) -> None:
    # torch seeds each worker differently; numpy and random must follow, or every
    # worker would draw the same degradations.
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    cv2.setNumThreads(0)


def train_loader(
    clips_dir: str | Path,
    num_frame: int,
    sr_scale: int,
    crop_size: Optional[Sequence[int]],
    hole_prob: float,
    batch_size: int = 1,
    num_workers: int = 0,
    augment: bool = True,
    texture_cache_dir: str | Path | None = None,
) -> DataLoader:
    dataset = TrainDataset(
        clips_dir, num_frame, sr_scale, crop_size, hole_prob, augment, texture_cache_dir
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
    )


def _as_is(item):
    return item


def eval_loader(
    root: str | Path, indices: Optional[Sequence[int]] = None, num_workers: int = 0
) -> DataLoader:
    """Clips one by one as frames (no batching, no tensors); ``indices`` picks a subset."""
    dataset = EvalDataset(root)
    if indices is not None:
        dataset = Subset(dataset, list(indices))
    return DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_as_is,
    )
