"""Data loading for REVIVID training / evaluation.

Reads the paired clips produced by :class:`dataset.dataset_creator.DatasetCreator`
(``data/training/{train,valid,test}/{degraded,gt}/*.mp4``), samples a temporal
window, upscales the low-quality input to the GT resolution and normalizes
everything to ``[-1, 1]``.

VFI frame masking
-----------------
During training, with probability ``vfi_prob`` a subset of middle frames is
masked out (set to all-zeros) to simulate Video Frame Interpolation supervision.
The first and last frames are always kept as anchor frames. The ``frame_mask``
boolean tensor returned in the sample indicates which frames are visible
(True = visible, False = masked / VFI-interpolated).

Training reads clean GT clips and applies :func:`dataset.degradation.process_video_frames`
on-the-fly (textures served from ``data/training/noise_textures/*.bin`` mmap cache).
Flip / rotation augmentations run only on that on-the-fly path.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dataset.augment import augment_frames
from dataset.degradation.cache import (
    build_texture_mmap,
    get_texture_cache,
    resolve_texture_dir,
)
from dataset.degradation.pipeline import apply_holes_to_window

VIDEO_EXTS = (".mp4", ".mkv", ".avi", ".mov")


def _video_frame_count(path: str) -> int:
    """Return the number of frames without decoding the video."""
    cap = cv2.VideoCapture(str(path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(count, 0)


def _read_video_frames(path: str, indices: List[int]) -> dict:
    """Decode only the frames at *indices* (must be sorted ascending).

    Returns a ``{frame_index: bgr_array}`` dict. Missing frames (seek failure)
    are filled with the last successfully decoded frame.
    """
    if not indices:
        return {}

    cap = cv2.VideoCapture(str(path))
    frames: dict = {}
    last_frame = None
    target = set(indices)
    max_idx = max(indices)
    start_idx = min(indices)

    if start_idx > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

    fi = start_idx
    while fi <= max_idx:
        ret, frame = cap.read()
        if not ret:
            break
        if fi in target:
            frames[fi] = frame
            last_frame = frame
        fi += 1
    cap.release()

    if last_frame is None and not frames:
        return {}
    fallback = last_frame if last_frame is not None else next(iter(frames.values()))
    return {i: frames.get(i, fallback) for i in indices}


def _to_tensor(frame_rgb: np.ndarray, normalize: bool) -> torch.Tensor:
    t = torch.from_numpy(frame_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    if normalize:
        t = t * 2.0 - 1.0
    return t


def _sample_frame_mask(
    num_frame: int,
    mask_ratio: float,
    vfi_prob: float,
    is_train: bool,
) -> torch.Tensor:
    """Return a boolean visibility mask for a temporal window.

    True  = frame is visible (use real pixel values).
    False = frame is masked for VFI supervision (fill with zeros).

    The first and last frames are always visible (anchor frames).
    Internal frames (indices 1..num_frame-2) are randomly masked when
    training is active and ``random.random() < vfi_prob``.
    """
    if not is_train or vfi_prob <= 0 or random.random() >= vfi_prob:
        return torch.ones(num_frame, dtype=torch.bool)

    mask = torch.ones(num_frame, dtype=torch.bool)
    num_internal = num_frame - 2
    if num_internal <= 0:
        return mask

    num_to_mask = round(num_internal * mask_ratio)
    if num_to_mask <= 0:
        return mask

    internal_indices = list(range(1, num_frame - 1))
    chosen = random.sample(internal_indices, min(num_to_mask, len(internal_indices)))
    for idx in chosen:
        mask[idx] = False
    return mask


class VideoFrameDataset(Dataset):
    def __init__(
        self,
        degraded_dir: str | Path,
        gt_dir: str | Path,
        num_frame: int = 7,
        is_train: bool = True,
        vfi_prob: float = 0.5,
        vfi_mask_ratio: float = 0.3,
        hole_prob: float = 0.15,
        normalize: bool = True,
        sr_scale: int = 1,
        patch_size: int = 0,
        texture_dir: str | None = None,
        use_flip: bool = True,
        use_rot: bool = True,
    ):
        super().__init__()
        self.degraded_dir = Path(degraded_dir)
        self.gt_dir = Path(gt_dir)
        self.num_frame = num_frame
        self.is_train = is_train
        self.vfi_prob = vfi_prob
        self.vfi_mask_ratio = vfi_mask_ratio
        self.hole_prob = hole_prob
        self.normalize = normalize
        self.sr_scale = max(1, int(sr_scale))
        self.patch_size = int(patch_size)
        self.use_flip = use_flip
        self.use_rot = use_rot

        self.texture_dir = (
            str(resolve_texture_dir(texture_dir).resolve()) if self.is_train else None
        )
        self.pairs = self._index_pairs()

    def _index_pairs(self) -> List[Tuple[Path, Path]]:
        if not self.gt_dir.exists():
            return []
        pairs = []
        if self.is_train:
            for gt in sorted(self.gt_dir.iterdir()):
                if gt.suffix.lower() not in VIDEO_EXTS:
                    continue
                pairs.append((gt, gt))
        else:
            if not self.degraded_dir.exists():
                return []
            for deg in sorted(self.degraded_dir.iterdir()):
                if deg.suffix.lower() not in VIDEO_EXTS:
                    continue
                gt = self.gt_dir / deg.name
                if gt.exists():
                    pairs.append((deg, gt))
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def _sample_window(self, total: int) -> List[int]:
        if total <= 0:
            return [0] * self.num_frame
        if not self.is_train:
            return list(range(total))
        if total <= self.num_frame:
            idx = list(range(total)) + [total - 1] * (self.num_frame - total)
            return idx
        start = random.randint(0, total - self.num_frame)
        return list(range(start, start + self.num_frame))

    def __getitem__(self, index: int):
        deg_path, gt_path = self.pairs[index]

        gt_count = _video_frame_count(str(gt_path))
        if self.is_train:
            total = max(gt_count, 1)
        else:
            deg_count = _video_frame_count(str(deg_path))
            total = min(gt_count, deg_count) if deg_count > 0 else gt_count
            total = max(total, 1)

        window = self._sample_window(total)
        sorted_window = sorted(set(window))

        gt_map = _read_video_frames(str(gt_path), sorted_window)
        if self.is_train:
            deg_map = {}
        else:
            deg_map = _read_video_frames(str(deg_path), sorted_window)

        sr = self.sr_scale

        first_gt = gt_map[sorted_window[0]]
        nh, nw = first_gt.shape[:2]
        gh = max(sr, nh // sr * sr)
        gw = max(sr, nw // sr * sr)
        lh, lw = gh // sr, gw // sr

        if self.patch_size > 0:
            ph = min(self.patch_size, gh) // sr * sr
            pw = min(self.patch_size, gw) // sr * sr
            if self.is_train:
                gy = (random.randint(0, gh - ph) // sr) * sr
                gx = (random.randint(0, gw - pw) // sr) * sr
            else:
                gy = ((gh - ph) // 2 // sr) * sr
                gx = ((gw - pw) // 2 // sr) * sr
        else:
            ph, pw, gy, gx = gh, gw, 0, 0
        ly, lx, lph, lpw = gy // sr, gx // sr, ph // sr, pw // sr

        frame_mask = _sample_frame_mask(
            self.num_frame, self.vfi_mask_ratio, self.vfi_prob, self.is_train
        )

        window_gt_bgrs = []
        for frame_idx in window:
            gt_bgr = gt_map[frame_idx]
            if gt_bgr.shape[:2] != (gh, gw):
                gt_bgr = cv2.resize(gt_bgr, (gw, gh), interpolation=cv2.INTER_AREA)
            window_gt_bgrs.append(gt_bgr[gy : gy + ph, gx : gx + pw])

        deg_bgrs = None
        if self.is_train and self.texture_dir is not None:
            from dataset.degradation.pipeline import process_video_frames, sample_degree

            texture_cache = get_texture_cache(self.texture_dir)
            degree = sample_degree()
            deg_bgrs = process_video_frames(
                window_gt_bgrs,
                texture_cache,
                degree=degree,
                downscale_factor=self.sr_scale,
                device=torch.device("cpu"),
                out_size=(lph, lpw),
            )

            deg_bgrs = apply_holes_to_window(deg_bgrs, self.hole_prob)

            if self.use_flip or self.use_rot:
                n_gt = len(window_gt_bgrs)
                combined = augment_frames(
                    window_gt_bgrs + deg_bgrs,
                    hflip=self.use_flip,
                    rotation=self.use_rot,
                )
                window_gt_bgrs = combined[:n_gt]
                deg_bgrs = combined[n_gt:]
                lph, lpw = deg_bgrs[0].shape[:2]

        gts = []
        for gt_crop in window_gt_bgrs:

            gt_gray = cv2.cvtColor(gt_crop, cv2.COLOR_BGR2GRAY)
            gt_rgb = cv2.cvtColor(gt_gray, cv2.COLOR_GRAY2RGB)
            gts.append(_to_tensor(gt_rgb, self.normalize))

        lqs = []
        for pos_idx, frame_idx in enumerate(window):
            if not frame_mask[pos_idx]:

                lq_t = torch.zeros(3, lph, lpw)
            else:
                if self.is_train and deg_bgrs is not None:
                    deg_bgr = deg_bgrs[pos_idx]
                else:
                    if deg_map:
                        deg_bgr = deg_map[frame_idx]
                        if deg_bgr.shape[:2] != (lh, lw):
                            deg_bgr = cv2.resize(
                                deg_bgr, (lw, lh), interpolation=cv2.INTER_LINEAR
                            )
                    else:
                        deg_bgr = cv2.resize(
                            window_gt_bgrs[pos_idx],
                            (lw, lh),
                            interpolation=cv2.INTER_AREA,
                        )
                    deg_bgr = deg_bgr[ly : ly + lph, lx : lx + lpw]

                deg_rgb = cv2.cvtColor(deg_bgr, cv2.COLOR_BGR2RGB)
                lq_t = _to_tensor(deg_rgb, self.normalize)

            lqs.append(lq_t)

        return {
            "lq": torch.stack(lqs, 0),
            "gt": torch.stack(gts, 0),
            "frame_mask": frame_mask,
            "key": deg_path.stem,
        }


class SyntheticHoleVideoDataset(Dataset):
    """In-memory random clips (for tests / dry runs)."""

    def __init__(
        self, length: int = 8, num_frame: int = 5, size: int = 64, sr_scale: int = 1
    ):
        self.length = length
        self.num_frame = num_frame
        self.size = size
        self.sr_scale = max(1, int(sr_scale))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        t, s, sr = self.num_frame, self.size, self.sr_scale
        hs = s * sr
        gt = torch.rand(t, 3, hs, hs) * 2 - 1
        lq = torch.rand(t, 3, s, s) * 2 - 1

        frame_mask = torch.ones(t, dtype=torch.bool)
        return {
            "lq": lq,
            "gt": gt,
            "frame_mask": frame_mask,
            "key": f"synthetic_{index:04d}",
        }


def _default_root() -> Path:
    return Path(__file__).parent.parent / "data" / "training"


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    cv2.setNumThreads(0)


def warmup_dataloader(loader: DataLoader, name: str = "train") -> None:
    """Spawn workers and fetch one batch before epoch progress tracking starts."""
    num_workers = int(getattr(loader, "num_workers", 0) or 0)
    if num_workers <= 0:
        return
    import time

    print(f"[dataloader] starting {num_workers} {name} worker(s)...")
    t0 = time.time()
    it = iter(loader)
    next(it)
    del it
    print(f"[dataloader] {name} workers ready ({time.time() - t0:.1f}s)")


def get_loader(
    split: str = "train",
    batch_size: int = 1,
    num_frame: int = 7,
    num_workers: int = 0,
    vfi_prob: float = 0.5,
    vfi_mask_ratio: float = 0.3,
    hole_prob: float = 0.15,
    root: Optional[str] = None,
    shuffle: Optional[bool] = None,
    sr_scale: int = 1,
    patch_size: int = 0,
    texture_dir: str | None = None,
    use_flip: bool = True,
    use_rot: bool = True,
) -> DataLoader:
    root = Path(root) if root is not None else _default_root()
    is_train = split == "train"
    dataset = VideoFrameDataset(
        degraded_dir=root / split / "degraded",
        gt_dir=root / split / "gt",
        num_frame=num_frame,
        is_train=is_train,
        vfi_prob=vfi_prob,
        vfi_mask_ratio=vfi_mask_ratio,
        hole_prob=hole_prob,
        sr_scale=sr_scale,
        patch_size=patch_size,
        texture_dir=texture_dir,
        use_flip=use_flip,
        use_rot=use_rot,
    )
    if len(dataset) == 0:
        raise FileNotFoundError(
            f"No paired clips found under {root / split}. Run DatasetCreator first "
            f"or pass a different root."
        )
    shuffle = is_train if shuffle is None else shuffle
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        drop_last=is_train,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )


def build_training_loaders(trainer) -> tuple[DataLoader, Optional[DataLoader]]:
    """Build train + val loaders with identical settings for main and trainer CLI."""
    tc = trainer.train_cfg
    val_cfg = trainer.val_cfg
    num_frame = int(tc.get("num_frame", 7))
    vfi_prob = float(trainer.model_cfg.vfi_prob)
    vfi_mask_ratio = float(trainer.model_cfg.vfi_mask_ratio)
    hole_prob = float(trainer.model_cfg.hole_prob)
    sr_scale = int(trainer.model_cfg.sr_scale)
    patch_size = int(tc.get("patch_size", 0))
    val_patch_size = int(val_cfg.get("patch_size", patch_size))
    texture_dir = tc.get("texture_dir")

    build_texture_mmap(texture_dir)

    train_loader = get_train_loader(
        batch_size=int(tc.get("batch_size", 1)),
        num_frame=num_frame,
        num_workers=int(tc.get("num_workers", 0)),
        vfi_prob=vfi_prob,
        vfi_mask_ratio=vfi_mask_ratio,
        hole_prob=hole_prob,
        sr_scale=sr_scale,
        patch_size=patch_size,
        texture_dir=texture_dir,
        use_flip=bool(tc.get("use_flip", True)),
        use_rot=bool(tc.get("use_rot", True)),
    )
    try:
        val_loader = get_valid_loader(
            num_frame=num_frame,
            num_workers=0,
            vfi_prob=vfi_prob,
            vfi_mask_ratio=vfi_mask_ratio,
            sr_scale=sr_scale,
            patch_size=val_patch_size,
        )
    except FileNotFoundError:
        val_loader = None
    return train_loader, val_loader


def get_train_loader(**kwargs) -> DataLoader:
    return get_loader(split="train", **kwargs)


def get_valid_loader(**kwargs) -> DataLoader:
    kwargs.setdefault("batch_size", 1)
    return get_loader(split="valid", **kwargs)


def get_test_loader(**kwargs) -> DataLoader:

    kwargs.setdefault("batch_size", 1)
    kwargs.setdefault("vfi_prob", 0.0)
    return get_loader(split="test", **kwargs)
