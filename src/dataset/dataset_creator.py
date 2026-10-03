"""Builds datasets for training or evaluation.

Input (a path, or a list of paths) may be a video file, a folder of frames
(one folder = one clip), or a directory tree containing either.
Output clips are always .mp4.

Training:
    <out>/train/*.mp4              train clips (no degradation)
    <out>/valid/gt/*.mp4           validation clips
    <out>/valid/degraded/*.mp4     validation clips, degraded

Texture cache: <textures_dir>, by default the shared data/training/noise_textures.
Evaluation does not build it, it reuses the one made for training.

Evaluation:
    <out>/gt/*.mp4
    <out>/degraded/*.mp4
"""

import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .degradation.cache import build_texture_mmap, get_texture_cache
from .degradation.pipeline import process_video_frames, sample_degree

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _natural_key(path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def _images_in(folder):
    return sorted((p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS), key=_natural_key)


def find_clips(source):
    """Video files and frame folders found under ``source``."""
    source = Path(source)
    if source.is_file():
        return [source] if source.suffix.lower() in VIDEO_EXTS else []
    if not source.is_dir():
        return []
    if _images_in(source):
        return [source]
    return [c for child in sorted(source.iterdir(), key=_natural_key) for c in find_clips(child)]


def clip_name(clip):
    return clip.name if clip.is_dir() else clip.stem


def read_clip(clip, fps_default):
    """All frames of a clip (BGR uint8) and its fps."""
    if clip.is_dir():
        frames = []
        for path in _images_in(clip):
            data = np.fromfile(str(path), dtype=np.uint8)  # unicode-safe on Windows
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if frame is not None:
                frames.append(frame)
        return frames, fps_default

    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or fps_default
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def write_video(path, frames, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot write {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def to_gray(frame):
    return cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def resize(frame, size):
    """``size`` is (height, width)."""
    if frame.shape[:2] == size:
        return frame
    return cv2.resize(frame, (size[1], size[0]), interpolation=cv2.INTER_AREA)


class DatasetCreator:
    def __init__(
        self,
        output_dir,
        textures_dir=None,
        sr_scale=2,
        num_frame=7,
        hole_prob=0.15,
        gt_size=None,
        fps=24.0,
    ):
        self.output_dir = Path(output_dir)
        self.textures_dir = textures_dir
        self.sr_scale = sr_scale
        self.num_frame = num_frame
        self.hole_prob = hole_prob
        self.gt_size = gt_size  # (height, width) or None = keep source size
        self.fps = fps

    # ---------------------------------------------------------------- public

    def create_training(self, train_source, valid_source, texture_source):
        self.create_textures(texture_source)
        self.copy_clips(train_source, self.output_dir / "train")
        self.create_degraded(valid_source, self.output_dir / "valid")

    def create_evaluation(self, source):
        self.create_degraded(source, self.output_dir)

    def create_textures(self, texture_source):
        """Cache the degradation textures (memory-mapped, read by the training loader)."""
        return build_texture_mmap(texture_source, self.textures_dir)

    def copy_clips(self, source, dest):
        """Every clip in ``source`` -> ``dest/<name>.mp4``, without degradation."""
        for clip, name in tqdm(self._clips(source), desc=f"clips -> {dest}"):
            target = dest / f"{name}.mp4"
            if clip.is_file() and clip.suffix.lower() == ".mp4":
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(clip, target)
                continue
            frames, fps = read_clip(clip, self.fps)
            if frames:
                write_video(target, frames, fps)

    def create_degraded(self, source, dest):
        """Every clip in ``source`` -> ``dest/gt/<name>.mp4`` + ``dest/degraded/<name>.mp4``."""
        textures = get_texture_cache(cache_dir=self.textures_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for clip, name in tqdm(self._clips(source), desc=f"degrade -> {dest}"):
            frames, fps = read_clip(clip, self.fps)
            if not frames:
                continue

            size = self.gt_size or self._round_size(frames[0].shape[:2])
            gt = [to_gray(resize(f, size)) for f in frames]

            degraded = process_video_frames(
                gt,
                textures,
                degree=sample_degree(),
                device=device,
                out_size=(size[0] // self.sr_scale, size[1] // self.sr_scale),
                hole_prob=self.hole_prob,
                hole_window=self.num_frame,
            )

            write_video(dest / "gt" / f"{name}.mp4", gt, fps)
            write_video(dest / "degraded" / f"{name}.mp4", degraded, fps)

    # --------------------------------------------------------------- helpers

    def _clips(self, source):
        """``[(clip, unique_name)]`` for one path or a list of paths."""
        sources = [source] if isinstance(source, (str, Path)) else source
        clips = [c for s in sources for c in find_clips(s)]
        if not clips:
            raise FileNotFoundError(f"no videos or frame folders in {source}")

        seen, named = set(), []
        for i, clip in enumerate(clips):
            name = clip_name(clip)
            if name in seen:
                name = f"{name}_{i}"
            seen.add(name)
            named.append((clip, name))
        return named

    def _round_size(self, size):
        """Round (h, w) to a multiple of sr_scale * 8, so the degraded size is an integer."""
        m = self.sr_scale * 8
        return tuple(max(m, round(s / m) * m) for s in size)
