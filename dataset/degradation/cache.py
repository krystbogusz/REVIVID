"""Texture cache backed by a memory-mapped pixel archive.

The mmap files live under ``data/training/noise_textures/`` and are built by
:class:`dataset.dataset_creator.DatasetCreator`.  All training processes mmap
the same read-only file; the OS shares physical RAM pages across workers.
"""

from __future__ import annotations

import glob
import json
import os
import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_TEXTURE_DIR = _PROJECT_ROOT / "data" / "raw" / "noise_data"
_DEFAULT_MMAP_CACHE_DIR = _PROJECT_ROOT / "data" / "training" / "noise_textures"
_MANIFEST_NAME = "noise_textures_manifest.json"
_PIXELS_NAME = "noise_textures_pixels.bin"
_SOURCE_NAME = "noise_textures_source.txt"
_BUILD_LOCK_NAME = ".building.lock"

# One open mmap handle per process (cheap); physical pages are shared by the OS.
_process_caches: dict[str, "TextureCache"] = {}


def default_texture_dir() -> Path:
    return _DEFAULT_TEXTURE_DIR


def default_mmap_cache_dir() -> Path:
    return _DEFAULT_MMAP_CACHE_DIR


def resolve_texture_dir(texture_dir: str | Path | None) -> Path:
    """Return an absolute source texture directory."""
    if texture_dir is None or str(texture_dir).strip() == "":
        return _DEFAULT_TEXTURE_DIR
    path = Path(texture_dir)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path


def resolve_mmap_cache_dir(cache_dir: str | Path | None) -> Path:
    """Return the mmap archive directory under ``data/training`` by default."""
    if cache_dir is None or str(cache_dir).strip() == "":
        return _DEFAULT_MMAP_CACHE_DIR
    path = Path(cache_dir)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _iter_texture_files(texture_dir: Path) -> Iterator[str]:
    pattern = os.path.join(str(texture_dir), "**", "*")
    for file_path in sorted(glob.glob(pattern, recursive=True)):
        if Path(file_path).suffix.lower() not in _IMAGE_EXTS:
            continue
        yield file_path.replace("\\", "/")


def _needs_rebuild(texture_dir: Path, cache_dir: Path) -> bool:
    manifest_path = cache_dir / _MANIFEST_NAME
    pixels_path = cache_dir / _PIXELS_NAME
    if not manifest_path.exists() or not pixels_path.exists():
        return True
    source_path = cache_dir / _SOURCE_NAME
    resolved_source = str(texture_dir.resolve())
    if not source_path.exists() or source_path.read_text(encoding="utf-8").strip() != resolved_source:
        return True
    manifest_mtime = manifest_path.stat().st_mtime
    for file_path in _iter_texture_files(texture_dir):
        if os.path.getmtime(file_path) > manifest_mtime:
            return True
    return False


def _build_mmap_cache(texture_dir: Path, cache_dir: Path) -> None:
    entries: list[dict] = []
    moving_line_keys: list[str] = []
    chunks: list[np.ndarray] = []
    offset = 0

    for file_path in _iter_texture_files(texture_dir):
        img_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            continue
        h, w = img_gray.shape
        flat = np.ascontiguousarray(img_gray).ravel()
        folder_name = os.path.basename(os.path.dirname(file_path))
        entries.append(
            {
                "key": file_path,
                "folder": folder_name,
                "h": int(h),
                "w": int(w),
                "offset": int(offset),
            }
        )
        if folder_name == "001":
            moving_line_keys.append(file_path)
        chunks.append(flat)
        offset += int(h * w)

    if not entries:
        raise FileNotFoundError(
            f"No texture images found under {texture_dir}. "
            "Download noise_data or set training.texture_dir in config."
        )

    pixels = np.concatenate(chunks).astype(np.uint8, copy=False)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_pixels = cache_dir / f"{_PIXELS_NAME}.tmp"
    tmp_manifest = cache_dir / f"{_MANIFEST_NAME}.tmp"
    tmp_pixels.write_bytes(pixels.tobytes())
    tmp_manifest.write_text(
        json.dumps({"entries": entries, "moving_line_keys": moving_line_keys}, indent=2),
        encoding="utf-8",
    )
    tmp_pixels.replace(cache_dir / _PIXELS_NAME)
    tmp_manifest.replace(cache_dir / _MANIFEST_NAME)
    (cache_dir / _SOURCE_NAME).write_text(str(texture_dir.resolve()), encoding="utf-8")
    print(
        f"[texture_mmap] built {len(entries)} textures "
        f"({pixels.nbytes / 1e6:.1f} MB) -> {cache_dir}"
    )


def build_texture_mmap(
    texture_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    """Build or refresh the mmap archive.  Called from DatasetCreator."""
    source = resolve_texture_dir(texture_dir)
    dest = resolve_mmap_cache_dir(cache_dir)
    dest.mkdir(parents=True, exist_ok=True)

    if not _needs_rebuild(source, dest):
        return dest

    lock_path = dest / _BUILD_LOCK_NAME
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if not _needs_rebuild(source, dest):
                return dest
            time.sleep(0.1)

    try:
        if _needs_rebuild(source, dest):
            _build_mmap_cache(source, dest)
    finally:
        lock_path.unlink(missing_ok=True)

    return dest


class TextureCache:
    """Read-only texture lookup backed by a shared memory-mapped pixel file."""

    def __init__(
        self,
        texture_dir: str | Path | None = None,
        cache_dir: str | Path | None = None,
    ):
        self.texture_dir = str(resolve_texture_dir(texture_dir))
        self.cache_dir = resolve_mmap_cache_dir(cache_dir)
        manifest_path = self.cache_dir / _MANIFEST_NAME
        pixels_path = self.cache_dir / _PIXELS_NAME
        if not manifest_path.exists() or not pixels_path.exists():
            raise FileNotFoundError(
                f"Texture mmap not found under {self.cache_dir}. "
                "Run DatasetCreator (or build_texture_mmap) first."
            )

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self._entries: dict[str, dict] = {e["key"]: e for e in manifest["entries"]}
        self._keys = list(self._entries.keys())
        self.moving_line_textures = list(manifest.get("moving_line_keys", []))

        total_pixels = sum(int(e["h"]) * int(e["w"]) for e in manifest["entries"])
        self._pixels = np.memmap(
            pixels_path,
            dtype=np.uint8,
            mode="r",
            shape=(total_pixels,),
        )

    def get_all_keys(self) -> list[str]:
        return self._keys

    def get_moving_line_keys(self) -> list[str]:
        return self.moving_line_textures

    def get_texture(self, key: str) -> tuple[np.ndarray, str]:
        entry = self._entries[key]
        h, w = int(entry["h"]), int(entry["w"])
        offset = int(entry["offset"])
        view = self._pixels[offset : offset + h * w].reshape(h, w)
        return np.asarray(view), entry["folder"]


def get_texture_cache(
    texture_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> TextureCache:
    """Open the mmap archive (must already exist — built by DatasetCreator)."""
    resolved_cache = str(resolve_mmap_cache_dir(cache_dir).resolve())
    cache = _process_caches.get(resolved_cache)
    if cache is None:
        cache = TextureCache(texture_dir=texture_dir, cache_dir=resolved_cache)
        _process_caches[resolved_cache] = cache
    return cache
