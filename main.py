import os
import shutil
# Reduce CUDA fragmentation OOMs (must be set before torch is imported).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path

from dataset import DatasetDownloader, DatasetCreator
from dataset.dataset_loader import build_training_loaders
from trainer.trainer import Trainer, load_config

PROJECT_ROOT = Path(__file__).parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TRAINING_DIR = PROJECT_ROOT / "data" / "training"


def _has_clips(split: str) -> bool:
    degraded = TRAINING_DIR / split / "degraded"
    return degraded.exists() and any(degraded.glob("*.mp4"))


def download_data():
    if RAW_DIR.exists():
        print("[pipeline] data/raw directory already exists, skipping download.")
        return
    DatasetDownloader().download_all()


def _reds_source_paths() -> list[str]:
    """All frame folders under REDS (train + val)."""
    paths = []
    for sub in ("train", "val"):
        p = RAW_DIR / "reds" / sub
        if p.exists():
            paths.append(str(p))
    return paths


def _reds_split_ready() -> bool:
    return all(_has_clips(s) for s in ("train", "valid", "test"))


def build_datasets(seed: int = 2021):
    """Degrade only REDS (train + val), then split clips 8:1:1 -> train/valid/test."""
    if TRAINING_DIR.exists():
        print("[pipeline] data/training directory already exists, skipping creation.")
        DatasetCreator.from_config().ensure_texture_mmap()
        return

    print("[pipeline] Building datasets...")
    for split in ("train", "valid"):
        for sub in ("degraded", "gt"):
            d = TRAINING_DIR / split / sub
            if d.exists():
                shutil.rmtree(d)
    
    test_dir = TRAINING_DIR / "test"
    if test_dir.exists():
        shutil.rmtree(test_dir)

    creator = DatasetCreator.from_config()

    reds_train = RAW_DIR / "reds" / "train"
    if reds_train.exists() and any(reds_train.iterdir()):
        print(f"[pipeline] degrading REDS train from {reds_train}...")
        creator.create_dataset("train", str(reds_train))

    reds_val = RAW_DIR / "reds" / "val"
    if reds_val.exists() and any(reds_val.iterdir()):
        print(f"[pipeline] degrading REDS val from {reds_val}...")
        creator.create_dataset("valid", str(reds_val))

    old_films = RAW_DIR / "old_films"
    if old_films.exists() and any(old_films.iterdir()):
        print(f"[pipeline] creating test dataset from {old_films}...")
        creator.create_test_dataset(str(old_films))


def train():
    trainer = Trainer()
    start_epoch = trainer.maybe_resume()
    train_loader, val_loader = build_training_loaders(trainer)
    trainer.fit(train_loader, val_loader=val_loader, start_epoch=start_epoch)


def main():
    cfg = load_config()
    sr_scale = int(cfg.get("model", {}).get("sr_scale", 1))
    seed = int(cfg.get("seed", 2021))
    download_data()
    build_datasets(seed=seed)
    train()


if __name__ == '__main__':
    main()
