"""Start REVIVID training (the ``Trainer`` class lives in ``trainer/trainer.py``).

No arguments needed — run it from anywhere (or from an IDE):

    python src/trainer/train.py                       # phase 1: restoration + SR (config/REVIVID.yaml)
    python src/trainer/train.py --config config/REVIVID_inpaint.yaml   # phase 2: inpainting
    python src/trainer/train.py --resume path.pth     # continue from a given checkpoint

Training continues automatically from <logging.exp_dir>/checkpoints/latest.pth
when it exists; a phase-2 run without one starts from ``training.init_from``
(the phase-1 weights).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    # Run as a script: make the packages under src/ importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if os.name != "nt":
    # Less fragmentation of the CUDA caching allocator; must be set before torch loads.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from trainer.trainer import Trainer  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Train REVIVID.")
    p.add_argument("--config", default=None, help="YAML config (default: config/REVIVID.yaml)")
    p.add_argument("--resume", default=None, help="checkpoint (default: <exp_dir>/checkpoints/latest.pth)")
    args = p.parse_args()

    trainer = Trainer(args.config)
    start_epoch = trainer.resume(args.resume)
    trainer.save_config()
    train_loader, val_loader = trainer.build_loaders()
    trainer.fit(train_loader, val_loader, start_epoch)


if __name__ == "__main__":
    main()
