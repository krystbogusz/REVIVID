"""CLI entry point for training REVIVID — the same as running the trainer itself.

All hyper-parameters live in ``config/REVIVID.yaml``. Without ``--resume``,
training auto-continues from ``<logging.exp_dir>/checkpoints/latest.pth`` when
that file exists.

Examples (run from the repo root)
--------
    python src/trainer/trainer.py                                # simplest: just train
    PYTHONPATH=src python -m trainer.train                       # train or auto-resume
    PYTHONPATH=src python -m trainer.train --config my.yaml      # custom config
    PYTHONPATH=src python -m trainer.train --resume path.pth     # resume from a specific checkpoint
"""

from __future__ import annotations

from trainer.trainer import main

if __name__ == "__main__":
    main()
