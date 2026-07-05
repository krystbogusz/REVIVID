"""CLI entry point for training REVIVID.

All hyper-parameters live in ``config/REVIVID.yaml`` (model + training only).
The CLI is intentionally thin: pick a config, optionally resume, or do a quick
synthetic dry run. Without ``--resume``, training auto-continues from
``experiments/revivid/checkpoints/latest.pth`` when that file exists.

Examples
--------
    python -m trainer.train                       # train or auto-resume
    python -m trainer.train --config my.yaml      # custom config
    python -m trainer.train --resume path.pth     # resume from a specific checkpoint
    python -m trainer.train --dry_run             # synthetic smoke run (no dataset)
"""

from __future__ import annotations

import argparse
import copy

from torch.utils.data import DataLoader

from dataset.dataset_loader import SyntheticHoleVideoDataset, build_training_loaders
from trainer.trainer import Trainer, load_config


def parse_args():
    p = argparse.ArgumentParser(description="Train REVIVID DiffMambaOFR")
    p.add_argument("--config", type=str, default=None, help="path to a YAML config")
    p.add_argument("--resume", type=str, default=None, help="checkpoint to resume from")
    p.add_argument("--dry_run", action="store_true", help="few steps on synthetic data")
    return p.parse_args()


def _dry_run_config(path):
    """A small config so the synthetic dry run is fast even on CPU."""
    cfg = copy.deepcopy(load_config(path))
    cfg.setdefault("model", {}).update(
        {
            "num_block": 1,
            "embed_dim": 32,
            "d_state": 8,
            "num_timesteps": 50,
            "refine_steps": 2,
        }
    )
    cfg.setdefault("training", {}).update({"num_frame": 3, "use_amp": False})
    return cfg


def main():
    args = parse_args()

    if args.dry_run:
        trainer = Trainer(config=_dry_run_config(args.config))
        ds = SyntheticHoleVideoDataset(
            length=6, num_frame=3, size=32, sr_scale=int(trainer.model_cfg.sr_scale)
        )
        loader = DataLoader(ds, batch_size=1, shuffle=True)
        trainer.fit(loader, val_loader=loader, epochs=1)
        return

    trainer = Trainer(config=args.config)
    start_epoch = trainer.maybe_resume(args.resume)
    train_loader, val_loader = build_training_loaders(trainer)
    trainer.fit(train_loader, val_loader=val_loader, start_epoch=start_epoch)


if __name__ == "__main__":
    main()
