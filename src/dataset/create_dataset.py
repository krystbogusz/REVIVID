"""CLI for DatasetCreator. Run from the repo root:

    PYTHONPATH=src python -m dataset.create_dataset train --train DIR --valid DIR --textures DIR
    PYTHONPATH=src python -m dataset.create_dataset eval --source DIR --name val_x2 --scale 2
    # phase 2 validation pairs (with holes, into validation.dir of that config):
    PYTHONPATH=src python -m dataset.create_dataset --config config/REVIVID_inpaint.yaml valid --valid DIR

Sources are video files, frame folders, or directories containing them.
Evaluation reuses the texture cache built by the train mode.
Defaults for scale, window, hole probability, textures and validation size
come from ``--config`` (default config/REVIVID.yaml; hole probability: 0 while
training.stage is restore).
"""

import argparse
import sys
from pathlib import Path

from utils.config import PROJECT_ROOT as ROOT, load_config as _load_yaml

from .dataset_creator import DatasetCreator


def load_config(path=None):
    cfg = _load_yaml(path)
    return cfg.get("model") or {}, cfg.get("training") or {}, cfg.get("validation") or {}


def _config_path():
    """``--config`` from the command line, read before the full parse (its values are the defaults)."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    return pre.parse_known_args(sys.argv[1:])[0].config


def parse_args(model_cfg, train_cfg, val_cfg):
    parser = argparse.ArgumentParser(description="Create REVIVID training or evaluation datasets.")
    parser.add_argument("--config", default=None, help="YAML config the defaults come from (default: config/REVIVID.yaml)")
    sub = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--num-frame", type=int, default=train_cfg.get("num_frame", 7),
                        help="window length for holes")
    # The restore stage trains without holes, so its validation pairs have none either.
    restore = train_cfg.get("stage", "inpaint") == "restore"
    common.add_argument("--hole-prob", type=float,
                        default=0.0 if restore else model_cfg.get("hole_prob", 0.15),
                        help="0 in the restore stage, else model.hole_prob")
    common.add_argument("--fps", type=float, default=24.0, help="fps for frame folders")

    train = sub.add_parser("train", parents=[common], help="build data/training")
    train.add_argument("--textures", default=train_cfg.get("texture_dir"),
                       help="folder with degradation textures (default: training.texture_dir)")
    train.add_argument("--scale", type=int, default=model_cfg.get("sr_scale", 2),
                       help="GT size / degraded size")
    train.add_argument("--train", nargs="+", required=True, help="train sources")
    train.add_argument("--valid", nargs="+", required=True, help="validation sources")
    train.add_argument("--gt-size", type=int, nargs=2, metavar=("W", "H"),
                       default=val_cfg.get("gt_size"),
                       help="validation GT size (default: validation.gt_size)")
    train.add_argument("--output", default=str(ROOT / "data" / "training"))

    va = sub.add_parser("valid", parents=[common], help="build only validation pairs (validation.dir)")
    va.add_argument("--scale", type=int, default=model_cfg.get("sr_scale", 2), help="GT size / degraded size")
    va.add_argument("--valid", nargs="+", required=True, help="validation sources")
    va.add_argument("--gt-size", type=int, nargs=2, metavar=("W", "H"), default=val_cfg.get("gt_size"),
                    help="validation GT size (default: validation.gt_size)")
    va.add_argument("--output", default=str(ROOT / (val_cfg.get("dir") or "data/training/valid")),
                    help="folder of the pairs (default: validation.dir)")

    ev = sub.add_parser("eval", parents=[common], help="build data/evaluation/<name>")
    ev.add_argument("--scale", type=int, default=2, help="GT size / degraded size")
    ev.add_argument("--source", nargs="+", required=True, help="evaluation sources")
    ev.add_argument("--name", required=True, help="dataset name")
    ev.add_argument("--gt-size", type=int, nargs=2, metavar=("W", "H"),
                    help="GT size (default: source size)")
    ev.add_argument("--output", default=str(ROOT / "data" / "evaluation"),
                    help="parent folder; the dataset goes to <output>/<name>")

    return parser.parse_args()


def main():
    args = parse_args(*load_config(_config_path()))
    gt_size = (args.gt_size[1], args.gt_size[0]) if args.gt_size else None  # W H -> (H, W)

    output = Path(args.output)
    if args.mode == "eval":
        output = output / args.name

    creator = DatasetCreator(
        output,
        sr_scale=args.scale,
        num_frame=args.num_frame,
        hole_prob=args.hole_prob,
        gt_size=gt_size,
        fps=args.fps,
    )

    if args.mode == "train":
        creator.create_training(args.train, args.valid, args.textures)
    elif args.mode == "valid":
        creator.create_degraded(args.valid, output)
    else:
        creator.create_evaluation(args.source)


if __name__ == "__main__":
    main()
