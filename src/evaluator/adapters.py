"""Model discovery and inference for the evaluator.

Every folder in the models directory that contains ``inference.py`` is one
self-contained model (code + weights). Its ``load_restorer(checkpoint, device,
**options)`` must return an object with ``restore(frames) -> frames``
(lists of BGR uint8 arrays; the output may be larger than the input, e.g. x2 SR).

Optional ``model.yaml`` (relative paths are resolved against the model folder)::

    checkpoint: weights/net_G_240000.pth
    options:                       # passed to load_restorer
      temporal_length: 20

or several checkpoints of the same model, each evaluated as ``<folder>_<key>``::

    checkpoints:
      best: ../../experiments/revivid/best.pth
      latest: ../../experiments/revivid/latest.pth

Without a checkpoint the folder's ``inference.py`` uses its own default.
"""

from __future__ import annotations

import importlib
import importlib.util
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class ModelSpec:
    name: str
    checkpoint: Path | None
    type: str = "package"  # "package" or "input"
    options: dict = field(default_factory=dict)
    package: Path | None = None


def discover_models(models_dir: Path) -> List[ModelSpec]:
    specs = []
    for entry in sorted(models_dir.iterdir()):
        if entry.name.startswith((".", "_")) or not entry.is_dir():
            continue
        if not (entry / "inference.py").exists():
            print(f"[eval] skipping {entry.name}: no inference.py")
            continue
        desc = entry / "model.yaml"
        cfg = (yaml.safe_load(desc.read_text(encoding="utf-8")) or {}) if desc.exists() else {}
        options = dict(cfg.get("options") or {})
        if cfg.get("checkpoints"):
            for key, path in cfg["checkpoints"].items():
                specs.append(ModelSpec(f"{entry.name}_{key}", (entry / path).resolve(), options=options,
                                       package=entry.resolve()))
        else:
            ckpt = (entry / cfg["checkpoint"]).resolve() if cfg.get("checkpoint") else None
            specs.append(ModelSpec(entry.name, ckpt, options=options, package=entry.resolve()))
    return specs


def _import_package(folder: Path):
    """Import a model folder as a uniquely named package so folders never clash."""
    name = "revivid_models_" + re.sub(r"\W", "_", folder.name)
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, folder / "__init__.py", submodule_search_locations=[str(folder)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return importlib.import_module(f"{name}.inference")


class InputRestorer:
    """Baseline: returns the degraded input unchanged."""

    def restore(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        return frames


def build_restorer(spec: ModelSpec, device: torch.device):
    if spec.type == "input":
        return InputRestorer()
    return _import_package(spec.package).load_restorer(spec.checkpoint, device, **spec.options)
