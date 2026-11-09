"""Evaluate every model in a models directory on the same data and compare them.

Run from the repo root (MambaOFR and REVIVID need mamba_ssm):
    PYTHONPATH=src python -m evaluator.evaluate
    PYTHONPATH=src python -m evaluator.evaluate --input-baseline --save-outputs
    PYTHONPATH=src python -m evaluator.evaluate --paired-clips 5 --paired-frames 40   # quick run

Paired set (``--paired-root``, default data/training/valid = the training
validation pairs): ``degraded/<name>.mp4`` is every model's input,
``gt/<name>.mp4`` its GT (read with ``dataset.dataset_loader``). All models get
the same input, so x1 models (MambaOFR) and x2 models (REVIVID) are scored on
the same clips, at two scales:
    LR  the input size. The GT is downscaled the way the degraded clips were
        made (PIL bicubic of the grey frame), outputs larger than the input
        the same way: the restoration alone - MambaOFR's task.
    HR  the GT size. Outputs smaller than the GT (x1 models, the input
        baseline) are upscaled (PIL bicubic): restoration + SR, what a viewer sees.
Full-reference metrics at both scales: PSNR, SSIM, LPIPS, FID (MambaOFR
protocol, ``quality.py``). At HR also the no-reference suite: pyiqa metrics per
frame (``--nr-metrics``, none = skip) + the temporal warping error E*warp.
Real set (old films, no GT): ``*.mp4`` under ``--real-root``, evenly spaced
clips no taller than ``--real-max-height``; no-reference metrics on the native output.
Per-model results are cached in ``<results-dir>/<model>.json`` and reused while
the checkpoint and the settings are unchanged (``--force``).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import gc
import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from dataset.dataset_loader import EvalDataset, eval_loader, frame_count, list_mp4, read_mp4
from evaluator.adapters import PROJECT_ROOT, ModelSpec, build_restorer, discover_models
from evaluator.quality import (
    DEFAULT_NO_REFERENCE,
    FULL_REFERENCE,
    LOWER_IS_BETTER,
    FullReferenceMetrics,
    InceptionFeatures,
    NoReferenceMetrics,
    WarpingError,
    frechet_distance,
    mean_scores,
)

PROTOCOL_VERSION = 4
SCALES = ("lr", "hr")


def clip_height(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return h


def evenly_spaced(items: list, k: Optional[int]) -> list:
    if k is None or k >= len(items):
        return items
    if k <= 0:
        return []
    return [items[i] for i in np.linspace(0, len(items) - 1, k).round().astype(int)]


def _display_path(p: Optional[str]) -> str:
    if not p:
        return "–"
    try:
        return Path(p).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(p)


def resize_gray(frame_bgr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Grey frame resized to ``size`` (h, w) with PIL bicubic - how the degradation
    pipeline downscales the GT to the input size. BGR uint8 in and out."""
    if frame_bgr.shape[:2] == tuple(size):
        return frame_bgr
    grey = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
    return cv2.cvtColor(np.asarray(grey.resize((size[1], size[0]), Image.BICUBIC)), cv2.COLOR_GRAY2BGR)


class Evaluation:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.results_dir = Path(args.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.nr_names = list(args.nr_metrics)

        self.paired_root = Path(args.paired_root).resolve()
        self.paired: List[Tuple[int, str]] = []  # (index in the dataset, clip name)
        if args.paired_clips != 0:
            try:
                pairs = EvalDataset(self.paired_root).pairs
                self.paired = evenly_spaced([(i, d.stem) for i, (d, _) in enumerate(pairs)], args.paired_clips)
            except FileNotFoundError:
                print(f"[eval] no degraded/gt pairs in {self.paired_root} - GT metrics skipped")

        self.real = []
        if args.real_root and args.real_clips != 0:
            limit = args.real_max_height
            eligible = [p for p in list_mp4(args.real_root) if not limit or clip_height(p) <= limit]
            self.real = [(p.stem, p) for p in evenly_spaced(eligible, args.real_clips)]

        self._fr: Optional[FullReferenceMetrics] = None
        self._nr: Optional[NoReferenceMetrics] = None
        self._warp: Optional[WarpingError] = None
        self._inception: Optional[InceptionFeatures] = None
        self._gt_features: Dict[str, np.ndarray] = {}  # per scale, shared by all models

    @property
    def fr(self) -> FullReferenceMetrics:
        if self._fr is None:
            self._fr = FullReferenceMetrics(self.device)
        return self._fr

    @property
    def nr(self) -> NoReferenceMetrics:
        if self._nr is None:
            self._nr = NoReferenceMetrics(self.device, self.nr_names)
        return self._nr

    @property
    def warp(self) -> WarpingError:
        if self._warp is None:
            self._warp = WarpingError(self.device, self.args.flow_max_side)
        return self._warp

    @property
    def inception(self) -> InceptionFeatures:
        if self._inception is None:
            self._inception = InceptionFeatures(self.device)
        return self._inception

    def lower_better(self) -> set:
        return LOWER_IS_BETTER | (self.nr.lower_better if self.nr_names else set())

    def settings(self) -> dict:
        a = self.args
        return {
            "protocol": PROTOCOL_VERSION,
            "paired_root": str(self.paired_root) if self.paired else None,
            "paired": [n for _, n in self.paired],
            "paired_frames": a.paired_frames,
            "real": [n for n, _ in self.real],
            "real_root": str(Path(a.real_root).resolve()) if self.real else None,
            "real_frames": a.real_frames,
            "real_max_height": a.real_max_height,
            "nr_metrics": self.nr_names,
            "flow_max_side": a.flow_max_side,
            "seed": a.seed,
        }

    def fingerprint(self, spec: ModelSpec) -> str:
        ckpt = None
        if spec.checkpoint is not None:
            st = spec.checkpoint.stat()
            ckpt = [str(spec.checkpoint), st.st_size, st.st_mtime_ns]
        blob = json.dumps(
            {"settings": self.settings(), "checkpoint": ckpt, "type": spec.type, "options": spec.options},
            sort_keys=True,
        )
        return hashlib.sha1(blob.encode()).hexdigest()

    def _save_video(self, frames: List[np.ndarray], model: str, split: str, clip: str):
        path = self.results_dir / "outputs" / model / split / f"{clip}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()

    def _restore(self, restorer, frames: List[np.ndarray]) -> List[np.ndarray]:
        torch.manual_seed(self.args.seed)
        return restorer.restore(frames)

    def _no_reference(self, frames: List[np.ndarray]) -> tuple:
        torch.manual_seed(self.args.seed)
        rows = [self.nr(f) for f in frames] if self.nr_names else [{} for _ in frames]
        return rows, self.warp(frames)

    def eval_paired(self, restorer, name: str) -> dict:
        rows = {s: [] for s in SCALES}
        feats = {s: [] for s in SCALES}
        gt_feats = {s: [] for s in SCALES if s not in self._gt_features}
        per_clip, ewarps, scale = {}, [], None
        loader = eval_loader(self.paired_root, [index for index, _ in self.paired])
        for sample in tqdm(loader, total=len(self.paired), desc=f"{name} | paired", unit="clip"):
            clip, n = sample["name"], self.args.paired_frames
            lq, gt = sample["lq"][:n], sample["gt"][:n]
            out = self._restore(restorer, lq)
            scale = out[0].shape[1] / lq[0].shape[1]
            lr_size, hr_size = lq[0].shape[:2], gt[0].shape[:2]
            views = {
                "lr": ([resize_gray(o, lr_size) for o in out], [resize_gray(g, lr_size) for g in gt]),
                "hr": ([resize_gray(o, hr_size) for o in out], gt),
            }
            nr_rows, ewarp = self._no_reference(views["hr"][0])
            per_clip[clip] = {}
            for s, (outs, gts) in views.items():
                clip_rows = [self.fr(o, g) for o, g in zip(outs, gts)]
                if s == "hr":
                    clip_rows = [{**c, **r} for c, r in zip(clip_rows, nr_rows)]
                per_clip[clip][s] = mean_scores(clip_rows)
                rows[s] += clip_rows
                feats[s].append(self.inception(outs))
                if s in gt_feats:
                    gt_feats[s].append(self.inception(gts))
            per_clip[clip]["hr"]["ewarp"] = ewarp
            ewarps.append(ewarp)
            if self.args.save_outputs:
                self._save_video(out, name, "paired", clip)
        for s, f in gt_feats.items():
            self._gt_features[s] = np.concatenate(f)
        result = {"dataset": str(self.paired_root), "scale": scale, "frames": len(rows["hr"]), "per_clip": per_clip}
        for s in SCALES:
            result[s] = {**mean_scores(rows[s]),
                         "fid": frechet_distance(np.concatenate(feats[s]), self._gt_features[s])}
        result["hr"]["ewarp"] = float(np.nanmean(ewarps))
        return result

    def eval_real(self, restorer, name: str) -> dict:
        rows, per_clip, ewarps = [], {}, []
        for clip, path in tqdm(self.real, desc=f"{name} | real", unit="clip"):
            # read_mp4 pads short clips with their last frame; never ask for more than exist
            n = min(self.args.real_frames or frame_count(path), frame_count(path))
            if n <= 0:
                continue
            out = self._restore(restorer, read_mp4(path, 0, n))
            clip_rows, ewarp = self._no_reference(out)
            per_clip[clip] = {**mean_scores(clip_rows), "ewarp": ewarp}
            rows += clip_rows
            ewarps.append(ewarp)
            if self.args.save_outputs:
                self._save_video(out, name, "real", clip)
        return {"dataset": str(Path(self.args.real_root).resolve()), "frames": len(rows),
                "summary": {**mean_scores(rows), "ewarp": float(np.nanmean(ewarps))}, "per_clip": per_clip}

    def evaluate(self, spec: ModelSpec) -> dict:
        cache = self.results_dir / f"{spec.name}.json"
        fp = self.fingerprint(spec)
        if cache.exists() and not self.args.force:
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if cached.get("fingerprint") == fp:
                print(f"[eval] {spec.name}: cached results reused")
                return cached

        print(f"[eval] {spec.name}: {spec.checkpoint or 'degraded input'}")
        restorer = build_restorer(spec, self.device)
        paired = self.eval_paired(restorer, spec.name) if self.paired else None
        real = self.eval_real(restorer, spec.name) if self.real else None
        result = {
            "model": spec.name,
            "checkpoint": str(spec.checkpoint) if spec.checkpoint else None,
            "fingerprint": fp,
            "settings": self.settings(),
            "lower_better": sorted(self.lower_better()),
            "evaluated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "paired": paired,
            "real": real,
        }
        del restorer
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        cache.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result


# ------------------------------------------------------------------ report

def _value(result: dict, table: str, metric: str) -> float:
    """``table``: lr / hr (paired set) or real."""
    if table == "real":
        part = (result.get("real") or {}).get("summary")
    else:
        part = (result.get("paired") or {}).get(table)
    return (part or {}).get(metric, float("nan"))


def _table(results: List[dict], table: str, metrics: List[str], lower: set) -> List[str]:
    split = "real" if table == "real" else "paired"
    rows = [r for r in results if r.get(split)]
    if not rows:
        return []
    best = {}
    for r in rows:
        for m in metrics:
            v = _value(r, table, m)
            if np.isfinite(v) and (m not in best or (v < best[m] if m in lower else v > best[m])):
                best[m] = v

    def head(m):
        name = "E*warp (×10⁻³)" if m == "ewarp" else m.upper()
        return f"{name} {'↓' if m in lower else '↑'}"

    lines = [
        "| model | output | frames | " + " | ".join(head(m) for m in metrics) + " |",
        "|---|---|---:|" + "---:|" * len(metrics),
    ]
    for r in rows:
        cells = []
        for m in metrics:
            v = _value(r, table, m)
            text = "–" if not np.isfinite(v) else f"{v:.4f}"
            cells.append(f"**{text}**" if np.isfinite(v) and v == best.get(m) else text)
        scale = r["paired"].get("scale") if split == "paired" else None
        out = f"×{scale:g}" if scale else "native"
        lines.append(f"| {r['model']} | {out} | {r[split]['frames']} | " + " | ".join(cells) + " |")
    return lines


def write_comparison(results: List[dict], results_dir: Path, nr_names: List[str]) -> str:
    lower = set().union(*(r.get("lower_better", []) for r in results)) | LOWER_IS_BETTER
    nr = [m for m in nr_names if any(np.isfinite(_value(r, t, m)) for r in results for t in ("hr", "real"))]
    tables = {
        "lr": list(FULL_REFERENCE),
        "hr": list(FULL_REFERENCE) + nr + ["ewarp"],
        "real": nr + ["ewarp"],
    }
    paired_root = next((r["paired"]["dataset"] for r in results if r.get("paired")), None)

    lines = [f"# Model comparison ({datetime.datetime.now():%Y-%m-%d %H:%M})", ""]
    if paired_root:
        lines += [f"Paired set: `{_display_path(paired_root)}` (degraded = the input of every model).", ""]
    sections = {
        "lr": ("## LR - restoration at the input resolution (MambaOFR's task)",
               "GT and outputs larger than the input downscaled to the input size (PIL bicubic)."),
        "hr": ("## HR - restoration + SR at the GT resolution",
               "Outputs smaller than the GT (×1) upscaled (PIL bicubic). No-reference metrics on these frames."),
        "real": ("## Real old films - no reference", "Native outputs."),
    }
    for table, (title, note) in sections.items():
        body = _table(results, table, tables[table], lower)
        if body:
            lines += [title, "", note, "", *body, ""]
    report = "\n".join(lines)

    (results_dir / "comparison.md").write_text(report, encoding="utf-8")
    columns = list(dict.fromkeys(m for t in tables.values() for m in t))
    with open(results_dir / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "table", "output_scale", "frames"] + columns)
        for r in results:
            for table in tables:
                split = "real" if table == "real" else "paired"
                if r.get(split):
                    scale = r["paired"].get("scale") if split == "paired" else ""
                    w.writerow([r["model"], table, scale, r[split]["frames"]]
                               + [_value(r, table, m) if m in tables[table] else "" for m in columns])
    return report


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate and compare all models in a folder on the same data.")
    p.add_argument("--models-dir", default=str(PROJECT_ROOT / "models"))
    p.add_argument("--results-dir", default=str(PROJECT_ROOT / "results" / "eval"))
    p.add_argument("--paired-root", default=str(PROJECT_ROOT / "data" / "training" / "valid"),
                   help="degraded/ + gt/ pairs, the same for every model")
    p.add_argument("--paired-clips", type=int, default=None, help="evenly spaced subset; default all, 0 skips")
    p.add_argument("--paired-frames", type=int, default=None, help="max frames per clip; default all")
    p.add_argument("--real-root", default=str(PROJECT_ROOT / "data" / "training" / "test"),
                   help="real degraded clips without GT, shared by all models")
    p.add_argument("--real-clips", type=int, default=20, help="evenly spaced subset; 0 skips")
    p.add_argument("--real-frames", type=int, default=100)
    p.add_argument("--real-max-height", type=int, default=480,
                   help="skip real clips taller than this (0 = no limit); clips are never resized")
    p.add_argument("--nr-metrics", nargs="*", default=list(DEFAULT_NO_REFERENCE),
                   help="pyiqa no-reference metric names; none = skip them")
    p.add_argument("--flow-max-side", type=int, default=1024, help="RAFT resolution cap for E*warp")
    p.add_argument("--input-baseline", action="store_true", help="also score the degraded input itself")
    p.add_argument("--only", nargs="+", help="evaluate only these model names")
    p.add_argument("--save-outputs", action="store_true", help="write restored clips as mp4")
    p.add_argument("--force", action="store_true", help="ignore cached results")
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    specs = discover_models(Path(args.models_dir))
    if args.only:
        specs = [s for s in specs if s.name in set(args.only)]
    if args.input_baseline:
        specs.insert(0, ModelSpec("degraded_input", None, "input"))
    if not specs:
        raise SystemExit(f"No models found in {args.models_dir}")

    ev = Evaluation(args)
    print(f"[eval] models: {', '.join(s.name for s in specs)}")
    print(f"[eval] paired clips: {len(ev.paired)} from {ev.paired_root} | real clips: {len(ev.real)} | "
          f"no-reference: {', '.join(ev.nr_names) or 'none'} + ewarp | device: {ev.device}")

    results = []
    for spec in specs:
        try:
            results.append(ev.evaluate(spec))
        except Exception:
            print(f"[eval] {spec.name}: FAILED, skipping")
            traceback.print_exc()
    if results:
        print()
        print(write_comparison(results, ev.results_dir, ev.nr_names))
        print(f"[eval] saved {ev.results_dir / 'comparison.md'} and comparison.csv")


if __name__ == "__main__":
    main()
