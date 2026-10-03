"""Evaluate every model in a models directory and compare them.

Run from the repo root:
    PYTHONPATH=src python -m evaluator.evaluate
    PYTHONPATH=src python -m evaluator.evaluate --input-baseline --save-outputs
    PYTHONPATH=src python -m evaluator.evaluate --paired-clips 5 --real-clips 0   # quick run

All data is read with ``dataset.dataset_loader`` (.mp4 only).
GT set: each model uses its own ``dataset:`` from model.yaml (``degraded/*.mp4`` +
``gt/*.mp4``, read clip by clip as frames by ``eval_loader``), falling back to ``--paired-root``. Inputs and
outputs are never resized: the model output must already match the GT resolution.
Scores: PSNR, SSIM, LPIPS, FID (MambaOFR protocol) + the no-reference suite.
Real set (old films, no GT): ``*.mp4`` shared by all models, evenly spaced clips whose
height is <= ``--real-max-height`` (taller clips are skipped, not resized).
No-reference suite: pyiqa metrics per frame (``--nr-metrics``) + temporal E*warp.
Per-model results are cached in ``<results-dir>/<model>.json`` and reused while
the checkpoint, the model's dataset and the settings are unchanged (``--force``).
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
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
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

PROTOCOL_VERSION = 3


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


class Evaluation:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.results_dir = Path(args.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.nr_names = list(args.nr_metrics)

        self.real = []
        if args.real_root and args.real_clips != 0:
            limit = args.real_max_height
            eligible = [p for p in list_mp4(args.real_root) if not limit or clip_height(p) <= limit]
            self.real = [(p.stem, p) for p in evenly_spaced(eligible, args.real_clips)]

        self._fr: Optional[FullReferenceMetrics] = None
        self._nr: Optional[NoReferenceMetrics] = None
        self._warp: Optional[WarpingError] = None
        self._inception: Optional[InceptionFeatures] = None
        self._gt_features: Dict[str, np.ndarray] = {}

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

    def paired_set(self, spec: ModelSpec):
        """(root, EvalDataset or None, [(index in the dataset, clip name)])."""
        root = Path(spec.dataset or self.args.paired_root).resolve()
        if self.args.paired_clips == 0:
            return root, None, []
        dataset = EvalDataset(root)
        clips = [(i, deg.stem) for i, (deg, _) in enumerate(dataset.pairs)]
        return root, dataset, evenly_spaced(clips, self.args.paired_clips)

    def settings(self) -> dict:
        a = self.args
        return {
            "protocol": PROTOCOL_VERSION,
            "paired_clips": a.paired_clips,
            "paired_frames": a.paired_frames,
            "real": [n for n, _ in self.real],
            "real_root": str(Path(a.real_root).resolve()) if self.real else None,
            "real_frames": a.real_frames,
            "real_max_height": a.real_max_height,
            "nr_metrics": self.nr_names,
            "flow_max_side": a.flow_max_side,
            "seed": a.seed,
        }

    def fingerprint(self, spec: ModelSpec, paired_root: Path, paired_clips) -> str:
        ckpt = None
        if spec.checkpoint is not None:
            st = spec.checkpoint.stat()
            ckpt = [str(spec.checkpoint), st.st_size, st.st_mtime_ns]
        blob = json.dumps(
            {
                "settings": self.settings(),
                "paired": [str(paired_root), [n for _, n in paired_clips]],
                "checkpoint": ckpt,
                "type": spec.type,
                "options": spec.options,
            },
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
        return [self.nr(f) for f in frames], self.warp(frames)

    def eval_paired(self, restorer, name: str, root: Path, clips) -> dict:
        rows, per_clip, fake_feats, ewarps = [], {}, [], []
        gt_key = str(root)
        gt_feats = None if gt_key in self._gt_features else []
        loader = eval_loader(root, [index for index, _ in clips])
        for sample in tqdm(loader, total=len(clips), desc=f"{name} | paired", unit="clip"):
            clip, n = sample["name"], self.args.paired_frames
            lq, gt = sample["lq"][:n], sample["gt"][:n]
            out = self._restore(restorer, lq)
            if out[0].shape != gt[0].shape:
                raise ValueError(
                    f"clip {clip}: output {out[0].shape[1]}x{out[0].shape[0]} != GT "
                    f"{gt[0].shape[1]}x{gt[0].shape[0]} in {root}. Point this model to a GT set "
                    f"matching its scale via 'dataset:' in its model.yaml."
                )
            nr_rows, ewarp = self._no_reference(out)
            clip_rows = [{**self.fr(o, g), **r} for o, g, r in zip(out, gt, nr_rows)]
            per_clip[clip] = {**mean_scores(clip_rows), "ewarp": ewarp}
            rows += clip_rows
            ewarps.append(ewarp)
            fake_feats.append(self.inception(out))
            if gt_feats is not None:
                gt_feats.append(self.inception(gt))
            if self.args.save_outputs:
                self._save_video(out, name, "paired", clip)
        if gt_feats is not None:
            self._gt_features[gt_key] = np.concatenate(gt_feats)
        summary = mean_scores(rows)
        summary["fid"] = frechet_distance(np.concatenate(fake_feats), self._gt_features[gt_key])
        summary["ewarp"] = float(np.nanmean(ewarps))
        return {"dataset": str(root), "summary": summary, "frames": len(rows), "per_clip": per_clip}

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
        summary = {**mean_scores(rows), "ewarp": float(np.nanmean(ewarps))}
        return {"dataset": str(Path(self.args.real_root).resolve()), "summary": summary,
                "frames": len(rows), "per_clip": per_clip}

    def evaluate(self, spec: ModelSpec) -> dict:
        root, dataset, clips = self.paired_set(spec)
        if spec.type == "input" and clips:
            deg_path, gt_path = dataset.pairs[clips[0][0]]
            if read_mp4(deg_path, 0, 1)[0].shape != read_mp4(gt_path, 0, 1)[0].shape:
                print(f"[eval] {spec.name}: input and GT sizes differ in {root}, GT metrics skipped")
                clips = []

        cache = self.results_dir / f"{spec.name}.json"
        fp = self.fingerprint(spec, root, clips)
        if cache.exists() and not self.args.force:
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if cached.get("fingerprint") == fp:
                print(f"[eval] {spec.name}: cached results reused")
                return cached

        print(f"[eval] {spec.name}: {spec.checkpoint or 'degraded input'} | GT set: {root}")
        restorer = build_restorer(spec, self.device)
        paired = self.eval_paired(restorer, spec.name, root, clips) if clips else None
        real = self.eval_real(restorer, spec.name) if self.real else None
        result = {
            "model": spec.name,
            "checkpoint": str(spec.checkpoint) if spec.checkpoint else None,
            "fingerprint": fp,
            "settings": self.settings(),
            "lower_better": sorted(LOWER_IS_BETTER | self.nr.lower_better),
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


def _value(result: dict, split: str, metric: str) -> float:
    return ((result.get(split) or {}).get("summary") or {}).get(metric, float("nan"))


def _table(results, split, metrics, lower, group_by_dataset) -> List[str]:
    rows = [r for r in results if r.get(split)]
    if not rows:
        return []
    best = {}
    for r in rows:
        group = r[split]["dataset"] if group_by_dataset else None
        for m in metrics:
            v = _value(r, split, m)
            if not np.isfinite(v):
                continue
            cur = best.get((group, m))
            if cur is None or (v < cur if m in lower else v > cur):
                best[(group, m)] = v

    def head(m):
        name = "E*warp (×10⁻³)" if m == "ewarp" else m.upper()
        return f"{name} {'↓' if m in lower else '↑'}"

    lines = [
        "| model | dataset | frames | " + " | ".join(head(m) for m in metrics) + " |",
        "|---|---|---:|" + "---:|" * len(metrics),
    ]
    for r in rows:
        group = r[split]["dataset"] if group_by_dataset else None
        cells = []
        for m in metrics:
            v = _value(r, split, m)
            text = "–" if not np.isfinite(v) else f"{v:.4f}"
            cells.append(f"**{text}**" if np.isfinite(v) and v == best.get((group, m)) else text)
        lines.append(
            f"| {r['model']} | {_display_path(r[split]['dataset'])} | {r[split]['frames']} | "
            + " | ".join(cells) + " |"
        )
    return lines


def write_comparison(results: List[dict], results_dir: Path, nr_names: List[str]) -> str:
    lower = set().union(*(r.get("lower_better", []) for r in results)) | LOWER_IS_BETTER
    nr = [m for m in nr_names if any(np.isfinite(_value(r, s, m)) for r in results for s in ("paired", "real"))]
    paired_metrics = list(FULL_REFERENCE) + nr + ["ewarp"]
    real_metrics = nr + ["ewarp"]

    lines = [f"# Model comparison ({datetime.datetime.now():%Y-%m-%d %H:%M})", ""]
    paired = _table(results, "paired", paired_metrics, lower, group_by_dataset=True)
    if paired:
        lines += [
            "## GT metrics — each model on its own GT set",
            "",
            "Best values are bold only among models evaluated on the same dataset.",
            "",
            *paired,
            "",
        ]
    real = _table(results, "real", real_metrics, lower, group_by_dataset=False)
    if real:
        lines += ["## No-reference metrics — shared set of real old films", "", *real, ""]
    report = "\n".join(lines)

    (results_dir / "comparison.md").write_text(report, encoding="utf-8")
    with open(results_dir / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "split", "dataset", "frames"] + paired_metrics)
        for r in results:
            for split, metrics in (("paired", paired_metrics), ("real", real_metrics)):
                if r.get(split):
                    vals = {m: _value(r, split, m) for m in metrics}
                    w.writerow([r["model"], split, r[split]["dataset"], r[split]["frames"]]
                               + [vals.get(m, "") for m in paired_metrics])
    return report


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate and compare all models in a folder.")
    p.add_argument("--models-dir", default=str(PROJECT_ROOT / "models"))
    p.add_argument("--results-dir", default=str(PROJECT_ROOT / "results" / "eval"))
    p.add_argument("--paired-root", default=str(PROJECT_ROOT / "data" / "training" / "valid"),
                   help="GT set (degraded/ + gt/) for models without 'dataset:' in model.yaml")
    p.add_argument("--paired-clips", type=int, default=None, help="evenly spaced subset; default all, 0 skips")
    p.add_argument("--paired-frames", type=int, default=None, help="max frames per clip; default all")
    p.add_argument("--real-root", default=str(PROJECT_ROOT / "data" / "training" / "test"),
                   help="real degraded clips without GT, shared by all models")
    p.add_argument("--real-clips", type=int, default=20, help="evenly spaced subset; 0 skips")
    p.add_argument("--real-frames", type=int, default=100)
    p.add_argument("--real-max-height", type=int, default=480,
                   help="skip real clips taller than this (0 = no limit); clips are never resized")
    p.add_argument("--nr-metrics", nargs="+", default=list(DEFAULT_NO_REFERENCE),
                   help="pyiqa no-reference metric names")
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
    print(f"[eval] real clips: {len(ev.real)}, no-reference: {', '.join(ev.nr_names)} + ewarp, device: {ev.device}")

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
