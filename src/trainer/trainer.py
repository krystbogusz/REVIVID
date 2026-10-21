"""``Trainer`` — training of REVIVID (MambaOFR restoration + 2x SR + inpainting
of holes burned through the whole window).

Started from ``trainer/train.py`` (``python src/trainer/train.py``).

Data (fixed layout, built by ``dataset.create_dataset``):
    data/training/train/*.mp4                  clean clips, degraded on the fly
    data/training/valid/{gt,degraded}/*.mp4    validation pairs

Two phases (``training.stage`` in config/REVIVID.yaml), each its own experiment;
a config value written as ``{restore: a, inpaint: b}`` takes the phase's one:
    restore  phase 1: removal of all degradations + SR; MambaOFR's network and
             recipe, trained from scratch. No holes in the data, the phase-2
             modules are off (``model.use_holes``)
    inpaint  phase 2: fine-tuning of the whole phase-1 model
             (``training.init_from``): holes burned in with ``model.hole_prob``;
             hole detector, flow completion and inpainter train at ``lr``, the
             phase-1 part at ``backbone_lr_mult``; the model finds the holes itself

Generator loss (weights: ``training.loss_weights``; a 0 weight skips the term):
    pix        Charbonnier(out, gt)              MambaOFR: L1; phase 2: true-hole
                                                 pixels weigh 1 + hole_loss_boost
    perceptual MambaOFR's VGG19 loss(out, gt)    relu1_1..relu5_1 weighted 1/32..1
    gan        hinge vs MambaOFR's 3-D clip discriminator (from ``gan.start_iter``)
    detect     BCE of the hole detector vs the true hole mask (phase 2)
Discriminator: hinge, GT vs out, own Adam.

The true hole mask of a training sample is a TARGET only — the model never
receives it. Validation runs the model on the stored .mp4 pairs and reports
PSNR / SSIM, the same at LR (``psnr_lr`` / ``ssim_lr``: output and GT
downscaled to the input size - MambaOFR's task on the same input), LPIPS,
flicker and sharpness; best.pth = best ``validation.best_metric``.
"""

from __future__ import annotations

import csv
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import lpips
import torch
import torch.nn.functional as F
import yaml
from torchvision.utils import save_image
from tqdm import tqdm

from dataset import dataset_loader as data
from evaluator.metrics import evaluate_clip
from model import ModelConfig, Video_Backbone
from model.discriminator import TemporalPatchDiscriminator
from model.flow import build_flow_estimator
from model.losses import (
    CharbonnierLoss,
    HingeGANLoss,
    HoleDetectionLoss,
    TemporalConsistencyLoss,
    VGGFeatures,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "REVIVID.yaml"
DATA_DIR = PROJECT_ROOT / "data" / "training"

# MambaOFR: L1 1.0, perceptual 1.0, adversarial 0.01.
DEFAULT_WEIGHTS = {"pix": 1.0, "perceptual": 1.0, "gan": 0.01, "detect": 0.05}
STAGES = ("restore", "inpaint")
# Modules that only phase 2 trains (full lr there); the rest is the phase-1
# model (backbone_lr_mult).
INPAINT_MODULES = ("hole_detector.", "flow_completion.", "inpainter.")
FROZEN = "flow_net."  # the frozen RAFT: not averaged by the EMA
HIGHER_IS_BETTER = ("psnr", "ssim", "psnr_lr", "ssim_lr")


def load_config(path=None) -> dict:
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_stage(cfg, stage: str):
    """A value written as ``{restore: a, inpaint: b}`` becomes the one of ``stage``."""
    if isinstance(cfg, dict):
        if cfg and set(cfg) == set(STAGES):
            return cfg[stage]
        return {k: resolve_stage(v, stage) for k, v in cfg.items()}
    return cfg


class ModelEMA:
    """Exponential moving average of the model weights: used for validation and
    saved in checkpoints (``restore_video.py`` reads its ``shadow``).

    With warmup: the effective decay is ``min(decay, (1 + n) / (10 + n))`` after
    ``n`` updates, so early on the average follows the weights closely instead
    of being dominated by the random init (with a plain 0.999, ~35 % of the
    average is still the init after 1000 optimizer steps).
    """

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.updates = 0
        self.shadow = {
            k: v.detach().clone().float()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point and not k.startswith(FROZEN)
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)

    @contextmanager
    def applied(self, model: torch.nn.Module):
        """Temporarily swap the EMA weights into ``model``."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items() if k in self.shadow}
        model.load_state_dict({k: v.to(backup[k].dtype) for k, v in self.shadow.items()}, strict=False)
        try:
            yield
        finally:
            model.load_state_dict(backup, strict=False)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "updates": self.updates, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.shadow = {k: v.float() for k, v in state["shadow"].items()}
        self.updates = int(state.get("updates", 0))


class Trainer:
    def __init__(self, config=None):
        cfg = config if isinstance(config, dict) else load_config(config)
        self.stage = str((cfg.get("training") or {}).get("stage", "restore"))
        if self.stage not in STAGES:
            raise ValueError(f"training.stage must be one of {STAGES}, got {self.stage!r}")
        self.inpaint = self.stage == "inpaint"
        cfg = resolve_stage(cfg, self.stage)
        self.cfg = cfg
        self.tc = cfg.get("training") or {}
        self.vc = cfg.get("validation") or {}
        self.lc = cfg.get("logging") or {}
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required.")
        self.device = dev = torch.device("cuda")
        torch.manual_seed(int(cfg.get("seed", 2026)))

        self.exp_dir = Path(self.lc.get("exp_dir", "experiments/revivid"))
        if not self.exp_dir.is_absolute():
            self.exp_dir = PROJECT_ROOT / self.exp_dir
        (self.exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.exp_dir / "samples").mkdir(parents=True, exist_ok=True)

        # ---- networks
        self.model_cfg = ModelConfig.from_dict(cfg.get("model"))
        self.model_cfg.use_holes = self.inpaint  # stored with the checkpoints' model_config
        self.net = Video_Backbone(self.model_cfg).to(dev)
        self.net_d = TemporalPatchDiscriminator().to(dev)
        self.ema = ModelEMA(self.net, float(self.tc.get("ema_decay", 0.999)))

        # ---- losses
        self.w = {**DEFAULT_WEIGHTS, **(self.tc.get("loss_weights") or {})}
        self.vgg = VGGFeatures().to(dev)
        self.pix_loss = CharbonnierLoss()
        self.gan_loss = HingeGANLoss()
        self.detect_loss = HoleDetectionLoss(float(self.tc.get("hole_pos_weight", 10.0))).to(dev)
        self.hole_loss_boost = float(self.tc.get("hole_loss_boost", 3.0))
        # Validation metrics.
        self.temporal_loss = TemporalConsistencyLoss()
        self.gt_flow = build_flow_estimator("large").to(dev)  # frozen RAFT large: motion of the GT
        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()

        # ---- optimisation
        self.lr = float(self.tc.get("lr", 2e-4))
        self.lr_min = float(self.tc.get("lr_min", 4e-5))
        self.lr_steady = float(self.tc.get("lr_steady", 0.6))
        self.lr_steps = max(1, int(self.tc.get("lr_steps", 8)))
        # Adam without weight decay, as MambaOFR.
        betas = (float(self.tc.get("beta1", 0.9)), float(self.tc.get("beta2", 0.999)))
        new, old = [], []
        for name, p in self.net.named_parameters():
            if p.requires_grad:
                (new if name.startswith(INPAINT_MODULES) else old).append(p)
        backbone_mult = float(self.tc.get("backbone_lr_mult", 1.0)) if self.inpaint else 1.0
        self.opt_g = torch.optim.Adam(
            [{"params": old, "lr_mult": backbone_mult}, {"params": new, "lr_mult": 1.0}], lr=self.lr, betas=betas
        )
        gan = self.tc.get("gan") or {}
        self.gan_start_iter = int(gan.get("start_iter", 0))
        self.d_lr = float(gan.get("d_lr", 2e-4))
        self.opt_d = torch.optim.Adam(
            self.net_d.parameters(),
            lr=self.d_lr,
            betas=(float(gan.get("beta1", 0.9)), float(gan.get("beta2", 0.999))),
        )
        self.grad_accum = max(1, int(self.tc.get("grad_accum", 1)))
        self.grad_clip = float(self.tc.get("grad_clip", 0.0))  # 0 = off (MambaOFR)

        # ---- state
        self.iteration = 0
        self._accum = 0
        self.best_metric = str(self.vc.get("best_metric", "lpips"))
        self.best_value = float("-inf") if self.best_metric in HIGHER_IS_BETTER else float("inf")
        self.best_epoch = 0
        self.history: list[dict] = []

    # ------------------------------------------------------------ schedule

    def _set_lr(self, epoch: int, epochs: int) -> float:
        """Constant ``lr`` over the first ``lr_steady`` fraction of the epochs, then
        down to ``lr_min`` in ``lr_steps`` equal steps, the last one over the
        final epochs. MambaOFR: 60 % at 2e-4, then -2e-5 per 12k iterations
        (8 steps, the last at 4e-5); D follows in proportion."""
        steady = int(round(self.lr_steady * epochs))
        step = math.ceil(max(epoch - steady, 0) * self.lr_steps / max(epochs - steady, 1))
        frac = min(step / self.lr_steps, 1.0)
        lr = self.lr + (self.lr_min - self.lr) * frac
        for g in self.opt_g.param_groups:
            g["lr"] = lr * g["lr_mult"]
        for g in self.opt_d.param_groups:
            g["lr"] = self.d_lr * lr / self.lr
        return lr

    # ---------------------------------------------------------------- train

    def train_step(self, batch: dict) -> dict:
        self.net.train()
        dev = self.device
        lq = batch["lq"].to(dev, non_blocking=True)
        gt = batch["gt"].to(dev, non_blocking=True)
        n, t, c, H, W = gt.shape
        gt_f = gt.reshape(n * t, c, H, W)

        out = self.net(lq)
        pred = out["out"]
        pred_f = pred.reshape(n * t, c, H, W)
        w = self.w
        losses = {}

        weight = None
        if self.inpaint:
            hole = batch["hole_mask"].to(dev, non_blocking=True)  # TRUE mask: a target only
            hole_lr = hole.reshape(n * t, 1, *hole.shape[-2:])
            weight = 1.0 + self.hole_loss_boost * F.interpolate(hole_lr, size=(H, W), mode="nearest")
            losses["detect"] = self.detect_loss(out["hole_logits"].reshape(hole_lr.shape), hole_lr)
        losses["pix"] = self.pix_loss(pred_f, gt_f, weight)

        with torch.no_grad():
            real_feats = self.vgg(gt_f)
        losses["perceptual"] = VGGFeatures.perceptual(self.vgg(pred_f), real_feats)

        gan_on = w["gan"] > 0 and self.iteration >= self.gan_start_iter
        if gan_on:
            # D's parameters are frozen while this graph is built, so the
            # generator loss sends them no gradient.
            self.net_d.requires_grad_(False)
            losses["gan"] = self.gan_loss.generator(self.net_d(pred))
            self.net_d.requires_grad_(True)

        total = sum(w[k] * v for k, v in losses.items())
        if not torch.isfinite(total):
            print(f"[iter {self.iteration}] non-finite loss, step skipped: "
                  + ", ".join(f"{k}={float(v):.4g}" for k, v in losses.items()))
            self.opt_g.zero_grad(set_to_none=True)
            self.opt_d.zero_grad(set_to_none=True)
            self._accum = 0
            self.iteration += 1
            return {}
        (total / self.grad_accum).backward()
        log = {k: float(v.detach()) for k, v in losses.items()}
        log["total"] = float(total.detach())

        if gan_on:
            d_real = self.net_d(gt)
            d_fake = self.net_d(pred.detach())
            loss_d = self.gan_loss.discriminator(d_real, d_fake)
            (loss_d / self.grad_accum).backward()
            log["d"] = float(loss_d.detach())
            log["d_real"] = float(d_real.detach().mean())
            log["d_fake"] = float(d_fake.detach().mean())

        self._accum += 1
        if self._accum == self.grad_accum:
            self._accum = 0
            clip = self.grad_clip if self.grad_clip > 0 else float("inf")
            log["gnorm"] = float(torch.nn.utils.clip_grad_norm_(self.net.parameters(), clip))
            self.opt_g.step()
            self.opt_g.zero_grad(set_to_none=True)
            self.opt_d.step()
            self.opt_d.zero_grad(set_to_none=True)
            self.ema.update(self.net)
        self.iteration += 1
        return log

    # ----------------------------------------------------------- validation

    @staticmethod
    def _val_clip(item: dict, max_frames: int = 0):
        """Stored frames → lq (1, T, 3, h, w) and gt (T, 3, H, W), CPU, [-1, 1]."""
        n = min(len(item["lq"]), len(item["gt"]))
        if max_frames > 0:
            n = min(n, max_frames)
        lq = data.to_tensor(item["lq"][:n]).unsqueeze(0)
        gt = data.to_tensor(item["gt"][:n], gray=True)
        return lq, gt

    @staticmethod
    def _down(x: torch.Tensor, size) -> torch.Tensor:
        """Bicubic downscale (antialiased, as PIL's resize that made the LQ input)."""
        if tuple(x.shape[-2:]) == tuple(size):
            return x
        return F.interpolate(x, size=tuple(size), mode="bicubic", antialias=True, align_corners=False).clamp(-1, 1)

    @torch.no_grad()
    def _lpips(self, out: torch.Tensor, gt: torch.Tensor, chunk: int = 8) -> float:
        vals = [
            self.lpips(out[i : i + chunk].to(self.device), gt[i : i + chunk].to(self.device)).flatten()
            for i in range(0, out.shape[0], chunk)
        ]
        return float(torch.cat(vals).mean())

    @torch.no_grad()
    def _flicker(self, out: torch.Tensor, gt: torch.Tensor, chunk: int = 8) -> float:
        """Temporal loss over all consecutive frame pairs (chunks overlap by one frame)."""
        total, pairs = 0.0, 0
        for i in range(0, out.shape[0] - 1, chunk):
            o = out[i : i + chunk + 1].to(self.device)
            g = gt[i : i + chunk + 1].to(self.device)
            flow = self.gt_flow(g[1:], g[:-1]).unsqueeze(0)
            total += float(self.temporal_loss(o[None], g[None], flow)) * (len(g) - 1)
            pairs += len(g) - 1
        return total / max(pairs, 1)

    @torch.no_grad()
    def _sharpness(self, out: torch.Tensor, gt: torch.Tensor, chunk: int = 8) -> float:
        """High-frequency energy of the output relative to the GT: the mean FFT
        amplitude between 0.45 and 0.9 of the Nyquist frequency, over the clip.
        1.0 = as sharp as the GT; below = softer; above = over-sharpened / noise."""
        h, w = gt.shape[-2:]
        fy = torch.fft.fftfreq(h, device=self.device)[:, None] * 2  # in units of Nyquist
        fx = torch.fft.rfftfreq(w, device=self.device)[None, :] * 2
        r = torch.sqrt(fy**2 + fx**2)
        band = ((r >= 0.45) & (r < 0.9)).float()

        def energy(x: torch.Tensor) -> float:
            total = 0.0
            for i in range(0, x.shape[0], chunk):
                y = x[i : i + chunk].to(self.device).float().mean(1)  # grey frames
                y = y - y.mean((-2, -1), keepdim=True)
                total += float((torch.fft.rfft2(y, norm="ortho").abs() * band).sum())
            return total

        return energy(out) / max(energy(gt), 1e-8)

    @torch.no_grad()
    def validate(self, loader, epoch: int) -> dict:
        """The model on the stored .mp4 pairs (inpaint: it finds the holes
        itself), in windows of ``validation.window`` frames; metrics of the output."""
        self.net.eval()
        win = int(self.vc.get("window", self.tc.get("num_frame", 7)))
        max_clips = int(self.vc.get("max_clips", 0))
        max_frames = int(self.vc.get("max_frames", 0))
        thr = self.model_cfg.hole_threshold
        sums, count = {}, 0
        with self.ema.applied(self.net):
            for item in tqdm(loader, desc=f"Val {epoch}", unit="clip", leave=False, dynamic_ncols=True):
                if max_clips and count >= max_clips:
                    break
                lq, gt = self._val_clip(item, max_frames)
                outs, holes = [], []
                for i in range(0, lq.shape[1], win):
                    r = self.net(lq[:, i : i + win].to(self.device))
                    outs.append(r["out"][0].cpu())
                    if "hole_logits" in r:
                        holes.append((torch.sigmoid(r["hole_logits"][0]) > thr).float().cpu())
                out = torch.cat(outs)
                m = evaluate_clip(out, gt)
                m["psnr"] = min(m["psnr"], 100.0)
                lr_size = lq.shape[-2:]
                m_lr = evaluate_clip(self._down(out, lr_size), self._down(gt, lr_size))
                m["psnr_lr"], m["ssim_lr"] = min(m_lr["psnr"], 100.0), m_lr["ssim"]
                m["lpips"] = self._lpips(out, gt)
                m["flicker"] = self._flicker(out, gt)
                m["sharpness"] = self._sharpness(out, gt)
                if holes:
                    m["hole_frac"] = float(torch.cat(holes).mean())  # detected holes, share of LR pixels
                for k, v in m.items():
                    sums[k] = sums.get(k, 0.0) + v
                count += 1
        return {k: v / max(count, 1) for k, v in sums.items()}

    @torch.no_grad()
    def save_sample(self, loader, epoch: int, tag: str) -> None:
        """First validation window: rows LQ / output / GT (+ detected holes in phase 2)."""
        self.net.eval()
        lq, gt = self._val_clip(next(iter(loader)), int(self.tc.get("num_frame", 7)))
        size = gt.shape[-2:]
        with self.ema.applied(self.net):
            r = self.net(lq.to(self.device))
        rows = [F.interpolate(lq[0], size=size, mode="bilinear", align_corners=False), r["out"][0].cpu(), gt]
        if "hole_logits" in r:
            hole = F.interpolate(torch.sigmoid(r["hole_logits"][0]).cpu(), size=size, mode="nearest")
            rows.append(hole.expand(-1, 3, -1, -1) * 2 - 1)
        grid = torch.cat([x.float().clamp(-1, 1).add(1).div(2) for x in rows])
        save_image(grid, self.exp_dir / "samples" / f"epoch{epoch:04d}_{tag}.png", nrow=lq.shape[1], padding=2)

    # ------------------------------------------------------------------ fit

    def _validate_epoch(self, val_loader, epoch: int, entry: dict) -> bool:
        """Validate, add the metrics to ``entry``, save a sample; True when this
        epoch is the best so far (``validation.best_metric``)."""
        m = self.validate(val_loader, epoch)
        entry.update({f"val_{k}": v for k, v in m.items()})
        print(
            f"[epoch {epoch}] VAL psnr {m['psnr']:.3f} ssim {m['ssim']:.4f} | "
            f"LR psnr {m['psnr_lr']:.3f} ssim {m['ssim_lr']:.4f} | lpips {m['lpips']:.4f} | "
            f"flicker {m['flicker']:.4f} | sharpness {100 * m['sharpness']:.0f}% of GT"
            + (f" | holes {100 * m['hole_frac']:.2f}% px" if "hole_frac" in m else "")
        )
        v = m[self.best_metric]
        best = v > self.best_value if self.best_metric in HIGHER_IS_BETTER else v < self.best_value
        if best:
            self.best_value, self.best_epoch = v, epoch
        self.save_sample(val_loader, epoch, "best" if best else "val")
        return best

    def fit(self, train_loader, val_loader=None, start_epoch: int = 1) -> None:
        epochs = int(self.tc.get("epochs", 2000))
        val_every = max(1, int(self.vc.get("val_every", 10)))
        save_every = max(1, int(self.lc.get("save_checkpoint_every", 1)))
        keep_every = int(self.lc.get("keep_checkpoint_every", 0))
        log_every = max(1, int(self.lc.get("log_every", 100)))

        for epoch in range(start_epoch, epochs + 1):
            t0 = time.time()
            lr = self._set_lr(epoch, epochs)
            sums, counts = {}, {}
            bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} (lr {lr:.2e})", unit="batch", dynamic_ncols=True)
            for batch in bar:
                log = self.train_step(batch)
                if not log:
                    continue
                for k, v in log.items():
                    sums[k] = sums.get(k, 0.0) + v
                    counts[k] = counts.get(k, 0) + 1
                bar.set_postfix(loss=f"{log['total']:.4f}")
                if self.iteration % log_every == 0:
                    bar.write(f"[iter {self.iteration}] " + " ".join(f"{k}:{v:.4f}" for k, v in log.items()))
            entry = {"epoch": epoch, "lr": lr, **{k: sums[k] / counts[k] for k in sums}}

            best = False
            if val_loader is not None and (epoch % val_every == 0 or epoch == epochs):
                best = self._validate_epoch(val_loader, epoch, entry)
            self.history.append(entry)
            self._write_history()

            if best:
                self._save("best.pth", epoch)
            if epoch % save_every == 0 or epoch == epochs:
                self._save("latest.pth", epoch)
            if keep_every > 0 and (epoch % keep_every == 0 or epoch == epochs):
                self._save(f"epoch{epoch:04d}.pth", epoch)
            print(f"[epoch {epoch}] done in {time.time() - t0:.1f}s")

    # ---------------------------------------------------------- checkpoints

    def _save(self, name: str, epoch: int) -> None:
        """Atomic: written next to the target, then renamed over it, so an
        interrupted save never leaves a broken checkpoint behind."""
        path = self.exp_dir / "checkpoints" / name
        tmp = path.with_name(path.name + ".tmp")
        torch.save(
            {
                "epoch": epoch,
                "iteration": self.iteration,
                "model": self.net.state_dict(),
                "ema": self.ema.state_dict(),
                "net_d": self.net_d.state_dict(),
                "opt_g": self.opt_g.state_dict(),
                "opt_d": self.opt_d.state_dict(),
                "best_metric": self.best_metric,
                "best_value": self.best_value,
                "best_epoch": self.best_epoch,
                "history": self.history,
                "model_config": self.model_cfg.to_dict(),
                "config": self.cfg,
            },
            tmp,
        )
        os.replace(tmp, path)

    def init_from(self, path: str | Path) -> None:
        """Start from the weights of another run (phase 2 from phase 1): its EMA
        weights where it has them, for the model and the new EMA alike, and its
        discriminator. Optimizers and schedule start fresh."""
        path = Path(path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        s = torch.load(path, map_location=self.device, weights_only=False)
        weights = {**s["model"], **s["ema"]["shadow"]}
        own = self.net.state_dict()
        weights = {k: v.to(own[k].dtype) for k, v in weights.items() if k in own and own[k].shape == v.shape}
        missing = sorted({k.split(".")[0] for k in own if k not in weights})
        self.net.load_state_dict(weights, strict=False)
        self.ema = ModelEMA(self.net, self.ema.decay)
        self.net_d.load_state_dict(s["net_d"])
        print(f"[trainer] initialised from {path} ({len(weights)} tensors"
              + (f"; fresh: {missing}" if missing else "") + ")")

    def resume(self, path: Optional[str] = None) -> int:
        """Continue from ``path`` or <exp_dir>/checkpoints/latest.pth (a checkpoint
        of this same model); returns the next epoch, 1 when starting fresh (phase
        2: from ``training.init_from``)."""
        path = Path(path) if path else self.exp_dir / "checkpoints" / "latest.pth"
        if not path.exists():
            if self.inpaint and self.tc.get("init_from"):
                self.init_from(self.tc["init_from"])
            else:
                print("[trainer] no checkpoint - starting fresh")
            return 1
        s = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(s["model"])
        self.ema.load_state_dict(s["ema"])
        self.net_d.load_state_dict(s["net_d"])
        self.opt_g.load_state_dict(s["opt_g"])
        self.opt_d.load_state_dict(s["opt_d"])
        self.iteration = s["iteration"]
        self.history = s["history"]
        if s["best_metric"] == self.best_metric:
            self.best_value, self.best_epoch = s["best_value"], s["best_epoch"]
        else:
            print(f"[trainer] best metric changed {s['best_metric']} -> {self.best_metric}: best.pth starts over")
        print(f"[trainer] resumed from {path} (next epoch {s['epoch'] + 1}, iter {self.iteration}, "
              f"best {self.best_metric} {self.best_value:.4f} @ epoch {self.best_epoch})")
        return s["epoch"] + 1

    def save_config(self) -> None:
        """Store the active config next to the run (<exp_dir>/config.yaml)."""
        with open(self.exp_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(self.cfg, f, sort_keys=False, allow_unicode=True)

    def _write_history(self) -> None:
        keys = list(dict.fromkeys(k for e in self.history for k in e))
        with open(self.exp_dir / "loss_history.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, restval="")
            writer.writeheader()
            writer.writerows(self.history)

    # ----------------------------------------------------------------- data

    def build_loaders(self):
        """Train clips (degraded on the fly, with their true hole masks; no holes
        in the restore stage) and the stored validation pairs (``None`` when
        there are none)."""
        tc = self.tc
        train = data.train_loader(
            DATA_DIR / "train",
            num_frame=int(tc.get("num_frame", 7)),
            sr_scale=self.model_cfg.sr_scale,
            crop_size=tc.get("gt_size"),
            hole_prob=self.model_cfg.hole_prob if self.inpaint else 0.0,
            batch_size=int(tc.get("batch_size", 1)),
            num_workers=int(tc.get("num_workers", 0)),
            augment=bool(tc.get("augment", True)),
            gt_usm=float(tc.get("gt_usm", 0.0)),
        )
        val_dir = DATA_DIR / "valid"
        try:
            val = data.eval_loader(val_dir, num_workers=int(self.vc.get("num_workers", 0)))
        except FileNotFoundError:
            print(f"[trainer] no validation pairs in {val_dir} - training without validation")
            val = None
        return train, val
