"""``Trainer`` — training of REVIVID (restoration + 2x SR + hole inpainting),
aimed at REALISM: a viewer should not be able to tell what was repaired.

Started from ``trainer/train.py`` (``python src/trainer/train.py``).

Data (fixed layout, built by ``dataset.create_dataset``):
    data/training/train/*.mp4                  clean clips, degraded on the fly
    data/training/valid/{gt,degraded}/*.mp4    validation pairs

Generator loss (weights: ``training.loss_weights``):
    pix        Charbonnier(coarse, gt), true-hole pixels weighted 1 + hole_loss_boost
    perceptual VGG19 feature distance(coarse, gt)
    gan        hinge, projected patch discriminator on the same VGG features
               (from ``training.gan.start_iter``)
    temporal   flicker of coarse vs the GT's own motion
    detect     BCE of the hole detector vs the true hole mask
    v          diffusion v-loss of the refiner inside the true holes
Discriminator: hinge on VGG features of GT vs coarse, own AdamW.

The true hole mask of a training sample is a TARGET only — the model never
receives it. Validation runs the full inference path on the stored .mp4 pairs
and reports LPIPS / flicker / PSNR / SSIM of the final output; best.pth is the
lowest LPIPS.
"""

from __future__ import annotations

import csv
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
from model.discriminator import ProjectedPatchDiscriminator
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

DEFAULT_WEIGHTS = {"pix": 1.0, "perceptual": 0.1, "gan": 0.05, "temporal": 0.5, "detect": 0.05, "v": 1.0}


def load_config(path=None) -> dict:
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class ModelEMA:
    """Exponential moving average of the model weights: used for validation and
    saved in checkpoints (``restore_video.py`` reads its ``shadow``)."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone().float()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1.0 - self.decay)

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
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.shadow = {k: v.float() for k, v in state["shadow"].items()}


class Trainer:
    def __init__(self, config=None):
        cfg = config if isinstance(config, dict) else load_config(config)
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
        self.net = Video_Backbone(self.model_cfg).to(dev)
        self.net_d = ProjectedPatchDiscriminator().to(dev)
        self.ema = ModelEMA(self.net, float(self.tc.get("ema_decay", 0.999)))

        # ---- losses
        self.w = {**DEFAULT_WEIGHTS, **(self.tc.get("loss_weights") or {})}
        self.vgg = VGGFeatures().to(dev)
        self.pix_loss = CharbonnierLoss()
        self.temporal_loss = TemporalConsistencyLoss()
        self.gan_loss = HingeGANLoss()
        self.detect_loss = HoleDetectionLoss(float(self.tc.get("hole_pos_weight", 10.0))).to(dev)
        self.hole_loss_boost = float(self.tc.get("hole_loss_boost", 3.0))
        self.gt_flow = build_flow_estimator().to(dev)  # frozen RAFT: motion of the GT
        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()

        # ---- optimisation
        self.lr = float(self.tc.get("lr", 2e-4))
        self.lr_min = float(self.tc.get("lr_min", 1e-6))
        betas = (float(self.tc.get("beta1", 0.9)), float(self.tc.get("beta2", 0.99)))
        params = [p for p in self.net.parameters() if p.requires_grad]
        self.opt_g = torch.optim.AdamW([{"params": params, "lr_mult": 1.0}], lr=self.lr, betas=betas)
        gan = self.tc.get("gan") or {}
        self.gan_start_iter = int(gan.get("start_iter", 5000))
        self.d_lr = float(gan.get("d_lr", 1e-4))
        self.opt_d = torch.optim.AdamW(
            self.net_d.parameters(),
            lr=self.d_lr,
            betas=(float(gan.get("beta1", 0.0)), float(gan.get("beta2", 0.99))),
        )
        self.grad_accum = max(1, int(self.tc.get("grad_accum", 1)))
        self.grad_clip = float(self.tc.get("grad_clip", 1.0))
        # RAFT starts frozen and is fine-tuned after a warmup (-1 = never).
        self.raft_unfreeze_iter = int(self.tc.get("raft_unfreeze_iter", 20000))
        self.flow_lr_mul = float(self.tc.get("flow_lr_mul", 0.125))
        self.raft_unfrozen = False

        # ---- state
        self.iteration = 0
        self._accum = 0
        self.best_lpips = float("inf")
        self.best_epoch = 0
        self.history: list[dict] = []

    # ------------------------------------------------------------ schedule

    def _set_lr(self, epoch: int, epochs: int) -> float:
        """Linear decay from ``lr`` (first epoch) to ``lr_min`` (last epoch)."""
        frac = min(max((epoch - 1) / max(epochs - 1, 1), 0.0), 1.0)
        lr = self.lr + (self.lr_min - self.lr) * frac
        for g in self.opt_g.param_groups:
            g["lr"] = lr * g["lr_mult"]
        for g in self.opt_d.param_groups:
            g["lr"] = self.d_lr * lr / self.lr
        return lr

    def _unfreeze_raft(self) -> None:
        flow = self.net.backbone.flow_net
        flow.set_trainable(True)
        lr = self.opt_g.param_groups[0]["lr"]
        self.opt_g.add_param_group(
            {"params": list(flow.parameters()), "lr": lr * self.flow_lr_mul, "lr_mult": self.flow_lr_mul}
        )
        self.raft_unfrozen = True
        print(f"[iter {self.iteration}] RAFT unfrozen (lr x{self.flow_lr_mul})")

    # ---------------------------------------------------------------- train

    def train_step(self, batch: dict) -> dict:
        self.net.train()
        if not self.raft_unfrozen and 0 <= self.raft_unfreeze_iter <= self.iteration:
            self._unfreeze_raft()
        dev = self.device
        lq = batch["lq"].to(dev, non_blocking=True)
        gt = batch["gt"].to(dev, non_blocking=True)
        hole = batch["hole_mask"].to(dev, non_blocking=True)  # TRUE mask: a target only
        n, t, c, H, W = gt.shape
        gt_f = gt.reshape(n * t, c, H, W)
        hole_lr = hole.reshape(n * t, 1, *hole.shape[-2:])
        hole_hr = F.interpolate(hole_lr, size=(H, W), mode="nearest")

        out = self.net(lq)
        coarse, coarse_f = out["coarse"], out["coarse_f"]
        losses = {}

        losses["pix"] = self.pix_loss(coarse_f, gt_f, 1.0 + self.hole_loss_boost * hole_hr)

        fake_feats = self.vgg(coarse_f)
        with torch.no_grad():
            real_feats = self.vgg(gt_f)
        losses["perceptual"] = VGGFeatures.perceptual(fake_feats, real_feats)

        gan_on = self.iteration >= self.gan_start_iter
        if gan_on:
            # D's parameters are frozen while this graph is built, so the
            # generator loss sends them no gradient.
            self.net_d.requires_grad_(False)
            losses["gan"] = self.gan_loss.generator(self.net_d(fake_feats))
            self.net_d.requires_grad_(True)

        if t > 1:
            with torch.no_grad():
                flow = self.gt_flow(
                    gt[:, 1:].reshape(-1, c, H, W), gt[:, :-1].reshape(-1, c, H, W)
                ).view(n, t - 1, 2, H, W)
            losses["temporal"] = self.temporal_loss(coarse, gt, flow)

        losses["detect"] = self.detect_loss(out["hole_logits_f"], hole_lr)

        # Refiner: diffusion inside the TRUE holes; batches without holes skip it.
        gen = self.net.generation_mask(hole_hr)
        if bool((gen > 0).any()):
            residual = gt_f - coarse_f.detach()
            std = self.net.update_residual_std(residual, gen)
            # One timestep per clip: the temporal attention mixes its frames.
            t_diff = torch.randint(
                0, self.net.diffusion.num_timesteps, (n,), device=dev
            ).repeat_interleave(t)
            losses["v"] = self.net.diffusion.training_loss(
                self.net.refine_unet, residual / std, t_diff, gen,
                cond=out["refine_cond"], num_frames=t,
            )

        total = sum(self.w[k] * v for k, v in losses.items())
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
            d_real = self.net_d(real_feats)
            d_fake = self.net_d([f.detach() for f in fake_feats])
            loss_d = self.gan_loss.discriminator(d_real, d_fake)
            (loss_d / self.grad_accum).backward()
            log["d"] = float(loss_d.detach())
            log["d_real"] = float(sum(r.detach().mean() for r in d_real)) / len(d_real)
            log["d_fake"] = float(sum(f.detach().mean() for f in d_fake)) / len(d_fake)

        self._accum += 1
        if self._accum == self.grad_accum:
            self._accum = 0
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
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
    def validate(self, loader, epoch: int) -> dict:
        """Full inference path on the stored .mp4 pairs (the model finds the holes
        itself, in windows of num_frame); metrics of the final output."""
        self.net.eval()
        win = int(self.tc.get("num_frame", 7))
        max_clips = int(self.vc.get("max_clips", 0))
        max_frames = int(self.vc.get("max_frames", 0))
        sums, count = {}, 0
        with self.ema.applied(self.net):
            for item in tqdm(loader, desc=f"Val {epoch}", unit="clip", leave=False, dynamic_ncols=True):
                if max_clips and count >= max_clips:
                    break
                lq, gt = self._val_clip(item, max_frames)
                outs, masks = [], []
                for i in range(0, lq.shape[1], win):
                    r = self.net.restore_full(lq[:, i : i + win].to(self.device))
                    outs.append(r["refined"][0].cpu())
                    masks.append((r["generation_mask"][0] > 0).float().cpu())
                out = torch.cat(outs)
                m = evaluate_clip(out, gt)
                m["psnr"] = min(m["psnr"], 100.0)
                m["lpips"] = self._lpips(out, gt)
                m["flicker"] = self._flicker(out, gt)
                m["hole_frac"] = float(torch.cat(masks).mean())  # where the refiner acted
                for k, v in m.items():
                    sums[k] = sums.get(k, 0.0) + v
                count += 1
        return {k: v / max(count, 1) for k, v in sums.items()}

    @torch.no_grad()
    def save_sample(self, loader, epoch: int, tag: str) -> None:
        """First validation window: rows LQ / coarse / refined / GT / refiner mask."""
        self.net.eval()
        lq, gt = self._val_clip(next(iter(loader)), int(self.tc.get("num_frame", 7)))
        with self.ema.applied(self.net):
            r = self.net.restore_full(lq.to(self.device))
        size = gt.shape[-2:]
        rows = [
            F.interpolate(lq[0], size=size, mode="bilinear", align_corners=False),
            r["coarse"][0].cpu(),
            r["refined"][0].cpu(),
            gt,
            r["generation_mask"][0].cpu().expand(-1, 3, -1, -1) * 2 - 1,
        ]
        grid = torch.cat([x.float().clamp(-1, 1).add(1).div(2) for x in rows])
        save_image(grid, self.exp_dir / "samples" / f"epoch{epoch:04d}_{tag}.png", nrow=lq.shape[1], padding=2)

    # ------------------------------------------------------------------ fit

    def fit(self, train_loader, val_loader=None, start_epoch: int = 1) -> None:
        epochs = int(self.tc.get("epochs", 4000))
        val_every = max(1, int(self.vc.get("val_every", 10)))
        save_every = max(1, int(self.lc.get("save_checkpoint_every", 10)))
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

            m = None
            if val_loader is not None and (epoch % val_every == 0 or epoch == epochs):
                m = self.validate(val_loader, epoch)
                entry.update({f"val_{k}": v for k, v in m.items()})
                print(
                    f"[epoch {epoch}] VAL lpips {m['lpips']:.4f} | flicker {m['flicker']:.4f} | "
                    f"psnr {m['psnr']:.3f} ssim {m['ssim']:.4f} | refiner on {100 * m['hole_frac']:.2f}% px"
                )
            self.history.append(entry)
            self._write_history()

            if m is not None and m["lpips"] < self.best_lpips:
                self.best_lpips, self.best_epoch = m["lpips"], epoch
                self._save("best.pth", epoch)
                self.save_sample(val_loader, epoch, "best")
            if epoch % save_every == 0 or epoch == epochs:
                self._save(f"epoch{epoch:04d}.pth", epoch)
                self._save("latest.pth", epoch)
                if val_loader is not None:
                    self.save_sample(val_loader, epoch, "checkpoint")
            print(f"[epoch {epoch}] done in {time.time() - t0:.1f}s")

    # ---------------------------------------------------------- checkpoints

    def _save(self, name: str, epoch: int) -> None:
        torch.save(
            {
                "epoch": epoch,
                "iteration": self.iteration,
                "model": self.net.state_dict(),
                "ema": self.ema.state_dict(),
                "net_d": self.net_d.state_dict(),
                "opt_g": self.opt_g.state_dict(),
                "opt_d": self.opt_d.state_dict(),
                "raft_unfrozen": self.raft_unfrozen,
                "best_lpips": self.best_lpips,
                "best_epoch": self.best_epoch,
                "history": self.history,
                "model_config": self.model_cfg.to_dict(),
                "config": self.cfg,
            },
            self.exp_dir / "checkpoints" / name,
        )

    def resume(self, path: Optional[str] = None) -> int:
        """Continue from ``path`` or <exp_dir>/checkpoints/latest.pth (a checkpoint
        of this same model); returns the next epoch, 1 when starting fresh."""
        path = Path(path) if path else self.exp_dir / "checkpoints" / "latest.pth"
        if not path.exists():
            print("[trainer] no checkpoint - starting fresh")
            return 1
        s = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(s["model"])
        self.ema.load_state_dict(s["ema"])
        self.net_d.load_state_dict(s["net_d"])
        if s["raft_unfrozen"]:
            self._unfreeze_raft()  # recreate the param group before loading opt_g
        self.opt_g.load_state_dict(s["opt_g"])
        self.opt_d.load_state_dict(s["opt_d"])
        self.iteration = s["iteration"]
        self.best_lpips, self.best_epoch = s["best_lpips"], s["best_epoch"]
        self.history = s["history"]
        print(f"[trainer] resumed from {path} (next epoch {s['epoch'] + 1}, "
              f"best lpips {self.best_lpips:.4f} @ epoch {self.best_epoch})")
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
        """Train clips (degraded on the fly, with their true hole masks) and the
        stored validation pairs (no masks; ``None`` when there are none)."""
        tc = self.tc
        train = data.train_loader(
            DATA_DIR / "train",
            num_frame=int(tc.get("num_frame", 7)),
            sr_scale=self.model_cfg.sr_scale,
            crop_size=tc.get("gt_size"),
            hole_prob=self.model_cfg.hole_prob,
            batch_size=int(tc.get("batch_size", 1)),
            num_workers=int(tc.get("num_workers", 0)),
            augment=bool(tc.get("augment", True)),
        )
        try:
            val = data.eval_loader(DATA_DIR / "valid", num_workers=int(self.vc.get("num_workers", 0)))
        except FileNotFoundError:
            print(f"[trainer] no validation pairs in {DATA_DIR / 'valid'} - training without validation")
            val = None
        return train, val
