"""Training loop for the REVIVID DiffMambaOFR model.

The trainer is driven entirely by ``config/REVIVID.yaml`` (model + training
hyper-parameters only). Data locations are fixed by the pipeline, so no paths
are configured here.

It performs joint optimization of:
    * the coarse restoration (charbonnier + optional VGG perceptual),
    * the persistent-hole detector (BCE),
    * the v-prediction diffusion head,
with AMP mixed precision, a per-epoch LR scheduler (cosine / step / plateau,
configurable via ``training.scheduler``), checkpointing and PSNR/SSIM validation
(DDIM sampling). This is a pure diffusion model - there is no adversarial / GAN
component.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn.functional as F
import yaml
from torchvision.utils import save_image
from tqdm import tqdm

from dataset.dataset_loader import warmup_dataloader
from evaluator.metrics import evaluate_clip
from model import ModelConfig, Video_Backbone
from model.losses import (
    CharbonnierLoss,
    DiffusionLoss,
    FocalFrequencyLoss,
    HoleDetectionLoss,
    MaskedReconstructionLoss,
    VGGPerceptualLoss,
)

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "REVIVID.yaml"


def load_config(path: Union[str, Path, None] = None) -> dict:
    path = Path(path) if path is not None else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class Trainer:
    def __init__(self, config: Union[str, Path, dict, None] = None):
        cfg = config if isinstance(config, dict) else load_config(config)
        self.cfg = cfg
        self.train_cfg = cfg.get("training", {})
        self.val_cfg = cfg.get("validation", {})
        self.log_cfg = cfg.get("logging", {})

        # Validation cost controls (validation on a diffusion model is expensive:
        # every window costs `refine_steps` DDIM forward passes).
        self.val_every = max(1, int(self.val_cfg.get("val_every", 1)))
        self.val_max_clips = int(self.val_cfg.get("max_clips", 0))
        self.val_max_frames = int(self.val_cfg.get("max_frames", 0))
        self.val_refine_steps = int(self.val_cfg.get("refine_steps", 0))

        self.device = torch.device("cuda")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required. No CUDA device found.")
        torch.manual_seed(int(cfg.get("seed", 2021)))

        self.exp_dir = Path(self.log_cfg.get("exp_dir", "./experiments/revivid"))
        (self.exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.exp_dir / "samples").mkdir(parents=True, exist_ok=True)

        self.model_cfg = ModelConfig.from_dict(cfg.get("model", {}))
        self.net = Video_Backbone(self.model_cfg).to(self.device)

        default_w = {
            "pix": 1.0,
            "perceptual": 1.0,
            "detect": 0.5,
            "v": 1.0,
            "fft": 0.1,
            "vfi": 10.0,
        }
        default_w.update(self.train_cfg.get("loss_weights", {}) or {})
        self.weights = default_w

        self.loss_pix = CharbonnierLoss().to(self.device)
        self.use_perceptual = bool(self.train_cfg.get("use_perceptual", True))
        self.loss_perceptual = (
            VGGPerceptualLoss().to(self.device) if self.use_perceptual else None
        )
        self.loss_detect = HoleDetectionLoss().to(self.device)
        self.loss_diffusion = DiffusionLoss().to(self.device)
        self.loss_fft = FocalFrequencyLoss().to(self.device)
        self.loss_vfi = MaskedReconstructionLoss().to(self.device)

        lr = float(self.train_cfg.get("lr", 2e-4))
        betas = (
            float(self.train_cfg.get("beta1", 0.9)),
            float(self.train_cfg.get("beta2", 0.99)),
        )
        self.optimizer_g = torch.optim.AdamW(self._param_groups(lr), lr=lr, betas=betas)
        self.grad_clip = float(self.train_cfg.get("grad_clip", 1.0))

        self.base_lr = lr
        self.total_epochs = int(self.train_cfg.get("epochs", 20))
        self.scheduler_type = "none"
        self.scheduler = self._build_scheduler()

        self.use_amp = bool(self.train_cfg.get("use_amp", True))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.iteration = 0
        self.best_psnr = -1.0
        self.best_epoch = 0

    def _param_groups(self, lr: float):
        return [p for p in self.net.parameters() if p.requires_grad]

    def _build_scheduler(self):
        """Build the LR scheduler (stepped once per epoch) from config.

        ``training.scheduler.type``:
            * ``cosine``  — CosineAnnealingLR (lr → eta_min) with optional
                            linear warmup.
            * ``step``    — StepLR (drop by ``gamma`` every ``step_size`` epochs).
            * ``plateau`` — ReduceLROnPlateau on validation PSNR (mode=max).
            * ``none``    — no scheduling (constant LR).
        """
        sch_cfg = self.train_cfg.get("scheduler", {}) or {}
        sch_type = str(sch_cfg.get("type", "none")).lower()
        self.scheduler_type = sch_type

        if sch_type in ("none", "off", ""):
            self.scheduler_type = "none"
            return None

        opt = self.optimizer_g
        epochs = self.total_epochs
        warmup = max(0, int(sch_cfg.get("warmup_epochs", 0)))
        eta_min = float(sch_cfg.get("eta_min", self.base_lr * 0.01))

        if sch_type == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode="max",
                factor=float(sch_cfg.get("factor", 0.5)),
                patience=int(sch_cfg.get("patience", 50)),
                min_lr=eta_min,
            )

        if sch_type == "cosine":
            t_max = max(1, epochs - warmup)
            main = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=t_max, eta_min=eta_min
            )
        elif sch_type == "step":
            step_size = max(1, int(sch_cfg.get("step_size", max(1, epochs // 4))))
            gamma = float(sch_cfg.get("gamma", 0.5))
            main = torch.optim.lr_scheduler.StepLR(
                opt, step_size=step_size, gamma=gamma
            )
        else:
            raise ValueError(
                f"Unknown scheduler type '{sch_type}'. "
                f"Use one of: cosine, step, plateau, none."
            )

        if warmup > 0:
            warm = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=1e-3, end_factor=1.0, total_iters=warmup
            )
            return torch.optim.lr_scheduler.SequentialLR(
                opt, schedulers=[warm, main], milestones=[warmup]
            )
        return main

    def _step_scheduler(self, epoch: int, metrics: Optional[dict]) -> None:
        """Advance the LR scheduler once per epoch and log the new LR."""
        if self.scheduler is None:
            return

        if self.scheduler_type == "plateau":
            if metrics is None:
                return
            self.scheduler.step(metrics["psnr"])
        else:
            self.scheduler.step()

        cur_lr = self.optimizer_g.param_groups[0]["lr"]
        print(f"[epoch {epoch}] lr -> {cur_lr:.3e}")

    def _coarse_perceptual(
        self, coarse: torch.Tensor, gt: torch.Tensor
    ) -> torch.Tensor:
        n, t, c, h, w = coarse.shape
        return self.loss_perceptual(
            coarse.reshape(-1, c, h, w), gt.reshape(-1, c, h, w)
        )

    def train_step(self, batch) -> dict:
        self.net.train()
        lq = batch["lq"].to(self.device, non_blocking=True)
        gt = batch["gt"].to(self.device, non_blocking=True)
        frame_mask = batch.get("frame_mask")
        frame_mask = (
            frame_mask.to(self.device, non_blocking=True)
            if frame_mask is not None
            else None
        )
        w = self.weights

        if not (torch.isfinite(lq).all() and torch.isfinite(gt).all()):
            print(
                f"[iter {self.iteration}] SKIPPED: NaN/Inf in input batch (lq={lq.isnan().any()}, gt={gt.isnan().any()})"
            )
            self.iteration += 1
            return {"loss_total": float("nan"), "skipped": 1.0}

        if self.use_amp and self.scaler.get_scale() < 1.0:
            self.scaler._scale.fill_(128.0)
            print(
                f"[iter {self.iteration}] AMP scaler reset to 128.0 (was stuck at ~{self.scaler.get_scale():.2e})"
            )

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            out = self.net(lq, frame_mask)

            coarse = out["coarse"]
            coarse_f = out["coarse_f"]

            n_b, t_b, c_b, h_b, w_b = gt.shape
            gt_f = gt.reshape(n_b * t_b, c_b, h_b, w_b)
            residual_target = (gt_f - coarse_f).detach()

            loss_pix = self.loss_pix(coarse, gt)
            loss_detect = self.loss_detect(out["hole_logits_f"], out["hole_mask_f"])
            loss_v = self.loss_diffusion(
                self.net.diffusion,
                self.net.refine_unet,
                residual_target,
                out["refine_cond"],
            )
            loss_fft = self.loss_fft(coarse, gt)
            loss_vfi = self.loss_vfi(coarse, gt, frame_mask)

            total = (
                w["pix"] * loss_pix
                + w["detect"] * loss_detect
                + w["v"] * loss_v
                + w["fft"] * loss_fft
                + w["vfi"] * loss_vfi
            )

            log = {
                "loss_pix": float(loss_pix.detach()),
                "loss_detect": float(loss_detect.detach()),
                "loss_v": float(loss_v.detach()),
                "loss_fft": float(loss_fft.detach()),
                "loss_vfi": float(loss_vfi.detach()),
            }

            if self.use_perceptual:
                loss_perc = self._coarse_perceptual(coarse, gt)
                total = total + w["perceptual"] * loss_perc
                log["loss_perc"] = float(loss_perc.detach())

        self.optimizer_g.zero_grad(set_to_none=True)
        self.scaler.scale(total).backward()
        self.scaler.unscale_(self.optimizer_g)
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.net.parameters(), max_norm=self.grad_clip
            )
        scale_before = self.scaler.get_scale()
        self.scaler.step(self.optimizer_g)
        self.scaler.update()
        scale_after = self.scaler.get_scale()

        self.iteration += 1
        log["loss_total"] = float(total.detach())

        if not torch.isfinite(total):
            bad = {
                k: v
                for k, v in log.items()
                if not (isinstance(v, float) and v == v and v < float("inf"))
            }
            print(
                f"\n[WARN iter {self.iteration}] NaN/Inf loss detected! "
                f"Culprits: {bad} | "
                f"AMP scale: {scale_before:.3g} -> {scale_after:.3g}"
            )

        elif scale_after < scale_before:
            print(
                f"[iter {self.iteration}] AMP scale dropped "
                f"{scale_before:.3g} -> {scale_after:.3g} (overflow, step skipped)"
            )

        return log

    @torch.no_grad()
    def validate(self, val_loader, epoch: Optional[int] = None) -> dict:
        """Run validation on the full validation set (MambaOFR style windowing)."""
        self.net.eval()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        psnr_sum, ssim_sum, count = 0.0, 0.0, 0
        window_size = int(self.train_cfg.get("num_frame", 7))
        refine_steps = self.val_refine_steps or None

        try:
            total_clips = len(val_loader)
        except TypeError:
            total_clips = None
        if self.val_max_clips > 0 and total_clips is not None:
            total_clips = min(total_clips, self.val_max_clips)
        desc = f"Val {epoch}" if epoch is not None else "Validation"
        vbar = tqdm(
            val_loader,
            total=total_clips,
            desc=desc,
            unit="clip",
            dynamic_ncols=True,
            leave=False,
        )

        for batch in vbar:
            if self.val_max_clips > 0 and count >= self.val_max_clips:
                break

            lq = batch["lq"]
            gt = batch["gt"]

            all_len = lq.shape[1]
            if self.val_max_frames > 0:
                all_len = min(all_len, self.val_max_frames)
                lq = lq[:, :all_len]
                gt = gt[:, :all_len]
            all_output = []

            for i in range(0, all_len, window_size):
                end = min(i + window_size, all_len)
                part_lq = lq[:, i:end].to(self.device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    part_out = self.net.restore(part_lq, refine_steps=refine_steps)

                all_output.append(part_out.detach().cpu())
                del part_lq, part_out

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            full_out = torch.cat(all_output, dim=1)

            m = evaluate_clip(full_out[0].float(), gt[0].float())
            psnr_sum += m["psnr"] if m["psnr"] != float("inf") else 0.0
            ssim_sum += m["ssim"]
            count += 1

            vbar.set_postfix(
                psnr=f"{psnr_sum / count:.3f}", ssim=f"{ssim_sum / count:.4f}"
            )

            del lq, gt, full_out, all_output

        vbar.close()

        count = max(count, 1)
        return {"psnr": psnr_sum / count, "ssim": ssim_sum / count}

    @torch.no_grad()
    def _save_validation_sample(
        self, epoch: int, val_loader, tag: str = "checkpoint"
    ) -> None:
        """Save LQ / restored / GT frames from the first validation clip."""
        self.net.eval()
        batch = next(iter(val_loader))

        window_size = int(self.train_cfg.get("num_frame", 7))

        lq = batch["lq"][:, :window_size].to(self.device, non_blocking=True)
        gt = batch["gt"][:, :window_size].to(self.device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            out = self.net.restore(lq)

        def _to_grid(clip: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
            frames = clip[0].float().clamp(-1.0, 1.0)
            if frames.shape[-2:] != target_hw:
                frames = F.interpolate(
                    frames, size=target_hw, mode="bilinear", align_corners=False
                )
            return frames.add(1.0).div(2.0)

        target_hw = (int(out.shape[-2]), int(out.shape[-1]))
        nrow = int(lq.shape[1])
        grid = torch.cat(
            [
                _to_grid(lq, target_hw),
                _to_grid(out, target_hw),
                _to_grid(gt, target_hw),
            ],
            dim=0,
        )
        path = self.exp_dir / "samples" / f"epoch{epoch:03d}_{tag}.png"
        save_image(grid, path, nrow=nrow, padding=2)
        print(f"[epoch {epoch}] saved validation sample: {path}")

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: Optional[int] = None,
        start_epoch: int = 1,
    ):
        total_epochs = (
            epochs if epochs is not None else int(self.train_cfg.get("epochs", 20))
        )
        log_every = int(self.log_cfg.get("log_every", 50))
        save_every = max(1, int(self.log_cfg.get("save_checkpoint_every", 5)))

        if start_epoch > total_epochs:
            print(
                f"[trainer] training already complete "
                f"({total_epochs}/{total_epochs} epochs). "
                f"Increase training.epochs in config to continue."
            )
            return

        warmup_dataloader(train_loader, "train")

        for epoch in range(start_epoch, total_epochs + 1):
            t0 = time.time()
            try:
                total_iters = len(train_loader)
            except TypeError:
                total_iters = None
            pbar = tqdm(
                train_loader,
                total=total_iters,
                desc=f"Epoch {epoch}/{total_epochs}",
                unit="batch",
                dynamic_ncols=True,
            )
            for batch in pbar:
                log = self.train_step(batch)
                pbar.set_postfix(loss=f"{log['loss_total']:.4f}")
                if self.iteration % log_every == 0:
                    msg = " ".join(f"{k}:{v:.4f}" for k, v in log.items())
                    pbar.write(f"[epoch {epoch} iter {self.iteration}] {msg}")
            pbar.close()

            metrics = None
            do_validate = val_loader is not None and (
                epoch % self.val_every == 0 or epoch == total_epochs
            )
            if do_validate:
                metrics = self.validate(val_loader, epoch=epoch)
                print(
                    f"[epoch {epoch}] VAL psnr:{metrics['psnr']:.3f} ssim:{metrics['ssim']:.4f}"
                )
                if metrics["psnr"] > self.best_psnr:
                    self.best_psnr = metrics["psnr"]
                    self.best_epoch = epoch
                    self._save_checkpoint_file("best.pth", epoch, metrics)
                    self._save_validation_sample(epoch, val_loader, tag="best")

            self._step_scheduler(epoch, metrics)

            is_last = epoch == total_epochs
            if epoch % save_every == 0 or is_last:
                self._save_checkpoint(epoch, metrics)
                if val_loader is not None:
                    self._save_validation_sample(epoch, val_loader, tag="checkpoint")

            print(f"[epoch {epoch}] done in {time.time() - t0:.1f}s")

    def _checkpoint_state(self, epoch: int, metrics: Optional[dict] = None) -> dict:
        state = {
            "epoch": epoch,
            "epoch_numbering": 1,
            "iteration": self.iteration,
            "model": self.net.state_dict(),
            "optimizer_g": self.optimizer_g.state_dict(),
            "scheduler": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "scaler": self.scaler.state_dict(),
            "model_config": self.model_cfg.to_dict(),
            "config": self.cfg,
            "best_psnr": self.best_psnr,
            "best_epoch": self.best_epoch,
        }
        if metrics is not None:
            state["val_metrics"] = metrics
        return state

    def _save_checkpoint_file(
        self, filename: str, epoch: int, metrics: Optional[dict] = None
    ):
        path = self.exp_dir / "checkpoints" / filename
        torch.save(self._checkpoint_state(epoch, metrics), path)
        print(f"[epoch {epoch}] saved checkpoint: {path}")

    def _save_checkpoint(self, epoch: int, metrics: Optional[dict] = None):
        self._save_checkpoint_file(f"revivid_epoch{epoch:03d}.pth", epoch, metrics)
        self._save_checkpoint_file("latest.pth", epoch, metrics)

    def save_training_config(self) -> Path:
        """Persist the active config when starting a fresh training run."""
        path = self.exp_dir / "config.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.cfg, f, sort_keys=False, allow_unicode=True)
        print(f"[trainer] saved training config: {path}")
        return path

    def load_checkpoint(self, path: Union[str, Path], strict: bool = True) -> int:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(state["model"], strict=strict)
        if "optimizer_g" in state:
            self.optimizer_g.load_state_dict(state["optimizer_g"])
        if self.scheduler is not None and state.get("scheduler") is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        if "scaler" in state and self.use_amp:
            self.scaler.load_state_dict(state["scaler"])
        self.iteration = state.get("iteration", 0)
        self.best_psnr = float(state.get("best_psnr", -1.0))
        stored_epoch = int(state.get("epoch", 0))
        if state.get("epoch_numbering") == 1:
            self.best_epoch = int(state.get("best_epoch", 0))
            return stored_epoch + 1

        self.best_epoch = (
            int(state.get("best_epoch", -1)) + 1
            if state.get("best_epoch", -1) >= 0
            else 0
        )
        return stored_epoch + 2

    def maybe_resume(self, path: Optional[Union[str, Path]] = None) -> int:
        """Resume from an explicit path or ``latest.pth``; start fresh otherwise."""
        if path is not None:
            resume_path = Path(path)
        else:
            resume_path = self.exp_dir / "checkpoints" / "latest.pth"

        if resume_path.exists():
            start_epoch = self.load_checkpoint(resume_path, strict=False)
            best_at = f" @ epoch {self.best_epoch}" if self.best_epoch > 0 else ""
            print(
                f"[trainer] resumed from {resume_path} "
                f"(next epoch {start_epoch}, best psnr {self.best_psnr:.3f}{best_at})"
            )
            self.save_training_config()
            return start_epoch

        self.save_training_config()
        print("[trainer] no checkpoint found — starting fresh training")
        return 1
