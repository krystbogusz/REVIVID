"""Training loop for the REVIVID DiffMambaOFR model.

The trainer is driven entirely by ``config/REVIVID.yaml`` (model + training
hyper-parameters only). Data locations are fixed by the pipeline, so no paths
are configured here.

It performs joint optimization of:
    * the coarse restoration (charbonnier + optional VGG perceptual),
    * the persistent-hole detector (BCE),
    * the v-prediction diffusion head,
    * sharpness losses (perceptual / frequency / gradient) on the final refined
      output to combat over-smoothing.
The learning rate decays linearly from ``training.lr`` to ``training.lr_min``
over the configured number of epochs (MambaOFR recipe). Gradients are
accumulated over ``training.grad_accum`` batches before each optimizer step,
and an exponential moving average (EMA) of the weights is maintained and used
for validation / checkpointed inference. Training runs in full precision
(float32) with checkpointing and PSNR/SSIM validation (DDIM sampling). This is
a pure diffusion model - there is no adversarial / GAN component.
"""

from __future__ import annotations

import csv
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
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
    GradientLoss,
    HoleDetectionLoss,
    VGGPerceptualLoss,
)

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "REVIVID.yaml"


class ModelEMA:
    """Exponential moving average of a model's floating-point weights.

    ``apply_to`` / ``restore`` temporarily swap the EMA weights into the live
    model (used for validation and when saving samples); the raw training
    weights are kept as a backup and restored afterwards.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {
            k: v.detach().clone().float()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }
        self._backup: Optional[dict] = None

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for k, v in model.state_dict().items():
            s = self.shadow.get(k)
            if s is not None:
                s.mul_(d).add_(v.detach().float(), alpha=1.0 - d)

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        self._backup = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if k in self.shadow
        }
        model.load_state_dict(
            {k: v.to(dtype=self._backup[k].dtype) for k, v in self.shadow.items()},
            strict=False,
        )

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        if self._backup is not None:
            model.load_state_dict(self._backup, strict=False)
            self._backup = None

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state.get("decay", self.decay))
        for k, v in (state.get("shadow") or {}).items():
            if k in self.shadow and self.shadow[k].shape == v.shape:
                self.shadow[k] = v.detach().clone().float()


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
        # every window costs `model.refine_steps` DDIM forward passes). The
        # protocol itself is fixed: full frames at `validation.gt_size`, whole
        # clips, and the model's own refine_steps — never a crop or a reduced
        # step count, so val numbers always describe the real inference path.
        self.val_every = max(1, int(self.val_cfg.get("val_every", 1)))
        self.val_max_clips = int(self.val_cfg.get("max_clips", 0))
        self.val_max_frames = int(self.val_cfg.get("max_frames", 0))

        self.device = torch.device("cuda")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required. No CUDA device found.")
        torch.manual_seed(int(cfg.get("seed", 2026)))

        self.exp_dir = Path(self.log_cfg.get("exp_dir", "./experiments/revivid"))
        (self.exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.exp_dir / "samples").mkdir(parents=True, exist_ok=True)

        self.model_cfg = ModelConfig.from_dict(cfg.get("model", {}))
        self.net = Video_Backbone(self.model_cfg).to(self.device)

        default_w = {
            "pix": 1.0,
            "perceptual": 0.1,
            "detect": 0.05,
            "v": 1.0,
            "refine_pix": 1.0,
            "refine_perceptual": 0.1,
            "refine_fft": 0.5,
            "refine_grad": 0.5,
        }
        default_w.update(self.train_cfg.get("loss_weights", {}) or {})
        self.weights = default_w
        self.refine_snr_weight = bool(self.train_cfg.get("refine_snr_weight", True))

        self.loss_pix = CharbonnierLoss().to(self.device)
        self.use_perceptual = bool(self.train_cfg.get("use_perceptual", True))
        self.loss_perceptual = (
            VGGPerceptualLoss().to(self.device) if self.use_perceptual else None
        )
        hole_pos_weight = float(self.train_cfg.get("hole_pos_weight", 10.0))
        self.loss_detect = HoleDetectionLoss(pos_weight=hole_pos_weight).to(self.device)
        self.loss_diffusion = DiffusionLoss().to(self.device)
        self.loss_fft = FocalFrequencyLoss().to(self.device)
        self.loss_grad = GradientLoss().to(self.device)

        lr = float(self.train_cfg.get("lr", 2e-4))
        betas = (
            float(self.train_cfg.get("beta1", 0.9)),
            float(self.train_cfg.get("beta2", 0.999)),
        )
        self.optimizer_g = torch.optim.AdamW(
            [
                {
                    "params": [p for p in self.net.parameters() if p.requires_grad],
                    "lr": lr,
                    "lr_mult": 1.0,
                }
            ],
            lr=lr,
            betas=betas,
        )
        self.grad_clip = float(self.train_cfg.get("grad_clip", 1.0))

        # RAFT warmup (MambaOFR recipe): the flow starts frozen and is unfrozen
        # after `raft_unfreeze_iter` iterations at `flow_lr_mul` * base LR.
        # Set raft_unfreeze_iter < 0 to keep the flow frozen forever.
        self.raft_unfreeze_iter = int(self.train_cfg.get("raft_unfreeze_iter", 20000))
        self.flow_lr_mul = float(self.train_cfg.get("flow_lr_mul", 0.125))
        self._raft_unfrozen = False

        # Pixels inside persistent holes are re-weighted by (1 + hole_loss_boost)
        # in the coarse pixel loss and the diffusion v-loss, so the few hole
        # pixels are not drowned out by the full-frame average.
        self.hole_loss_boost = float(self.train_cfg.get("hole_loss_boost", 3.0))

        # Linear LR decay: lr (epoch 1) -> lr_min (last epoch).
        self.base_lr = lr
        self.lr_min = float(self.train_cfg.get("lr_min", 1e-6))

        # Gradient accumulation: optimizer steps every `grad_accum` batches.
        self.grad_accum = max(1, int(self.train_cfg.get("grad_accum", 1)))
        self._accum_count = 0

        # EMA of the model weights, used for validation / saved with checkpoints.
        ema_decay = float(self.train_cfg.get("ema_decay", 0.999))
        self.ema = ModelEMA(self.net, ema_decay) if ema_decay > 0 else None

        self.iteration = 0
        self.best_psnr = -1.0
        self.best_epoch = 0

        # Per-epoch mean losses (+ val metrics when available). Stored inside
        # every checkpoint and mirrored to <exp_dir>/loss_history.csv so loss
        # curves can be plotted without loading the .pth files.
        self.loss_history: list[dict] = []

    def _set_epoch_lr(self, epoch: int, total_epochs: int) -> float:
        """Linearly decay the LR from ``base_lr`` (epoch 1) to ``lr_min`` (last epoch)."""
        if total_epochs <= 1:
            lr = self.base_lr
        else:
            frac = min(max((epoch - 1) / (total_epochs - 1), 0.0), 1.0)
            lr = self.base_lr + (self.lr_min - self.base_lr) * frac
        for group in self.optimizer_g.param_groups:
            group["lr"] = lr * group.get("lr_mult", 1.0)
        return lr

    def _maybe_unfreeze_raft(self, force: bool = False) -> None:
        """Unfreeze RAFT once ``iteration`` reaches ``raft_unfreeze_iter``."""
        if self._raft_unfrozen:
            return
        if not force and (
            self.raft_unfreeze_iter < 0 or self.iteration < self.raft_unfreeze_iter
        ):
            return
        flow = self.net.backbone.flow_net
        flow.set_trainable(True)
        base_lr = self.optimizer_g.param_groups[0]["lr"]
        self.optimizer_g.add_param_group(
            {
                "params": list(flow.parameters()),
                "lr": base_lr * self.flow_lr_mul,
                "lr_mult": self.flow_lr_mul,
            }
        )
        self._raft_unfrozen = True
        print(
            f"[iter {self.iteration}] RAFT unfrozen for fine-tuning "
            f"(lr mult {self.flow_lr_mul})"
        )

    @contextmanager
    def _ema_weights(self):
        """Temporarily swap the EMA weights into ``self.net``."""
        if self.ema is not None:
            self.ema.apply_to(self.net)
        try:
            yield
        finally:
            if self.ema is not None:
                self.ema.restore(self.net)

    def _coarse_perceptual(
        self, coarse: torch.Tensor, gt: torch.Tensor
    ) -> torch.Tensor:
        n, t, c, h, w = coarse.shape
        return self.loss_perceptual(
            coarse.reshape(-1, c, h, w), gt.reshape(-1, c, h, w)
        )

    def train_step(self, batch) -> dict:
        self.net.train()
        self._maybe_unfreeze_raft()
        lq = batch["lq"].to(self.device, non_blocking=True)
        gt = batch["gt"].to(self.device, non_blocking=True)
        w = self.weights

        if not (torch.isfinite(lq).all() and torch.isfinite(gt).all()):
            print(
                f"[iter {self.iteration}] SKIPPED: NaN/Inf in input batch (lq={lq.isnan().any()}, gt={gt.isnan().any()})"
            )
            self.iteration += 1
            return {"loss_total": float("nan"), "skipped": 1.0}

        out = self.net(lq)

        coarse = out["coarse"]
        coarse_f = out["coarse_f"]

        n_b, t_b, c_b, h_b, w_b = gt.shape
        gt_f = gt.reshape(n_b * t_b, c_b, h_b, w_b)
        # Normalise the residual by a running estimate of its own std so the
        # diffusion target always sits in the schedule's native ~N(0, 1) range.
        # A fixed scale collapses as `coarse` improves: the target shrinks, v
        # becomes a closed form of x_t, the loss falls to ~0 and the refiner
        # stops learning anything about the image. Inverted below and in
        # restore().
        residual = gt_f - coarse_f
        res_std = self.net.update_residual_std(residual)
        residual_target = (residual / res_std).detach()

        # Boost the loss inside persistent holes so the sparse hole pixels are
        # not averaged away (hole_mask_f comes from the LQ fill-value threshold,
        # i.e. it is ground truth during training).
        pix_weight = (1.0 + self.hole_loss_boost * out["hole_mask_f"]).expand(
            -1, 3, -1, -1
        )

        loss_pix = self.loss_pix(coarse_f, gt_f, weight=pix_weight)
        loss_detect = self.loss_detect(out["hole_logits_f"], out["hole_mask_f"])
        loss_v, diff_info = self.loss_diffusion(
            self.net.diffusion,
            self.net.refine_unet,
            residual_target,
            out["refine_cond"],
            loss_mask=pix_weight,
        )

        # Sharpness losses on the FINAL refiner output (coarse + predicted
        # residual). x0_pred is the diffusion estimate of the clean residual;
        # coarse is detached so these losses train the refiner (and the
        # backbone `cond` features), not the coarse branch.
        refined_pred = torch.clamp(
            coarse_f.detach() + diff_info["x0_pred"] * res_std, -1.0, 1.0
        )
        # Pixel loss on the FINAL output: without it nothing anchors the
        # refined image to the GT in pixel space, which is exactly what
        # val_psnr measures.
        loss_r_pix = self.loss_pix(refined_pred, gt_f, weight=pix_weight)
        loss_r_fft = self.loss_fft(refined_pred, gt_f)
        loss_r_grad = self.loss_grad(refined_pred, gt_f)
        if self.use_perceptual:
            loss_r_perc = self.loss_perceptual(refined_pred, gt_f)
        else:
            loss_r_perc = coarse_f.new_zeros(())

        # Down-weight the refined losses when x0_pred is unreliable (high
        # noise / large timestep); alphas_cumprod[t] ~ 1 at low noise, ~ 0
        # at high noise.
        if self.refine_snr_weight:
            w_snr = self.net.diffusion.alphas_cumprod[
                diff_info["t"]
            ].float().mean()
        else:
            w_snr = coarse_f.new_ones(())

        total = (
            w["pix"] * loss_pix
            + w["detect"] * loss_detect
            + w["v"] * loss_v
            + w_snr
            * (
                w["refine_pix"] * loss_r_pix
                + w["refine_perceptual"] * loss_r_perc
                + w["refine_fft"] * loss_r_fft
                + w["refine_grad"] * loss_r_grad
            )
        )

        log = {
            "loss_pix": float(loss_pix.detach()),
            "loss_detect": float(loss_detect.detach()),
            "loss_v": float(loss_v.detach()),
            "loss_r_pix": float(loss_r_pix.detach()),
            "loss_r_fft": float(loss_r_fft.detach()),
            "loss_r_grad": float(loss_r_grad.detach()),
            "residual_std": float(res_std.detach()),
        }
        if self.use_perceptual:
            log["loss_r_perc"] = float(loss_r_perc.detach())

        if self.use_perceptual:
            loss_perc = self._coarse_perceptual(coarse, gt)
            total = total + w["perceptual"] * loss_perc
            log["loss_perc"] = float(loss_perc.detach())

        # Accumulate gradients over `grad_accum` batches, then step once.
        (total / self.grad_accum).backward()
        self._accum_count += 1

        if self._accum_count >= self.grad_accum:
            self._accum_count = 0
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), max_norm=self.grad_clip
                )
            self.optimizer_g.step()
            self.optimizer_g.zero_grad(set_to_none=True)
            if self.ema is not None:
                self.ema.update(self.net)

        self.iteration += 1
        log["loss_total"] = float(total.detach())

        if not torch.isfinite(total):
            bad = {
                k: v
                for k, v in log.items()
                if not (isinstance(v, float) and v == v and v < float("inf"))
            }
            print(f"\n[WARN iter {self.iteration}] NaN/Inf loss detected! Culprits: {bad}")

        return log

    @torch.no_grad()
    def validate(self, val_loader, epoch: Optional[int] = None) -> dict:
        """Run validation on the full validation set (MambaOFR style windowing).

        Uses the EMA weights (if enabled) and reports metrics both for the
        final refined output and for the backbone's coarse output, so it is
        visible whether the diffusion refinement helps or hurts PSNR/SSIM.
        """
        with self._ema_weights():
            return self._validate_impl(val_loader, epoch=epoch)

    def _validate_impl(self, val_loader, epoch: Optional[int] = None) -> dict:
        self.net.eval()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        psnr_sum, ssim_sum, count = 0.0, 0.0, 0
        c_psnr_sum, c_ssim_sum = 0.0, 0.0
        window_size = int(self.train_cfg.get("num_frame", 7))

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
            all_coarse = []

            for i in range(0, all_len, window_size):
                end = min(i + window_size, all_len)
                part_lq = lq[:, i:end].to(self.device, non_blocking=True)

                part_out, part_coarse = self.net.restore(
                    part_lq, return_coarse=True
                )

                all_output.append(part_out.detach().cpu())
                all_coarse.append(part_coarse.detach().clamp(-1.0, 1.0).cpu())
                del part_lq, part_out, part_coarse

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            full_out = torch.cat(all_output, dim=1)
            full_coarse = torch.cat(all_coarse, dim=1)

            m = evaluate_clip(full_out[0].float(), gt[0].float())
            psnr_sum += m["psnr"] if m["psnr"] != float("inf") else 0.0
            ssim_sum += m["ssim"]

            mc = evaluate_clip(full_coarse[0].float(), gt[0].float())
            c_psnr_sum += mc["psnr"] if mc["psnr"] != float("inf") else 0.0
            c_ssim_sum += mc["ssim"]
            count += 1

            vbar.set_postfix(
                psnr=f"{psnr_sum / count:.3f}",
                ssim=f"{ssim_sum / count:.4f}",
                coarse_psnr=f"{c_psnr_sum / count:.3f}",
            )

            del lq, gt, full_out, full_coarse, all_output, all_coarse

        vbar.close()

        count = max(count, 1)
        return {
            "psnr": psnr_sum / count,
            "ssim": ssim_sum / count,
            "psnr_coarse": c_psnr_sum / count,
            "ssim_coarse": c_ssim_sum / count,
        }

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

        with self._ema_weights():
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
            lr_now = self._set_epoch_lr(epoch, total_epochs)
            try:
                total_iters = len(train_loader)
            except TypeError:
                total_iters = None
            pbar = tqdm(
                train_loader,
                total=total_iters,
                desc=f"Epoch {epoch}/{total_epochs} (lr {lr_now:.2e})",
                unit="batch",
                dynamic_ncols=True,
            )
            epoch_sums: dict[str, float] = {}
            epoch_count = 0
            for batch in pbar:
                log = self.train_step(batch)
                if not log.get("skipped"):
                    for k, v in log.items():
                        if isinstance(v, float) and v == v:  # skip NaN
                            epoch_sums[k] = epoch_sums.get(k, 0.0) + v
                    epoch_count += 1
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
                # `delta` is the whole point of the refiner: positive means the
                # diffusion head improved on the coarse output, negative means
                # it is actively damaging it. Printed explicitly so a broken
                # refiner is visible within the first few validations instead
                # of thousands of epochs later.
                delta = metrics["psnr"] - metrics["psnr_coarse"]
                verdict = "refiner POMAGA" if delta > 0 else "refiner SZKODZI"
                print(
                    f"[epoch {epoch}] VAL refined psnr:{metrics['psnr']:.3f} "
                    f"ssim:{metrics['ssim']:.4f} | coarse "
                    f"psnr:{metrics['psnr_coarse']:.3f} "
                    f"ssim:{metrics['ssim_coarse']:.4f} "
                    f"| delta:{delta:+.3f} dB ({verdict}) "
                    f"| residual_std:{float(self.net.residual_std):.4f}"
                )

            # Record the epoch's mean losses (and val metrics when available)
            # BEFORE saving any checkpoint, so every checkpoint carries the
            # loss history up to and including its own epoch.
            entry = {"epoch": epoch, "lr": lr_now}
            if epoch_count > 0:
                entry.update({k: v / epoch_count for k, v in epoch_sums.items()})
            if metrics is not None:
                entry.update(
                    {
                        "val_psnr": metrics["psnr"],
                        "val_ssim": metrics["ssim"],
                        "val_psnr_coarse": metrics["psnr_coarse"],
                        "val_ssim_coarse": metrics["ssim_coarse"],
                    }
                )
            self.loss_history.append(entry)
            self._write_loss_history()

            if do_validate:
                # Select best on the coarse PSNR: it is the stable measure of
                # restoration progress (refined has DDIM sampling variance).
                if metrics["psnr_coarse"] > self.best_psnr:
                    self.best_psnr = metrics["psnr_coarse"]
                    self.best_epoch = epoch
                    self._save_checkpoint_file("best.pth", epoch, metrics)
                    self._save_validation_sample(epoch, val_loader, tag="best")

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
            "ema": self.ema.state_dict() if self.ema is not None else None,
            "optimizer_g": self.optimizer_g.state_dict(),
            "raft_unfrozen": self._raft_unfrozen,
            "model_config": self.model_cfg.to_dict(),
            "config": self.cfg,
            "best_psnr": self.best_psnr,
            "best_epoch": self.best_epoch,
            "loss_history": self.loss_history,
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

    def _write_loss_history(self) -> None:
        """Mirror the loss history to <exp_dir>/loss_history.csv for plotting."""
        if not self.loss_history:
            return
        fieldnames: list[str] = []
        for entry in self.loss_history:
            for k in entry:
                if k not in fieldnames:
                    fieldnames.append(k)
        path = self.exp_dir / "loss_history.csv"
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
            writer.writeheader()
            writer.writerows(self.loss_history)

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
        if self.ema is not None:
            if state.get("ema"):
                self.ema.load_state_dict(state["ema"])
            else:
                # Older checkpoint without EMA — seed the shadow from the
                # freshly loaded weights instead of the random init.
                self.ema = ModelEMA(self.net, self.ema.decay)
        self.iteration = state.get("iteration", 0)
        self.loss_history = list(state.get("loss_history") or [])
        if state.get("raft_unfrozen"):
            # Recreate the flow param group BEFORE loading the optimizer state
            # so the group structure matches the checkpoint.
            self._maybe_unfreeze_raft(force=True)
        if "optimizer_g" in state:
            self.optimizer_g.load_state_dict(state["optimizer_g"])
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
