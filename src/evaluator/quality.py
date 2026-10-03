"""Quality metrics for the evaluator.

Full-reference metrics follow the MambaOFR protocol (``VP_code/metrics/measure.py``,
``fid.py``) 1:1: grayscale-as-RGB frames, PSNR, SSIM (win 21), LPIPS (alex) and
FID on torchvision InceptionV3 pool3 features.

No-reference metrics follow current practice in video restoration: image IQA
models from pyiqa (IQA-PyTorch) averaged over frames at the native output
resolution, plus the flow-based temporal warping error E*warp.
Frames are BGR uint8 arrays (OpenCV layout).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

FULL_REFERENCE = ("psnr", "ssim", "lpips", "fid")
DEFAULT_NO_REFERENCE = ("niqe", "brisque", "musiq", "clipiqa", "maniqa")
TEMPORAL = ("ewarp",)
LOWER_IS_BETTER = {"lpips", "fid", "ewarp"}


def _gray_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    rgb = np.ascontiguousarray(frame_bgr[..., ::-1])
    return np.asarray(Image.fromarray(rgb).convert("L").convert("RGB"))


def _rgb_tensor(frame_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = np.ascontiguousarray(frame_bgr[..., ::-1])
    return torch.from_numpy(rgb).permute(2, 0, 1)[None].float().div(255.0).to(device)


class FullReferenceMetrics:
    def __init__(self, device: torch.device):
        import lpips

        self.device = device
        self._lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    @torch.no_grad()
    def __call__(self, out_bgr: np.ndarray, gt_bgr: np.ndarray) -> Dict[str, float]:
        fake, real = _gray_rgb(out_bgr), _gray_rgb(gt_bgr)
        fake01 = fake.astype(np.float32) / 255.0
        real01 = real.astype(np.float32) / 255.0
        to_t = lambda a: torch.from_numpy(a).permute(2, 0, 1)[None].to(self.device) * 2 - 1
        return {
            "psnr": float(peak_signal_noise_ratio(real01, fake01, data_range=1.0)),
            "ssim": float(
                structural_similarity(real01, fake01, data_range=1.0, win_size=21, channel_axis=-1)
            ),
            "lpips": float(self._lpips(to_t(fake01), to_t(real01)).item()),
        }


class NoReferenceMetrics:
    """pyiqa image-quality models, scored per frame on the RGB output as-is."""

    def __init__(self, device: torch.device, names: Sequence[str] = DEFAULT_NO_REFERENCE):
        import packaging.version
        import pkg_resources

        # openai-clip (used by pyiqa's CLIP-IQA) imports pkg_resources.packaging,
        # which setuptools >= 70 no longer exposes.
        if not hasattr(pkg_resources, "packaging"):
            pkg_resources.packaging = packaging
        import pyiqa

        self.device = device
        self.models = {n: pyiqa.create_metric(n, device=device, as_loss=False) for n in names}

    @property
    def lower_better(self) -> set:
        return {n for n, m in self.models.items() if m.lower_better}

    @torch.no_grad()
    def __call__(self, out_bgr: np.ndarray) -> Dict[str, float]:
        x = _rgb_tensor(out_bgr, self.device)
        return {n: float(m(x).mean().item()) for n, m in self.models.items()}


class WarpingError:
    """E*warp (Lai et al., ECCV 2018) on the output video itself, reported x1e-3.

    RAFT flow between consecutive output frames, forward-backward occlusion
    check, squared RGB error in [0, 1] over non-occluded pixels. Flow is
    estimated with the longer side capped at ``flow_max_side`` (RAFT's all-pairs
    correlation does not fit in memory at 4K) and upsampled to the native size;
    warping and the error itself are always computed at native resolution.
    """

    def __init__(self, device: torch.device, flow_max_side: int = 1024):
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

        weights = Raft_Large_Weights.DEFAULT
        self.raft = raft_large(weights=weights).to(device).eval()
        self.prep = weights.transforms()
        self.device = device
        self.flow_max_side = flow_max_side

    def _flow(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        h, w = a.shape[-2:]
        scale = min(1.0, self.flow_max_side / max(h, w))
        fh, fw = max(8, round(h * scale / 8) * 8), max(8, round(w * scale / 8) * 8)
        a_s = F.interpolate(a, size=(fh, fw), mode="bilinear", align_corners=False)
        b_s = F.interpolate(b, size=(fh, fw), mode="bilinear", align_corners=False)
        a_s, b_s = self.prep(a_s, b_s)
        flow = self.raft(a_s, b_s)[-1]
        flow = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False)
        return flow * torch.tensor([w / fw, h / fh], device=flow.device).view(1, 2, 1, 1)

    @staticmethod
    def _warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        ys, xs = torch.meshgrid(
            torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing="ij"
        )
        gx = (xs[None] + flow[:, 0]) / max(w - 1, 1) * 2 - 1
        gy = (ys[None] + flow[:, 1]) / max(h - 1, 1) * 2 - 1
        grid = torch.stack((gx, gy), dim=-1)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)

    @torch.no_grad()
    def __call__(self, frames_bgr: List[np.ndarray]) -> float:
        errs = []
        prev = _rgb_tensor(frames_bgr[0], self.device)
        for frame in frames_bgr[1:]:
            cur = _rgb_tensor(frame, self.device)
            fw, bw = self._flow(prev, cur), self._flow(cur, prev)
            bw_at_fw = self._warp(bw, fw)
            mismatch = ((fw + bw_at_fw) ** 2).sum(1)
            occluded = mismatch > 0.01 * ((fw**2).sum(1) + (bw_at_fw**2).sum(1)) + 0.5
            valid = (~occluded).float()
            sq = ((prev - self._warp(cur, fw)) ** 2).sum(1)
            errs.append(float((sq * valid).sum() / valid.sum().clamp_min(1.0)))
            prev = cur
        return float(np.mean(errs)) * 1e3 if errs else float("nan")


class InceptionFeatures:
    """Pool3 activations of InceptionV3, preprocessed exactly like MambaOFR's fid.py."""

    def __init__(self, device: torch.device, batch_size: int = 64):
        from torchvision.models import Inception_V3_Weights, inception_v3

        net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        self.net = nn.Sequential(
            net.Conv2d_1a_3x3, net.Conv2d_2a_3x3, net.Conv2d_2b_3x3, nn.MaxPool2d(3, 2),
            net.Conv2d_3b_1x1, net.Conv2d_4a_3x3, nn.MaxPool2d(3, 2),
            net.Mixed_5b, net.Mixed_5c, net.Mixed_5d, net.Mixed_6a,
            net.Mixed_6b, net.Mixed_6c, net.Mixed_6d, net.Mixed_6e,
            net.Mixed_7a, net.Mixed_7b, net.Mixed_7c, nn.AdaptiveAvgPool2d(1),
        ).to(device).eval()
        self.device = device
        self.batch_size = batch_size

    @torch.no_grad()
    def __call__(self, frames_bgr: List[np.ndarray]) -> np.ndarray:
        feats = []
        for i in range(0, len(frames_bgr), self.batch_size):
            batch = np.stack([f[..., ::-1] for f in frames_bgr[i : i + self.batch_size]])
            x = torch.from_numpy(batch).permute(0, 3, 1, 2).float().div(255.0).to(self.device)
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
            x = torch.stack(
                [
                    x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5,
                    x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5,
                    x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5,
                ],
                dim=1,
            )
            feats.append(self.net(x).flatten(1).double().cpu().numpy())
        return np.concatenate(feats) if feats else np.empty((0, 2048))


def frechet_distance(feats_a: np.ndarray, feats_b: np.ndarray, eps: float = 1e-6) -> float:
    mu1, mu2 = feats_a.mean(0), feats_b.mean(0)
    sigma1 = np.atleast_2d(np.cov(feats_a, rowvar=False))
    sigma2 = np.atleast_2d(np.cov(feats_b, rowvar=False))
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"Imaginary component {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def mean_scores(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    out = {}
    for key in rows[0]:
        vals = [r[key] for r in rows if np.isfinite(r[key])]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    return out
