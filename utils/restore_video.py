import os
import argparse
from pathlib import Path

import cv2
import torch
import numpy as np
import yaml
from tqdm import tqdm

from model import Video_Backbone
from model.config import ModelConfig

_MEAN = 0.5
_STD = 0.5


def _to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    """Convert a BGR uint8 HxWx3 frame to a normalised float32 3xHxW tensor."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1)
    return (t - _MEAN) / _STD


def _to_frame(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised 3xHxW float32 tensor back to a BGR uint8 frame."""
    t = tensor.clamp(-1.0, 1.0).cpu().float()
    t = (t * _STD + _MEAN).clamp(0.0, 1.0)
    rgb = (t.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _load_model(
    checkpoint: str, config_path: str, device: torch.device
) -> Video_Backbone:
    """Load a Video_Backbone from a trainer checkpoint file."""
    state = torch.load(checkpoint, map_location=device, weights_only=False)

    if "model_config" in state:
        model_cfg = ModelConfig.from_dict(state["model_config"])
    else:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        model_cfg = ModelConfig.from_dict(cfg.get("model", {}))

    net = Video_Backbone(model_cfg).to(device)
    net.load_state_dict(state["model"], strict=True)
    net.eval()
    return net


def _read_all_frames(cap: cv2.VideoCapture) -> list[np.ndarray]:
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames


def main():
    parser = argparse.ArgumentParser(
        description="Restore a degraded video using the trained REVIVID model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to the degraded input video file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path where the restored video will be saved.",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default="./experiments/revivid/checkpoints/latest.pth",
        help="Path to the model checkpoint (.pth).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config/REVIVID.yaml",
        help="Path to REVIVID.yaml (used when the checkpoint has no embedded config).",
    )
    parser.add_argument(
        "--clip-length",
        type=int,
        default=7,
        help="Number of frames per clip fed to the model. Should match training num_frame.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=3,
        help="Frame overlap between consecutive clips (reduces boundary artefacts).",
    )
    parser.add_argument(
        "--refine-steps",
        type=int,
        default=None,
        help="DDIM denoising steps at inference (default: value from model config).",
    )
    parser.add_argument(
        "--downscale-factor",
        type=int,
        default=1,
        help="Optional spatial downscale applied to frames before restoration (1 = no downscale).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device string, e.g. 'cuda', 'cuda:1', 'cpu'. Auto-detected if not set.",
    )

    args = parser.parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[restore_video] device: {device}")

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print(f"[restore_video] loading model from: {checkpoint}")
    net = _load_model(str(checkpoint), args.config, device)
    sr_scale = net.cfg.sr_scale
    print(f"[restore_video] sr_scale={sr_scale}, refine_steps={net.cfg.refine_steps}")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    src_w = width // args.downscale_factor
    src_h = height // args.downscale_factor
    dst_w = src_w * sr_scale
    dst_h = src_h * sr_scale

    print(f"[restore_video] input  : {args.input}  ({width}x{height} @ {fps:.2f} fps)")
    print(f"[restore_video] output : {args.output}  ({dst_w}x{dst_h})")

    print("[restore_video] reading all frames into memory …")
    all_frames = _read_all_frames(cap)
    cap.release()
    total_frames = len(all_frames)
    print(f"[restore_video] {total_frames} frames read")

    if total_frames == 0:
        raise ValueError("Input video has no frames.")

    if args.downscale_factor != 1:
        all_frames = [
            cv2.resize(f, (src_w, src_h), interpolation=cv2.INTER_AREA)
            for f in all_frames
        ]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (dst_w, dst_h))
    if not out.isOpened():
        raise IOError(f"Cannot open VideoWriter for: {args.output}")

    T = args.clip_length
    ovl = min(args.overlap, T - 1)
    step = T - ovl

    restored_sum = [None] * total_frames
    restored_count = [0] * total_frames

    starts = list(range(0, total_frames, step))

    if starts[-1] + T > total_frames:
        last_start = max(0, total_frames - T)
        if starts[-1] != last_start:
            starts.append(last_start)

    with torch.no_grad():
        for start in tqdm(starts, desc="Restoring clips", unit="clip"):
            end = min(start + T, total_frames)
            indices = list(range(start, end))

            frames_clip = [all_frames[i] for i in indices]
            while len(frames_clip) < T:
                frames_clip.append(frames_clip[-1])

            tensors = [_to_tensor(f) for f in frames_clip]
            lq = torch.stack(tensors, dim=0).unsqueeze(0).to(device)

            restored = net.restore(lq, refine_steps=args.refine_steps)

            for clip_i, frame_i in enumerate(indices):
                r_frame = restored[0, clip_i].cpu().float()
                if restored_sum[frame_i] is None:
                    restored_sum[frame_i] = r_frame
                else:
                    restored_sum[frame_i] = restored_sum[frame_i] + r_frame
                restored_count[frame_i] += 1

    print("[restore_video] writing output …")
    for i in tqdm(range(total_frames), desc="Writing frames", unit="frame"):
        avg = restored_sum[i] / restored_count[i]
        out.write(_to_frame(avg))

    out.release()
    print(f"[restore_video] done → {args.output}")


if __name__ == "__main__":
    main()
