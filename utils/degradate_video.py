import os
import cv2
import torch
import argparse
from tqdm import tqdm
from dataset.degradation import (
    build_texture_mmap,
    get_texture_cache,
    process_video_frames,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("-d", "--degree", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("-t", "--textures", type=str, default="./data/raw/noise_data")
    parser.add_argument("-b", "--batch_size", type=int, default=30)
    parser.add_argument("--downscale_factor", type=int, default=1)

    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    build_texture_mmap(args.textures)
    texture_cache = get_texture_cache(args.textures)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    target_width = width // args.downscale_factor
    target_height = height // args.downscale_factor

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (target_width, target_height))

    if not out.isOpened():
        cap.release()
        return

    frame_batch = []

    with tqdm(total=total_frames) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_batch.append(frame)

            if len(frame_batch) >= args.batch_size:
                degraded_batch = process_video_frames(
                    frame_batch,
                    texture_cache,
                    degree=args.degree,
                    downscale_factor=args.downscale_factor,
                    device=device,
                    bake_holes=True,
                )
                for df in degraded_batch:
                    out.write(df)
                frame_batch = []
                pbar.update(args.batch_size)

        if frame_batch:
            degraded_batch = process_video_frames(
                frame_batch,
                texture_cache,
                degree=args.degree,
                downscale_factor=args.downscale_factor,
                device=device,
                bake_holes=True,
            )
            for df in degraded_batch:
                out.write(df)
            pbar.update(len(frame_batch))

    cap.release()
    out.release()


if __name__ == "__main__":
    main()
