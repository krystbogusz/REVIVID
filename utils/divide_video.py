import argparse
import os
import random

import cv2
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-n', '--num-frames', type=int, default=None)

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.num_frames is not None:
        num_to_save = min(args.num_frames, total_frames)
        target_frames = sorted(random.sample(range(total_frames), num_to_save))

        with tqdm(total=num_to_save) as pbar:
            for frame_idx in target_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame_filename = os.path.join(
                        args.output, f"{frame_idx:07d}.jpg"
                    )
                    cv2.imwrite(frame_filename, frame)
                pbar.update(1)
    else:
        frame_count = 0
        with tqdm(total=total_frames) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_filename = os.path.join(
                    args.output, f"{frame_count:07d}.jpg"
                )
                cv2.imwrite(frame_filename, frame)
                frame_count += 1
                pbar.update(1)

    cap.release()


if __name__ == "__main__":
    main()