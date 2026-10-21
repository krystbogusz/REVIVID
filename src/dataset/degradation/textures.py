"""Film-damage textures (MambaOFR's ``texture_augmentation.py``, one to one) and
the burned-hole masks.

The textures are greyscale PIL images ("L"), dark marks on a light ground,
blended over the frames by ``pipeline.process_video_frames``.
"""

from __future__ import annotations

import math
import random

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

DILATION_FOLDERS = ("002", "003", "004", "005", "006", "007", "009", "012")


def rotated_rect_with_max_area(w, h, angle):
    """Width and height of the largest axis-aligned rectangle inside a w x h
    rectangle rotated by ``angle`` (radians)."""
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    return wr, hr


def center_crop(img, new_width, new_height):
    width, height = img.size
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    return img.crop((left, top, left + new_width, top + new_height))


def _crop_around_mark(texture, darker_by):
    """A random 150-360 px square of ``texture`` holding a pixel darker than
    its mean by ``darker_by`` (none within 40 px of the left / right edge);
    ``None`` when there is no such pixel."""
    arr = np.asarray(texture)
    mask = arr < arr.mean() - darker_by
    mask[:, :40] = False
    mask[:, -40:] = False
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    i = random.randrange(len(ys))
    anchor_y, anchor_x = int(ys[i]), int(xs[i])

    W, H = texture.size
    box = random.randint(150, 360)
    shift_x = random.randint(max(box - W + anchor_x, 0), min(anchor_x, box))
    shift_y = random.randint(max(box - H + anchor_y, 0), min(anchor_y, box))
    left, top = anchor_x - shift_x, anchor_y - shift_y
    return texture.crop((left, top, left + box, top + box))


def texture_generator(texture, folder_name, target_h, target_w):
    """A per-frame texture at the frame size: the whole template (p 0.15, and
    always for 008), else a crop around a mark, rotated by any angle; dilated
    marks for some folders, a contrast boost with p 0.7."""
    if random.uniform(0.0, 1.0) < 0.15 or folder_name == "008":
        return texture.resize((target_w, target_h), resample=Image.LANCZOS)

    cropped = _crop_around_mark(texture, 15)
    if cropped is None:
        return texture.resize((target_w, target_h), resample=Image.LANCZOS)

    box = cropped.size[0]
    angle = random.randint(0, 360)
    rotated = cropped.rotate(angle=angle, expand=True, resample=Image.BILINEAR)
    max_w, max_h = rotated_rect_with_max_area(box, box, math.radians(angle))
    final = center_crop(rotated, max_w, max_h).resize((target_w, target_h), resample=Image.LANCZOS)

    if folder_name in DILATION_FOLDERS:
        final = final.filter(ImageFilter.MinFilter(random.randint(0, 1) * 2 + 1))
    if random.uniform(0.0, 1.0) < 0.7:
        final = ImageEnhance.Contrast(final).enhance(random.uniform(2.0, 4.0))
    return final


def moving_line_texture_generator(texture, last_texture, target_h, target_w):
    """Scratches that stay over the clip: on the first frame a crop of a 001
    template around a line, then the previous frame's texture shifted
    sideways by 5-15 px (the edge columns repeated)."""
    if last_texture is None:
        return _crop_around_mark(texture, 25).resize((target_w, target_h), resample=Image.LANCZOS)

    x = np.asarray(last_texture)
    leftwards = random.uniform(0, 1) > 0.5
    d = random.randint(5, 15)
    if leftwards:
        x = np.hstack((x[:, d:], x[:, -d:]))
    else:
        x = np.hstack((x[:, :d], x[:, : target_w - d]))
    return Image.fromarray(np.uint8(x), "L")


def generate_persistent_hole_mask(h, w):
    mask = np.zeros((h, w), dtype=np.uint8)

    center_x = random.randint(int(w * 0.1), int(w * 0.9))
    center_y = random.randint(int(h * 0.1), int(h * 0.9))

    base_h = random.randint(h // 8, h // 2)
    base_w = random.randint(w // 20, w // 5)

    num_nodes = random.randint(4, 10)
    points = []
    current_y = center_y - base_h // 2

    for _ in range(num_nodes):
        offset_x = center_x + random.randint(-base_w, base_w)
        points.append([offset_x, current_y])
        current_y += base_h // num_nodes

    points = np.array(points, np.int32).reshape((-1, 1, 2))

    thickness = random.randint(10, max(11, base_w))
    cv2.polylines(mask, [points], isClosed=False, color=255, thickness=thickness)

    num_blisters = random.randint(3, 8)
    for _ in range(num_blisters):
        bx = center_x + random.randint(-base_w * 2, base_w * 2)
        by = center_y + random.randint(-base_h // 2, base_h // 2)

        axis_x = random.randint(5, max(6, base_w // 2))
        axis_y = random.randint(5, max(6, base_h // 4))

        angle = random.randint(0, 180)
        cv2.ellipse(mask, (bx, by), (axis_x, axis_y), angle, 0, 360, 255, -1)

    noise = np.random.randint(0, 256, (h, w), dtype=np.uint8)
    noise = cv2.GaussianBlur(noise, (41, 41), 0)
    _, noise_thresh = cv2.threshold(noise, 120, 255, cv2.THRESH_BINARY)

    mask = cv2.bitwise_and(mask, noise_thresh)

    kernel_erode = np.ones((3, 3), np.uint8)
    kernel_dilate = np.ones((7, 7), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_erode, iterations=1)
    mask = cv2.dilate(mask, kernel_dilate, iterations=2)

    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    return mask
