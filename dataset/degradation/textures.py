"""Texture generation and manipulation for film-grain simulation.

Functions produce greyscale textures (uint8 NumPy arrays) that are blended
over degraded frames by ``pipeline.process_video_frames``.
"""

from __future__ import annotations

import math
import os
import random

import cv2
import numpy as np


def rotated_rect_with_max_area(w, h, angle):
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
    height, width = img.shape[:2]
    left = int((width - new_width) / 2)
    top = int((height - new_height) / 2)
    right = left + new_width
    bottom = top + new_height
    return img[top:bottom, left:right]


def generate_texture(texture_input, folder_name, target_h=256, target_w=256):
    random_prob = random.uniform(0.0, 1.0)
    dilation_group = ["002", "003", "004", "005", "006", "007", "009", "012"]
    dilation_flag = folder_name in dilation_group
    img_h_orig, img_w_orig = texture_input.shape[:2]

    if random_prob < 0.15 or folder_name == "008":
        if img_h_orig != target_h or img_w_orig != target_w:
            return cv2.resize(
                texture_input, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4
            )
        return texture_input

    texture_mean = np.mean(texture_input)
    texture_mask = (texture_input < (texture_mean - 15)).astype(np.float32)

    texture_mask[:, :40] = 0.0
    texture_mask[:, -40:] = 0.0

    if np.sum(texture_mask) < 1:
        if img_h_orig != target_h or img_w_orig != target_w:
            return cv2.resize(
                texture_input, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4
            )
        return texture_input

    y_indices, x_indices = np.where(texture_mask > 0)
    idx = random.randint(0, len(y_indices) - 1)
    anchor_y, anchor_x = y_indices[idx], x_indices[idx]

    bounding_box_size = random.randint(150, 360)

    shift_x = random.randint(
        max(bounding_box_size - img_w_orig + anchor_x, 0),
        min(anchor_x, bounding_box_size),
    )
    shift_y = random.randint(
        max(bounding_box_size - img_h_orig + anchor_y, 0),
        min(anchor_y, bounding_box_size),
    )

    left_up_x = anchor_x - shift_x
    left_up_y = anchor_y - shift_y

    padded_texture = cv2.copyMakeBorder(
        texture_input,
        max(0, -left_up_y),
        max(0, left_up_y + bounding_box_size - img_h_orig),
        max(0, -left_up_x),
        max(0, left_up_x + bounding_box_size - img_w_orig),
        cv2.BORDER_REFLECT,
    )

    crop_x1 = max(0, left_up_x)
    crop_y1 = max(0, left_up_y)
    cropped_texture = padded_texture[
        crop_y1 : crop_y1 + bounding_box_size, crop_x1 : crop_x1 + bounding_box_size
    ]

    rotation_angle = random.randint(0, 360)
    center = (bounding_box_size // 2, bounding_box_size // 2)
    rot_mat = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

    abs_cos = abs(rot_mat[0, 0])
    abs_sin = abs(rot_mat[0, 1])
    bound_w = int(bounding_box_size * abs_cos + bounding_box_size * abs_sin)
    bound_h = int(bounding_box_size * abs_sin + bounding_box_size * abs_cos)
    rot_mat[0, 2] += bound_w / 2 - center[0]
    rot_mat[1, 2] += bound_h / 2 - center[1]

    rotated_texture = cv2.warpAffine(
        cropped_texture, rot_mat, (bound_w, bound_h), flags=cv2.INTER_LINEAR
    )
    max_w, max_h = rotated_rect_with_max_area(
        bounding_box_size, bounding_box_size, math.radians(rotation_angle)
    )

    final_texture = center_crop(rotated_texture, int(max_w), int(max_h))

    h_final, w_final = final_texture.shape[:2]
    if h_final != target_h or w_final != target_w:
        final_texture = cv2.resize(
            final_texture, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4
        )

    if dilation_flag:
        dilation_kernel_size = random.randint(0, 1) * 2 + 1
        kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
        final_texture = cv2.erode(final_texture, kernel, iterations=1)

    if random.uniform(0.0, 1.0) < 0.7:
        alpha_contrast = random.uniform(2.0, 4.0)
        final_texture = cv2.convertScaleAbs(
            final_texture, alpha=alpha_contrast, beta=128 * (1 - alpha_contrast)
        )

    return final_texture


def generate_moving_line_texture(
    texture_input, last_texture, target_h=256, target_w=256
):
    if last_texture is None:
        img_h, img_w = texture_input.shape[:2]
        texture_mean = np.mean(texture_input)
        texture_mask = (texture_input < (texture_mean - 25)).astype(np.float32)

        texture_mask[:, :40] = 0.0
        texture_mask[:, -40:] = 0.0

        y_indices, x_indices = np.where(texture_mask > 0)
        if len(y_indices) > 0:
            idx = random.randint(0, len(y_indices) - 1)
            anchor_y, anchor_x = y_indices[idx], x_indices[idx]
        else:
            anchor_y, anchor_x = img_h // 2, img_w // 2

        bounding_box_size = random.randint(150, 360)

        shift_x = random.randint(
            max(bounding_box_size - img_w + anchor_x, 0),
            min(anchor_x, bounding_box_size),
        )
        shift_y = random.randint(
            max(bounding_box_size - img_h + anchor_y, 0),
            min(anchor_y, bounding_box_size),
        )

        left_up_x = anchor_x - shift_x
        left_up_y = anchor_y - shift_y

        padded_texture = cv2.copyMakeBorder(
            texture_input,
            max(0, -left_up_y),
            max(0, left_up_y + bounding_box_size - img_h),
            max(0, -left_up_x),
            max(0, left_up_x + bounding_box_size - img_w),
            cv2.BORDER_REFLECT,
        )

        crop_x1 = max(0, left_up_x)
        crop_y1 = max(0, left_up_y)
        cropped_texture = padded_texture[
            crop_y1 : crop_y1 + bounding_box_size, crop_x1 : crop_x1 + bounding_box_size
        ]

        h_crop, w_crop = cropped_texture.shape[:2]
        if h_crop != target_h or w_crop != target_w:
            final_texture = cv2.resize(
                cropped_texture, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4
            )
        else:
            final_texture = cropped_texture
    else:
        random_direction = random.uniform(0, 1) > 0.5
        random_distance = random.randint(5, 15)
        texture_np = last_texture.copy()

        if random_direction:
            final_texture = np.roll(texture_np, -random_distance, axis=1)
        else:
            final_texture = np.roll(texture_np, random_distance, axis=1)

    return final_texture, final_texture


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
