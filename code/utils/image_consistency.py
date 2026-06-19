from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from .image_quality import analyze_image_quality


def _average_hash(image_path: str, hash_size: int = 8) -> tuple[int, int, int]:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        fallback = hash(image_path)
        return (fallback, 0, 0)

    resized = cv2.resize(image, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    mean = float(resized.mean())
    std = float(resized.std())
    bits = resized > mean
    hash_value = 0
    for bit in bits.flatten():
        hash_value = (hash_value << 1) | int(bit)
    mean_bucket = int(mean // 8)
    std_bucket = int(std // 8)
    return (hash_value, mean_bucket, std_bucket)


def analyze_image_set(image_paths: list[str]) -> dict[str, Any]:
    """Return deterministic consistency metrics for a set of images."""
    image_count = len(image_paths)
    if image_count == 0:
        return {
            "image_count": 0,
            "duplicate_images": False,
            "mixed_object_types": False,
            "consistency_score": 1.0,
            "supporting_image_ids": [],
        }

    image_hashes: list[tuple[int, int, int]] = []
    image_areas: list[int] = []
    quality_penalty = 0.0
    supporting_image_ids: list[str] = []

    for image_path in image_paths:
        image_hashes.append(_average_hash(image_path))
        supporting_image_ids.append(Path(image_path).stem)

        quality = analyze_image_quality(image_path)
        if quality["blurry"] or quality["low_light"] or quality.get("cropped_or_obstructed", False):
            quality_penalty += 0.05

        image = cv2.imread(image_path)
        if image is not None:
            height, width = image.shape[:2]
            image_areas.append(width * height)

    duplicate_images = len(set(image_hashes)) < len(image_hashes)
    mixed_object_types = False

    tiny_penalty = 0.0
    if image_areas:
        average_area = sum(image_areas) / len(image_areas)
        if average_area > 0:
            tiny_images = [area for area in image_areas if area < average_area * 0.3]
            if tiny_images:
                tiny_penalty = 0.15

    duplicate_penalty = 0.15 if duplicate_images else 0.0
    quality_penalty = min(quality_penalty, 0.3)
    consistency_score = 1.0 - duplicate_penalty - tiny_penalty - quality_penalty
    consistency_score = round(max(0.0, consistency_score), 3)

    return {
        "image_count": image_count,
        "duplicate_images": duplicate_images,
        "mixed_object_types": mixed_object_types,
        "consistency_score": consistency_score,
        "supporting_image_ids": supporting_image_ids,
    }
