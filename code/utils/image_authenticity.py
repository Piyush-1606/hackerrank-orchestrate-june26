from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

SCREENSHOT_KEYWORDS = (
    "screenshot",
    "screen_shot",
    "capture",
)


def _average_hash(image_path: str, hash_size: int = 8) -> int:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return hash(image_path) & ((1 << (hash_size * hash_size)) - 1)

    resized = cv2.resize(image, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    mean = float(resized.mean())
    stddev = float(resized.std())
    bits = resized > mean
    hash_value = 0
    for bit in bits.flatten():
        hash_value = (hash_value << 1) | int(bit)
    mean_component = int(round(mean)) & 0xFF
    std_component = int(round(stddev)) & 0xFF
    return (hash_value << 16) | (mean_component << 8) | std_component


def analyze_image_authenticity(image_paths: list[str]) -> dict[str, Any]:
    """Return simple authenticity signals for one or more submitted images."""
    image_hashes: list[int] = []
    dimensions: list[tuple[int, int]] = []
    screenshot_detected = False

    for image_path in image_paths:
        image_hashes.append(_average_hash(image_path))
        filename = Path(image_path).name.lower()
        if any(keyword in filename for keyword in SCREENSHOT_KEYWORDS):
            screenshot_detected = True

        image = cv2.imread(image_path)
        if image is not None:
            height, width = image.shape[:2]
            dimensions.append((width, height))

    duplicate_images = len(image_hashes) > 1 and len(set(image_hashes)) < len(image_hashes)
    duplicate_submission = len(image_hashes) > 1 and len(set(image_hashes)) == 1
    possible_manipulation = duplicate_images

    if len(dimensions) > 1:
        areas = [width * height for width, height in dimensions]
        average_area = sum(areas) / len(areas)
        if any(area < average_area * 0.3 or area > average_area * 3.0 for area in areas):
            possible_manipulation = True

    if screenshot_detected:
        possible_manipulation = True

    reasons: list[str] = []
    if duplicate_images:
        reasons.append("Duplicate images detected")
    if duplicate_submission:
        reasons.append("All submitted images are identical")
    if len(dimensions) > 1 and any(
        (width * height) < sum(width * height for width, height in dimensions) / len(dimensions) * 0.3
        or (width * height) > sum(width * height for width, height in dimensions) / len(dimensions) * 3.0
        for width, height in dimensions
    ):
        reasons.append("Image dimensions differ drastically")
    if screenshot_detected:
        reasons.append("Filename suggests a screenshot or screen capture")

    reason = "; ".join(reasons) if reasons else "No obvious authenticity issues detected."

    return {
        "possible_manipulation": possible_manipulation,
        "non_original_image": screenshot_detected,
        "duplicate_images": duplicate_images,
        "duplicate_submission": duplicate_submission,
        "reason": reason,
    }
