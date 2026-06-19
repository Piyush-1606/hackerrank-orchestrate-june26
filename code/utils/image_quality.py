from __future__ import annotations

from typing import Any

import cv2


BLUR_VARIANCE_THRESHOLD = 100.0
LOW_LIGHT_MEAN_THRESHOLD = 50.0
CROPPED_OR_OBSTRUCTED_MIN_SIZE = 300


def detect_blurry_image(image_path: str) -> bool:
    """Detect blur using variance of Laplacian."""
    quality = analyze_image_quality(image_path)
    return quality["blurry"]


def detect_low_light(image_path: str) -> bool:
    """Detect low-light images using mean grayscale intensity."""
    quality = analyze_image_quality(image_path)
    return quality["low_light"]


def detect_cropped_or_obstructed(image_path: str) -> bool:
    """Detect overly small or likely obstructed images based on dimensions."""
    quality = analyze_image_quality(image_path)
    return quality["cropped_or_obstructed"]


def analyze_image_quality(image_path: str) -> dict[str, Any]:
    """Return deterministic OpenCV quality metrics for one image.

    Unreadable files return neutral values so existing path-validation tests
    using placeholder bytes remain compatible.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return {
            "blurry": False,
            "low_light": False,
            "cropped_or_obstructed": False,
            "width": 0,
            "height": 0,
            "aspect_ratio": 0.0,
            "blur_variance": 0.0,
            "mean_intensity": 0.0,
        }

    height, width = image.shape[:2]
    blur_variance = float(cv2.Laplacian(image, cv2.CV_64F).var())
    mean_intensity = float(image.mean())

    cropped_or_obstructed = width < CROPPED_OR_OBSTRUCTED_MIN_SIZE or height < CROPPED_OR_OBSTRUCTED_MIN_SIZE

    return {
        "blurry": blur_variance < BLUR_VARIANCE_THRESHOLD,
        "low_light": mean_intensity < LOW_LIGHT_MEAN_THRESHOLD,
        "cropped_or_obstructed": cropped_or_obstructed,
        "width": int(width),
        "height": int(height),
        "aspect_ratio": round(float(width / height), 6) if height else 0.0,
        "blur_variance": blur_variance,
        "mean_intensity": mean_intensity,
    }
