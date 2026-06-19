from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main

import cv2
import numpy as np

from code.agents.vision_agent import detect_blurry_image


def laplacian_variance(image_path: Path) -> float:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read test image: {image_path}")
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def create_sharp_image(path: Path) -> None:
    image = np.zeros((256, 256), dtype=np.uint8)
    image[32:224, 32:224] = 255
    image[64:192, 64:192] = 0
    cv2.line(image, (0, 0), (255, 255), 255, 3)
    cv2.imwrite(str(path), image)


class BlurDetectionTest(TestCase):
    def test_sharp_synthetic_image(self) -> None:
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sharp.png"
            create_sharp_image(image_path)

            variance = laplacian_variance(image_path)
            print(f"sharp variance: {variance:.2f}")

            self.assertGreaterEqual(variance, 100.0)
            self.assertFalse(detect_blurry_image(str(image_path)))

    def test_blurred_synthetic_image(self) -> None:
        with TemporaryDirectory() as temp_dir:
            sharp_path = Path(temp_dir) / "sharp.png"
            blurred_path = Path(temp_dir) / "blurred.png"
            create_sharp_image(sharp_path)

            sharp = cv2.imread(str(sharp_path), cv2.IMREAD_GRAYSCALE)
            blurred = cv2.GaussianBlur(sharp, (31, 31), 0)
            cv2.imwrite(str(blurred_path), blurred)

            variance = laplacian_variance(blurred_path)
            print(f"blurred variance: {variance:.2f}")

            self.assertLess(variance, 100.0)
            self.assertTrue(detect_blurry_image(str(blurred_path)))


if __name__ == "__main__":
    main()
