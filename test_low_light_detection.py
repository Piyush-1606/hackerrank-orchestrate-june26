from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main

import cv2
import numpy as np

from code.agents.vision_agent import detect_low_light


def mean_intensity(image_path: Path) -> float:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read test image: {image_path}")
    return float(image.mean())


def create_solid_image(path: Path, intensity: int) -> None:
    image = np.full((256, 256), intensity, dtype=np.uint8)
    cv2.imwrite(str(path), image)


class LowLightDetectionTest(TestCase):
    def test_bright_synthetic_image(self) -> None:
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "bright.png"
            create_solid_image(image_path, 180)

            value = mean_intensity(image_path)
            print(f"bright mean intensity: {value:.2f}")

            self.assertGreaterEqual(value, 50.0)
            self.assertFalse(detect_low_light(str(image_path)))

    def test_dark_synthetic_image(self) -> None:
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "dark.png"
            create_solid_image(image_path, 20)

            value = mean_intensity(image_path)
            print(f"dark mean intensity: {value:.2f}")

            self.assertLess(value, 50.0)
            self.assertTrue(detect_low_light(str(image_path)))


if __name__ == "__main__":
    main()
