from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main

import cv2
import numpy as np

from code.utils.image_quality import analyze_image_quality


def create_test_image(path: Path) -> None:
    image = np.zeros((120, 240), dtype=np.uint8)
    image[:, :] = 180
    cv2.rectangle(image, (20, 20), (220, 100), 20, 3)
    cv2.imwrite(str(path), image)


class ImageQualityTest(TestCase):
    def test_analyze_image_quality_returns_expected_metrics(self) -> None:
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "quality.png"
            create_test_image(image_path)

            result = analyze_image_quality(str(image_path))

        self.assertEqual(
            set(result),
            {
                "blurry",
                "low_light",
                "cropped_or_obstructed",
                "width",
                "height",
                "aspect_ratio",
                "blur_variance",
                "mean_intensity",
            },
        )
        self.assertEqual(result["width"], 240)
        self.assertEqual(result["height"], 120)
        self.assertEqual(result["aspect_ratio"], 2.0)
        self.assertIsInstance(result["blurry"], bool)
        self.assertIsInstance(result["low_light"], bool)
        self.assertGreater(result["blur_variance"], 0.0)
        self.assertGreater(result["mean_intensity"], 50.0)
        self.assertTrue(result["cropped_or_obstructed"])

    def test_analyze_image_quality_detects_cropped_or_obstructed_small_image(self) -> None:
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "small.png"
            image = np.full((250, 250), 200, dtype=np.uint8)
            cv2.imwrite(str(image_path), image)

            result = analyze_image_quality(str(image_path))

        self.assertTrue(result["cropped_or_obstructed"])
        self.assertEqual(result["width"], 250)
        self.assertEqual(result["height"], 250)

    def test_analyze_image_quality_accepts_normal_image(self) -> None:
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "normal.png"
            image = np.full((400, 500), 200, dtype=np.uint8)
            cv2.imwrite(str(image_path), image)

            result = analyze_image_quality(str(image_path))

        self.assertFalse(result["cropped_or_obstructed"])
        self.assertEqual(result["width"], 500)
        self.assertEqual(result["height"], 400)

    def test_analyze_unreadable_image_returns_neutral_metrics(self) -> None:
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "placeholder.jpg"
            image_path.write_bytes(b"not a real image")

            result = analyze_image_quality(str(image_path))

        self.assertFalse(result["blurry"])
        self.assertFalse(result["low_light"])
        self.assertEqual(result["width"], 0)
        self.assertEqual(result["height"], 0)
        self.assertEqual(result["aspect_ratio"], 0.0)


if __name__ == "__main__":
    main()
