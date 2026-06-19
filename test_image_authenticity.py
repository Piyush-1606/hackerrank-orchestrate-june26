from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main

import cv2
import numpy as np

from code.utils.image_authenticity import analyze_image_authenticity


def _create_image(path: Path, color: tuple[int, int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = np.full((32, 32, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), canvas)
    return path


class ImageAuthenticityTest(TestCase):
    def test_identical_images_trigger_duplicate_submission(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_image = _create_image(root / "img_1.jpg", (100, 100, 100))
            second_image = _create_image(root / "img_2.jpg", (100, 100, 100))

            result = analyze_image_authenticity([str(first_image), str(second_image)])

        self.assertTrue(result["duplicate_images"])
        self.assertTrue(result["duplicate_submission"])
        self.assertTrue(result["possible_manipulation"])
        self.assertIn("Duplicate images detected", result["reason"])

    def test_screenshot_filename_flags_non_original_image(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            screenshot_image = _create_image(root / "screenshot_001.png", (120, 120, 120))

            result = analyze_image_authenticity([str(screenshot_image)])

        self.assertFalse(result["duplicate_images"])
        self.assertFalse(result["duplicate_submission"])
        self.assertTrue(result["possible_manipulation"])
        self.assertTrue(result["non_original_image"])
        self.assertIn("screenshot or screen capture", result["reason"].lower())

    def test_distinct_images_do_not_raise_authenticity_flags(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_image = _create_image(root / "img_1.jpg", (10, 20, 30))
            second_image = _create_image(root / "img_2.jpg", (200, 220, 230))

            result = analyze_image_authenticity([str(first_image), str(second_image)])

        self.assertFalse(result["duplicate_images"])
        self.assertFalse(result["duplicate_submission"])
        self.assertFalse(result["non_original_image"])
        self.assertFalse(result["possible_manipulation"])
        self.assertEqual(result["reason"], "No obvious authenticity issues detected.")


if __name__ == "__main__":
    main()
