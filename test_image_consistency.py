from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main

import cv2
import numpy as np

from code.utils.image_consistency import analyze_image_set


class ImageConsistencyTest(TestCase):
    def test_duplicate_image_set(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path_1 = root / "img_a.png"
            image_path_2 = root / "img_b.png"
            image = np.full((320, 320, 3), 180, dtype=np.uint8)
            cv2.imwrite(str(image_path_1), image)
            cv2.imwrite(str(image_path_2), image)

            result = analyze_image_set([str(image_path_1), str(image_path_2)])

        self.assertEqual(result["image_count"], 2)
        self.assertTrue(result["duplicate_images"])
        self.assertFalse(result["mixed_object_types"])
        self.assertEqual(result["supporting_image_ids"], ["img_a", "img_b"])
        self.assertLess(result["consistency_score"], 1.0)

    def test_unique_image_set(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path_1 = root / "img_a.png"
            image_path_2 = root / "img_b.png"
            image_a = np.full((320, 320, 3), 180, dtype=np.uint8)
            image_b = np.full((320, 320, 3), 120, dtype=np.uint8)
            cv2.imwrite(str(image_path_1), image_a)
            cv2.imwrite(str(image_path_2), image_b)

            result = analyze_image_set([str(image_path_1), str(image_path_2)])

        self.assertEqual(result["image_count"], 2)
        self.assertFalse(result["duplicate_images"])
        self.assertFalse(result["mixed_object_types"])
        self.assertEqual(result["supporting_image_ids"], ["img_a", "img_b"])
        self.assertGreaterEqual(result["consistency_score"], 0.0)
        self.assertLessEqual(result["consistency_score"], 1.0)


if __name__ == "__main__":
    main()
