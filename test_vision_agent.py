from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main

import cv2
import numpy as np

from code.agents.vision_agent import VisionAgent
from code.models.schemas import ClaimInput, IssueType, ObjectType, VisionQualityFlag


def _claim(image_paths: list[str]) -> ClaimInput:
    return ClaimInput(
        claim_id="claim_001",
        user_id="user_001",
        image_paths=image_paths,
        claim_text="Customer: The rear bumper is damaged.",
        object_type=ObjectType.CAR,
        user_history=None,
        evidence_requirements=[],
        metadata={},
    )


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        height = 320
        width = 320
        image = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                image[y, x] = [(x + y) % 256, (x * 2 + y * 3) % 256, (x * 3 + y * 2) % 256]
        cv2.imwrite(str(path), image)
    else:
        path.write_bytes(b"placeholder")
    return path


class VisionAgentTest(TestCase):
    def test_valid_image_path(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = _touch(root / "img_1.jpg")
            result = VisionAgent(project_root=root).run(_claim([str(image_path)]))

        self.assertEqual(result.detected_object, ObjectType.CAR)
        self.assertEqual(result.detected_issue_type, IssueType.UNKNOWN)
        self.assertFalse(result.damage_visible)
        self.assertEqual(result.supporting_image_ids, ["img_1"])
        self.assertIn(VisionQualityFlag.DAMAGE_NOT_VISIBLE, result.image_quality_flags)

    def test_missing_image_path(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            missing_path = root / "missing.jpg"

            with self.assertRaises(FileNotFoundError):
                VisionAgent(project_root=root).run(_claim([str(missing_path)]))

    def test_multiple_image_paths(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_image = _touch(root / "case" / "img_1.png")
            second_image = _touch(root / "case" / "img_2.webp")

            result = VisionAgent(project_root=root).run(
                _claim([str(first_image), str(second_image)])
            )

        self.assertEqual(result.supporting_image_ids, ["img_1", "img_2"])
        self.assertEqual(result.confidence, 0.0)
        self.assertIn("not implemented yet", result.reasoning)

    def test_unsupported_image_format(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            text_file = _touch(root / "img_1.txt")

            with self.assertRaisesRegex(ValueError, "Unsupported image format"):
                VisionAgent(project_root=root).run(_claim([str(text_file)]))

    def test_authenticity_risk_flags_from_screenshot_filename(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            screenshot_image = _touch(root / "screenshot_001.png")

            result = VisionAgent(project_root=root).run(_claim([str(screenshot_image)]))

        self.assertIn("non_original_image", result.risk_flags)
        self.assertIn("possible_manipulation", result.risk_flags)


if __name__ == "__main__":
    main()
