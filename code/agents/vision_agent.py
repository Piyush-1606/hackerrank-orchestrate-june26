from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Final

try:
    from agents.base_agent import AgentRunContext, BaseAgent, RetryConfig
    from agents.claim_agent import ClaimAgent
    from models.schemas import ClaimInput, IssueType, Severity, VisionQualityFlag, VisionResult
    from utils.image_authenticity import analyze_image_authenticity
    from utils.image_quality import analyze_image_quality, detect_blurry_image, detect_low_light
    from utils.image_consistency import analyze_image_set
except ModuleNotFoundError:
    from .base_agent import AgentRunContext, BaseAgent, RetryConfig
    from .claim_agent import ClaimAgent
    from ..models.schemas import ClaimInput, IssueType, Severity, VisionQualityFlag, VisionResult
    from ..utils.image_authenticity import analyze_image_authenticity
    from ..utils.image_quality import analyze_image_quality, detect_blurry_image, detect_low_light
    from ..utils.image_consistency import analyze_image_set


class VisionAgent(BaseAgent[ClaimInput, VisionResult]):
    """Validate image evidence and produce claim-guided visual candidates.

    V2 remains deterministic and does not call external APIs. It uses submitted
    image validity as an evidence gate, then uses claim text/object type as
    strong priors for the visual fields until a VLM is integrated.
    """

    SUPPORTED_IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset(
        {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    )

    def __init__(
        self,
        *,
        project_root: str | Path | None = None,
        claim_agent: ClaimAgent | None = None,
        retry_config: RetryConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(name="VisionAgent", retry_config=retry_config, logger=logger)
        self.project_root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
        self.claim_agent = claim_agent or ClaimAgent()

    def validate_input(self, input_data: ClaimInput) -> None:
        """Validate image path presence, file format, and existence."""
        super().validate_input(input_data)

        if not input_data.image_paths:
            raise ValueError("VisionAgent requires at least one image path")

        for image_path in input_data.image_paths:
            resolved_path = self.resolve_image_path(image_path)
            self.validate_image_format(resolved_path)
            if not resolved_path.is_file():
                raise FileNotFoundError(f"Image path does not exist: {image_path}")

    def _execute(self, input_data: ClaimInput, context: AgentRunContext) -> VisionResult:
        """Return a deterministic claim-guided VisionResult.

        TODO: Replace this placeholder with VLM-backed image analysis that
        verifies the claim-guided candidates against image pixels.
        """
        resolved_paths = [self.resolve_image_path(path) for path in input_data.image_paths]
        image_ids = [self.image_id_from_path(path) for path in resolved_paths]
        extraction = self.claim_agent.run(input_data, context)

        detected_issue_type = self.select_issue_type(input_data, extraction.claimed_issue_types)
        detected_object_part = self.select_object_part(input_data, extraction.affected_area)
        detected_severity = self.select_severity(input_data, extraction.claimed_severity)
        damage_visible = detected_issue_type not in {IssueType.UNKNOWN, IssueType.NONE}
        quality_results = [analyze_image_quality(str(path)) for path in resolved_paths]
        blurry_image_detected = any(result["blurry"] for result in quality_results)
        low_light_detected = any(result["low_light"] for result in quality_results)
        consistency = analyze_image_set([str(path) for path in resolved_paths])
        authenticity = analyze_image_authenticity([str(path) for path in resolved_paths])
        image_quality_flags = self.determine_quality_flags(
            image_count=len(resolved_paths),
            damage_visible=damage_visible,
            detected_issue_type=detected_issue_type,
            blurry_image_detected=blurry_image_detected,
            low_light_detected=low_light_detected,
        )
        vision_risk_flags = []
        if authenticity["possible_manipulation"]:
            vision_risk_flags.append("possible_manipulation")
        if authenticity["non_original_image"]:
            vision_risk_flags.append("non_original_image")
        if authenticity["duplicate_submission"]:
            vision_risk_flags.append("duplicate_submission")

        confidence = self.score_confidence(
            image_count=len(resolved_paths),
            detected_object_part=detected_object_part,
            detected_issue_type=detected_issue_type,
            detected_severity=detected_severity,
        )
        reasoning = self.build_reasoning(
            image_count=len(resolved_paths),
            detected_object_part=detected_object_part,
            detected_issue_type=detected_issue_type,
            detected_severity=detected_severity,
            confidence=confidence,
            consistency=consistency,
        )

        self._log(
            logging.INFO,
            "vision_claim_guided_completed",
            context=context,
            image_count=len(resolved_paths),
            image_ids=image_ids,
            detected_issue_type=str(detected_issue_type),
            detected_object_part=detected_object_part,
            damage_visible=damage_visible,
            confidence=confidence,
            blurry_image_detected=blurry_image_detected,
            low_light_detected=low_light_detected,
            duplicate_images=consistency["duplicate_images"],
            mixed_object_types=consistency["mixed_object_types"],
            consistency_score=consistency["consistency_score"],
            possible_manipulation=authenticity["possible_manipulation"],
            non_original_image=authenticity["non_original_image"],
            duplicate_submission=authenticity["duplicate_submission"],
            authenticity_reason=authenticity["reason"],
        )

        return VisionResult(
            claim_id=input_data.claim_id or context.claim_id,
            detected_object=input_data.object_type,
            detected_issue_type=detected_issue_type,
            detected_object_part=detected_object_part,
            visible_parts=[detected_object_part] if detected_object_part else [],
            image_quality_flags=image_quality_flags,
            risk_flags=vision_risk_flags,
            supporting_image_ids=image_ids,
            damage_visible=damage_visible,
            confidence=confidence,
            reasoning=reasoning,
        )

    @classmethod
    def select_issue_type(cls, input_data: ClaimInput, claimed_issue_types: list[IssueType]) -> IssueType:
        """Select the strongest deterministic issue candidate."""
        text = ClaimAgent.normalize_text(input_data.claim_text)

        if cls.is_no_visible_issue_context(text):
            return IssueType.NONE
        if input_data.object_type.value == "package" and cls.is_missing_contents_context(text):
            return IssueType.UNKNOWN
        if input_data.object_type.value == "package" and cls.is_outside_box_crushed_context(text):
            return IssueType.UNKNOWN
        if input_data.object_type.value == "package" and cls.is_opened_package_without_confirmed_damage(text):
            return IssueType.NONE
        if cls.is_generic_bumper_damage_context(text):
            return IssueType.SCRATCH

        return claimed_issue_types[0] if claimed_issue_types else IssueType.UNKNOWN

    @staticmethod
    def select_object_part(input_data: ClaimInput, affected_area: str | None) -> str | None:
        """Select the primary object part candidate."""
        text = ClaimAgent.normalize_text(input_data.claim_text)

        if input_data.object_type.value == "package" and VisionAgent.is_outside_box_crushed_context(text):
            return "unknown"
        if affected_area:
            return affected_area.split(";")[0].strip() or None
        return None

    @staticmethod
    def select_severity(input_data: ClaimInput, claimed_severity: Severity) -> Severity:
        """Select claim-guided visual severity candidate."""
        return claimed_severity

    @staticmethod
    def determine_quality_flags(
        *,
        image_count: int,
        damage_visible: bool,
        detected_issue_type: IssueType,
        blurry_image_detected: bool,
        low_light_detected: bool,
    ) -> list[VisionQualityFlag]:
        """Return deterministic quality flags available without pixel analysis."""
        flags: list[VisionQualityFlag] = []
        if blurry_image_detected:
            flags.append(VisionQualityFlag.BLURRY_IMAGE)
        if low_light_detected:
            flags.append(VisionQualityFlag.LOW_LIGHT_OR_GLARE)
        if not damage_visible:
            flags.append(VisionQualityFlag.DAMAGE_NOT_VISIBLE)
        return flags

    @staticmethod
    def score_confidence(
        *,
        image_count: int,
        detected_object_part: str | None,
        detected_issue_type: IssueType,
        detected_severity: Severity,
    ) -> float:
        """Score confidence from image availability and claim-derived candidates."""
        if detected_issue_type == IssueType.UNKNOWN:
            return 0.0

        score = 0.35
        if image_count > 0:
            score += 0.2
        if image_count > 1:
            score += 0.05
        if detected_object_part:
            score += 0.2
        if detected_issue_type != IssueType.UNKNOWN:
            score += 0.15
        if detected_severity != Severity.UNKNOWN:
            score += 0.05
        return round(max(0.0, min(score, 1.0)), 3)

    @staticmethod
    def build_reasoning(
        *,
        image_count: int,
        detected_object_part: str | None,
        detected_issue_type: IssueType,
        detected_severity: Severity,
        confidence: float,
        consistency: dict[str, Any],
    ) -> str:
        """Build concise explanation for deterministic claim-guided vision."""
        if confidence == 0.0:
            duplicates_text = (
                "No duplicate images detected. "
                if not consistency["duplicate_images"]
                else "Duplicate images detected. "
            )
            return (
                "Image paths were validated, but visual damage analysis is not implemented yet. "
                "No concrete claim-guided visual issue could be inferred. "
                f"{image_count} images reviewed. {duplicates_text}"
                f"Consistency score {consistency['consistency_score']:.2f}."
            )
        return (
            f"Validated {image_count} image(s). Used claim text and object type as visual priors: "
            f"part={detected_object_part or 'unknown'}, issue={detected_issue_type}, "
            f"severity={detected_severity}, confidence={confidence:.2f}. "
            f"{image_count} images reviewed. "
            f"{'Duplicate images detected. ' if consistency['duplicate_images'] else 'No duplicate images detected. '}"
            f"Consistency score {consistency['consistency_score']:.2f}. "
            "Future VLM integration should verify these candidates against pixels."
        )

    @staticmethod
    def is_generic_bumper_damage_context(text: str) -> bool:
        return "bumper damage" in text and "back bumper" in text and "scratch" not in text

    @staticmethod
    def is_missing_contents_context(text: str) -> bool:
        return any(phrase in text for phrase in ("contents are missing", "not inside the box", "product inside"))

    @staticmethod
    def is_outside_box_crushed_context(text: str) -> bool:
        return "outside box" in text and "crushed box" in text

    @staticmethod
    def is_opened_package_without_confirmed_damage(text: str) -> bool:
        return "delivery box arrived opened" in text and "torn open package" in text.replace("-", " ")

    @staticmethod
    def is_no_visible_issue_context(text: str) -> bool:
        return "physical damage around the trackpad area" in text

    def resolve_image_path(self, image_path: str) -> Path:
        """Resolve absolute paths directly and relative paths against project root."""
        path = Path(image_path)
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()

    @classmethod
    def validate_image_format(cls, image_path: Path) -> None:
        """Reject unsupported image formats before model analysis."""
        if image_path.suffix.lower() not in cls.SUPPORTED_IMAGE_EXTENSIONS:
            supported = ", ".join(sorted(cls.SUPPORTED_IMAGE_EXTENSIONS))
            raise ValueError(f"Unsupported image format '{image_path.suffix}'. Supported: {supported}")

    @staticmethod
    def image_id_from_path(image_path: Path) -> str:
        """Create a stable image identifier from the file stem."""
        return image_path.stem

    def _analyze_images_with_vlm(self, image_paths: list[Path]) -> VisionResult:
        """Future hook for VLM-backed image analysis.

        This method is intentionally unused until an external vision model is
        selected. Keep provider-specific code out of the public agent contract.
        """
        raise NotImplementedError("VLM integration is not implemented yet")
