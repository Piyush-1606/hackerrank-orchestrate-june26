from __future__ import annotations

import logging
from collections.abc import Sequence

try:
    from agents.base_agent import AgentRunContext, BaseAgent, RetryConfig
    from models.schemas import (
        AlignmentStatus,
        ClaimExtractionResult,
        ClaimStatus,
        EvidenceValidationResult,
        IssueType,
        Reviewability,
        VisionQualityFlag,
        VisionResult,
    )
except ModuleNotFoundError:
    from .base_agent import AgentRunContext, BaseAgent, RetryConfig
    from ..models.schemas import (
        AlignmentStatus,
        ClaimExtractionResult,
        ClaimStatus,
        EvidenceValidationResult,
        IssueType,
        Reviewability,
        VisionQualityFlag,
        VisionResult,
    )


EvidenceInput = tuple[ClaimExtractionResult, VisionResult]


class EvidenceAgent(BaseAgent[EvidenceInput, EvidenceValidationResult]):
    """Validate whether image evidence supports the extracted claim.

    Inputs are passed as `(claim_extraction, vision_result)` to avoid adding a
    new public schema. The agent is evidence-only: it does not use risk history
    and does not make final claim decisions.
    """

    REVIEWABILITY_REDUCING_FLAGS = frozenset(
        {
            VisionQualityFlag.BLURRY_IMAGE,
            VisionQualityFlag.CROPPED_OR_OBSTRUCTED,
            VisionQualityFlag.LOW_LIGHT_OR_GLARE,
        }
    )

    def __init__(
        self,
        *,
        retry_config: RetryConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(name="EvidenceAgent", retry_config=retry_config, logger=logger)

    def validate_input(self, input_data: EvidenceInput) -> None:
        """Validate the tuple contract and required evidence inputs."""
        super().validate_input(input_data)

        if not isinstance(input_data, tuple) or len(input_data) != 2:
            raise ValueError("EvidenceAgent input must be (ClaimExtractionResult, VisionResult)")

        claim, vision = input_data
        if not isinstance(claim, ClaimExtractionResult):
            raise TypeError("EvidenceAgent first input must be ClaimExtractionResult")
        if not isinstance(vision, VisionResult):
            raise TypeError("EvidenceAgent second input must be VisionResult")

    def _execute(self, input_data: EvidenceInput, context: AgentRunContext) -> EvidenceValidationResult:
        claim, vision = input_data

        object_alignment = self.determine_object_alignment(claim, vision)
        issue_alignment = self.determine_issue_alignment(claim, vision)
        area_alignment = self.determine_area_alignment(claim, vision)
        reviewability = self.determine_reviewability(vision)
        missing_evidence = self.determine_missing_evidence(
            object_alignment=object_alignment,
            issue_alignment=issue_alignment,
            area_alignment=area_alignment,
            reviewability=reviewability,
            vision=vision,
        )
        evidence_standard_met = self.determine_evidence_standard_met(
            reviewability=reviewability,
            object_alignment=object_alignment,
            area_alignment=area_alignment,
        )
        recommended_status = self.determine_recommended_status(
            reviewability=reviewability,
            object_alignment=object_alignment,
            issue_alignment=issue_alignment,
            area_alignment=area_alignment,
            evidence_standard_met=evidence_standard_met,
        )
        confidence = self.determine_confidence(
            vision=vision,
            reviewability=reviewability,
            object_alignment=object_alignment,
            issue_alignment=issue_alignment,
            area_alignment=area_alignment,
        )
        reason = self.build_evidence_reason(
            evidence_standard_met=evidence_standard_met,
            reviewability=reviewability,
            object_alignment=object_alignment,
            issue_alignment=issue_alignment,
            area_alignment=area_alignment,
            missing_evidence=missing_evidence,
        )

        self._log(
            logging.INFO,
            "evidence_validation_completed",
            context=context,
            object_alignment=str(object_alignment),
            issue_alignment=str(issue_alignment),
            area_alignment=str(area_alignment),
            reviewability=str(reviewability),
            evidence_standard_met=evidence_standard_met,
            recommended_status=str(recommended_status),
            confidence=confidence,
        )

        return EvidenceValidationResult(
            claim_id=claim.claim_id or vision.claim_id or context.claim_id,
            evidence_standard_met=evidence_standard_met,
            evidence_standard_met_reason=reason,
            reviewability=reviewability,
            object_alignment=object_alignment,
            issue_alignment=issue_alignment,
            area_alignment=area_alignment,
            severity_alignment=AlignmentStatus.UNKNOWN,
            missing_evidence=missing_evidence,
            supporting_image_ids=list(vision.supporting_image_ids),
            recommended_status=recommended_status,
            confidence=confidence,
        )

    @staticmethod
    def determine_object_alignment(
        claim: ClaimExtractionResult,
        vision: VisionResult,
    ) -> AlignmentStatus:
        """Compare extracted claimed object with detected image object."""
        if vision.detected_object is None:
            return AlignmentStatus.UNKNOWN
        if vision.detected_object == claim.object_type:
            return AlignmentStatus.SUPPORTS
        return AlignmentStatus.CONTRADICTS

    @classmethod
    def determine_issue_alignment(
        cls,
        claim: ClaimExtractionResult,
        vision: VisionResult,
    ) -> AlignmentStatus:
        """Compare claimed issue type with detected visual issue type."""
        claimed_issues = cls.concrete_claimed_issues(claim.claimed_issue_types)

        if vision.detected_issue_type in claimed_issues:
            return AlignmentStatus.SUPPORTS

        if claimed_issues and vision.detected_issue_type == IssueType.NONE:
            return AlignmentStatus.CONTRADICTS

        if claimed_issues and vision.damage_visible is False and vision.detected_issue_type == IssueType.UNKNOWN:
            return AlignmentStatus.UNKNOWN

        if vision.detected_issue_type not in {IssueType.UNKNOWN, IssueType.NONE} and claimed_issues:
            return AlignmentStatus.UNKNOWN

        return AlignmentStatus.UNKNOWN

    @staticmethod
    def determine_area_alignment(
        claim: ClaimExtractionResult,
        vision: VisionResult,
    ) -> AlignmentStatus:
        """Compare claimed affected area with detected image part."""
        claimed_area = EvidenceAgent.normalize_area(claim.affected_area)
        detected_area = EvidenceAgent.normalize_area(vision.detected_object_part)

        if not claimed_area or not detected_area:
            return AlignmentStatus.UNKNOWN
        if claimed_area == detected_area:
            return AlignmentStatus.SUPPORTS
        return AlignmentStatus.CONTRADICTS

    @classmethod
    def determine_reviewability(cls, vision: VisionResult) -> Reviewability:
        """Classify image reviewability from support images and quality flags."""
        if not vision.supporting_image_ids:
            return Reviewability.NOT_REVIEWABLE
        if VisionQualityFlag.WRONG_OBJECT in vision.image_quality_flags:
            return Reviewability.NOT_REVIEWABLE
        if VisionQualityFlag.DAMAGE_NOT_VISIBLE in vision.image_quality_flags and not vision.damage_visible:
            return Reviewability.PARTIALLY_REVIEWABLE
        if cls.REVIEWABILITY_REDUCING_FLAGS.intersection(vision.image_quality_flags):
            return Reviewability.PARTIALLY_REVIEWABLE
        return Reviewability.REVIEWABLE

    @staticmethod
    def determine_evidence_standard_met(
        *,
        reviewability: Reviewability,
        object_alignment: AlignmentStatus,
        area_alignment: AlignmentStatus,
    ) -> bool:
        """Evidence standard requires reviewable images and object/area support."""
        return (
            reviewability == Reviewability.REVIEWABLE
            and object_alignment == AlignmentStatus.SUPPORTS
            and area_alignment == AlignmentStatus.SUPPORTS
        )

    @staticmethod
    def determine_recommended_status(
        *,
        reviewability: Reviewability,
        object_alignment: AlignmentStatus,
        issue_alignment: AlignmentStatus,
        area_alignment: AlignmentStatus,
        evidence_standard_met: bool,
    ) -> ClaimStatus:
        """Recommend evidence-only status from alignments."""
        if object_alignment == AlignmentStatus.CONTRADICTS:
            return ClaimStatus.CONTRADICTED
        if area_alignment == AlignmentStatus.CONTRADICTS:
            return ClaimStatus.CONTRADICTED
        if issue_alignment == AlignmentStatus.CONTRADICTS:
            return ClaimStatus.CONTRADICTED
        if evidence_standard_met and issue_alignment == AlignmentStatus.SUPPORTS:
            return ClaimStatus.SUPPORTED
        if reviewability == Reviewability.NOT_REVIEWABLE:
            return ClaimStatus.NOT_ENOUGH_INFORMATION
        return ClaimStatus.NOT_ENOUGH_INFORMATION

    @staticmethod
    def determine_missing_evidence(
        *,
        object_alignment: AlignmentStatus,
        issue_alignment: AlignmentStatus,
        area_alignment: AlignmentStatus,
        reviewability: Reviewability,
        vision: VisionResult,
    ) -> list[str]:
        """Explain missing evidence needed for a supported determination."""
        missing: list[str] = []

        if not vision.supporting_image_ids:
            missing.append("No supporting image evidence was available.")
        if reviewability != Reviewability.REVIEWABLE:
            missing.append("Image quality or visibility limits reviewability.")
        if object_alignment != AlignmentStatus.SUPPORTS:
            missing.append("Claimed object is not clearly supported by image evidence.")
        if area_alignment != AlignmentStatus.SUPPORTS:
            missing.append("Claimed affected area is not clearly supported by image evidence.")
        if issue_alignment == AlignmentStatus.UNKNOWN:
            missing.append("Claimed issue type is not clearly visible in image evidence.")

        return missing

    @staticmethod
    def determine_confidence(
        *,
        vision: VisionResult,
        reviewability: Reviewability,
        object_alignment: AlignmentStatus,
        issue_alignment: AlignmentStatus,
        area_alignment: AlignmentStatus,
    ) -> float:
        """Calibrate validation confidence from visual confidence and alignments."""
        score = vision.confidence

        if object_alignment == AlignmentStatus.SUPPORTS:
            score += 0.08
        if issue_alignment == AlignmentStatus.SUPPORTS:
            score += 0.08
        if area_alignment == AlignmentStatus.SUPPORTS:
            score += 0.08
        if reviewability == Reviewability.PARTIALLY_REVIEWABLE:
            score -= 0.15
        if reviewability == Reviewability.NOT_REVIEWABLE:
            score -= 0.3

        return round(max(0.0, min(1.0, score)), 3)

    @staticmethod
    def build_evidence_reason(
        *,
        evidence_standard_met: bool,
        reviewability: Reviewability,
        object_alignment: AlignmentStatus,
        issue_alignment: AlignmentStatus,
        area_alignment: AlignmentStatus,
        missing_evidence: Sequence[str],
    ) -> str:
        """Generate concise evidence standard rationale."""
        if evidence_standard_met:
            return "Evidence is reviewable and supports the claimed object and affected area."

        if object_alignment == AlignmentStatus.CONTRADICTS:
            return "Evidence contradicts the claimed object."
        if area_alignment == AlignmentStatus.CONTRADICTS:
            return "Evidence contradicts the claimed affected area."
        if issue_alignment == AlignmentStatus.CONTRADICTS:
            return "Evidence contradicts the claimed issue type."
        if reviewability != Reviewability.REVIEWABLE:
            return "Evidence is not fully reviewable due to image quality or visibility limits."
        if missing_evidence:
            return missing_evidence[0]
        return "Evidence does not meet the required visual support standard."

    @staticmethod
    def concrete_claimed_issues(issue_types: Sequence[IssueType]) -> set[IssueType]:
        """Drop unknown from claimed issue types for alignment comparison."""
        return {issue for issue in issue_types if issue != IssueType.UNKNOWN}

    @staticmethod
    def normalize_area(area: str | None) -> str | None:
        """Normalize semicolon-delimited or whitespace area labels."""
        if not area:
            return None
        first_area = area.split(";")[0].strip().casefold()
        return first_area.replace(" ", "_") or None
