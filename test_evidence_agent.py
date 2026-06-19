from __future__ import annotations

from unittest import TestCase, main

from code.agents.evidence_agent import EvidenceAgent
from code.models.schemas import (
    AlignmentStatus,
    ClaimExtractionResult,
    ClaimStatus,
    EvidenceValidationResult,
    IssueType,
    ObjectType,
    Reviewability,
    Severity,
    VisionQualityFlag,
    VisionResult,
)


def _claim(
    *,
    object_type: ObjectType = ObjectType.CAR,
    issue_type: IssueType = IssueType.DENT,
    affected_area: str | None = "rear_bumper",
) -> ClaimExtractionResult:
    return ClaimExtractionResult(
        claim_id="claim_001",
        user_id="user_001",
        object_type=object_type,
        claimed_issue_types=[issue_type],
        claimed_severity=Severity.MEDIUM,
        affected_area=affected_area,
        incident_summary="Claimed damage.",
        evidence_requirements=[],
        prompt_injection_detected=False,
        confidence=0.9,
    )


def _vision(
    *,
    detected_object: ObjectType | None = ObjectType.CAR,
    detected_issue_type: IssueType = IssueType.DENT,
    detected_object_part: str | None = "rear_bumper",
    supporting_image_ids: list[str] | None = None,
    quality_flags: list[VisionQualityFlag] | None = None,
    damage_visible: bool = True,
    confidence: float = 0.8,
) -> VisionResult:
    return VisionResult(
        claim_id="claim_001",
        detected_object=detected_object,
        detected_issue_type=detected_issue_type,
        detected_object_part=detected_object_part,
        visible_parts=[detected_object_part] if detected_object_part else [],
        image_quality_flags=quality_flags or [],
        supporting_image_ids=["img_1"] if supporting_image_ids is None else supporting_image_ids,
        damage_visible=damage_visible,
        confidence=confidence,
        reasoning="Synthetic vision result for evidence tests.",
    )


class EvidenceAgentTest(TestCase):
    def test_supported_evidence(self) -> None:
        result = EvidenceAgent().run((_claim(), _vision()))

        self.assertIsInstance(result, EvidenceValidationResult)
        self.assertTrue(result.evidence_standard_met)
        self.assertEqual(result.object_alignment, AlignmentStatus.SUPPORTS)
        self.assertEqual(result.issue_alignment, AlignmentStatus.SUPPORTS)
        self.assertEqual(result.area_alignment, AlignmentStatus.SUPPORTS)
        self.assertEqual(result.reviewability, Reviewability.REVIEWABLE)
        self.assertEqual(result.recommended_status, ClaimStatus.SUPPORTED)

    def test_contradicted_object(self) -> None:
        result = EvidenceAgent().run(
            (
                _claim(object_type=ObjectType.CAR),
                _vision(detected_object=ObjectType.LAPTOP),
            )
        )

        self.assertFalse(result.evidence_standard_met)
        self.assertEqual(result.object_alignment, AlignmentStatus.CONTRADICTS)
        self.assertEqual(result.recommended_status, ClaimStatus.CONTRADICTED)
        self.assertIn("object", result.evidence_standard_met_reason)

    def test_contradicted_area(self) -> None:
        result = EvidenceAgent().run(
            (
                _claim(affected_area="rear_bumper"),
                _vision(detected_object_part="front_bumper"),
            )
        )

        self.assertFalse(result.evidence_standard_met)
        self.assertEqual(result.area_alignment, AlignmentStatus.CONTRADICTS)
        self.assertEqual(result.recommended_status, ClaimStatus.CONTRADICTED)

    def test_blurry_image(self) -> None:
        result = EvidenceAgent().run(
            (
                _claim(),
                _vision(quality_flags=[VisionQualityFlag.BLURRY_IMAGE]),
            )
        )

        self.assertFalse(result.evidence_standard_met)
        self.assertEqual(result.reviewability, Reviewability.PARTIALLY_REVIEWABLE)
        self.assertEqual(result.recommended_status, ClaimStatus.NOT_ENOUGH_INFORMATION)

    def test_insufficient_evidence(self) -> None:
        result = EvidenceAgent().run(
            (
                _claim(),
                _vision(
                    detected_issue_type=IssueType.UNKNOWN,
                    detected_object_part=None,
                    supporting_image_ids=[],
                    damage_visible=False,
                    confidence=0.0,
                ),
            )
        )

        self.assertFalse(result.evidence_standard_met)
        self.assertEqual(result.reviewability, Reviewability.NOT_REVIEWABLE)
        self.assertEqual(result.area_alignment, AlignmentStatus.UNKNOWN)
        self.assertEqual(result.recommended_status, ClaimStatus.NOT_ENOUGH_INFORMATION)
        self.assertTrue(result.missing_evidence)

    def test_multiple_supporting_images(self) -> None:
        result = EvidenceAgent().run(
            (
                _claim(),
                _vision(supporting_image_ids=["img_1", "img_2", "img_3"]),
            )
        )

        self.assertTrue(result.evidence_standard_met)
        self.assertEqual(result.supporting_image_ids, ["img_1", "img_2", "img_3"])
        self.assertEqual(result.recommended_status, ClaimStatus.SUPPORTED)

    def test_reviewable_object_and_area_but_unknown_issue_is_not_enough_information(self) -> None:
        result = EvidenceAgent().run(
            (
                _claim(issue_type=IssueType.SCRATCH, affected_area="rear_bumper"),
                _vision(
                    detected_issue_type=IssueType.UNKNOWN,
                    detected_object_part="rear_bumper",
                    damage_visible=False,
                    confidence=0.8,
                ),
            )
        )

        self.assertFalse(result.evidence_standard_met)
        self.assertEqual(result.object_alignment, AlignmentStatus.SUPPORTS)
        self.assertEqual(result.area_alignment, AlignmentStatus.SUPPORTS)
        self.assertEqual(result.issue_alignment, AlignmentStatus.UNKNOWN)
        self.assertEqual(result.recommended_status, ClaimStatus.NOT_ENOUGH_INFORMATION)


if __name__ == "__main__":
    main()
