from __future__ import annotations

from unittest import TestCase, main

from code.agents.decision_agent import DecisionAgent
from code.models.schemas import (
    AlignmentStatus,
    ClaimExtractionResult,
    ClaimStatus,
    EvidenceValidationResult,
    IssueType,
    ObjectType,
    Reviewability,
    RiskAssessmentResult,
    Severity,
    VisionResult,
)


def _claim() -> ClaimExtractionResult:
    return ClaimExtractionResult(
        claim_id="claim_001",
        user_id="user_001",
        object_type=ObjectType.CAR,
        claimed_issue_types=[IssueType.DENT],
        claimed_severity=Severity.MEDIUM,
        affected_area="rear_bumper",
        incident_summary="Rear bumper dent.",
        evidence_requirements=[],
        prompt_injection_detected=False,
        confidence=0.9,
    )


def _vision(
    *,
    issue_type: IssueType = IssueType.DENT,
    object_part: str | None = "rear_bumper",
    supporting_image_ids: list[str] | None = None,
    damage_visible: bool = True,
) -> VisionResult:
    return VisionResult(
        claim_id="claim_001",
        detected_object=ObjectType.CAR,
        detected_issue_type=issue_type,
        detected_object_part=object_part,
        visible_parts=[object_part] if object_part else [],
        image_quality_flags=[],
        supporting_image_ids=["img_1"] if supporting_image_ids is None else supporting_image_ids,
        damage_visible=damage_visible,
        confidence=0.85,
        reasoning="Synthetic vision output.",
    )


def _evidence(
    *,
    evidence_standard_met: bool = True,
    reviewability: Reviewability = Reviewability.REVIEWABLE,
    object_alignment: AlignmentStatus = AlignmentStatus.SUPPORTS,
    issue_alignment: AlignmentStatus = AlignmentStatus.SUPPORTS,
    area_alignment: AlignmentStatus = AlignmentStatus.SUPPORTS,
    supporting_image_ids: list[str] | None = None,
    confidence: float = 0.9,
) -> EvidenceValidationResult:
    status = ClaimStatus.SUPPORTED
    if AlignmentStatus.CONTRADICTS in {object_alignment, issue_alignment, area_alignment}:
        status = ClaimStatus.CONTRADICTED
    elif not evidence_standard_met:
        status = ClaimStatus.NOT_ENOUGH_INFORMATION

    return EvidenceValidationResult(
        claim_id="claim_001",
        evidence_standard_met=evidence_standard_met,
        evidence_standard_met_reason="Evidence alignment was evaluated.",
        reviewability=reviewability,
        object_alignment=object_alignment,
        issue_alignment=issue_alignment,
        area_alignment=area_alignment,
        severity_alignment=AlignmentStatus.UNKNOWN,
        missing_evidence=[] if evidence_standard_met else ["Insufficient visible evidence."],
        supporting_image_ids=["img_1"] if supporting_image_ids is None else supporting_image_ids,
        recommended_status=status,
        confidence=confidence,
    )


def _risk(*, risk_flags: list[str] | None = None, risk_score: float = 0.0) -> RiskAssessmentResult:
    return RiskAssessmentResult(
        user_id="user_001",
        risk_flags=["none"] if risk_flags is None else risk_flags,
        risk_score=risk_score,
        justification="Risk context for testing.",
    )


class DecisionAgentTest(TestCase):
    def test_supported_claim(self) -> None:
        result = DecisionAgent().run((_claim(), _vision(), _evidence(), _risk()))

        self.assertEqual(result.claim_status, ClaimStatus.SUPPORTED)
        self.assertEqual(result.issue_type, IssueType.DENT)
        self.assertEqual(result.object_part, "rear_bumper")
        self.assertTrue(result.evidence_standard_met)
        self.assertTrue(result.valid_image)

    def test_contradicted_claim(self) -> None:
        result = DecisionAgent().run(
            (
                _claim(),
                _vision(),
                _evidence(object_alignment=AlignmentStatus.CONTRADICTS),
                _risk(),
            )
        )

        self.assertEqual(result.claim_status, ClaimStatus.CONTRADICTED)
        self.assertIn("contradicts", result.claim_status_justification)
        self.assertEqual(result.review_priority, "high")

    def test_insufficient_evidence(self) -> None:
        result = DecisionAgent().run(
            (
                _claim(),
                _vision(issue_type=IssueType.UNKNOWN, object_part=None, supporting_image_ids=[], damage_visible=False),
                _evidence(
                    evidence_standard_met=False,
                    reviewability=Reviewability.NOT_REVIEWABLE,
                    issue_alignment=AlignmentStatus.UNKNOWN,
                    area_alignment=AlignmentStatus.UNKNOWN,
                    supporting_image_ids=[],
                    confidence=0.2,
                ),
                _risk(),
            )
        )

        self.assertEqual(result.claim_status, ClaimStatus.NOT_ENOUGH_INFORMATION)
        self.assertFalse(result.valid_image)
        self.assertEqual(result.supporting_image_ids, [])

    def test_high_risk_supported_claim(self) -> None:
        result = DecisionAgent().run(
            (
                _claim(),
                _vision(),
                _evidence(),
                _risk(risk_flags=["high_recent_claim_volume"], risk_score=0.82),
            )
        )

        self.assertEqual(result.claim_status, ClaimStatus.SUPPORTED)
        self.assertEqual(result.review_priority, "high")
        self.assertIn("Risk flags affect review priority only", result.claim_status_justification)

    def test_multi_image_supported_claim(self) -> None:
        image_ids = ["img_1", "img_2", "img_3"]
        result = DecisionAgent().run(
            (
                _claim(),
                _vision(supporting_image_ids=image_ids),
                _evidence(supporting_image_ids=image_ids),
                _risk(),
            )
        )

        self.assertEqual(result.claim_status, ClaimStatus.SUPPORTED)
        self.assertEqual(result.supporting_image_ids, image_ids)
        self.assertEqual(result.review_priority, "normal")


if __name__ == "__main__":
    main()
