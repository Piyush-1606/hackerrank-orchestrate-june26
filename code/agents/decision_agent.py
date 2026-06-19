from __future__ import annotations

import logging
from typing import Any

try:
    from agents.base_agent import AgentRunContext, BaseAgent, RetryConfig
    from models.schemas import (
        AlignmentStatus,
        ClaimExtractionResult,
        ClaimStatus,
        EvidenceValidationResult,
        FinalDecisionResult,
        IssueType,
        Reviewability,
        RiskAssessmentResult,
        Severity,
        VisionResult,
    )
except ModuleNotFoundError:
    from .base_agent import AgentRunContext, BaseAgent, RetryConfig
    from ..models.schemas import (
        AlignmentStatus,
        ClaimExtractionResult,
        ClaimStatus,
        EvidenceValidationResult,
        FinalDecisionResult,
        IssueType,
        Reviewability,
        RiskAssessmentResult,
        Severity,
        VisionResult,
    )


DecisionInput = tuple[
    ClaimExtractionResult,
    VisionResult,
    EvidenceValidationResult,
    RiskAssessmentResult,
]


class DecisionAgent(BaseAgent[DecisionInput, FinalDecisionResult]):
    """Fuse upstream agent outputs into the final claim review decision.

    Evidence alignment determines claim status. Risk context only affects the
    final explanation and operational review priority.
    """

    def __init__(
        self,
        *,
        retry_config: RetryConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(name="DecisionAgent", retry_config=retry_config, logger=logger)

    def validate_input(self, input_data: DecisionInput) -> None:
        """Validate the four-agent tuple contract."""
        super().validate_input(input_data)

        if not isinstance(input_data, tuple) or len(input_data) != 4:
            raise ValueError(
                "DecisionAgent input must be "
                "(ClaimExtractionResult, VisionResult, EvidenceValidationResult, RiskAssessmentResult)"
            )

        claim, vision, evidence, risk = input_data
        if not isinstance(claim, ClaimExtractionResult):
            raise TypeError("DecisionAgent first input must be ClaimExtractionResult")
        if not isinstance(vision, VisionResult):
            raise TypeError("DecisionAgent second input must be VisionResult")
        if not isinstance(evidence, EvidenceValidationResult):
            raise TypeError("DecisionAgent third input must be EvidenceValidationResult")
        if not isinstance(risk, RiskAssessmentResult):
            raise TypeError("DecisionAgent fourth input must be RiskAssessmentResult")

    def _execute(self, input_data: DecisionInput, context: AgentRunContext) -> FinalDecisionResult:
        claim, vision, evidence, risk = input_data

        claim_status = self.determine_claim_status(evidence)
        issue_type = self.select_issue_type(claim, vision)
        object_part = self.select_object_part(claim, vision)
        severity = self.select_severity(claim, vision)
        valid_image = evidence.reviewability != Reviewability.NOT_REVIEWABLE
        risk_flags = self.get_risk_flags(risk)
        risk_score = self.get_risk_score(risk)
        review_priority = self.determine_review_priority(
            claim_status=claim_status,
            evidence=evidence,
            risk_flags=risk_flags,
            risk_score=risk_score,
        )
        confidence = self.determine_confidence(
            evidence=evidence,
            risk_score=risk_score,
            claim_status=claim_status,
        )
        justification = self.build_justification(
            claim_status=claim_status,
            evidence=evidence,
            risk=risk,
            review_priority=review_priority,
        )

        self._log(
            logging.INFO,
            "final_decision_completed",
            context=context,
            claim_status=str(claim_status),
            issue_type=str(issue_type),
            object_part=object_part,
            severity=str(severity),
            evidence_standard_met=evidence.evidence_standard_met,
            risk_score=risk_score,
            review_priority=review_priority,
            confidence=confidence,
        )

        return FinalDecisionResult(
            claim_id=claim.claim_id or vision.claim_id or evidence.claim_id or context.claim_id,
            claim_status=claim_status,
            issue_type=issue_type,
            object_part=object_part,
            severity=severity,
            confidence=confidence,
            evidence_standard_met=evidence.evidence_standard_met,
            evidence_standard_met_reason=evidence.evidence_standard_met_reason,
            claim_status_justification=justification,
            supporting_image_ids=list(evidence.supporting_image_ids),
            valid_image=valid_image,
            risk_flags=risk_flags,
            risk_score=risk_score,
            review_priority=review_priority,
        )

    @staticmethod
    def determine_claim_status(evidence: EvidenceValidationResult) -> ClaimStatus:
        """Determine final claim status from evidence-only results."""
        if evidence.recommended_status == ClaimStatus.CONTRADICTED:
            return ClaimStatus.CONTRADICTED

        if (
            evidence.evidence_standard_met
            and evidence.object_alignment == AlignmentStatus.SUPPORTS
            and evidence.area_alignment == AlignmentStatus.SUPPORTS
            and evidence.issue_alignment == AlignmentStatus.SUPPORTS
        ):
            return ClaimStatus.SUPPORTED

        if evidence.reviewability == Reviewability.NOT_REVIEWABLE:
            return ClaimStatus.NOT_ENOUGH_INFORMATION

        if not evidence.evidence_standard_met:
            return ClaimStatus.NOT_ENOUGH_INFORMATION

        return ClaimStatus.NOT_ENOUGH_INFORMATION

    @staticmethod
    def select_issue_type(claim: ClaimExtractionResult, vision: VisionResult) -> IssueType:
        """Prefer concrete vision issue type, otherwise use claim issue type."""
        if vision.detected_issue_type not in {IssueType.UNKNOWN, IssueType.NONE}:
            return vision.detected_issue_type
        if claim.claimed_issue_types:
            return claim.claimed_issue_types[0]
        return IssueType.UNKNOWN

    @staticmethod
    def select_object_part(claim: ClaimExtractionResult, vision: VisionResult) -> str:
        """Prefer detected object part, otherwise use extracted claim area."""
        if vision.detected_object_part:
            return vision.detected_object_part
        if claim.affected_area:
            return claim.affected_area.split(";")[0].strip() or "unknown"
        return "unknown"

    @staticmethod
    def select_severity(claim: ClaimExtractionResult, vision: VisionResult) -> Severity:
        """Prefer vision severity if future VisionResult variants provide one."""
        vision_severity = getattr(vision, "severity", None) or getattr(vision, "visible_severity", None)
        if isinstance(vision_severity, Severity) and vision_severity != Severity.UNKNOWN:
            return vision_severity
        if isinstance(vision_severity, str) and vision_severity != Severity.UNKNOWN.value:
            return Severity(vision_severity)
        return claim.claimed_severity

    @staticmethod
    def get_risk_flags(risk: RiskAssessmentResult) -> list[str]:
        """Extract normalized risk flags from current RiskAgent schema."""
        flags = getattr(risk, "risk_flags", None) or ["none"]
        normalized = [flag for flag in flags if flag and flag != "none"]
        return normalized or ["none"]

    @staticmethod
    def get_risk_score(risk: RiskAssessmentResult) -> float:
        """Extract risk score from current or earlier RiskAssessmentResult shapes."""
        value = getattr(risk, "risk_score", None)
        if value is None:
            return 0.0
        return round(max(0.0, min(float(value), 1.0)), 3)

    @staticmethod
    def determine_review_priority(
        *,
        claim_status: ClaimStatus,
        evidence: EvidenceValidationResult,
        risk_flags: list[str],
        risk_score: float,
    ) -> str:
        """Determine review priority without changing claim status."""
        has_risk = any(flag != "none" for flag in risk_flags)
        if claim_status == ClaimStatus.CONTRADICTED:
            return "high"
        if risk_score >= 0.7 or has_risk:
            return "high"
        if evidence.reviewability == Reviewability.PARTIALLY_REVIEWABLE or risk_score >= 0.3:
            return "medium"
        return "normal"

    @staticmethod
    def determine_confidence(
        *,
        evidence: EvidenceValidationResult,
        risk_score: float,
        claim_status: ClaimStatus,
    ) -> float:
        """Calibrate confidence primarily from evidence, lightly adjusted for review risk."""
        confidence = evidence.confidence
        if claim_status == ClaimStatus.NOT_ENOUGH_INFORMATION:
            confidence = min(confidence, 0.65)
        if risk_score >= 0.7:
            confidence -= 0.05
        return round(max(0.0, min(confidence, 1.0)), 3)

    @classmethod
    def build_justification(
        cls,
        *,
        claim_status: ClaimStatus,
        evidence: EvidenceValidationResult,
        risk: RiskAssessmentResult,
        review_priority: str,
    ) -> str:
        """Generate concise final decision justification."""
        if claim_status == ClaimStatus.SUPPORTED:
            base = "Image evidence supports the claimed object, affected area, and issue type."
        elif claim_status == ClaimStatus.CONTRADICTED:
            base = "Visible evidence contradicts the claimed damage."
        else:
            base = "There is not enough reviewable visual evidence to support or contradict the claim."

        risk_flags = cls.get_risk_flags(risk)
        risk_text = ""
        if any(flag != "none" for flag in risk_flags):
            risk_text = f" Risk flags affect review priority only: {', '.join(risk_flags)}."
        elif getattr(risk, "justification", ""):
            risk_text = f" Risk context: {getattr(risk, 'justification')}"

        return f"{base} {evidence.evidence_standard_met_reason}{risk_text} Review priority: {review_priority}."

    @staticmethod
    def contradiction_reason(evidence: EvidenceValidationResult) -> str:
        """Explain the strongest contradiction."""
        if evidence.object_alignment == AlignmentStatus.CONTRADICTS:
            return "Image evidence contradicts the claimed object."
        if evidence.area_alignment == AlignmentStatus.CONTRADICTS:
            return "Image evidence contradicts the claimed affected area."
        if evidence.issue_alignment == AlignmentStatus.CONTRADICTS:
            return "Image evidence contradicts the claimed issue type."
        return "Image evidence contradicts the claim."
