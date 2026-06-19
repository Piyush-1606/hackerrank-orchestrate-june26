from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

try:
    from agents.base_agent import AgentRunContext
    from agents.claim_agent import ClaimAgent
    from agents.decision_agent import DecisionAgent
    from agents.evidence_agent import EvidenceAgent
    from agents.risk_agent import RiskAgent
    from agents.vision_agent import VisionAgent
    from models.schemas import (
        AlignmentStatus,
        ClaimInput,
        ClaimStatus,
        EvidenceValidationResult,
        FinalDecisionResult,
        IssueType,
        ObjectType,
        Reviewability,
        Severity,
        UserHistory,
        VisionQualityFlag,
        VisionResult,
    )
except ModuleNotFoundError:
    from ..agents.base_agent import AgentRunContext
    from ..agents.claim_agent import ClaimAgent
    from ..agents.decision_agent import DecisionAgent
    from ..agents.evidence_agent import EvidenceAgent
    from ..agents.risk_agent import RiskAgent
    from ..agents.vision_agent import VisionAgent
    from ..models.schemas import (
        AlignmentStatus,
        ClaimInput,
        ClaimStatus,
        EvidenceValidationResult,
        FinalDecisionResult,
        IssueType,
        ObjectType,
        Reviewability,
        Severity,
        UserHistory,
        VisionQualityFlag,
        VisionResult,
    )


class ReviewPipeline:
    """Orchestrates claim, vision, evidence, risk, and decision agents."""

    def __init__(
        self,
        *,
        claim_agent: ClaimAgent | None = None,
        vision_agent: VisionAgent | None = None,
        evidence_agent: EvidenceAgent | None = None,
        risk_agent: RiskAgent | None = None,
        decision_agent: DecisionAgent | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.claim_agent = claim_agent or ClaimAgent()
        self.vision_agent = vision_agent or VisionAgent()
        self.evidence_agent = evidence_agent or EvidenceAgent()
        self.risk_agent = risk_agent or RiskAgent()
        self.decision_agent = decision_agent or DecisionAgent()
        self.logger = logger or logging.getLogger("pipelines.ReviewPipeline")

    def process_claim(self, claim: ClaimInput) -> FinalDecisionResult:
        """Process a single claim and return a final deterministic decision."""
        if claim is None:
            raise ValueError("process_claim requires a ClaimInput")

        context = AgentRunContext(claim_id=claim.claim_id, user_id=claim.user_id)
        self.logger.info(
            "review_pipeline_started",
            extra={"claim_id": claim.claim_id, "user_id": claim.user_id},
        )

        try:
            claim_result = self.claim_agent.run(claim, context)
            vision_result = self.vision_agent.run(claim, context)
            evidence_result = self.evidence_agent.run((claim_result, vision_result), context)
            risk_result = self.risk_agent.run(self.build_user_history(claim), context)
            decision = self.decision_agent.run(
                (claim_result, vision_result, evidence_result, risk_result),
                context,
            )

            self.logger.info(
                "review_pipeline_succeeded",
                extra={
                    "claim_id": claim.claim_id,
                    "user_id": claim.user_id,
                    "claim_status": str(decision.claim_status),
                },
            )
            return decision

        except Exception as exc:
            self.logger.exception(
                "review_pipeline_failed",
                extra={
                    "claim_id": claim.claim_id,
                    "user_id": claim.user_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            return self.failure_decision(claim=claim, error=exc)

    def process_batch(self, claims: Any) -> list[FinalDecisionResult]:
        """Process a dataframe or iterable of claims."""
        if claims is None:
            raise ValueError("process_batch requires a dataframe or iterable of claims")

        claim_inputs = self.claims_from_batch(claims)
        if not claim_inputs:
            return []

        return [self.process_claim(claim) for claim in claim_inputs]

    @classmethod
    def claims_from_batch(cls, claims: Any) -> list[ClaimInput]:
        """Normalize a dataframe-like object or iterable into ClaimInput objects."""
        if hasattr(claims, "iterrows"):
            return [cls.claim_from_mapping(row.to_dict(), index=index) for index, row in claims.iterrows()]

        if isinstance(claims, Iterable) and not isinstance(claims, (str, bytes, dict)):
            normalized: list[ClaimInput] = []
            for index, item in enumerate(claims):
                if isinstance(item, ClaimInput):
                    normalized.append(item)
                elif isinstance(item, dict):
                    normalized.append(cls.claim_from_mapping(item, index=index))
                else:
                    raise TypeError("Batch iterable items must be ClaimInput or dict")
            return normalized

        raise TypeError("process_batch expects a dataframe-like object or iterable")

    @staticmethod
    def claim_from_mapping(row: dict[str, Any], *, index: Any) -> ClaimInput:
        """Build ClaimInput from dataframe row or dictionary."""
        metadata = dict(row.get("metadata") or {})
        for key in (
            "past_claim_count",
            "accept_claim",
            "manual_review_claim",
            "rejected_claim",
            "last_90_days_claim_count",
            "history_flags",
            "history_summary",
        ):
            if key in row and key not in metadata:
                metadata[key] = row[key]

        return ClaimInput(
            claim_id=str(row.get("claim_id") or f"row_{index}"),
            user_id=str(row["user_id"]),
            image_paths=row.get("image_paths", []),
            claim_text=str(row.get("claim_text") or row.get("user_claim") or ""),
            object_type=ObjectType(row.get("object_type") or row.get("claim_object")),
            user_history=row.get("user_history"),
            evidence_requirements=row.get("evidence_requirements") or [],
            metadata=metadata,
        )

    @staticmethod
    def build_user_history(claim: ClaimInput) -> UserHistory:
        """Create RiskAgent input from claim metadata with safe defaults."""
        metadata = claim.metadata or {}
        return UserHistory(
            user_id=claim.user_id,
            past_claim_count=int(metadata.get("past_claim_count", 0) or 0),
            accept_claim=int(metadata.get("accept_claim", 0) or 0),
            manual_review_claim=int(metadata.get("manual_review_claim", 0) or 0),
            rejected_claim=int(metadata.get("rejected_claim", 0) or 0),
            last_90_days_claim_count=int(metadata.get("last_90_days_claim_count", 0) or 0),
            history_flags=str(metadata.get("history_flags") or "none"),
            history_summary=str(metadata.get("history_summary") or claim.user_history or ""),
        )

    @staticmethod
    def failure_decision(claim: ClaimInput, error: Exception) -> FinalDecisionResult:
        """Return a valid not-enough-information decision after pipeline failure."""
        image_ids = [
            path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].rsplit(".", 1)[0]
            for path in claim.image_paths
        ]
        return FinalDecisionResult(
            claim_id=claim.claim_id,
            claim_status=ClaimStatus.NOT_ENOUGH_INFORMATION,
            issue_type=IssueType.UNKNOWN,
            object_part="unknown",
            severity=Severity.UNKNOWN,
            confidence=0.0,
            evidence_standard_met=False,
            evidence_standard_met_reason=f"Pipeline could not complete evidence review: {type(error).__name__}.",
            claim_status_justification="The claim could not be fully reviewed because an upstream processing step failed.",
            supporting_image_ids=image_ids,
            valid_image=False,
            risk_flags=["pipeline_error"],
            risk_score=0.0,
            review_priority="high",
        )


class DeterministicVisionAgent(VisionAgent):
    """Optional test/helper vision agent that avoids filesystem validation."""

    def validate_input(self, input_data: ClaimInput) -> None:
        if input_data is None:
            raise ValueError("Vision input is required")

    def _execute(self, input_data: ClaimInput, context: AgentRunContext) -> VisionResult:
        image_ids = [
            path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].rsplit(".", 1)[0]
            for path in input_data.image_paths
        ]
        return VisionResult(
            claim_id=input_data.claim_id or context.claim_id,
            detected_object=input_data.object_type,
            detected_issue_type=IssueType.DENT,
            detected_object_part="rear_bumper",
            visible_parts=["rear_bumper"],
            image_quality_flags=[] if image_ids else [VisionQualityFlag.DAMAGE_NOT_VISIBLE],
            supporting_image_ids=image_ids,
            damage_visible=True,
            confidence=0.85,
            reasoning="Deterministic pipeline helper vision result.",
        )


class DeterministicEvidenceAgent(EvidenceAgent):
    """Optional test/helper evidence agent for deterministic pipeline tests."""

    def _execute(self, input_data: Any, context: AgentRunContext) -> EvidenceValidationResult:
        claim_result, vision_result = input_data
        has_images = bool(vision_result.supporting_image_ids)
        return EvidenceValidationResult(
            claim_id=claim_result.claim_id or context.claim_id,
            evidence_standard_met=has_images,
            evidence_standard_met_reason="Deterministic evidence validation completed.",
            reviewability=Reviewability.REVIEWABLE if has_images else Reviewability.NOT_REVIEWABLE,
            object_alignment=AlignmentStatus.SUPPORTS,
            issue_alignment=AlignmentStatus.SUPPORTS if has_images else AlignmentStatus.UNKNOWN,
            area_alignment=AlignmentStatus.SUPPORTS if has_images else AlignmentStatus.UNKNOWN,
            severity_alignment=AlignmentStatus.UNKNOWN,
            missing_evidence=[] if has_images else ["No image evidence."],
            supporting_image_ids=list(vision_result.supporting_image_ids),
            recommended_status=ClaimStatus.SUPPORTED if has_images else ClaimStatus.NOT_ENOUGH_INFORMATION,
            confidence=0.9 if has_images else 0.1,
        )
