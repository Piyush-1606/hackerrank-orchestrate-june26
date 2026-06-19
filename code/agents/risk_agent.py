from __future__ import annotations

from code.agents.base_agent import BaseAgent, AgentRunContext
from code.models.schemas import (
    UserHistory,
    RiskAssessmentResult,
)


class RiskAgent(BaseAgent[UserHistory, RiskAssessmentResult]):
    """
    Evaluates user claim history and produces
    risk context for downstream review.

    IMPORTANT:
    User history never determines claim validity.
    It only provides review context.
    """

    def _execute(
        self,
        input_data: UserHistory,
        context: AgentRunContext,
    ) -> RiskAssessmentResult:

        risk_flags: list[str] = []

        score = 0.0
        reasons: list[str] = []

        # ---------------------------------------------------
        # Parse explicit history flags
        # ---------------------------------------------------

        if input_data.history_flags:
            parsed_flags = [
                flag.strip()
                for flag in input_data.history_flags.split(";")
                if flag.strip() and flag.strip().lower() != "none"
            ]

            risk_flags.extend(parsed_flags)

        # ---------------------------------------------------
        # Rejection ratio
        # ---------------------------------------------------

        if input_data.past_claim_count > 0:
            rejection_ratio = (
                input_data.rejected_claim /
                input_data.past_claim_count
            )

            if rejection_ratio >= 0.50:
                score += 0.40
                reasons.append(
                    "high rejected claim ratio"
                )

            elif rejection_ratio >= 0.25:
                score += 0.25
                reasons.append(
                    "moderate rejected claim ratio"
                )

        # ---------------------------------------------------
        # Recent claim frequency
        # ---------------------------------------------------

        recent = input_data.last_90_days_claim_count

        if recent >= 8:
            score += 0.30
            reasons.append(
                "very high recent claim volume"
            )

        elif recent >= 4:
            score += 0.15
            reasons.append(
                "elevated recent claim volume"
            )

        # ---------------------------------------------------
        # Manual review history
        # ---------------------------------------------------

        if input_data.manual_review_claim >= 3:
            score += 0.20
            reasons.append(
                "multiple manual reviews"
            )

        elif input_data.manual_review_claim >= 1:
            score += 0.10
            reasons.append(
                "prior manual review"
            )

        # ---------------------------------------------------
        # Positive history adjustment
        # ---------------------------------------------------

        if (
            input_data.accept_claim > 0
            and input_data.rejected_claim == 0
        ):
            score -= 0.10
            reasons.append(
                "strong accepted claim history"
            )

        # ---------------------------------------------------
        # Clamp score
        # ---------------------------------------------------

        score = max(0.0, min(score, 1.0))

        # ---------------------------------------------------
        # Justification
        # ---------------------------------------------------

        if not reasons:
            justification = (
                "No significant user history risk detected."
            )
        else:
            justification = (
                "Risk factors: "
                + ", ".join(reasons)
                + "."
            )

        if not risk_flags:
            risk_flags = ["none"]

        return RiskAssessmentResult(
            user_id=input_data.user_id,
            risk_flags=risk_flags,
            risk_score=round(score, 2),
            justification=justification,
        )