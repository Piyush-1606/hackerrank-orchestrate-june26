from code.agents.risk_agent import RiskAgent
from code.models.schemas import UserHistory


def run_case(name: str, user: UserHistory):
    print("\n" + "=" * 80)
    print(f"TEST: {name}")

    result = RiskAgent().run(user)

    print("Flags        :", result.risk_flags)
    print("Risk Score   :", result.risk_score)
    print("Justification:", result.justification)


if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Low Risk User (user_001)
    # ------------------------------------------------------------------
    run_case(
        "Low Risk User",
        UserHistory(
            user_id="user_001",
            past_claim_count=2,
            accept_claim=2,
            manual_review_claim=0,
            rejected_claim=0,
            last_90_days_claim_count=1,
            history_flags="none",
            history_summary="Low-risk user with prior accepted car damage claims",
        ),
    )

    # ------------------------------------------------------------------
    # New User (user_006)
    # ------------------------------------------------------------------
    run_case(
        "New User",
        UserHistory(
            user_id="user_006",
            past_claim_count=0,
            accept_claim=0,
            manual_review_claim=0,
            rejected_claim=0,
            last_90_days_claim_count=0,
            history_flags="none",
            history_summary="New user with no prior claim history",
        ),
    )

    # ------------------------------------------------------------------
    # Medium Risk User (user_016)
    # ------------------------------------------------------------------
    run_case(
        "Medium Risk User",
        UserHistory(
            user_id="user_016",
            past_claim_count=11,
            accept_claim=2,
            manual_review_claim=2,
            rejected_claim=7,
            last_90_days_claim_count=6,
            history_flags="user_history_risk;manual_review_required",
            history_summary="Frequent rejected car scratch claims",
        ),
    )

    # ------------------------------------------------------------------
    # High Risk User (user_037)
    # ------------------------------------------------------------------
    run_case(
        "High Risk User",
        UserHistory(
            user_id="user_037",
            past_claim_count=14,
            accept_claim=4,
            manual_review_claim=4,
            rejected_claim=6,
            last_90_days_claim_count=9,
            history_flags="user_history_risk;manual_review_required",
            history_summary="Unusually frequent package damage claims",
        ),
    )