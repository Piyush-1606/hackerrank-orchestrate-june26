from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from code.agents.vision_agent import VisionAgent
from code.models.schemas import ClaimInput, ClaimStatus, FinalDecisionResult, IssueType, ObjectType, Severity
from code.pipelines.review_pipeline import ReviewPipeline


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
CLAIMS_PATH = DATASET_ROOT / "claims.csv"
USER_HISTORY_PATH = DATASET_ROOT / "user_history.csv"
EVIDENCE_REQUIREMENTS_PATH = DATASET_ROOT / "evidence_requirements.csv"
OUTPUT_PATH = DATASET_ROOT / "output.csv"

OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(CLAIMS_PATH),
        pd.read_csv(USER_HISTORY_PATH),
        pd.read_csv(EVIDENCE_REQUIREMENTS_PATH),
    )


def build_history_lookup(history: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {
        str(row["user_id"]): row.to_dict()
        for _, row in history.iterrows()
    }


def requirements_for_object(requirements: pd.DataFrame, claim_object: str) -> list[str]:
    relevant = requirements[
        (requirements["claim_object"].astype(str) == "all")
        | (requirements["claim_object"].astype(str) == str(claim_object))
    ]
    return relevant["minimum_image_evidence"].dropna().astype(str).tolist()


def build_claim_input(
    row: pd.Series,
    *,
    index: int,
    history_lookup: dict[str, dict[str, Any]],
    requirements: pd.DataFrame,
) -> ClaimInput:
    user_id = str(row["user_id"])
    claim_object = str(row["claim_object"])
    history = history_lookup.get(user_id, {})

    metadata = {
        "past_claim_count": history.get("past_claim_count", 0),
        "accept_claim": history.get("accept_claim", 0),
        "manual_review_claim": history.get("manual_review_claim", 0),
        "rejected_claim": history.get("rejected_claim", 0),
        "last_90_days_claim_count": history.get("last_90_days_claim_count", 0),
        "history_flags": history.get("history_flags", "none"),
        "history_summary": history.get("history_summary", ""),
    }

    return ClaimInput(
        claim_id=f"claim_{index + 1:03d}",
        user_id=user_id,
        image_paths=row["image_paths"],
        claim_text=row["user_claim"],
        object_type=ObjectType(claim_object),
        user_history=str(history.get("history_summary", "")),
        evidence_requirements=requirements_for_object(requirements, claim_object),
        metadata=metadata,
    )


def fallback_decision(row: pd.Series, error: Exception) -> FinalDecisionResult:
    image_ids = [
        path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].rsplit(".", 1)[0]
        for path in str(row.get("image_paths", "")).split(";")
        if path.strip()
    ]
    return FinalDecisionResult(
        claim_id=None,
        claim_status=ClaimStatus.NOT_ENOUGH_INFORMATION,
        issue_type=IssueType.UNKNOWN,
        object_part="unknown",
        severity=Severity.UNKNOWN,
        confidence=0.0,
        evidence_standard_met=False,
        evidence_standard_met_reason=f"Pipeline failed during submission generation: {type(error).__name__}.",
        claim_status_justification="The claim could not be fully reviewed because processing failed.",
        supporting_image_ids=image_ids,
        valid_image=False,
        risk_flags=["pipeline_error"],
        risk_score=0.0,
        review_priority="high",
    )


def output_row(original: pd.Series, decision: FinalDecisionResult) -> dict[str, Any]:
    return {
        "user_id": original["user_id"],
        "image_paths": original["image_paths"],
        "user_claim": original["user_claim"],
        "claim_object": original["claim_object"],
        "evidence_standard_met": decision.evidence_standard_met,
        "evidence_standard_met_reason": decision.evidence_standard_met_reason,
        "risk_flags": ";".join(decision.risk_flags or ["none"]),
        "issue_type": decision.issue_type.value,
        "object_part": decision.object_part,
        "claim_status": decision.claim_status.value,
        "claim_status_justification": decision.claim_status_justification,
        "supporting_image_ids": ";".join(decision.supporting_image_ids),
        "valid_image": decision.valid_image,
        "severity": decision.severity.value,
    }


def validate_output(output: pd.DataFrame, claims: pd.DataFrame) -> None:
    if list(output.columns) != OUTPUT_COLUMNS:
        raise ValueError("Output column order does not match required template")
    if len(output) != len(claims):
        raise ValueError(f"Output row count {len(output)} does not match claims row count {len(claims)}")
    missing_columns = [column for column in OUTPUT_COLUMNS if column not in output.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def main() -> None:
    claims, history, requirements = load_inputs()
    history_lookup = build_history_lookup(history)
    pipeline = ReviewPipeline(vision_agent=VisionAgent(project_root=DATASET_ROOT))

    rows: list[dict[str, Any]] = []

    for index, row in claims.iterrows():
        try:
            claim = build_claim_input(
                row,
                index=index,
                history_lookup=history_lookup,
                requirements=requirements,
            )
            decision = pipeline.process_claim(claim)
        except Exception as exc:
            decision = fallback_decision(row, exc)

        rows.append(output_row(row, decision))

    output = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    validate_output(output, claims)
    output.to_csv(OUTPUT_PATH, index=False)

    status_counts = output["claim_status"].value_counts()
    print(f"total claims processed: {len(output)}")
    print(f"supported count: {int(status_counts.get('supported', 0))}")
    print(f"contradicted count: {int(status_counts.get('contradicted', 0))}")
    print(
        "not_enough_information count: "
        f"{int(status_counts.get('not_enough_information', 0))}"
    )
    print(f"saved output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
