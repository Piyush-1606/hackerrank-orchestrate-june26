from __future__ import annotations

from pathlib import Path

import pandas as pd

from code.agents.vision_agent import VisionAgent
from code.models.schemas import ClaimInput, ObjectType


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
DATASET_PATH = PROJECT_ROOT / "dataset" / "sample_claims.csv"


def build_claim(row: pd.Series, case_number: int) -> ClaimInput:
    return ClaimInput(
        claim_id=f"sample_{case_number:03d}",
        user_id=row["user_id"],
        image_paths=row["image_paths"],
        claim_text=row["user_claim"],
        object_type=ObjectType(row["claim_object"]),
        user_history=None,
        evidence_requirements=[],
        metadata={},
    )


def main() -> None:
    sample = pd.read_csv(DATASET_PATH)
    agent = VisionAgent(project_root=DATASET_ROOT)

    part_correct = 0
    issue_correct = 0

    for index, row in sample.iterrows():
        case_number = index + 1
        claim = build_claim(row, case_number)
        expected_issue = row["issue_type"]
        expected_part = row["object_part"]

        try:
            result = agent.run(claim)
            predicted_issue = result.detected_issue_type.value
            predicted_part = result.detected_object_part
        except Exception as exc:
            predicted_issue = "error"
            predicted_part = "error"
            result = None
            print(f"\nCASE {case_number:03d} failed: {type(exc).__name__}: {exc}")

        if predicted_issue == expected_issue:
            issue_correct += 1
        if predicted_part == expected_part:
            part_correct += 1

        if predicted_issue != expected_issue or predicted_part != expected_part:
            print("=" * 80)
            print(f"CASE {case_number:03d}")
            print("\nEXPECTED")
            print(f"Issue : {expected_issue}")
            print(f"Part  : {expected_part}")
            print("\nPREDICTED")
            print(f"Issue : {predicted_issue}")
            print(f"Part  : {predicted_part}")
            if result is not None:
                print(f"Reason: {result.reasoning}")
            print("=" * 80)

    total = len(sample)
    print(f"\nPart Accuracy: {part_correct}/{total}")
    print(f"Issue Accuracy: {issue_correct}/{total}")


if __name__ == "__main__":
    main()
