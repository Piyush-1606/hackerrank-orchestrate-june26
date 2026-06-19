from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from code.agents.vision_agent import VisionAgent
from code.models.schemas import ClaimInput, ObjectType
from code.pipelines.review_pipeline import ReviewPipeline


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
DATASET_PATH = DATASET_ROOT / "sample_claims.csv"


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
    pipeline = ReviewPipeline(vision_agent=VisionAgent(project_root=DATASET_ROOT))

    issue_correct = 0
    part_correct = 0
    issue_confusion: Counter[tuple[str, str]] = Counter()

    for index, row in sample.iterrows():
        case_number = index + 1
        expected_issue = str(row["issue_type"])
        expected_part = str(row["object_part"])

        result = pipeline.process_claim(build_claim(row, case_number))
        predicted_issue = result.issue_type.value
        predicted_part = result.object_part

        issue_confusion[(expected_issue, predicted_issue)] += 1

        if predicted_issue == expected_issue:
            issue_correct += 1
        if predicted_part == expected_part:
            part_correct += 1

        if predicted_issue != expected_issue or predicted_part != expected_part:
            print("=" * 50)
            print(f"CASE {case_number:03d}")
            print("\nEXPECTED")
            print(f"Issue : {expected_issue}")
            print(f"Part  : {expected_part}")
            print("\nPREDICTED")
            print(f"Issue : {predicted_issue}")
            print(f"Part  : {predicted_part}")
            print("=" * 12)

    total = len(sample)
    print(f"\nIssue Accuracy: {issue_correct}/{total}")
    print(f"Part Accuracy: {part_correct}/{total}")

    print("\nIssue Confusion Counts")
    print("======================")
    confusion_rows = [
        {
            "expected_issue": expected,
            "predicted_issue": predicted,
            "count": count,
        }
        for (expected, predicted), count in sorted(issue_confusion.items())
    ]
    print(pd.DataFrame(confusion_rows).to_string(index=False))


if __name__ == "__main__":
    main()
