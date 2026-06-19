from __future__ import annotations

from pathlib import Path

import pandas as pd

from code.models.schemas import ClaimInput, ObjectType
from code.agents.vision_agent import VisionAgent
from code.pipelines.review_pipeline import ReviewPipeline


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
    pipeline = ReviewPipeline(vision_agent=VisionAgent(project_root=DATASET_ROOT))
    rows: list[dict[str, str]] = []

    for index, row in sample.iterrows():
        case_number = index + 1
        claim = build_claim(row, case_number)
        result = pipeline.process_claim(claim)

        output = {
            "case": f"CASE {case_number:03d}",
            "claim_status": result.claim_status.value,
            "issue_type": result.issue_type.value,
            "object_part": result.object_part,
            "severity": result.severity.value,
        }
        rows.append(output)

        print("=" * 80)
        print(output["case"])
        print(f"claim_status : {output['claim_status']}")
        print(f"issue_type   : {output['issue_type']}")
        print(f"object_part  : {output['object_part']}")
        print(f"severity     : {output['severity']}")

    result_frame = pd.DataFrame(rows)

    print("\nSummary Distributions")
    print("=====================")
    for column in ["claim_status", "issue_type", "object_part", "severity"]:
        print(f"\n{column}")
        print("-" * len(column))
        print(result_frame[column].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
