import pandas as pd
from code.agents.claim_agent import ClaimAgent
from code.models.schemas import ClaimInput, ObjectType

agent = ClaimAgent()

sample = pd.read_csv("dataset/sample_claims.csv")

part_correct = 0
issue_correct = 0

for _, row in sample.iterrows():
    claim = ClaimInput(
        claim_id="eval",
        user_id=row["user_id"],
        image_paths=[],
        claim_text=row["user_claim"],
        object_type=ObjectType(row["claim_object"]),
        user_history=None,
        evidence_requirements=[],
        metadata={}
    )

    result = agent.run(claim)

    predicted_part = result.affected_area.split(";")[0] if result.affected_area else None
    predicted_issue = (
        result.claimed_issue_types[0].value
        if result.claimed_issue_types
        else "unknown"
    )

    if predicted_part == row["object_part"]:
        part_correct += 1

    if predicted_issue == row["issue_type"]:
        issue_correct += 1

    if predicted_part != row["object_part"] or predicted_issue != row["issue_type"]:
        print("=" * 80)
        print("CLAIM:")
        print(row["user_claim"])

        print("\nEXPECTED:")
        print("Issue :", row["issue_type"])
        print("Part  :", row["object_part"])

        print("\nPREDICTED:")
        print("Issue :", predicted_issue)
        print("Part  :", predicted_part)

print("\n" + "=" * 80)
print(f"Part Accuracy: {part_correct}/{len(sample)}")
print(f"Issue Accuracy: {issue_correct}/{len(sample)}")