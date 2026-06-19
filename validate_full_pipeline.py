import pandas as pd

from code.pipelines.review_pipeline import (
    ReviewPipeline,
    DeterministicVisionAgent,
    DeterministicEvidenceAgent,
)

df = pd.read_csv("dataset/sample_claims.csv")

pipeline = ReviewPipeline(
    vision_agent=DeterministicVisionAgent(),
    evidence_agent=DeterministicEvidenceAgent(),
)

results = pipeline.process_batch(df.head(5))

for result in results:
    print("=" * 80)
    print("Claim ID :", result.claim_id)
    print("Status   :", result.claim_status)
    print("Issue    :", result.issue_type)
    print("Part     :", result.object_part)
    print("Severity :", result.severity)