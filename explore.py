import pandas as pd

sample = pd.read_csv("dataset/sample_claims.csv")
claims = pd.read_csv("dataset/claims.csv")
history = pd.read_csv("dataset/user_history.csv")
requirements = pd.read_csv("dataset/evidence_requirements.csv")

print("Sample:", sample.shape)
print("Claims:", claims.shape)
print("History:", history.shape)
print("Requirements:", requirements.shape)

print("\nClaim Status")
print(sample["claim_status"].value_counts())

print("\nIssue Type")
print(sample["issue_type"].value_counts())

print("\nSeverity")
print(sample["severity"].value_counts())

print("\nRisk Flags")
print(sample["risk_flags"].value_counts().head(20))
