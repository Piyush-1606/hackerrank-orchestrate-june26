# Mismatch Report

## Selected Cases

### row_0

- Claim text: Customer: I picked up my car after service and noticed a mark on the hood. | Support: What kind of mark is it? | Customer: It looks like a scratch across the top panel. | Support: Do you think it happened during service? | Customer: Yes, that is why I am asking for review. The photo is attached.
- Expected issue_type: broken_part
- Expected object_part: front_bumper
- Image paths:
  - C:\Users\piyus\OneDrive\Desktop\HackeRank Hackathon\hackerrank-orchestrate-june26\dataset\images\sample\case_001\img_1.jpg | exists=True

- Predicted issue_type: scratch
- Predicted object_part: hood
- Root-cause classifications:
  - vision_reasoning_issue
- Suggested improvements:
  - Integrate stronger VLM grounding or higher-resolution region proposals to localize damage.

### row_1

- Claim text: Customer: The item I ordered was not inside the box. | Support: Did the package look opened when you received it? | Customer: I checked it after delivery and could not find the product inside. | Support: What are you asking us to verify? | Customer: Please verify that the contents are missing from the package.
- Expected issue_type: unknown
- Expected object_part: contents
- Image paths:
  - C:\Users\piyus\OneDrive\Desktop\HackeRank Hackathon\hackerrank-orchestrate-june26\dataset\images\sample\case_001\img_1.jpg | exists=True
  - C:\Users\piyus\OneDrive\Desktop\HackeRank Hackathon\hackerrank-orchestrate-june26\dataset\images\sample\case_002\img_2.jpg | exists=True

- Predicted issue_type: missing_part
- Predicted object_part: contents
- Root-cause classifications:
  - evidence_logic_issue
  - label_ambiguity
- Suggested improvements:
  - Relax evidence sufficiency thresholds to surface partial matches and provide human review cues.
  - Add label normalization and annotator guidelines; handle 'none/unknown' with soft labels.

### row_2

- Claim text: Customer: The shipping box arrived in bad condition. | Support: What kind of condition issue are you reporting? | Customer: It looked badly crushed when it was delivered. | Support: Is the claim about the outside box or the product inside? | Customer: The outside box. I want the crushed box reviewed.
- Expected issue_type: unknown
- Expected object_part: unknown
- Image paths:
  - C:\Users\piyus\OneDrive\Desktop\HackeRank Hackathon\hackerrank-orchestrate-june26\dataset\images\sample\case_001\img_1.jpg | exists=True

- Predicted issue_type: crushed_packaging
- Predicted object_part: unknown
- Root-cause classifications:
  - evidence_logic_issue
  - label_ambiguity
- Suggested improvements:
  - Relax evidence sufficiency thresholds to surface partial matches and provide human review cues.
  - Add label normalization and annotator guidelines; handle 'none/unknown' with soft labels.

### row_3

- Claim text: Customer: My delivery box arrived opened. | Support: Was the package crushed or was the seal affected? | Customer: The seal area looked torn when I received it. | Support: Are you asking us to review the package condition or the item inside? | Customer: The package condition. I want the torn-open package reviewed.
- Expected issue_type: none
- Expected object_part: seal
- Image paths:
  - C:\Users\piyus\OneDrive\Desktop\HackeRank Hackathon\hackerrank-orchestrate-june26\dataset\images\sample\case_001\img_1.jpg | exists=True
  - C:\Users\piyus\OneDrive\Desktop\HackeRank Hackathon\hackerrank-orchestrate-june26\dataset\images\sample\case_002\img_2.jpg | exists=True

- Predicted issue_type: torn_packaging
- Predicted object_part: seal
- Root-cause classifications:
  - evidence_logic_issue
  - label_ambiguity
- Suggested improvements:
  - Relax evidence sufficiency thresholds to surface partial matches and provide human review cues.
  - Add label normalization and annotator guidelines; handle 'none/unknown' with soft labels.


## Root-cause summary and concrete improvements

See per-case recommendations above. Key themes:
- Improve vision grounding and region-level damage detection.
- Enhance claim parsing to reduce claim-object mismatches.
- Surface ambiguity cases for human review and request more images when necessary.