# Unit Test Results

- Tests run: 39
- Passed: 39
- Failed: 0
- Skipped: 0
- Execution time (s): 2.41

# Accuracy Evaluation

```
==================================================
CASE 008

EXPECTED
Issue : broken_part
Part  : front_bumper

PREDICTED
Issue : scratch
Part  : hood
============
==================================================
CASE 018

EXPECTED
Issue : unknown
Part  : contents

PREDICTED
Issue : missing_part
Part  : contents
============
==================================================
CASE 019

EXPECTED
Issue : unknown
Part  : unknown

PREDICTED
Issue : crushed_packaging
Part  : unknown
============
==================================================
CASE 020

EXPECTED
Issue : none
Part  : seal

PREDICTED
Issue : torn_packaging
Part  : seal
============

Issue Accuracy: 16/20
Part Accuracy: 19/20

Issue Confusion Counts
======================
   expected_issue   predicted_issue  count
      broken_part       broken_part      2
      broken_part           scratch      1
            crack             crack      3
crushed_packaging crushed_packaging      1
             dent              dent      3
             none              none      1
             none    torn_packaging      1
          scratch           scratch      2
            stain             stain      1
   torn_packaging    torn_packaging      1
          unknown crushed_packaging      1
          unknown      missing_part      1
          unknown           unknown      1
     water_damage      water_damage      1


```
- Issue Accuracy: 16/20
- Part Accuracy: 19/20

# Stress Testing Results

```
Stress test completed. Report written to stress_test_report.json
{
  "num_claims": 100,
  "failures": [
    "stress_2",
    "stress_23",
    "stress_25",
    "stress_27",
    "stress_28",
    "stress_49",
    "stress_65",
    "stress_67",
    "stress_70",
    "stress_86"
  ],
  "exceptions": [],
  "invalid_outputs": [],
  "status_counts": {
    "supported": 69,
    "not_enough_information": 31
  }
}

review_pipeline_failed
Traceback (most recent call last):
  File "C:\Users\piyus\OneDrive\Desktop\HackeRank Hackathon\hackerrank-orchestrate-june26\code\pipelines\review_pipeline.py", line 84, in process_claim
    vision_result = self.vision_agent.run(claim, context)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\piyus\OneDrive\Desktop\HackeRank Hackathon\hackerrank-orchestrate-june26\code\agents\base_agent.py", line 97, in run
    self.validate_input(input_data)
  File "C:\Users\piyus\OneDrive\Desktop\HackeRank Hackathon\hackerrank-orchestrate-june26\code\agents\vision_agent.py", line 58, in validate_input
    raise FileNotFoundError(f"Image path does not exist: {image_path}")
FileNotFoundError: Image path does not exist: stress_images\missing_2_1.jpg
review_pipeline_failed
Traceback (most recent call last):
  File "C:\Users\piyus\OneDrive\Desktop\HackeRank Hackathon\hackerrank-orchestrate-june26\code\pipelines\review_pipeline.py", line 84, in process_claim
    vision_result = self.vision_agent.run(claim, context)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\piyus\OneDrive\Desktop\HackeRank Hackathon\hackerrank-orchestrate-june26\code\agents\base_agent.py", line 97, in run
    self.validate_input(input_data)
  File "C:\Users\piyus\OneDrive\Desktop\HackeRank Hackathon\hackerrank-orchestrate-june26\code\agents\vision_agent.py", line 58, in validate_input
    raise FileNotFoundError(f"Image path does not exist: {image_path}")
FileNotFoundError: Image path does not exist: stress_images\missing_23_1.jpg
review_pipeline_failed
Traceback (most recent call last):
  File "C:\Users\piyus\OneDrive\Desktop\HackeRank Hackathon\hackerrank-orchestrate-june26\code\pipelines\review_pipeline.py", line 84, in process_claim
    vision_result = self.vision_agent.run(claim, context)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\piyus\OneDrive\Desktop\HackeRank Hackathon\hackerrank-orchestrate-june26\code\agents\base_agent.py", line 97, in run
    self.validate_input(input_data)
  File "C:\Users\piyus\OneDrive\Desktop\HackeRank Hackathon\hackerrank-orchestrate-june26\code\agents\vision_agent.py", line 58, in validate_input
    raise FileNotFoundError(f"Image path does not exist: {image_path}")
FileNotFoundError: Image path does not exist: stress_images\missing_25_1.jpg
review_pipeline_failed
Traceback (most recent call last):
  File "C:\Users\piyus\OneDrive\Desktop\HackeRank Hackathon\hackerrank-orchestrate-june26\code\pipelines\review_pipeline.py", line 84, in process_claim
    vision_result = self.vision_agent.run(claim, context)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\piyus\OneDrive\Desktop\HackeRank Hackathon\hackerrank-orchestrate-june26\code\agents\base_agent.py", line 97, in run
    self.validate_input(input_data)
  File "C:\Users\piyus\OneDrive\Desktop\HackeRank Hackathon\hackerrank-orchestrate-june26\code\agents\vision_agent.py", line 58, in validate_input
    raise FileNotFoundError(f"Image path does not exist: {image_path}")
FileNotFoundError: Image path does not exist: stress_images\missing_27_1.jpg
review_pipeline_failed
Traceback (most recent call last):
  File "C:\Users\piyus\OneDrive\Desktop\HackeRank Hackathon\hackerrank-orchestrate-june26\code\pipelines\review_pipeline.py", line 84, in process_claim
    vision_result = self.vision_agent.run(claim, context)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\piyus\OneDrive\Desktop\HackeRank Hackathon\hackerrank-orchestrate-june26\code\agents\base_agent.py", line 97, in run
    self.validate_input(input_data)
  File "C:\Users\piyus\OneDrive\Desktop\HackeRank Hackathon\hackerrank-orchestrate-june26\code\agents\vision_agent.py", line 58, in validate_input
    raise FileNotFoundError(f"Image path does not exist: {image_path}")
FileNotFoundError: Image path does not exist: stress_images\missing_28_1.jpg
review_pipeline_failed
Traceback (most recent call last):
  File "C:\Users\piyus\OneDrive\Desktop\HackeRank Hackathon\hackerrank-orchestrate-june26\code\pipelines\review_pipeline.py", line 84, in process_claim
    vision_result = self.vision_agent.run(claim, context)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\piyus\OneDrive\Desktop\HackeRank Hackathon\hackerrank-orchestrate-june26\code\agents\base_agent.py", line 97, in run
    self.validate_input(input_data)
  File "C:\Users\piyus\OneDrive\Desktop\HackeRank Hackathon\hackerrank-orchestrate-june26\code\agents\vision_agent.py", line 58, in validate_input
    raise FileNotFoundError(f"Image path does not exist: {image_path}")
FileNotFoundError: Image path does not exist: stress_images\missing_49_1.jpg
review_pipeline_failed
Traceback (most recent call last):
  File "C:\Users\piyus\OneDrive\Desktop\HackeRank Hackathon\hackerrank-orchestrate-june26\code\pipelines\review_pipeline.py", line 84, in process_claim
    vision_result = self.vision_agent.run(claim, context)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\piyus\OneDrive\Desktop\HackeRank Hackathon\hackerrank-orchestrate-june26\code\agents\base_agent.py", line 97, in run
    self.validate_input(input_data)
  File "C:\Users\piyus\OneDrive\Desktop\HackeRank Hackathon\hackerrank-orchestrate-june26\code\agents\vision_agent.py", line 58, in validate_input
    raise FileNotFoundError(f"Image path does not exist: {image_path}")
FileNotFoundError: Image path does not exist: stress_images\missing_65_1.jpg
review_pipeline_failed
Traceback (most recent call last):
  File "C:\Users\piyus\OneDrive\Desktop\HackeRank Hackathon\hackerrank-orchestrate-june26\code\pipelines\review_pipeline.py", line 84, in process_claim
    vision_result = self.vision_agent.run(claim, context)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\piyus\OneDrive\Desktop\HackeRank Hackathon\hackerrank-orchestrate-june26\code\agents\base_agent.py", line 97, in run
    self.validate_input(input_data)
  File "C:\Users\piyus\OneDrive\Desktop\HackeRank Hackathon\hackerrank-orchestrate-june26\code\agents\vision_agent.py", line 58, in validate_input
    raise FileNotFoundError(f"Image path does not exist: {image_path}")
FileNotFoundError: Image path does not exist: stress_images\missing_67_1.jpg
review_pipeline_failed
Traceback (most recent call last):
  File "C:\Users\piyus\OneDrive\Desktop\HackeRank Hackathon\hackerrank-orchestrate-june26\code\pipelines\review_pipeline.py", line 84, in process_claim
    vision_result = self.vision_agent.run(claim, context)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\piyus\OneDrive\Desktop\HackeRank Hackathon\hackerrank-orchestrate-june26\code\agents\base_agent.py", line 97, in run
    self.validate_input(input_data)
  File "C:\Users\piyus\OneDrive\Desktop\HackeRank Hackathon\hackerrank-orchestrate-june26\code\agents\vision_agent.py", line 58, in validate_input
    raise FileNotFoundError(f"Image path does not exist: {image_path}")
FileNotFoundError: Image path does not exist: stress_images\missing_70_1.jpg
review_pipeline_failed
Traceback (most recent call last):
  File "C:\Users\piyus\OneDrive\Desktop\HackeRank Hackathon\hackerrank-orchestrate-june26\code\pipelines\review_pipeline.py", line 84, in process_claim
    vision_result = self.vision_agent.run(claim, context)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\piyus\OneDrive\Desktop\HackeRank Hackathon\hackerrank-orchestrate-june26\code\agents\base_agent.py", line 97, in run
    self.validate_input(input_data)
  File "C:\Users\piyus\OneDrive\Desktop\HackeRank Hackathon\hackerrank-orchestrate-june26\code\agents\vision_agent.py", line 58, in validate_input
    raise FileNotFoundError(f"Image path does not exist: {image_path}")
FileNotFoundError: Image path does not exist: stress_images\missing_86_1.jpg

```
- total synthetic claims: 100
- failures: 10
- invalid outputs: 0
- claim status distribution: {'supported': 69, 'not_enough_information': 31}

# Demo Validation

```

========================================

##################################################
CAR CLAIM

## CLAIM
Object : car
User ID : user_001
Claim ID : row_0

## CLAIM AGENT
Issue : dent
Part : rear_bumper
Severity : medium
Confidence : 0.95

## VISION AGENT
Issue : dent
Part : rear_bumper
Images Used : img_1
Quality : none
Confidence : 0.95

## EVIDENCE AGENT
Evidence Met: True
Reviewable : reviewable
Status : supported
Supporting Images : img_1

## RISK AGENT
Risk Score : 0.00
Risk Flags : none

## DECISION AGENT
Claim Status: supported
Severity : medium
Supporting Images : img_1
Decision Risk Flags : none

## FINAL DECISION
Supported
Reason:
Image evidence supports the claimed object, affected area, and issue type. Evidence is reviewable and supports the claimed object, affected area, and issue type. Risk context: No significant user history risk detected. Review priority: normal.
##################################################


========================================

##################################################
LAPTOP CLAIM

## CLAIM
Object : laptop
User ID : user_009
Claim ID : row_8

## CLAIM AGENT
Issue : crack
Part : screen
Severity : medium
Confidence : 0.95

## VISION AGENT
Issue : crack
Part : screen
Images Used : img_1
Quality : none
Confidence : 0.95

## EVIDENCE AGENT
Evidence Met: True
Reviewable : reviewable
Status : supported
Supporting Images : img_1

## RISK AGENT
Risk Score : 0.00
Risk Flags : none

## DECISION AGENT
Claim Status: supported
Severity : medium
Supporting Images : img_1
Decision Risk Flags : none

## FINAL DECISION
Supported
Reason:
Image evidence supports the claimed object, affected area, and issue type. Evidence is reviewable and supports the claimed object, affected area, and issue type. Risk context: No significant user history risk detected. Review priority: normal.
##################################################


========================================

##################################################
PACKAGE CLAIM

## CLAIM
Object : package
User ID : user_015
Claim ID : row_14

## CLAIM AGENT
Issue : crushed_packaging
Part : package_corner
Severity : medium
Confidence : 0.95

## VISION AGENT
Issue : crushed_packaging
Part : package_corner
Images Used : img_1
Quality : none
Confidence : 0.95

## EVIDENCE AGENT
Evidence Met: True
Reviewable : reviewable
Status : supported
Supporting Images : img_1

## RISK AGENT
Risk Score : 0.00
Risk Flags : none

## DECISION AGENT
Claim Status: supported
Severity : medium
Supporting Images : img_1
Decision Risk Flags : none

## FINAL DECISION
Supported
Reason:
Image evidence supports the claimed object, affected area, and issue type. Evidence is reviewable and supports the claimed object, affected area, and issue type. Risk context: No significant user history risk detected. Review priority: normal.
##################################################



```
- Demo executed successfully: True

# Submission Generation and Schema Validation

```
total claims processed: 44
supported count: 32
contradicted count: 0
not_enough_information count: 12
saved output: C:\Users\piyus\OneDrive\Desktop\HackeRank Hackathon\hackerrank-orchestrate-june26\dataset\output.csv


```
```
submission_validation_report.md


```
# Overall System Health

PASS