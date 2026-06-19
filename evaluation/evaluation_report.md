# Evaluation Report

## 1. System Overview

This system is built as a modular claim review pipeline with five core agents plus a review orchestration layer.

- **ClaimAgent**: Extracts structured claim metadata from the incoming claim text. It identifies the claimed object type, issue type, affected part, and severity, providing the downstream vision and evidence agents with claim-guided context.
- **VisionAgent**: Analyzes submitted images and reconciles them with claim guidance. It detects whether damage is visible, infers the primary issue type and affected part, assesses image quality, and computes authenticity-related risk signals for potential manipulation or non-original image evidence.
- **EvidenceAgent**: Validates visual evidence against the extracted claim. It compares object alignment, issue alignment, and affected area alignment, and determines whether the submitted evidence meets reviewability and evidence standard thresholds.
- **RiskAgent**: Assesses broader review risk based on user history, suspicious flags, and submitted evidence context. It produces normalized risk flags and a risk score that influence review priority without changing the claim status.
- **DecisionAgent**: Fuses ClaimAgent, VisionAgent, EvidenceAgent, and RiskAgent outputs into a final claim decision. It selects the final issue type, part, severity, claim status, confidence, and review priority.
- **ReviewPipeline**: Orchestrates the agent flow, validating each claim, executing the agents in sequence, handling failures, and producing final review results for the dataset.

## 2. Operational Analysis

- **Approximate number of model/agent calls for sample dataset**
  - ClaimAgent: 20 calls
  - VisionAgent: 20 calls
  - EvidenceAgent: 20 calls
  - RiskAgent: 20 calls
  - DecisionAgent: 20 calls
  - ReviewPipeline orchestration: 20 pipeline executions

- **Approximate number of model/agent calls for claims.csv**
  - ClaimAgent: 100+ calls (equal to the number of rows in `claims.csv`)
  - VisionAgent: 100+ calls
  - EvidenceAgent: 100+ calls
  - RiskAgent: 100+ calls
  - DecisionAgent: 100+ calls
  - ReviewPipeline orchestration: 100+ pipeline executions

- **Images processed**
  - Sample dataset: roughly 20 submitted images or image sets, depending on claim batch size.
  - `claims.csv`: roughly 100 image submissions, potentially more if multiple image paths are attached per claim.

- **Runtime estimate**
  - Local image processing and deterministic rule-based analysis typically execute in under 1 second per claim for small image sets.
  - Full dataset runtime is approximately 1–3 minutes for 100 claims on a typical local machine, depending on image decoding and disk I/O.

- **Retry strategy**
  - Agents are designed with retry support in the base pipeline. Recoverable failures are retried up to the configured limit before the claim is marked as failed.
  - The ReviewPipeline handles transient errors by attempting execution retries on agent-level failures, preserving robustness for flaky image reads or temporary resource issues.

- **Caching strategy**
  - The current solution relies primarily on deterministic local processing and does not implement a dedicated cache layer.
  - Because the same image paths may be evaluated repeatedly, future caching can memoize image quality, consistency, and authenticity results by image hash or file path.

- **Batching strategy**
  - The pipeline processes claims sequentially in the current implementation.
  - Batching is limited to claim-by-claim orchestration, with each claim executing the full agent chain independently.
  - This approach preserves deterministic behavior for mixed datasets and avoids state leakage across claims.

## 3. Cost Analysis

- External API calls: 0
- Estimated API cost: $0

### Assumptions

- All processing is performed locally using OpenCV and deterministic Python logic.
- No cloud model or paid external inference API is invoked.
- Cost is limited to local compute and storage already available on the host machine.

## 4. Latency Analysis

- **Single claim latency**
  - Estimated at under 1 second for a typical claim with one or a few submitted images.
  - Primary latency sources are image file reads, OpenCV quality analysis, and lightweight rule evaluation.

- **Full dataset latency**
  - Estimated at 1–3 minutes for 100 claims on a standard local laptop or desktop.
  - If the dataset contains larger image volumes or additional preprocessing, runtime may grow proportionally.

## 5. Evaluation Results

- Issue Accuracy: 16/20
- Part Accuracy: 19/20

These values reflect current measured claim-level performance on the provided validation set.

## 6. Future Improvements

- **Vision grounding**: Integrate a stronger visual grounding component to better align image pixels with claimed object parts and issue descriptions.
- **Authenticity detection improvements**: Enhance manipulation and non-original image detection with richer image metadata, content-based forensics, and multi-frame consistency checks.
- **Better contradiction reasoning**: Improve evidence contradiction handling by reasoning about partial matches, multiple conflicting signals, and soft alignment confidence rather than strict binary contradiction.
