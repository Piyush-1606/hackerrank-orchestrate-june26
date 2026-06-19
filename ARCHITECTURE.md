# Multi-Modal Evidence Review System

## Pipeline

```text
ClaimInput
    |
    v
ClaimAgent
    |
    v
VisionAgent
    |-- Image Quality Analysis
    |-- Blur Detection
    |-- Low-Light Detection
    |-- Cropped Detection
    |-- Image Consistency Analysis
    |-- Authenticity Analysis
    |
    v
EvidenceAgent
    |
    v
RiskAgent
    |
    v
DecisionAgent
    |
    v
Output.csv
```

## Agent Responsibilities

### ClaimAgent

- Extract issue
- Extract object part
- Extract severity

### VisionAgent

- Image validation
- Quality analysis
- Consistency analysis
- Authenticity analysis
- Visual evidence extraction

### EvidenceAgent

- Alignment checks
- Evidence sufficiency
- Contradiction detection

### RiskAgent

- User history analysis
- Fraud risk signals

### DecisionAgent

- Final claim decision
- Risk aggregation
- Justification generation

## Data Flow

The system starts with a `ClaimInput` that contains the claim text, object type, image paths, and related metadata. `ClaimAgent` first extracts structured claim details such as issue type, affected object part, and severity.

Those extracted claim cues are passed into `VisionAgent` along with image references. `VisionAgent` validates submitted images, evaluates image quality, checks visual consistency across multiple images, and detects potential authenticity issues. It returns a structured `VisionResult` containing detected issue candidates, quality flags, risk flags, and supporting image identifiers.

`EvidenceAgent` consumes the claim extraction output and vision findings to determine whether the images align with the claimed object, issue, and affected area. It also assesses whether the evidence is sufficient for review and flags contradictions between the claim and the visual evidence.

`RiskAgent` analyzes user and submission context for fraud-related signals and assigns normalized risk flags and a risk score. This risk context is combined with vision-based authenticity signals.

Finally, `DecisionAgent` fuses outputs from ClaimAgent, VisionAgent, EvidenceAgent, and RiskAgent to produce a final claim decision. The decision includes claim status, issue type, object part, severity, confidence, review priority, and a justification narrative. The complete result is assembled into the final output dataset such as `output.csv`.
