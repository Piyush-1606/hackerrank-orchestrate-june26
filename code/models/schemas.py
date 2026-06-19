from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


class ObjectType(StrEnum):
    CAR = "car"
    LAPTOP = "laptop"
    PACKAGE = "package"


class ClaimStatus(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    NOT_ENOUGH_INFORMATION = "not_enough_information"


class IssueType(StrEnum):
    DENT = "dent"
    SCRATCH = "scratch"
    CRACK = "crack"
    GLASS_SHATTER = "glass_shatter"
    BROKEN_PART = "broken_part"
    MISSING_PART = "missing_part"
    TORN_PACKAGING = "torn_packaging"
    CRUSHED_PACKAGING = "crushed_packaging"
    WATER_DAMAGE = "water_damage"
    STAIN = "stain"
    NONE = "none"
    UNKNOWN = "unknown"


class Severity(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class ImageQuality(StrEnum):
    GOOD = "good"
    PARTIAL = "partial"
    POOR = "poor"
    UNUSABLE = "unusable"


class AlignmentStatus(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    UNKNOWN = "unknown"


class Reviewability(StrEnum):
    REVIEWABLE = "reviewable"
    PARTIALLY_REVIEWABLE = "partially_reviewable"
    NOT_REVIEWABLE = "not_reviewable"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class ClaimBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class ClaimInput(ClaimBaseModel):
    claim_id: str | None = Field(
        default=None,
        min_length=1,
        description="Optional unique claim identifier. Use row index when the dataset does not provide one.",
    )
    user_id: str = Field(..., min_length=1, description="User or policyholder identifier.")
    image_paths: list[str] = Field(
        default_factory=list,
        description="Submitted image paths or URIs. CSV semicolon-delimited strings are accepted.",
    )
    claim_text: str = Field(
        ...,
        min_length=1,
        validation_alias=AliasChoices("claim_text", "user_claim"),
        description="Raw user claim description.",
    )
    object_type: ObjectType = Field(
        ...,
        validation_alias=AliasChoices("object_type", "claim_object"),
        description="Object category involved in the claim.",
    )
    user_history: str | None = Field(
        default=None,
        description="Optional user claim history or risk context.",
    )
    evidence_requirements: list[str] = Field(
        default_factory=list,
        description="Evidence requirements relevant to the claimed object or issue.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional structured metadata supplied with the claim.",
    )

    @field_validator("image_paths", mode="before")
    @classmethod
    def parse_image_paths(cls, value: str | list[str] | None) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [path.strip() for path in value.split(";") if path.strip()]
        return value

    @field_validator("image_paths")
    @classmethod
    def validate_image_paths(cls, value: list[str]) -> list[str]:
        if any(not path.strip() for path in value):
            raise ValueError("image_paths cannot contain empty strings")
        return value


class ClaimExtractionResult(ClaimBaseModel):
    claim_id: str | None = Field(default=None, description="Claim identifier, when available.")
    user_id: str | None = Field(default=None, description="User or policyholder identifier.")
    object_type: ObjectType = Field(..., description="Extracted claimed object category.")
    claimed_issue_types: list[IssueType] = Field(
        default_factory=list,
        description="Issue types explicitly or implicitly claimed by the user.",
    )
    claimed_severity: Severity = Field(
        default=Severity.UNKNOWN,
        description="Severity claimed or implied by the user.",
    )
    affected_area: str | None = Field(
        default=None,
        description="Claimed affected area, such as rear bumper, laptop screen, or package corner.",
    )
    incident_summary: str = Field(
        default="",
        description="Concise normalized summary of the alleged incident.",
    )
    evidence_requirements: list[str] = Field(
        default_factory=list,
        description="Evidence required to review the claim.",
    )
    prompt_injection_detected: bool = Field(
        default=False,
        description="Whether the claim text contains adversarial instructions.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in extraction quality.",
    )

    @field_validator("claimed_issue_types")
    @classmethod
    def deduplicate_issue_types(cls, value: list[IssueType]) -> list[IssueType]:
        return list(dict.fromkeys(value))


class ImageAnalysisResult(ClaimBaseModel):
    claim_id: str | None = Field(default=None, description="Claim identifier, when available.")
    image_id: str = Field(..., min_length=1, description="Identifier for the analyzed image.")
    image_path: str | None = Field(default=None, description="Path or URI of analyzed image.")
    object_type_detected: ObjectType | None = Field(
        default=None,
        description="Object category visually detected in the image.",
    )
    visible_issue_types: list[IssueType] = Field(
        default_factory=list,
        description="Damage or issue types visible in the image.",
    )
    visible_severity: Severity = Field(
        default=Severity.UNKNOWN,
        description="Estimated visual severity.",
    )
    affected_areas: list[str] = Field(
        default_factory=list,
        description="Visible affected areas in the image.",
    )
    image_quality: ImageQuality = Field(
        default=ImageQuality.PARTIAL,
        description="Image quality and usefulness for claim review.",
    )
    valid_image: bool = Field(
        default=True,
        description="Whether the image is relevant and usable for claim verification.",
    )
    visual_summary: str = Field(
        default="",
        description="Concise summary of visible evidence.",
    )
    prompt_injection_detected: bool = Field(
        default=False,
        description="Whether the image contains adversarial text instructions.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in visual analysis.",
    )

    @field_validator("affected_areas")
    @classmethod
    def deduplicate_affected_areas(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(area for area in value if area))

    @field_validator("visible_issue_types")
    @classmethod
    def deduplicate_visible_issue_types(cls, value: list[IssueType]) -> list[IssueType]:
        return list(dict.fromkeys(value))

    @model_validator(mode="after")
    def normalize_unusable_or_no_damage(self) -> ImageAnalysisResult:
        if self.image_quality == ImageQuality.UNUSABLE:
            self.valid_image = False
        if IssueType.NONE in self.visible_issue_types:
            self.visible_issue_types = [IssueType.NONE]
            self.visible_severity = Severity.NONE
        return self


class EvidenceValidationResult(ClaimBaseModel):
    claim_id: str | None = Field(default=None, description="Claim identifier, when available.")
    evidence_standard_met: bool = Field(
        ...,
        description="Whether submitted evidence is sufficient for the claim type.",
    )
    evidence_standard_met_reason: str = Field(
        ...,
        min_length=1,
        description="Short explanation of whether evidence requirements are met.",
    )
    reviewability: Reviewability = Field(
        ...,
        description="Whether available evidence is sufficient for review.",
    )
    object_alignment: AlignmentStatus = Field(
        default=AlignmentStatus.UNKNOWN,
        description="Alignment between claimed and observed object.",
    )
    issue_alignment: AlignmentStatus = Field(
        default=AlignmentStatus.UNKNOWN,
        description="Alignment between claimed and observed issue type.",
    )
    area_alignment: AlignmentStatus = Field(
        default=AlignmentStatus.UNKNOWN,
        description="Alignment between claimed and observed affected area.",
    )
    severity_alignment: AlignmentStatus = Field(
        default=AlignmentStatus.UNKNOWN,
        description="Alignment between claimed and observed severity.",
    )
    missing_evidence: list[str] = Field(
        default_factory=list,
        description="Evidence gaps that limit reviewability.",
    )
    supporting_image_ids: list[str] = Field(
        default_factory=list,
        description="Image identifiers supporting the evidence assessment.",
    )
    recommended_status: ClaimStatus = Field(
        ...,
        description="Evidence-only recommended claim status.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in evidence validation.",
    )

    @model_validator(mode="after")
    def validate_reviewability_consistency(self) -> EvidenceValidationResult:
        if self.reviewability == Reviewability.NOT_REVIEWABLE and self.evidence_standard_met:
            raise ValueError("not_reviewable evidence cannot have evidence_standard_met=true")
        if self.recommended_status == ClaimStatus.NOT_ENOUGH_INFORMATION and self.evidence_standard_met:
            raise ValueError("not_enough_information should not have evidence_standard_met=true")
        return self


class RiskAssessmentResult(ClaimBaseModel):
    claim_id: str | None = Field(default=None, description="Claim identifier, when available.")
    risk_level: RiskLevel = Field(
        default=RiskLevel.UNKNOWN,
        description="Risk level based on user history and claim pattern.",
    )
    risk_flags: list[str] = Field(
        default_factory=list,
        description="Leaderboard-friendly risk flags. Use ['none'] when no risk is detected.",
    )
    risk_factors: list[str] = Field(
        default_factory=list,
        description="Factors increasing claim risk.",
    )
    mitigating_factors: list[str] = Field(
        default_factory=list,
        description="Factors reducing claim risk.",
    )
    should_escalate: bool = Field(
        default=False,
        description="Whether the claim should be escalated for manual review.",
    )
    should_reduce_confidence: bool = Field(
        default=False,
        description="Whether risk context should reduce final confidence.",
    )
    should_change_claim_status: bool = Field(
        default=False,
        description="Risk should generally not change claim status directly.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in risk assessment.",
    )

    @field_validator("risk_flags")
    @classmethod
    def normalize_risk_flags(cls, value: list[str]) -> list[str]:
        flags = list(dict.fromkeys(flag.strip() for flag in value if flag.strip()))
        return flags or ["none"]


class ClaimDecisionResult(ClaimBaseModel):
    claim_id: str | None = Field(default=None, description="Claim identifier, when available.")
    claim_status: ClaimStatus = Field(..., description="Final claim status.")
    claim_status_justification: str = Field(
        ...,
        min_length=1,
        description="Concise judge-facing justification for the final status.",
    )
    issue_type: IssueType = Field(..., description="Primary issue type for leaderboard output.")
    object_part: str = Field(
        default="unknown",
        min_length=1,
        description="Normalized affected object part, such as rear_bumper or laptop_screen.",
    )
    severity: Severity = Field(..., description="Final severity classification.")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Final decision confidence.",
    )
    evidence_standard_met: bool = Field(
        ...,
        description="Whether evidence requirements were met.",
    )
    evidence_standard_met_reason: str = Field(
        ...,
        min_length=1,
        description="Short reason explaining evidence sufficiency.",
    )
    risk_flags: list[str] = Field(
        default_factory=lambda: ["none"],
        description="Risk flags to include in the final output.",
    )
    supporting_image_ids: list[str] = Field(
        default_factory=list,
        description="Image identifiers supporting the decision.",
    )
    valid_image: bool = Field(
        default=True,
        description="Whether at least one submitted image was relevant and usable.",
    )
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="Evidence supporting the final decision.",
    )
    contradicting_evidence: list[str] = Field(
        default_factory=list,
        description="Evidence contradicting the claim.",
    )
    missing_evidence: list[str] = Field(
        default_factory=list,
        description="Evidence missing from the submission.",
    )
    review_flags: list[str] = Field(
        default_factory=list,
        description="Operational flags such as prompt injection or poor image quality.",
    )

    @field_validator("risk_flags", "supporting_image_ids", "review_flags")
    @classmethod
    def deduplicate_strings(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(item.strip() for item in value if item.strip()))

    @model_validator(mode="after")
    def validate_decision_consistency(self) -> ClaimDecisionResult:
        if self.issue_type == IssueType.NONE and self.severity != Severity.NONE:
            raise ValueError("issue_type=none requires severity=none")
        if self.severity == Severity.NONE and self.issue_type not in {IssueType.NONE, IssueType.UNKNOWN}:
            raise ValueError("severity=none requires issue_type none or unknown")
        if self.claim_status == ClaimStatus.NOT_ENOUGH_INFORMATION and self.evidence_standard_met:
            raise ValueError("not_enough_information should not have evidence_standard_met=true")
        return self


class FinalOutputRow(ClaimBaseModel):
    user_id: str = Field(..., min_length=1, description="User or policyholder identifier.")
    image_paths: str = Field(
        ...,
        min_length=1,
        description="Original semicolon-delimited image path string for the claim.",
    )
    user_claim: str = Field(..., min_length=1, description="Original raw user claim text.")
    claim_object: ObjectType = Field(..., description="Claimed object category.")
    evidence_standard_met: bool = Field(
        ...,
        description="Whether evidence requirements were met.",
    )
    evidence_standard_met_reason: str = Field(
        ...,
        min_length=1,
        description="Short evidence sufficiency rationale.",
    )
    risk_flags: str = Field(
        default="none",
        min_length=1,
        description="Delimited risk flags for CSV output.",
    )
    issue_type: IssueType = Field(..., description="Primary predicted issue type.")
    object_part: str = Field(
        default="unknown",
        min_length=1,
        description="Primary affected object part.",
    )
    claim_status: ClaimStatus = Field(..., description="Final predicted claim status.")
    claim_status_justification: str = Field(
        ...,
        min_length=1,
        description="Short final status justification.",
    )
    supporting_image_ids: str = Field(
        default="",
        description="Delimited image IDs that support the decision.",
    )
    valid_image: bool = Field(
        ...,
        description="Whether at least one submitted image was valid for review.",
    )
    severity: Severity = Field(..., description="Final predicted severity.")

    @classmethod
    def from_decision(cls, claim: ClaimInput, decision: ClaimDecisionResult) -> FinalOutputRow:
        return cls(
            user_id=claim.user_id,
            image_paths=";".join(claim.image_paths),
            user_claim=claim.claim_text,
            claim_object=claim.object_type,
            evidence_standard_met=decision.evidence_standard_met,
            evidence_standard_met_reason=decision.evidence_standard_met_reason,
            risk_flags=";".join(decision.risk_flags or ["none"]),
            issue_type=decision.issue_type,
            object_part=decision.object_part,
            claim_status=decision.claim_status,
            claim_status_justification=decision.claim_status_justification,
            supporting_image_ids=";".join(decision.supporting_image_ids),
            valid_image=decision.valid_image,
            severity=decision.severity,
        )

from pydantic import BaseModel, Field


class UserHistory(BaseModel):
    user_id: str
    past_claim_count: int
    accept_claim: int
    manual_review_claim: int
    rejected_claim: int
    last_90_days_claim_count: int
    history_flags: str
    history_summary: str


class RiskAssessmentResult(BaseModel):
    user_id: str
    risk_flags: list[str] = Field(default_factory=list)
    risk_score: float
    justification: str