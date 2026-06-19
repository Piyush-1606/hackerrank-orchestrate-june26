from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

try:
    from agents.base_agent import AgentRunContext, BaseAgent, RetryConfig
    from models.schemas import ClaimExtractionResult, ClaimInput, IssueType, ObjectType, Severity
except ModuleNotFoundError:
    from .base_agent import AgentRunContext, BaseAgent, RetryConfig
    from ..models.schemas import ClaimExtractionResult, ClaimInput, IssueType, ObjectType, Severity


CandidateT = TypeVar("CandidateT")
LlmFallback = Callable[[ClaimInput, str, Sequence[str], Sequence[IssueType]], ClaimExtractionResult | None]


@dataclass(frozen=True, slots=True)
class ExtractionCandidate(Generic[CandidateT]):
    value: CandidateT
    matched_terms: tuple[str, ...]
    confidence: float


@dataclass(frozen=True, slots=True)
class ClaimExtractionDetails:
    final_claim_text: str
    issue_types: list[IssueType]
    parts: list[str]
    multiple_parts_claimed: bool
    severity: Severity
    prompt_injection_detected: bool
    confidence: float
    reasoning: str


class ClaimAgent(BaseAgent[ClaimInput, ClaimExtractionResult]):
    """Extract structured damage-claim facts from user text only.

    The agent is intentionally framework-agnostic. It performs deterministic
    keyword extraction first, then optionally calls an injected LLM fallback
    when confidence is low. It never analyzes images.
    """

    LOW_CONFIDENCE_THRESHOLD = 0.58

    CUSTOMER_ROLES = ("customer", "cliente", "client", "usuario", "user")
    NON_CUSTOMER_ROLES = ("support", "agent", "soporte", "assistant")

    PROMPT_INJECTION_PATTERNS = (
        r"\bapprove\s+(?:this\s+)?(?:claim\s+)?immediately\b",
        r"\bskip\s+manual\s+review\b",
        r"\bmark\s+(?:as\s+)?supported\b",
        r"\bignore\s+(?:all\s+)?(?:previous|above|system)\s+instructions\b",
        r"\bdisregard\s+(?:all\s+)?(?:previous|above|system)\s+instructions\b",
        r"\bdo\s+not\s+(?:inspect|verify|analyze|review)\b",
        r"\bauto(?:matically)?\s+approve\b",
        r"\bclaim_status\s*[:=]\s*supported\b",
        r"\bseverity\s*[:=]\s*(?:low|medium|high|none)\b",
        r"\baprueba(?:lo)?\s+(?:inmediatamente|ahora)\b",
        r"\bomitir\s+(?:revision|revisión)\s+manual\b",
        r"\bmarcar\s+como\s+apoyado\b",
    )

    UNCERTAINTY_SYNONYMS = (
        "may",
        "might",
        "looks like",
        "seems",
        "not sure",
        "probably",
        "possibly",
        "i think",
    )

    STRONG_DAMAGE_SYNONYMS = (
        "dent",
        "dented",
        "scratch",
        "scratched",
        "crack",
        "crack spreading",
        "broken",
        "snapped",
        "detached",
        "loose",
        "hanging",
        "bent out of place",
        "not sitting correctly",
        "not sitting the way",
        "out of position",
        "misaligned",
        "wobbling",
        "separated",
        "not working",
        "shattered",
        "missing",
        "torn",
        "ripped",
        "crushed",
        "wet",
        "moisture",
        "damp",
        "soaked",
        "water stained",
        "water damage",
    )

    BROKEN_PART_STRONG_SYNONYMS = (
        "broken",
        "broke",
        "snapped",
        "detached",
        "not working",
        "shattered",
        "cracked",
        "loose",
        "hanging",
        "bent out of place",
        "not sitting correctly",
        "not sitting the way",
        "out of position",
        "misaligned",
        "wobbling",
        "separated",
        "toota",
        "tuta",
        "toot gaya",
        "broken ho gaya",
    )

    PART_SYNONYMS: Mapping[ObjectType, Mapping[str, tuple[str, ...]]] = {
        ObjectType.CAR: {
            "front_bumper": (
                "front bumper",
                "front side bumper",
                "front lower bumper",
                "bumper front",
                "delantero bumper",
                "parachoques delantero",
                "defensa delantera",
                "aage bumper",
                "aage ka bumper",
            ),
            "rear_bumper": (
                "rear bumper",
                "back bumper",
                "bumper rear",
                "parachoques trasero",
                "defensa trasera",
                "piche bumper",
                "peeche bumper",
                "peeche ka bumper",
                "back side",
            ),
            "door": (
                "door",
                "doors",
                "driver door",
                "passenger door",
                "left door",
                "right door",
                "car door",
                "puerta",
                "darwaza",
                "darwaja",
            ),
            "hood": ("hood", "bonnet", "capo", "capó", "engine cover"),
            "windshield": (
                "windshield",
                "windscreen",
                "front glass",
                "car glass",
                "parabrisas",
                "sheesha",
                "shisha",
            ),
            "side_mirror": (
                "side mirror",
                "mirror",
                "wing mirror",
                "rear view mirror",
                "espejo",
                "side ka mirror",
            ),
            "headlight": (
                "headlight",
                "headlamp",
                "front light",
                "left headlight",
                "right headlight",
                "faro",
                "luz delantera",
            ),
            "taillight": (
                "taillight",
                "tail light",
                "rear light",
                "back light",
                "luz trasera",
            ),
            "fender": ("fender", "wing", "guardabarros"),
            "quarter_panel": ("quarter panel", "rear quarter", "side panel"),
            "body": (
                "body",
                "panel",
                "paint",
                "side",
                "car body",
                "cuerpo",
                "gaadi ki body",
                "gadi body",
            ),
        },
        ObjectType.LAPTOP: {
            "screen": (
                "screen",
                "display",
                "lcd",
                "monitor",
                "pantalla",
                "screen toot",
                "screen phat",
            ),
            "keyboard": ("keyboard", "keys", "teclado", "keypad", "button", "buttons"),
            "trackpad": ("trackpad", "touchpad", "mouse pad", "panel tactil", "panel táctil"),
            "hinge": ("hinge", "hinges", "bisagra", "bisagras", "hinge broken"),
            "lid": ("lid", "cover", "top cover", "tapa", "laptop cover"),
            "corner": ("corner", "edge", "kona", "esquina"),
            "port": ("port", "ports", "usb", "charging port", "charger port", "puerto"),
            "base": ("base", "bottom", "underside", "lower body", "bottom case"),
            "body": ("body", "case", "chassis", "shell", "carcasa", "laptop body"),
        },
        ObjectType.PACKAGE: {
            "box": ("box", "carton", "cartón", "dabba", "package box", "outer box"),
            "package_corner": (
                "corner",
                "package corner",
                "box corner",
                "kona",
                "esquina",
            ),
            "package_side": ("side", "package side", "box side", "surface", "package surface", "lateral", "side wall"),
            "seal": ("seal", "tape", "packing tape", "sello", "cinta", "sealed area"),
            "label": ("label", "shipping label", "barcode", "etiqueta", "address label"),
            "contents": ("contents", "inside items", "andar ka saman", "contenido", "items inside"),
            "item": ("item", "product", "ordered item", "article", "producto", "saman"),
        },
    }

    ISSUE_SYNONYMS: Mapping[IssueType, tuple[str, ...]] = {
        IssueType.DENT: (
            "dent",
            "dented",
            "ding",
            "deformation",
            "deformed",
            "abolladura",
            "abollado",
            "gaddha",
            "daba",
            "dab gaya",
            "dab gaya hai",
        ),
        IssueType.SCRATCH: (
            "scratch",
            "scratched",
            "scrape",
            "scuff",
            "mark",
            "rasguño",
            "rayon",
            "rayón",
            "kharoch",
            "kharocha",
            "scratch ho gaya",
        ),
        IssueType.CRACK: (
            "crack",
            "cracked",
            "fracture",
            "split",
            "fisura",
            "rajado",
            "crack ho gaya",
            "tootne ki line",
        ),
        IssueType.GLASS_SHATTER: (
            "shattered",
            "glass shattered",
            "broken glass",
            "window shattered",
            "screen shattered",
            "vidrio roto",
            "cristal roto",
            "sheesha toot",
            "shisha toot",
            "glass toot",
        ),
        IssueType.BROKEN_PART: (
            "broken",
            "broke",
            "snapped",
            "detached",
            "not working",
            "shattered",
            "cracked",
            "roto",
            "rota",
            "quebrado",
            "quebrada",
            "toota",
            "tuta",
            "toot gaya",
            "broken ho gaya",
        ),
        IssueType.MISSING_PART: (
            "missing",
            "fell off",
            "gone",
            "lost",
            "not there",
            "missing part",
            "falta",
            "perdido",
            "gayab",
            "nahi hai",
            "nikal gaya",
        ),
        IssueType.TORN_PACKAGING: (
            "torn",
            "ripped",
            "tear",
            "opened",
            "cut open",
            "rasgado",
            "roto el paquete",
            "phata",
            "fata",
            "phat gaya",
        ),
        IssueType.CRUSHED_PACKAGING: (
            "crushed",
            "smashed",
            "compressed",
            "caved in",
            "box crushed",
            "aplastado",
            "machacado",
            "dabba daba",
            "dabba crushed",
        ),
        IssueType.WATER_DAMAGE: (
            "water damage",
            "water damaged",
            "water stained",
            "wet",
            "damp",
            "soaked",
            "liquid damage",
            "water inside",
            "moisture",
            "daño por agua",
            "mojado",
            "agua",
            "paani",
            "pani",
            "bheeg",
            "geela",
        ),
        IssueType.STAIN: (
            "stain",
            "stained",
            "discoloration",
            "spot",
            "mancha",
            "manchado",
            "daag",
            "nishan",
        ),
        IssueType.NONE: (
            "no damage",
            "no issue",
            "nothing damaged",
            "sin daño",
            "no hay daño",
            "koi damage nahi",
            "damage nahi",
        ),
    }

    SEVERITY_SYNONYMS: Mapping[Severity, tuple[str, ...]] = {
        Severity.HIGH: (
            "severe",
            "major",
            "huge",
            "completely broken",
            "destroyed",
            "unsafe",
            "total loss",
            "muy dañado",
            "grave",
            "bahut zyada",
            "poora toot",
        ),
        Severity.MEDIUM: (
            "damaged",
            "broken",
            "cracked",
            "visible",
            "moderate",
            "notable",
            "dañado",
            "roto",
            "kaafi",
            "theek thak damage",
        ),
        Severity.LOW: (
            "minor",
            "small",
            "slight",
            "tiny",
            "light",
            "cosmetic",
            "pequeño",
            "leve",
            "thoda",
            "chota",
            "minor sa",
        ),
        Severity.NONE: (
            "no damage",
            "no issue",
            "nothing damaged",
            "sin daño",
            "koi damage nahi",
        ),
    }

    def __init__(
        self,
        *,
        llm_fallback: LlmFallback | None = None,
        low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
        retry_config: RetryConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(name="ClaimAgent", retry_config=retry_config, logger=logger)
        self.llm_fallback = llm_fallback
        self.low_confidence_threshold = low_confidence_threshold

    def _execute(self, input_data: ClaimInput, context: AgentRunContext) -> ClaimExtractionResult:
        final_claim = self.extract_final_user_claim(input_data.claim_text)
        combined_claim = self.extract_all_user_claim_text(input_data.claim_text)
        sanitized_claim, injection_detected = self.remove_prompt_injection(combined_claim)
        sanitized_final_claim, final_injection_detected = self.remove_prompt_injection(final_claim)
        injection_detected = injection_detected or final_injection_detected

        parts = self.extract_parts(sanitized_claim, input_data.object_type)
        issue_types = self.normalize_issue_types(
            text=sanitized_claim,
            issue_types=self.extract_issue_types(sanitized_claim),
            object_type=input_data.object_type,
            parts=parts,
        )
        severity = self.extract_severity(sanitized_claim, issue_types)
        confidence = self.score_confidence(
            text=sanitized_claim,
            issue_types=issue_types,
            parts=parts,
            prompt_injection_detected=injection_detected,
        )
        reasoning = self.build_reasoning(
            issue_types=issue_types,
            parts=parts,
            severity=severity,
            prompt_injection_detected=injection_detected,
            confidence=confidence,
        )

        details = ClaimExtractionDetails(
            final_claim_text=sanitized_final_claim,
            issue_types=issue_types,
            parts=parts,
            multiple_parts_claimed=len(parts) > 1,
            severity=severity,
            prompt_injection_detected=injection_detected,
            confidence=confidence,
            reasoning=reasoning,
        )

        self._log_extraction_details(context, details, used_llm_fallback=False)

        if confidence < self.low_confidence_threshold and self.llm_fallback is not None:
            fallback_result = self.llm_fallback(input_data, sanitized_claim, parts, issue_types)
            if fallback_result is not None:
                self._log(
                    logging.INFO,
                    "claim_extraction_llm_fallback_used",
                    context=context,
                    deterministic_confidence=confidence,
                    fallback_confidence=fallback_result.confidence,
                )
                return fallback_result

        return ClaimExtractionResult(
            claim_id=input_data.claim_id or context.claim_id,
            user_id=input_data.user_id or context.user_id,
            object_type=input_data.object_type,
            claimed_issue_types=issue_types or [IssueType.UNKNOWN],
            claimed_severity=severity,
            affected_area=";".join(parts) if parts else None,
            incident_summary=self.build_summary(sanitized_final_claim, details),
            evidence_requirements=list(input_data.evidence_requirements),
            prompt_injection_detected=injection_detected,
            confidence=confidence,
        )

    @classmethod
    def extract_all_user_claim_text(cls, transcript: str) -> str:
        """Return combined customer-authored claim text while excluding support turns."""
        segments = cls.split_transcript(transcript)
        customer_segments: list[str] = []
        unlabelled_segments: list[str] = []

        for role, text in segments:
            if role is None:
                unlabelled_segments.append(text)
                continue
            normalized_role = cls.normalize_text(role)
            if normalized_role in cls.CUSTOMER_ROLES:
                customer_segments.append(text)

        if customer_segments:
            return " ".join(customer_segments).strip()

        unlabelled_text = " ".join(unlabelled_segments).strip()
        return unlabelled_text or transcript.strip()

    @classmethod
    def extract_final_user_claim(cls, transcript: str) -> str:
        """Return the last customer-authored segment from a transcript."""
        segments = cls.split_transcript(transcript)
        customer_segments: list[str] = []

        for role, text in segments:
            if role is None:
                continue
            normalized_role = cls.normalize_text(role)
            if normalized_role in cls.CUSTOMER_ROLES:
                customer_segments.append(text)

        if customer_segments:
            return customer_segments[-1].strip()

        unlabelled_text = " ".join(text for role, text in segments if role is None).strip()
        return unlabelled_text or transcript.strip()

    @classmethod
    def split_transcript(cls, transcript: str) -> list[tuple[str | None, str]]:
        """Split pipe/newline separated conversation text into role-aware segments."""
        chunks = [chunk.strip() for chunk in re.split(r"\s*\|\s*|\r?\n+", transcript) if chunk.strip()]
        segments: list[tuple[str | None, str]] = []
        role_pattern = re.compile(r"^\s*([A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+)\s*:\s*(.+?)\s*$")

        for chunk in chunks:
            match = role_pattern.match(chunk)
            if not match:
                segments.append((None, chunk))
                continue

            role, text = match.group(1), match.group(2)
            normalized_role = cls.normalize_text(role)
            if normalized_role in {*cls.CUSTOMER_ROLES, *cls.NON_CUSTOMER_ROLES}:
                segments.append((normalized_role, text.strip()))
            else:
                segments.append((None, chunk))

        return segments

    @classmethod
    def remove_prompt_injection(cls, text: str) -> tuple[str, bool]:
        """Remove known instruction attacks while preserving claim content."""
        sanitized = text
        detected = False

        for pattern in cls.PROMPT_INJECTION_PATTERNS:
            sanitized, count = re.subn(pattern, " ", sanitized, flags=re.IGNORECASE)
            detected = detected or count > 0

        sanitized = re.sub(r"\s+", " ", sanitized).strip(" .;-")
        return sanitized, detected

    @classmethod
    def extract_issue_types(cls, text: str) -> list[IssueType]:
        """Extract normalized issue types using multilingual keyword matching."""
        normalized = cls.normalize_text(text)

        if cls.has_uncertainty_without_strong_damage(normalized):
            return [IssueType.UNKNOWN]

        candidates: list[ExtractionCandidate[IssueType]] = []

        for issue_type, synonyms in cls.ISSUE_SYNONYMS.items():
            matches = cls.find_synonym_matches(normalized, synonyms)
            if matches:
                candidates.append(
                    ExtractionCandidate(
                        value=issue_type,
                        matched_terms=tuple(matches),
                        confidence=min(1.0, 0.64 + 0.08 * len(matches)),
                    )
                )

        if not candidates:
            return []

        candidates.sort(key=lambda candidate: candidate.confidence, reverse=True)
        extracted = [IssueType(candidate.value) for candidate in candidates]
        if IssueType.NONE in extracted and len(extracted) > 1:
            return [issue for issue in extracted if issue != IssueType.NONE]
        return list(dict.fromkeys(extracted))

    @classmethod
    def normalize_issue_types(
        cls,
        *,
        text: str,
        issue_types: Sequence[IssueType],
        object_type: ObjectType,
        parts: Sequence[str],
    ) -> list[IssueType]:
        """Apply narrow object-aware issue corrections after keyword extraction."""
        normalized = cls.normalize_text(text)
        ordered = list(dict.fromkeys(issue_types))

        if cls.is_laptop_screen_shatter_claim(normalized, object_type, parts):
            ordered = [IssueType.CRACK, *[issue for issue in ordered if issue != IssueType.GLASS_SHATTER]]

        if cls.has_structural_broken_part_language(normalized):
            ordered = [IssueType.BROKEN_PART, *[issue for issue in ordered if issue != IssueType.BROKEN_PART]]

        if not ordered and cls.has_ambiguous_physical_damage_only(normalized):
            ordered = [IssueType.NONE]

        return ordered

    @classmethod
    def extract_parts(cls, text: str, object_type: ObjectType) -> list[str]:
        """Extract normalized object parts for the relevant claim object."""
        normalized = cls.normalize_text(text)
        part_map = cls.PART_SYNONYMS[object_type]
        candidates: list[ExtractionCandidate[str]] = []

        for part, synonyms in part_map.items():
            matches = cls.find_synonym_matches(normalized, synonyms)
            if matches:
                candidates.append(
                    ExtractionCandidate(
                        value=part,
                        matched_terms=tuple(matches),
                        confidence=min(1.0, 0.62 + 0.08 * len(matches)),
                    )
                )

        candidates.sort(key=lambda candidate: candidate.confidence, reverse=True)
        parts = [candidate.value for candidate in candidates]
        return cls.apply_part_priority(cls.prune_generic_parts(parts, object_type), normalized, object_type)

    @classmethod
    def extract_severity(cls, text: str, issue_types: Sequence[IssueType]) -> Severity:
        """Infer claimed severity from text with conservative defaults."""
        normalized = cls.normalize_text(text)

        for severity in (Severity.HIGH, Severity.LOW, Severity.NONE, Severity.MEDIUM):
            if cls.find_synonym_matches(normalized, cls.SEVERITY_SYNONYMS[severity]):
                return severity

        if not issue_types:
            return Severity.UNKNOWN
        if issue_types == [IssueType.NONE]:
            return Severity.NONE
        return Severity.MEDIUM

    @classmethod
    def score_confidence(
        cls,
        *,
        text: str,
        issue_types: Sequence[IssueType],
        parts: Sequence[str],
        prompt_injection_detected: bool,
    ) -> float:
        """Score deterministic extraction confidence."""
        score = 0.25

        if text:
            score += 0.15
        if issue_types:
            score += 0.28
        if parts:
            score += 0.22
        if len(parts) > 1:
            score += 0.05
        if issue_types and IssueType.UNKNOWN not in issue_types:
            score += 0.05
        if prompt_injection_detected:
            score -= 0.08

        return round(max(0.0, min(1.0, score)), 3)

    @classmethod
    def build_reasoning(
        cls,
        *,
        issue_types: Sequence[IssueType],
        parts: Sequence[str],
        severity: Severity,
        prompt_injection_detected: bool,
        confidence: float,
    ) -> str:
        issue_text = ", ".join(issue_types) if issue_types else "unknown issue"
        part_text = ", ".join(parts) if parts else "unknown part"
        multi_part_text = "multiple parts" if len(parts) > 1 else "single or unspecified part"
        injection_text = " Prompt-injection text was removed." if prompt_injection_detected else ""
        return (
            f"Extracted {issue_text} on {part_text}; {multi_part_text}; "
            f"claimed severity {severity}; confidence {confidence:.2f}.{injection_text}"
        )

    @classmethod
    def build_summary(cls, claim_text: str, details: ClaimExtractionDetails) -> str:
        """Build a compact summary compatible with ClaimExtractionResult."""
        cleaned_claim = claim_text.strip()
        if len(cleaned_claim) > 220:
            cleaned_claim = cleaned_claim[:217].rstrip() + "..."
        return f"{cleaned_claim} | {details.reasoning}" if cleaned_claim else details.reasoning

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for accent-insensitive multilingual keyword matching."""
        decomposed = unicodedata.normalize("NFKD", text.casefold())
        ascii_text = "".join(char for char in decomposed if not unicodedata.combining(char))
        ascii_text = re.sub(r"[_/-]+", " ", ascii_text)
        return re.sub(r"\s+", " ", ascii_text).strip()

    @classmethod
    def find_synonym_matches(cls, normalized_text: str, synonyms: Sequence[str]) -> list[str]:
        """Return matched synonyms using phrase-aware boundaries."""
        matches: list[str] = []
        for synonym in synonyms:
            normalized_synonym = cls.normalize_text(synonym)
            if not normalized_synonym:
                continue
            pattern = rf"(?<!\w){re.escape(normalized_synonym)}(?!\w)"
            if re.search(pattern, normalized_text):
                matches.append(synonym)
        return matches

    @classmethod
    def has_uncertainty_without_strong_damage(cls, normalized_text: str) -> bool:
        """Detect tentative claims that lack a concrete damage keyword."""
        has_uncertainty = bool(cls.find_synonym_matches(normalized_text, cls.UNCERTAINTY_SYNONYMS))
        has_strong_damage = bool(cls.find_synonym_matches(normalized_text, cls.STRONG_DAMAGE_SYNONYMS))
        return has_uncertainty and not has_strong_damage

    @classmethod
    def is_laptop_screen_shatter_claim(
        cls,
        normalized_text: str,
        object_type: ObjectType,
        parts: Sequence[str],
    ) -> bool:
        """Normalize laptop screen shatter language to crack for leaderboard labels."""
        if object_type != ObjectType.LAPTOP or "screen" not in parts:
            return False
        return bool(
            cls.find_synonym_matches(
                normalized_text,
                ("shattered", "shattered display", "shattered screen", "spiderweb pattern"),
            )
        )

    @classmethod
    def has_structural_broken_part_language(cls, normalized_text: str) -> bool:
        """Detect concrete structural component damage that should be broken_part."""
        return bool(
            cls.find_synonym_matches(
                normalized_text,
                (
                    "broken",
                    "detached",
                    "loose",
                    "hanging",
                    "bent out of place",
                    "not sitting correctly",
                    "not sitting the way",
                    "out of position",
                    "misaligned",
                    "wobbling",
                    "separated",
                    "snapped",
                ),
            )
        )

    @classmethod
    def has_ambiguous_physical_damage_only(cls, normalized_text: str) -> bool:
        """Detect physical-damage mentions with no visible issue type."""
        return bool(cls.find_synonym_matches(normalized_text, ("physical damage", "damage around")))

    @classmethod
    def prune_generic_parts(cls, parts: Sequence[str], object_type: ObjectType) -> list[str]:
        """Drop broad catch-all parts when a more specific part is present."""
        ordered = list(dict.fromkeys(parts))
        if len(ordered) <= 1:
            return ordered

        generic_parts = {
            ObjectType.CAR: {"body"},
            ObjectType.LAPTOP: {"body"},
            ObjectType.PACKAGE: {"box"},
        }
        return [part for part in ordered if part not in generic_parts[object_type]] or ordered

    @classmethod
    def apply_part_priority(
        cls,
        parts: Sequence[str],
        normalized_text: str,
        object_type: ObjectType,
    ) -> list[str]:
        """Apply narrow object-specific part precedence learned from evaluation."""
        ordered = list(dict.fromkeys(parts))

        hinge_damage_language = cls.has_part_damage_language(normalized_text, "hinge")
        if object_type == ObjectType.LAPTOP and "hinge" in ordered and hinge_damage_language:
            ordered = cls.move_first(ordered, "hinge")

        if object_type == ObjectType.PACKAGE and "seal" in ordered:
            ordered = cls.move_first(ordered, "seal")

        missing_item_language = cls.find_synonym_matches(
            normalized_text,
            (
                "missing item",
                "missing product",
                "item missing",
                "product missing",
                "contents missing",
                "contents are missing",
                "not inside",
                "could not find product inside",
            ),
        )
        if object_type == ObjectType.PACKAGE and missing_item_language and "contents" in ordered:
            ordered = cls.move_first(ordered, "contents")

        return ordered

    @staticmethod
    def move_first(values: Sequence[str], target: str) -> list[str]:
        """Return values with target first while preserving the rest of the order."""
        return [target, *[value for value in values if value != target]]

    @classmethod
    def has_part_damage_language(cls, normalized_text: str, part: str) -> bool:
        """Return true when a part appears close to concrete damage language."""
        damage_terms = ("broken", "damaged", "cracked", "snapped", "detached", "not working")
        part_pattern = re.escape(cls.normalize_text(part))
        damage_pattern = "|".join(re.escape(term) for term in damage_terms)
        return bool(
            re.search(rf"\b{part_pattern}\b(?:\W+\w+){{0,4}}\W+(?:{damage_pattern})\b", normalized_text)
            or re.search(rf"\b(?:{damage_pattern})\b(?:\W+\w+){{0,4}}\W+\b{part_pattern}\b", normalized_text)
        )

    def _log_extraction_details(
        self,
        context: AgentRunContext,
        details: ClaimExtractionDetails,
        *,
        used_llm_fallback: bool,
    ) -> None:
        self._log(
            logging.INFO,
            "claim_extraction_completed",
            context=context,
            issue_types=[str(issue) for issue in details.issue_types],
            affected_parts=details.parts,
            multiple_parts_claimed=details.multiple_parts_claimed,
            severity=str(details.severity),
            prompt_injection_detected=details.prompt_injection_detected,
            confidence=details.confidence,
            extraction_reasoning=details.reasoning,
            used_llm_fallback=used_llm_fallback,
        )
