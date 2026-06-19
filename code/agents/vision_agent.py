from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

try:
    from agents.base_agent import AgentRunContext, BaseAgent, RetryConfig
    from models.schemas import ClaimInput, IssueType, VisionQualityFlag, VisionResult
except ModuleNotFoundError:
    from .base_agent import AgentRunContext, BaseAgent, RetryConfig
    from ..models.schemas import ClaimInput, IssueType, VisionQualityFlag, VisionResult


class VisionAgent(BaseAgent[ClaimInput, VisionResult]):
    """Validate submitted image evidence and return a stable visual contract.

    This placeholder does not perform image understanding. It exists to lock the
    downstream interface before integrating a VLM. Future model integration
    should happen behind `_analyze_images_with_vlm` without changing `run()`.
    """

    SUPPORTED_IMAGE_EXTENSIONS: Final[frozenset[str]] = frozenset(
        {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    )

    def __init__(
        self,
        *,
        project_root: str | Path | None = None,
        retry_config: RetryConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(name="VisionAgent", retry_config=retry_config, logger=logger)
        self.project_root = Path(project_root).resolve() if project_root else Path.cwd().resolve()

    def validate_input(self, input_data: ClaimInput) -> None:
        """Validate image path presence, file format, and existence."""
        super().validate_input(input_data)

        if not input_data.image_paths:
            raise ValueError("VisionAgent requires at least one image path")

        for image_path in input_data.image_paths:
            resolved_path = self.resolve_image_path(image_path)
            self.validate_image_format(resolved_path)
            if not resolved_path.is_file():
                raise FileNotFoundError(f"Image path does not exist: {image_path}")

    def _execute(self, input_data: ClaimInput, context: AgentRunContext) -> VisionResult:
        """Return a deterministic placeholder VisionResult.

        TODO: Replace this placeholder with VLM-backed image analysis that
        populates detected issue type, object part, visible parts, quality flags,
        and damage visibility from actual pixels.
        """
        resolved_paths = [self.resolve_image_path(path) for path in input_data.image_paths]
        image_ids = [self.image_id_from_path(path) for path in resolved_paths]

        self._log(
            logging.INFO,
            "vision_placeholder_completed",
            context=context,
            image_count=len(resolved_paths),
            image_ids=image_ids,
        )

        return VisionResult(
            claim_id=input_data.claim_id or context.claim_id,
            detected_object=input_data.object_type,
            detected_issue_type=IssueType.UNKNOWN,
            detected_object_part=None,
            visible_parts=[],
            image_quality_flags=[VisionQualityFlag.DAMAGE_NOT_VISIBLE],
            supporting_image_ids=image_ids,
            damage_visible=False,
            confidence=0.0,
            reasoning=(
                "Image paths were validated, but visual damage analysis is not implemented yet. "
                "A future VLM integration should populate visual findings from image pixels."
            ),
        )

    def resolve_image_path(self, image_path: str) -> Path:
        """Resolve absolute paths directly and relative paths against project root."""
        path = Path(image_path)
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()

    @classmethod
    def validate_image_format(cls, image_path: Path) -> None:
        """Reject unsupported image formats before model analysis."""
        if image_path.suffix.lower() not in cls.SUPPORTED_IMAGE_EXTENSIONS:
            supported = ", ".join(sorted(cls.SUPPORTED_IMAGE_EXTENSIONS))
            raise ValueError(f"Unsupported image format '{image_path.suffix}'. Supported: {supported}")

    @staticmethod
    def image_id_from_path(image_path: Path) -> str:
        """Create a stable image identifier from the file stem."""
        return image_path.stem

    def _analyze_images_with_vlm(self, image_paths: list[Path]) -> VisionResult:
        """Future hook for VLM-backed image analysis.

        This method is intentionally unused until an external vision model is
        selected. Keep provider-specific code out of the public agent contract.
        """
        raise NotImplementedError("VLM integration is not implemented yet")
