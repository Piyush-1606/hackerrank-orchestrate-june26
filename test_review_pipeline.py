from __future__ import annotations

from unittest import TestCase, main

import pandas as pd

from code.agents.base_agent import AgentRunContext, BaseAgent
from code.models.schemas import ClaimInput, ClaimStatus, FinalDecisionResult, ObjectType
from code.pipelines.review_pipeline import (
    DeterministicEvidenceAgent,
    DeterministicVisionAgent,
    ReviewPipeline,
)


def _claim(image_paths: list[str] | None = None) -> ClaimInput:
    return ClaimInput(
        claim_id="claim_001",
        user_id="user_001",
        image_paths=image_paths or ["images/sample/case_001/img_1.jpg"],
        claim_text="Customer: The rear bumper has a dent.",
        object_type=ObjectType.CAR,
        user_history=None,
        evidence_requirements=[],
        metadata={},
    )


def _pipeline(**overrides: object) -> ReviewPipeline:
    kwargs = {
        "vision_agent": DeterministicVisionAgent(),
        "evidence_agent": DeterministicEvidenceAgent(),
    }
    kwargs.update(overrides)
    return ReviewPipeline(**kwargs)


class FailingAgent(BaseAgent[ClaimInput, object]):
    def _execute(self, input_data: ClaimInput, context: AgentRunContext) -> object:
        raise RuntimeError("synthetic failure")


class ReviewPipelineTest(TestCase):
    def test_single_claim_flow(self) -> None:
        result = _pipeline().process_claim(_claim())

        self.assertIsInstance(result, FinalDecisionResult)
        self.assertEqual(result.claim_status, ClaimStatus.SUPPORTED)
        self.assertEqual(result.supporting_image_ids, ["img_1"])
        self.assertTrue(result.valid_image)

    def test_batch_flow(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "claim_id": "claim_001",
                    "user_id": "user_001",
                    "image_paths": "images/sample/case_001/img_1.jpg",
                    "user_claim": "Customer: The rear bumper has a dent.",
                    "claim_object": "car",
                },
                {
                    "claim_id": "claim_002",
                    "user_id": "user_002",
                    "image_paths": "images/sample/case_002/img_1.jpg",
                    "user_claim": "Customer: The rear bumper has a dent.",
                    "claim_object": "car",
                },
            ]
        )

        results = _pipeline().process_batch(frame)

        self.assertEqual(len(results), 2)
        self.assertTrue(all(result.claim_status == ClaimStatus.SUPPORTED for result in results))

    def test_agent_failure_handling(self) -> None:
        result = _pipeline(vision_agent=FailingAgent()).process_claim(_claim())

        self.assertEqual(result.claim_status, ClaimStatus.NOT_ENOUGH_INFORMATION)
        self.assertFalse(result.evidence_standard_met)
        self.assertFalse(result.valid_image)
        self.assertEqual(result.risk_flags, ["pipeline_error"])

    def test_empty_input_handling(self) -> None:
        self.assertEqual(_pipeline().process_batch([]), [])

        with self.assertRaises(ValueError):
            _pipeline().process_claim(None)  # type: ignore[arg-type]

    def test_multiple_image_claim(self) -> None:
        result = _pipeline().process_claim(
            _claim(
                [
                    "images/sample/case_001/img_1.jpg",
                    "images/sample/case_001/img_2.jpg",
                    "images/sample/case_001/img_3.jpg",
                ]
            )
        )

        self.assertEqual(result.claim_status, ClaimStatus.SUPPORTED)
        self.assertEqual(result.supporting_image_ids, ["img_1", "img_2", "img_3"])


if __name__ == "__main__":
    main()
