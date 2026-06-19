from __future__ import annotations

import csv
from pathlib import Path
from typing import List
import cv2
import os

from code.pipelines.review_pipeline import ReviewPipeline
from code.models.schemas import ClaimInput

CASES = ["case_008", "case_018", "case_019", "case_020"]
CSV_PATH = Path("dataset/sample_claims.csv")
OUT_MD = Path("mismatch_report.md")
ANALYSIS_OUT = Path("analysis_output")


def load_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def find_case_rows(rows: List[dict], cases: List[str]) -> List[dict]:
    selected = []
    for r in rows:
        img_paths = r.get("image_paths", "")
        for c in cases:
            if c in img_paths:
                selected.append(r)
                break
    return selected


def build_claim(row: dict, index: int) -> ClaimInput:
    # Adapt pipeline.claim_from_mapping: pass row mapping and index
    return ReviewPipeline.claim_from_mapping(row, index=index)


def run_analysis():
    rows = load_rows(CSV_PATH)
    selected = find_case_rows(rows, CASES)

    pipeline = ReviewPipeline()

    # Determine repository root and locate the sample images folder automatically
    repo_root = Path(__file__).resolve().parent
    candidate = repo_root / "dataset" / "images" / "sample"
    if candidate.exists() and candidate.is_dir():
        sample_root = candidate
    else:
        # search for a directory named 'sample' under a dataset/images path
        sample_root = None
        for p in repo_root.rglob("sample"):
            if p.is_dir() and any(part == "images" for part in [pp.name for pp in p.parents]) and any(part == "dataset" for part in [pp.name for pp in p.parents]):
                sample_root = p
                break
        if sample_root is None:
            # fallback to repo_root/dataset/images/sample even if missing
            sample_root = candidate

    ANALYSIS_OUT.mkdir(exist_ok=True)

    report_lines = ["# Mismatch Report", "", "## Selected Cases", ""]

    for i, row in enumerate(selected):
        claim = build_claim(row, index=i)
        # Resolve image paths against dataset images/sample automatically
        resolved_paths: List[str] = []
        image_exists_flags: List[bool] = []
        for img in claim.image_paths:
            orig = Path(img)
            resolved = None
            # absolute path provided
            if orig.is_absolute() and orig.exists():
                resolved = orig
            else:
                # try relative to repo
                cand = repo_root / orig
                if cand.exists():
                    resolved = cand
                else:
                    # try under sample_root using the exact relative path
                    cand2 = sample_root / orig
                    if cand2.exists():
                        resolved = cand2
                    else:
                        # try to find by basename under sample_root
                        matches = list(sample_root.rglob(orig.name)) if sample_root.exists() else []
                        if matches:
                            resolved = matches[0]

            if resolved is not None:
                resolved = resolved.resolve()
                resolved_paths.append(str(resolved))
                image_exists_flags.append(True)
                # save thumbnail
                try:
                    img_mtx = cv2.imread(str(resolved))
                    if img_mtx is not None:
                        h, w = img_mtx.shape[:2]
                        max_dim = 256
                        scale = max_dim / max(h, w) if max(h, w) > max_dim else 1.0
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        thumb = cv2.resize(img_mtx, (new_w, new_h))
                        out_name = f"{claim.claim_id or i}_{Path(resolved).name}"
                        out_path = ANALYSIS_OUT / out_name
                        cv2.imwrite(str(out_path), thumb)
                except Exception:
                    # don't fail analysis for thumbnailing
                    pass
            else:
                resolved_paths.append(str(orig))
                image_exists_flags.append(False)

        # update claim image paths to resolved ones (strings)
        claim.image_paths = resolved_paths
        context_info = f"Claim ID: {claim.claim_id} | User: {claim.user_id}"
        report_lines.append(f"### {claim.claim_id}")
        report_lines.append("")
        report_lines.append(f"- Claim text: {claim.claim_text}")
        expected_issue = row.get("issue_type", "")
        expected_part = row.get("object_part", "")
        report_lines.append(f"- Expected issue_type: {expected_issue}")
        report_lines.append(f"- Expected object_part: {expected_part}")
        report_lines.append(f"- Image paths:")
        for p, exists in zip(claim.image_paths, image_exists_flags):
            report_lines.append(f"  - {p} | exists={exists}")
        report_lines.append("")

        # Run agents step-by-step to capture intermediate outputs
        try:
            claim_result = pipeline.claim_agent.run(claim, pipeline.claim_agent.get_default_context() if hasattr(pipeline.claim_agent, 'get_default_context') else None)
        except Exception:
            # fallback: use pipeline.process_claim but note
            claim_result = None

        # Run Vision, Evidence, Risk, Decision collecting results
        try:
            vision_result = pipeline.vision_agent.run(claim, pipeline.vision_agent.get_default_context() if hasattr(pipeline.vision_agent, 'get_default_context') else None)
        except Exception as e:
            vision_result = None
            report_lines.append(f"- Vision agent error: {type(e).__name__}: {e}")

        try:
            # Evidence expects (ClaimExtractionResult, VisionResult)
            if claim_result is None:
                # Execute ClaimAgent via pipeline.process_claim to get decision only
                decision = pipeline.process_claim(claim)
                report_lines.append(f"- Pipeline returned decision (claim processing failed to get intermediates). Predicted issue: {decision.issue_type}, part: {decision.object_part}")
                predicted_issue = decision.issue_type
                predicted_part = decision.object_part
                classification = []
            else:
                evidence_result = pipeline.evidence_agent.run((claim_result, vision_result), None)
                risk_result = pipeline.risk_agent.run(pipeline.build_user_history(claim), None)
                decision = pipeline.decision_agent.run((claim_result, vision_result, evidence_result, risk_result), None)
                predicted_issue = decision.issue_type
                predicted_part = decision.object_part

                report_lines.append(f"- Predicted issue_type: {predicted_issue}")
                report_lines.append(f"- Predicted object_part: {predicted_part}")

                # For the requested CASES print a concise summary to stdout as well
                if any(c in (row.get("image_paths") or "") for c in CASES):
                    print("---")
                    print(f"Claim ID: {claim.claim_id}")
                    print(f"Claim text: {claim.claim_text}")
                    print(f"Expected issue: {row.get('issue_type')}")
                    print(f"Expected part: {row.get('object_part')}")
                    print(f"Predicted issue: {predicted_issue}")
                    print(f"Predicted part: {predicted_part}")
                    print("Image paths and exists:")
                    for p, ex in zip(claim.image_paths, image_exists_flags):
                        print(f" - {p} | exists={ex}")

                # Simple mismatch detection
                mismatches = []
                if str(predicted_issue) != expected_issue:
                    mismatches.append("issue_type_mismatch")
                if (predicted_part or "") != (expected_part or ""):
                    mismatches.append("object_part_mismatch")

                # Root-cause heuristics
                root_causes = []
                if vision_result is None:
                    root_causes.append("vision_agent_error")
                else:
                    # If vision detected wrong object or damage_not_visible flags, classify accordingly
                    vflags = getattr(vision_result, "image_quality_flags", [])
                    vrisk = getattr(vision_result, "risk_flags", [])
                    if "wrong_object" in [str(f) for f in vflags] or "wrong_object" in vrisk:
                        root_causes.append("claim_extraction_issue")
                    if any(str(f) == "damage_not_visible" for f in vflags):
                        root_causes.append("evidence_logic_issue")
                    # If image set cropped or obstructed
                    if any(str(f) == "cropped_or_obstructed" for f in vflags):
                        root_causes.append("hidden_visual_evidence")

                # Ambiguity if expected unknown or 'none'
                if expected_issue in ("unknown", "none"):
                    root_causes.append("label_ambiguity")

                if not root_causes:
                    root_causes.append("vision_reasoning_issue")

                report_lines.append("- Root-cause classifications:")
                for rc in root_causes:
                    report_lines.append(f"  - {rc}")

                # Suggest improvements
                report_lines.append("- Suggested improvements:")
                suggestions = []
                if "claim_extraction_issue" in root_causes:
                    suggestions.append("Improve NER and claim-guided parsing to better anchor object mentions to images.")
                if "vision_reasoning_issue" in root_causes:
                    suggestions.append("Integrate stronger VLM grounding or higher-resolution region proposals to localize damage.")
                if "evidence_logic_issue" in root_causes:
                    suggestions.append("Relax evidence sufficiency thresholds to surface partial matches and provide human review cues.")
                if "label_ambiguity" in root_causes:
                    suggestions.append("Add label normalization and annotator guidelines; handle 'none/unknown' with soft labels.")
                if "hidden_visual_evidence" in root_causes:
                    suggestions.append("Request additional images or implement heuristics to request specific angles from users.")

                for s in suggestions:
                    report_lines.append(f"  - {s}")

        except Exception as e:
            report_lines.append(f"- Error during evidence/decision: {type(e).__name__}: {e}")

        report_lines.append("")

    report_lines.append("\n## Root-cause summary and concrete improvements")
    report_lines.append("")
    report_lines.append("See per-case recommendations above. Key themes:\n- Improve vision grounding and region-level damage detection.\n- Enhance claim parsing to reduce claim-object mismatches.\n- Surface ambiguity cases for human review and request more images when necessary.")

    OUT_MD.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Wrote report to {OUT_MD}")


if __name__ == "__main__":
    run_analysis()
