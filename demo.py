from __future__ import annotations

from pathlib import Path
import csv
from pprint import pprint

from code.pipelines.review_pipeline import ReviewPipeline


CSV_PATH = Path("dataset/sample_claims.csv")


def load_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def resolve_image_paths(claim, repo_root: Path):
    # resolve image paths against dataset/images/sample when possible
    sample_root = repo_root / "dataset" / "images" / "sample"
    resolved = []
    for img in claim.image_paths:
        p = Path(img)
        if p.is_absolute() and p.exists():
            resolved.append(str(p.resolve()))
            continue
        cand = repo_root / img
        if cand.exists():
            resolved.append(str(cand.resolve()))
            continue
        cand2 = sample_root / img
        if cand2.exists():
            resolved.append(str(cand2.resolve()))
            continue
        # try basename search under sample_root
        found = None
        if sample_root.exists():
            matches = list(sample_root.rglob(p.name))
            if matches:
                found = matches[0]
        resolved.append(str(found.resolve()) if found is not None else img)
    claim.image_paths = resolved


def pick_samples(rows):
    car = next((r for r in rows if (r.get("claim_object") or r.get("object_type")) == "car"), None)
    laptop = next((r for r in rows if (r.get("claim_object") or r.get("object_type")) == "laptop"), None)
    package = next((r for r in rows if (r.get("claim_object") or r.get("object_type")) == "package"), None)
    return [car, laptop, package]


def _enum_to_str(val):
    try:
        return val.value
    except Exception:
        return str(val)


def print_claim_walkthrough(claim, claim_result, vision_result, evidence_result, risk_result, decision_result, final):
    obj_label = (claim.object_type.value if hasattr(claim.object_type, "value") else str(claim.object_type)).upper()
    print("#" * 50)
    print(f"{obj_label} CLAIM")
    print()
    # CLAIM
    print("## CLAIM")
    print(f"Object : {claim.object_type.value if hasattr(claim.object_type, 'value') else claim.object_type}")
    print(f"User ID : {claim.user_id}")
    print(f"Claim ID : {claim.claim_id}")
    print()

    # CLAIM AGENT
    print("## CLAIM AGENT")
    if claim_result is None:
        print("(claim agent failed)")
    else:
        issue = None
        if getattr(claim_result, "claimed_issue_types", None):
            first = claim_result.claimed_issue_types[0]
            issue = _enum_to_str(first)
        part = getattr(claim_result, "affected_area", "")
        sev = _enum_to_str(getattr(claim_result, "claimed_severity", ""))
        conf = getattr(claim_result, "confidence", "")
        print(f"Issue : {issue}")
        print(f"Part : {part}")
        print(f"Severity : {sev}")
        print(f"Confidence : {conf}")
    print()

    # VISION
    print("## VISION AGENT")
    if vision_result is None:
        print("(vision agent failed)")
    else:
        vis_issue = _enum_to_str(getattr(vision_result, "detected_issue_type", ""))
        vis_part = getattr(vision_result, "detected_object_part", "")
        images_used = ",".join(getattr(vision_result, "supporting_image_ids", []) or [])
        qflags = getattr(vision_result, "image_quality_flags", []) or []
        quality = "none" if not qflags else ",".join(_enum_to_str(q) for q in qflags)
        vconf = getattr(vision_result, "confidence", "")
        rflags = getattr(vision_result, "risk_flags", []) or []
        print(f"Issue : {vis_issue}")
        print(f"Part : {vis_part}")
        print(f"Images Used : {images_used}")
        print(f"Quality : {quality}")
        print(f"Confidence : {vconf}")
        if rflags:
            print(f"Vision Risk Flags : {','.join(rflags)}")
    print()

    # EVIDENCE
    print("## EVIDENCE AGENT")
    if evidence_result is None:
        print("(evidence agent failed)")
    else:
        met = getattr(evidence_result, "evidence_standard_met", "")
        rev = _enum_to_str(getattr(evidence_result, "reviewability", ""))
        status = _enum_to_str(getattr(evidence_result, "recommended_status", ""))
        sup_imgs = ",".join(getattr(evidence_result, "supporting_image_ids", []) or [])
        print(f"Evidence Met: {met}")
        print(f"Reviewable : {rev}")
        print(f"Status : {status}")
        if sup_imgs:
            print(f"Supporting Images : {sup_imgs}")
    print()

    # RISK
    print("## RISK AGENT")
    if risk_result is None:
        print("(risk agent failed)")
    else:
        score = getattr(risk_result, "risk_score", getattr(risk_result, "score", 0.0))
        flags = getattr(risk_result, "risk_flags", []) or []
        print(f"Risk Score : {score:.2f}")
        print(f"Risk Flags : {','.join(flags) if flags else 'none'}")
    print()

    # DECISION
    print("## DECISION AGENT")
    if decision_result is None:
        print("(decision agent failed)")
    else:
        status = _enum_to_str(getattr(decision_result, "claim_status", ""))
        sev = _enum_to_str(getattr(decision_result, "severity", ""))
        sup_imgs = ",".join(getattr(decision_result, "supporting_image_ids", []) or [])
        flags = getattr(decision_result, "risk_flags", []) or []
        print(f"Claim Status: {status}")
        print(f"Severity : {sev}")
        print(f"Supporting Images : {sup_imgs}")
        print(f"Decision Risk Flags : {','.join(flags) if flags else 'none'}")
    print()

    # FINAL
    print("## FINAL DECISION")
    if final is None:
        print("(final pipeline failed)")
    else:
        final_status = _enum_to_str(getattr(final, "claim_status", ""))
        reason = getattr(final, "claim_status_justification", "")
        print(final_status.capitalize())
        print("Reason:")
        print(reason)
    print("#" * 50)
    print()


def run_demo():
    rows = load_rows(CSV_PATH)
    samples = pick_samples(rows)
    pipeline = ReviewPipeline()
    repo_root = Path(__file__).resolve().parent

    for row in samples:
        if row is None:
            continue
        claim = ReviewPipeline.claim_from_mapping(row, index=rows.index(row))

        # resolve image paths so VisionAgent can access files
        resolve_image_paths(claim, repo_root)

        print("\n========================================\n")

        # Run ClaimAgent
        try:
            claim_result = pipeline.claim_agent.run(claim, pipeline.claim_agent.get_default_context() if hasattr(pipeline.claim_agent, 'get_default_context') else None)
        except Exception as e:
            claim_result = None
            print("Claim agent failed:", e)
        # Vision
        try:
            vision_result = pipeline.vision_agent.run(claim, pipeline.vision_agent.get_default_context() if hasattr(pipeline.vision_agent, 'get_default_context') else None)
        except Exception as e:
            vision_result = None
            print("Vision agent failed:", e)
        # Evidence
        try:
            evidence_result = pipeline.evidence_agent.run((claim_result, vision_result), None)
        except Exception as e:
            evidence_result = None
            print("Evidence agent failed:", e)
        # Risk
        try:
            risk_input = pipeline.build_user_history(claim)
            risk_result = pipeline.risk_agent.run(risk_input, None)
        except Exception as e:
            risk_result = None
            print("Risk agent failed:", e)
        # Decision
        try:
            decision_result = pipeline.decision_agent.run((claim_result, vision_result, evidence_result, risk_result), None)
        except Exception as e:
            decision_result = None
            print("Decision agent failed:", e)
        # Final pipeline call for comparison
        try:
            final = pipeline.process_claim(claim)
        except Exception as e:
            final = None
            print("Process claim failed:", e)

        # Print judge-friendly walkthrough
        print_claim_walkthrough(claim, claim_result, vision_result, evidence_result, risk_result, decision_result, final)


if __name__ == "__main__":
    run_demo()
