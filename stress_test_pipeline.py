from __future__ import annotations

import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from code.pipelines.review_pipeline import ReviewPipeline
from code.models.schemas import ClaimStatus, IssueType


RANDOM_SEED = 42
NUM_CLAIMS = 100
OUT_REPORT = "stress_test_report.json"
IMAGE_DIR = Path("stress_images")

random.seed(RANDOM_SEED)

ISSUE_TEMPLATES = [
    ("car", "dent", "rear_bumper"),
    ("car", "scratch", "front_bumper"),
    ("car", "crack", "windshield"),
    ("laptop", "crack", "screen"),
    ("package", "torn_packaging", "box"),
]


def make_image(path: Path, seed: int = 0, noisy: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = 640, 480
    rng = np.random.RandomState(seed)
    base = rng.randint(40, 220, size=(h, w, 3), dtype=np.uint8)
    if noisy:
        noise = (rng.randn(h, w, 3) * 25).clip(-50, 50).astype(np.int16)
        base = (base.astype(np.int16) + noise).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(path), base)
    return path


def make_blurry_image(path: Path, seed: int = 0):
    p = make_image(path, seed=seed, noisy=True)
    img = cv2.imread(str(p))
    blurred = cv2.GaussianBlur(img, (21, 21), 0)
    cv2.imwrite(str(p), blurred)
    return p


def make_low_light_image(path: Path, seed: int = 0):
    p = make_image(path, seed=seed)
    img = cv2.imread(str(p))
    dark = (img * 0.18).astype(np.uint8)
    cv2.imwrite(str(p), dark)
    return p


def create_synthetic_claims(out_dir: Path, n: int) -> list[dict]:
    claims = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        obj, issue, part = random.choice(ISSUE_TEMPLATES)
        user_id = f"user_{random.randint(1,20):03d}"
        claim_text = f"Customer: The {obj} has a {issue} on the {part}."

        # Decide image scenario
        scenario = random.choices(
            ["normal", "missing", "duplicate", "blurry", "low_light"],
            weights=[0.5, 0.1, 0.15, 0.15, 0.1],
            k=1,
        )[0]

        images = []
        if scenario == "missing":
            # add a non-existing path
            images = [str(out_dir / f"missing_{i}_1.jpg")]
        elif scenario == "duplicate":
            p = out_dir / f"dup_{i}_1.jpg"
            make_image(p, seed=i)
            images = [str(p), str(p)]
        elif scenario == "blurry":
            p = out_dir / f"blur_{i}_1.jpg"
            make_blurry_image(p, seed=i)
            # sometimes include a second normal
            if random.random() < 0.4:
                p2 = out_dir / f"img_{i}_2.jpg"
                make_image(p2, seed=i + 1000)
                images = [str(p), str(p2)]
            else:
                images = [str(p)]
        elif scenario == "low_light":
            p = out_dir / f"dark_{i}_1.jpg"
            make_low_light_image(p, seed=i)
            images = [str(p)]
        else:
            # normal: 1-2 images
            count = random.choice([1, 2])
            for j in range(count):
                p = out_dir / f"img_{i}_{j}.jpg"
                make_image(p, seed=i + j)
                images.append(str(p))

        claim = {
            "claim_id": f"stress_{i}",
            "user_id": user_id,
            "image_paths": images,
            "claim_text": claim_text,
            "object_type": obj,
            "metadata": {},
        }
        claims.append(claim)

    return claims


def validate_result(res) -> tuple[bool, list[str]]:
    errors = []
    ok = True
    # required fields
    try:
        cid = getattr(res, "claim_id", None)
        status = getattr(res, "claim_status", None)
        if status is None:
            ok = False
            errors.append("missing_claim_status")
        else:
            if status not in ClaimStatus:
                # allow str comparison
                try:
                    _ = ClaimStatus(status)
                except Exception:
                    ok = False
                    errors.append(f"invalid_claim_status:{status}")
        rf = getattr(res, "risk_flags", None)
        if not isinstance(rf, (list, tuple)):
            ok = False
            errors.append("risk_flags_not_list")
        sids = getattr(res, "supporting_image_ids", None)
        if not isinstance(sids, (list, tuple)):
            ok = False
            errors.append("supporting_image_ids_not_list")
    except Exception as exc:
        ok = False
        errors.append(f"exception_validating:{type(exc).__name__}")
    return ok, errors


def run_stress_test():
    if IMAGE_DIR.exists():
        shutil.rmtree(IMAGE_DIR)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    claims = create_synthetic_claims(IMAGE_DIR, NUM_CLAIMS)

    pipeline = ReviewPipeline()

    report = {
        "num_claims": len(claims),
        "failures": [],
        "exceptions": [],
        "invalid_outputs": [],
        "status_counts": {},
    }

    try:
        results = pipeline.process_batch(claims)
    except Exception as exc:
        report["exceptions"].append({"error": type(exc).__name__, "message": str(exc)})
        results = []

    counts = Counter()
    for res in results:
        try:
            ok, errors = validate_result(res)
            if not ok:
                report["invalid_outputs"].append({"claim_id": res.claim_id, "errors": errors})
            # detect pipeline failures
            rf = getattr(res, "risk_flags", []) or []
            if "pipeline_error" in rf:
                report["failures"].append(res.claim_id)
            counts[str(res.claim_status)] += 1
        except Exception as exc:
            report["exceptions"].append({"claim_id": getattr(res, 'claim_id', None), "error": type(exc).__name__, "message": str(exc)})

    report["status_counts"] = dict(counts)

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Stress test completed. Report written to {OUT_REPORT}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run_stress_test()
