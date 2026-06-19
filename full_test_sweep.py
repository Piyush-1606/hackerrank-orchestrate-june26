from __future__ import annotations

import subprocess
import time
import json
from pathlib import Path
import unittest
import sys

REPO_ROOT = Path(__file__).resolve().parent
REPORT_MD = REPO_ROOT / "full_test_report.md"
SUBMISSION_VALIDATOR = REPO_ROOT / "submission_validation.py"


def run_unit_tests():
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(REPO_ROOT), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    start = time.perf_counter()
    result = runner.run(suite)
    duration = time.perf_counter() - start

    passed = result.testsRun - len(result.failures) - len(result.errors) - len(getattr(result, 'skipped', []))
    failed = len(result.failures) + len(result.errors)
    skipped = len(getattr(result, 'skipped', []))

    failures_detail = [
        {
            "test": str(f[0]),
            "trace": f[1][:2000],
        }
        for f in list(result.failures) + list(result.errors)
    ]

    return {
        "tests_run": result.testsRun,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "duration_seconds": duration,
        "failures": failures_detail,
    }


def run_script(cmd, cwd=REPO_ROOT):
    proc = subprocess.run(cmd, cwd=str(cwd), shell=True, capture_output=True, text=True)
    return proc.returncode, proc.stdout + "\n" + proc.stderr


def main():
    report = []

    # 1. Unit tests
    report.append("# Unit Test Results\n")
    ut = run_unit_tests()
    report.append(f"- Tests run: {ut['tests_run']}")
    report.append(f"- Passed: {ut['passed']}")
    report.append(f"- Failed: {ut['failed']}")
    report.append(f"- Skipped: {ut['skipped']}")
    report.append(f"- Execution time (s): {ut['duration_seconds']:.2f}")
    report.append("")
    if ut['failed']:
        report.append("## Failed Tests Details")
        for f in ut['failures']:
            report.append(f"- {f['test']}")
            report.append("```\n" + f['trace'] + "\n```")
        report.append("")

    # 2. Accuracy evaluation
    report.append("# Accuracy Evaluation\n")
    # execute evaluate_full_pipeline_accuracy.py
    code, out = run_script("python evaluate_full_pipeline_accuracy.py")
    report.append("```")
    report.append(out)
    report.append("```")

    # parse evaluation file for Issue/Part Accuracy if present
    eval_md = REPO_ROOT / "evaluation" / "evaluation_report.md"
    issue_acc = None
    part_acc = None
    if eval_md.exists():
        text = eval_md.read_text(encoding='utf-8')
        for line in text.splitlines():
            if line.strip().startswith("Issue Accuracy"):
                issue_acc = line.split(":", 1)[1].strip()
            if line.strip().startswith("Part Accuracy"):
                part_acc = line.split(":", 1)[1].strip()

    report.append(f"- Issue Accuracy: {issue_acc}")
    report.append(f"- Part Accuracy: {part_acc}")
    report.append("")

    # 3. Stress test
    report.append("# Stress Testing Results\n")
    code, out = run_script("python stress_test_pipeline.py")
    report.append("```")
    report.append(out)
    report.append("```")

    stress_report = REPO_ROOT / "stress_test_report.json"
    if stress_report.exists():
        data = json.loads(stress_report.read_text(encoding='utf-8'))
        report.append(f"- total synthetic claims: {data.get('num_claims')}")
        report.append(f"- failures: {len(data.get('failures', []))}")
        report.append(f"- invalid outputs: {len(data.get('invalid_outputs', []))}")
        report.append(f"- claim status distribution: {data.get('status_counts')}")
    else:
        report.append("- Stress report not found; stress_test_pipeline.py may have failed")

    report.append("")

    # 4. Demo
    report.append("# Demo Validation\n")
    code, out = run_script("python demo.py")
    report.append("```")
    report.append(out)
    report.append("```")
    demo_ok = code == 0
    report.append(f"- Demo executed successfully: {demo_ok}")
    report.append("")

    # 5. Submission generation and schema validation
    report.append("# Submission Generation and Schema Validation\n")
    code, out = run_script("python generate_submission.py")
    report.append("```")
    report.append(out)
    report.append("```")
    # run submission validator
    val_code, val_out = run_script(f"python {SUBMISSION_VALIDATOR.name}")
    report.append("```")
    report.append(val_out)
    report.append("```")
    # The validator prints the report filename; read the generated report to determine PASS/FAIL
    val_report_path = REPO_ROOT / "submission_validation_report.md"
    submission_pass = False
    if val_report_path.exists():
        content = val_report_path.read_text(encoding='utf-8')
        # final verdict is on a line containing PASS or FAIL
        for line in reversed(content.splitlines()):
            if line.strip() in ("PASS", "FAIL"):
                submission_pass = line.strip() == "PASS"
                break
    else:
        submission_pass = False

    # Determine final verdict
    overall_pass = (ut['failed'] == 0) and submission_pass and demo_ok

    report.append("# Overall System Health\n")
    if overall_pass:
        report.append("PASS")
    else:
        report.append("FAIL")
        reasons = []
        if ut['failed'] != 0:
            reasons.append("unit tests failures")
        if not submission_pass:
            reasons.append("submission validation failed")
        if not demo_ok:
            reasons.append("demo execution failed")
        report.append("Reasons: " + ", ".join(reasons))

    REPORT_MD.write_text("\n".join(report), encoding='utf-8')
    print(f"Wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
