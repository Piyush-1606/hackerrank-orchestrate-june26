from __future__ import annotations

import csv
from pathlib import Path
from collections import Counter

CLAIMS_CSV = Path("dataset/claims.csv")
OUTPUT_CSV = Path("dataset/output.csv")
REPORT_MD = Path("submission_validation_report.md")

REQUIRED_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

ALLOWED_CLAIM_STATUS = {"supported", "contradicted", "not_enough_information"}
ALLOWED_SEVERITY = {"none", "low", "medium", "high", "unknown"}
ALLOWED_ISSUE = {
    "dent",
    "scratch",
    "crack",
    "glass_shatter",
    "broken_part",
    "missing_part",
    "torn_packaging",
    "crushed_packaging",
    "water_damage",
    "stain",
    "none",
    "unknown",
}


def load_csv(path: Path):
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows, reader.fieldnames


def validate():
    report_lines = ["# Submission Validation", ""]

    if not CLAIMS_CSV.exists():
        report_lines.append(f"- ERROR: {CLAIMS_CSV} not found")
        REPORT_MD.write_text("\n".join(report_lines), encoding="utf-8")
        print("Missing claims.csv")
        return

    if not OUTPUT_CSV.exists():
        report_lines.append(f"- ERROR: {OUTPUT_CSV} not found")
        REPORT_MD.write_text("\n".join(report_lines), encoding="utf-8")
        print("Missing output.csv")
        return

    claims_rows, _ = load_csv(CLAIMS_CSV)
    out_rows, out_cols = load_csv(OUTPUT_CSV)

    # Row count check
    pass_all = True
    claims_count = len(claims_rows)
    out_count = len(out_rows)
    report_lines.append(f"- Row count: claims.csv={claims_count}, output.csv={out_count}")
    if claims_count == out_count:
        report_lines.append("- ✓ Row count check: PASS")
    else:
        report_lines.append("- ✗ Row count check: FAIL")
        pass_all = False

    # Schema check
    report_lines.append("")
    report_lines.append("- Schema check:")
    if out_cols == REQUIRED_COLUMNS:
        report_lines.append("  - ✓ Schema matches required columns and order")
    else:
        report_lines.append("  - ✗ Schema mismatch")
        report_lines.append(f"    - Found: {out_cols}")
        report_lines.append(f"    - Expected: {REQUIRED_COLUMNS}")
        pass_all = False

    # Value checks
    missing_value_rows = []
    invalid_rows = []
    risk_flags_counter = Counter()
    supporting_id_issues = []

    for i, r in enumerate(out_rows, start=1):
        row_id = i
        # Required non-empty fields
        for fld in ["claim_status", "issue_type", "object_part", "severity"]:
            if fld not in r or (r[fld] is None) or (str(r[fld]).strip() == ""):
                missing_value_rows.append((row_id, fld))
                pass_all = False

        # Validate claim_status
        cs = r.get("claim_status", "").strip()
        if cs and cs not in ALLOWED_CLAIM_STATUS:
            invalid_rows.append((row_id, "claim_status", cs))
            pass_all = False

        # severity
        sev = r.get("severity", "").strip()
        if sev and sev not in ALLOWED_SEVERITY:
            invalid_rows.append((row_id, "severity", sev))
            pass_all = False

        # issue_type
        it = r.get("issue_type", "").strip()
        if it and it not in ALLOWED_ISSUE:
            invalid_rows.append((row_id, "issue_type", it))
            pass_all = False

        # supporting_image_ids checks: must be 'none' or semicolon-separated ids like img_1 or img_1;img_2
        sids = r.get("supporting_image_ids", "").strip()
        if sids == "":
            supporting_id_issues.append((row_id, "empty"))
            pass_all = False
        else:
            if sids.lower() != "none":
                parts = [p.strip() for p in sids.split(";") if p.strip()]
                for p in parts:
                    # Reject full paths or extensions
                    if "/" in p or "\\" in p:
                        supporting_id_issues.append((row_id, p, "contains_path"))
                        pass_all = False
                    if "." in p:
                        supporting_id_issues.append((row_id, p, "contains_extension"))
                        pass_all = False

        # risk flags distribution
        rf = r.get("risk_flags", "").strip()
        for flag in [f.strip() for f in rf.split(";") if f.strip()]:
            risk_flags_counter[flag] += 1

    report_lines.append("")
    # Missing values
    if not missing_value_rows:
        report_lines.append("- ✓ Missing value check: no empty required fields detected")
    else:
        report_lines.append("- ✗ Missing value check: some required fields are empty")
        for row_id, fld in missing_value_rows:
            report_lines.append(f"  - Row {row_id}: missing {fld}")

    # Invalid values
    if not invalid_rows:
        report_lines.append("- ✓ Allowed value check: claim_status/issue_type/severity within allowed lists")
    else:
        report_lines.append("- ✗ Allowed value check failures:")
        for row_id, fld, val in invalid_rows:
            report_lines.append(f"  - Row {row_id}: {fld}='{val}' is invalid")

    # Supporting image id issues
    if not supporting_id_issues:
        report_lines.append("- ✓ Supporting image ID check: formats look valid (img_1, img_1;img_2, none)")
    else:
        report_lines.append("- ✗ Supporting image ID check failures:")
        for item in supporting_id_issues:
            report_lines.append(f"  - Row {item[0]}: {item[1:]}")
        pass_all = False

    # Risk flag summary
    report_lines.append("")
    report_lines.append("- Risk flag distribution:")
    total_flags = sum(risk_flags_counter.values())
    if total_flags == 0:
        report_lines.append("  - (no risk flags present)")
    else:
        for flag, cnt in risk_flags_counter.most_common():
            report_lines.append(f"  - {flag}: {cnt}")

    report_lines.append("")
    report_lines.append("## Final Verdict")
    report_lines.append("")
    report_lines.append("PASS" if pass_all else "FAIL")

    # Include failing rows details if any
    if not pass_all:
        report_lines.append("")
        report_lines.append("## Failing Rows Details")
        if missing_value_rows:
            report_lines.append("- Missing required fields:")
            for row_id, fld in missing_value_rows:
                report_lines.append(f"  - Row {row_id}: missing {fld}")
        if invalid_rows:
            report_lines.append("- Invalid enumerated values:")
            for row_id, fld, val in invalid_rows:
                report_lines.append(f"  - Row {row_id}: {fld}='{val}'")
        if supporting_id_issues:
            report_lines.append("- Supporting image id issues:")
            for item in supporting_id_issues:
                report_lines.append(f"  - Row {item[0]}: {item[1:]}")

    REPORT_MD.write_text("\n".join(report_lines), encoding="utf-8")
    print(REPORT_MD)


if __name__ == "__main__":
    validate()
