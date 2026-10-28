from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchoring_study.cases import REQUIRED_COLUMNS, validate_cases
from anchoring_study.io_utils import utc_now


PROTECTED_COLUMNS = [
    "case_id", "domain", "gold_diagnosis", "acceptable_synonyms", "correct_anchor",
    "incorrect_anchor", "objective_case", "key_red_flags", "reference_triage",
    "mandatory_actions", "acceptable_actions", "unsafe_actions",
]
DOMAIN_ORDER = [
    "cardiovascular emergency", "neurologic emergency", "acute abdomen",
    "severe infection", "oncology", "postoperative complication",
]
ROUNDS = [
    ("revised_v1", ROOT / "ratings/formal_case_expert_review_revised_v1/reconciliation/formal_case_review_reconciliation.csv"),
    ("revised_v2", ROOT / "ratings/formal_case_expert_review_revised_v2/reconciliation/formal_case_review_reconciliation.csv"),
    ("revised_v3", ROOT / "ratings/formal_case_expert_review_revised_v3/reconciliation/formal_case_review_reconciliation.csv"),
    ("revised_v5", ROOT / "ratings/formal_case_expert_review_revised_v5/reconciliation/formal_case_review_reconciliation.csv"),
]
OUTPUT = ROOT / "data/cases_formal_approved_candidates_v2.csv"
REPORT = ROOT / "results/formal_case_registry_v2/registry_qc.json"


def read_csv_compatibly(path: Path) -> list[dict[str, str]]:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030", "gbk"):
        try:
            return list(csv.DictReader(raw.decode(encoding).splitlines()))
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path}.")


def main() -> int:
    approved: dict[str, dict[str, str]] = {}
    provenance: dict[str, str] = {}
    errors: list[str] = []
    for round_name, path in ROUNDS:
        for row in read_csv_compatibly(path):
            if row.get("recommended_next_action", "").strip() != "eligible_for_locking_after_final_version_check":
                continue
            case_id = row.get("case_id", "").strip()
            if case_id in approved:
                errors.append(f"Duplicate approval across rounds: {case_id} ({provenance[case_id]} and {round_name}).")
                continue
            missing = [column for column in PROTECTED_COLUMNS if not (row.get(column) or "").strip()]
            if missing:
                errors.append(f"{case_id}: missing protected fields {missing} in {round_name} reconciliation.")
                continue
            approved[case_id] = {column: (row.get(column) or "").strip() for column in PROTECTED_COLUMNS}
            provenance[case_id] = round_name

    domain_rank = {domain: index for index, domain in enumerate(DOMAIN_ORDER)}
    output_rows: list[dict[str, str]] = []
    for case_id, row in sorted(
        approved.items(),
        key=lambda item: (domain_rank.get(item[1]["domain"], 999), item[0]),
    ):
        output_rows.append({
            **row,
            "case_status": "reviewed",
            "expert_review_status": "expert_approved",
            "notes": (
                f"Two independent clinicians both recommended include after {provenance[case_id]}; "
                "approved candidate only, not yet locked for formal generation."
            ),
        })

    if len(output_rows) < 30:
        errors.append(f"Expected at least 30 cumulative approved candidates, found {len(output_rows)}.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)

    rows, validation_errors, warnings = validate_cases(OUTPUT)
    errors.extend(validation_errors)
    counts = Counter(row["domain"] for row in rows)
    for domain in DOMAIN_ORDER:
        if counts.get(domain, 0) < 5:
            errors.append(f"{domain}: fewer than 5 approved candidates ({counts.get(domain, 0)}).")
    report = {
        "timestamp_utc": utc_now(),
        "purpose": "Cumulative registry of double-clinician-approved formal case candidates; not the locked formal case set.",
        "candidate_count": len(rows),
        "domain_counts": {domain: counts.get(domain, 0) for domain in DOMAIN_ORDER},
        "approval_round_by_case": provenance,
        "errors": errors,
        "warnings": warnings,
        "passed": not errors,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"REGISTRY PASSED: {len(rows)} approved candidates")
    print("DOMAIN COUNTS: " + ", ".join(f"{domain}={counts.get(domain, 0)}" for domain in DOMAIN_ORDER))
    print(f"Output: {OUTPUT}")
    print(f"Report: {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
