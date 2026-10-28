from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchoring_study.cases import REQUIRED_COLUMNS, validate_cases
from anchoring_study.io_utils import utc_now


INPUT = ROOT / "data/cases_formal_approved_candidates_v2.csv"
LOCKED_OUTPUT = ROOT / "data/cases_formal_locked_v1.csv"
RESERVE_OUTPUT = ROOT / "data/cases_formal_approved_reserve_v1.csv"
REPORT = ROOT / "results/formal_case_lock_v1/lock_qc.json"
DOMAIN_ORDER = [
    "cardiovascular emergency", "neurologic emergency", "acute abdomen",
    "severe infection", "oncology", "postoperative complication",
]


def read_csv_compatibly(path: Path) -> list[dict[str, str]]:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030", "gbk"):
        try:
            return list(csv.DictReader(raw.decode(encoding).splitlines()))
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path}.")


def case_number(case_id: str) -> int:
    match = re.search(r"(\d+)$", case_id)
    if not match:
        raise ValueError(f"Case ID has no numeric suffix: {case_id}")
    return int(match.group(1))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    errors: list[str] = []
    source = read_csv_compatibly(INPUT)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in source:
        if row.get("expert_review_status", "").strip() != "expert_approved":
            errors.append(f"{row.get('case_id')}: candidate is not expert_approved.")
        groups[row.get("domain", "").strip()].append(row)

    selected: list[dict[str, str]] = []
    reserves: list[dict[str, str]] = []
    for domain in DOMAIN_ORDER:
        candidates = sorted(groups.get(domain, []), key=lambda row: (case_number(row["case_id"]), row["case_id"]))
        if len(candidates) < 5:
            errors.append(f"{domain}: only {len(candidates)} approved candidates; 5 required.")
            continue
        for row in candidates[:5]:
            new = dict(row)
            new["case_status"] = "locked"
            new["expert_review_status"] = "expert_approved"
            new["notes"] = (
                f"{row.get('notes', '').strip()} Selected for formal_v1 by deterministic within-domain "
                "ascending case-ID rule; no model outputs were consulted."
            ).strip()
            selected.append(new)
        for row in candidates[5:]:
            new = dict(row)
            new["case_status"] = "reviewed"
            new["expert_review_status"] = "expert_approved"
            new["notes"] = (
                f"{row.get('notes', '').strip()} Retained as an approved reserve after deterministic "
                "within-domain ascending case-ID selection for formal_v1."
            ).strip()
            reserves.append(new)

    selected_ids = {row["case_id"] for row in selected}
    reserve_ids = {row["case_id"] for row in reserves}
    source_ids = {row["case_id"] for row in source}
    if len(selected) != 30:
        errors.append(f"Locked set must contain 30 cases; found {len(selected)}.")
    if selected_ids & reserve_ids:
        errors.append(f"Locked and reserve sets overlap: {sorted(selected_ids & reserve_ids)}")
    if selected_ids | reserve_ids != source_ids:
        errors.append("Locked plus reserve IDs do not exactly reconstruct the approved-candidate registry.")
    counts = Counter(row["domain"] for row in selected)
    for domain in DOMAIN_ORDER:
        if counts.get(domain, 0) != 5:
            errors.append(f"{domain}: locked count is {counts.get(domain, 0)}, expected 5.")

    write_csv(LOCKED_OUTPUT, selected)
    write_csv(RESERVE_OUTPUT, reserves)
    _, locked_errors, locked_warnings = validate_cases(LOCKED_OUTPUT)
    _, reserve_errors, reserve_warnings = validate_cases(RESERVE_OUTPUT)
    errors.extend(locked_errors)
    errors.extend(reserve_errors)

    report = {
        "timestamp_utc": utc_now(),
        "purpose": "Final formal_v1 case-set locking audit before any formal model generation.",
        "selection_rule": (
            "Within each clinical domain, sort double-clinician-approved candidates by numeric case-ID suffix "
            "ascending and select the first five. Retain remaining approved cases as reserves. No model outputs "
            "or pilot performance are used."
        ),
        "source_candidate_count": len(source),
        "locked_case_count": len(selected),
        "reserve_case_count": len(reserves),
        "locked_domain_counts": {domain: counts.get(domain, 0) for domain in DOMAIN_ORDER},
        "locked_case_ids": [row["case_id"] for row in selected],
        "reserve_case_ids": [row["case_id"] for row in reserves],
        "files": {
            "approved_candidates": {"path": str(INPUT.relative_to(ROOT)), "sha256": sha256(INPUT)},
            "locked_cases": {"path": str(LOCKED_OUTPUT.relative_to(ROOT)), "sha256": sha256(LOCKED_OUTPUT)},
            "approved_reserves": {"path": str(RESERVE_OUTPUT.relative_to(ROOT)), "sha256": sha256(RESERVE_OUTPUT)},
        },
        "errors": errors,
        "warnings": {"locked": locked_warnings, "reserves": reserve_warnings},
        "passed": not errors,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"LOCK PASSED: {len(selected)} formal cases; {len(reserves)} approved reserves")
    print("LOCKED DOMAIN COUNTS: " + ", ".join(f"{domain}={counts.get(domain, 0)}" for domain in DOMAIN_ORDER))
    print("RESERVE IDS: " + ", ".join(row["case_id"] for row in reserves))
    print(f"Locked cases: {LOCKED_OUTPUT}")
    print(f"Reserves: {RESERVE_OUTPUT}")
    print(f"Report: {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
