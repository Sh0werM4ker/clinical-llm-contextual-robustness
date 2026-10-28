from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchoring_study.cases import ALLOWED_TRIAGE, DISALLOWED_CUE_PHRASES, REQUIRED_COLUMNS
from anchoring_study.io_utils import utc_now


SOURCE_FIELDS = [
    "case_id",
    "gold_diagnosis",
    "acceptable_synonyms",
    "correct_anchor",
    "incorrect_anchor",
    "objective_case",
    "key_red_flags",
    "reference_triage",
    "mandatory_actions",
    "acceptable_actions",
    "unsafe_actions",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{key: (value or "").strip() for key, value in row.items()} for row in csv.DictReader(handle)]


def parse_source_text(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8-sig")
    blocks = re.split(r"(?=^case_id:\s*)", text, flags=re.MULTILINE)
    records: list[dict[str, str]] = []
    for block in blocks:
        if not block.startswith("case_id:"):
            continue
        row: dict[str, str] = {}
        for field in SOURCE_FIELDS:
            match = re.search(rf"^{field}:\s*(.*?)(?:\s*;)?\s*$", block, flags=re.MULTILINE)
            row[field] = match.group(1).strip().rstrip(";").strip() if match else ""
        row["source_file"] = path.name
        records.append(row)
    return records


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def has_anchor_name_in_vignette(row: dict[str, str]) -> bool:
    vignette = row["objective_case"].casefold()
    for anchor_field in ("correct_anchor", "incorrect_anchor"):
        anchor = row[anchor_field].casefold()
        if len(anchor) >= 6 and anchor in vignette:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Import web-GPT synthetic case drafts into a non-locked study CSV.")
    parser.add_argument("--draft-dir", default="data/web_gpt_drafts")
    parser.add_argument("--register", default="data/formal_case_development_register.csv")
    parser.add_argument("--output", default="data/cases_formal_candidate_draft.csv")
    parser.add_argument("--report", default="results/formal_case_draft_import_v1/import_qc_report.json")
    args = parser.parse_args()

    draft_dir = ROOT / args.draft_dir
    register_path = ROOT / args.register
    output_path = ROOT / args.output
    report_path = ROOT / args.report

    if not draft_dir.is_dir():
        raise SystemExit(f"Draft directory not found: {draft_dir}")
    if not register_path.is_file():
        raise SystemExit(f"Development register not found: {register_path}")

    expected = {row["case_id"]: row for row in read_csv_rows(register_path)}
    records: list[dict[str, str]] = []
    source_hashes: dict[str, str] = {}
    for path in sorted(draft_dir.glob("*.txt")):
        source_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        records.extend(parse_source_text(path))

    errors: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()
    output_rows: list[dict[str, str]] = []
    item_report: list[dict[str, object]] = []
    for row in records:
        case_id = row["case_id"]
        if not case_id:
            errors.append(f"{row['source_file']}: blank case_id.")
            continue
        if case_id in seen:
            errors.append(f"Duplicate case_id: {case_id}.")
        seen.add(case_id)
        if case_id not in expected:
            errors.append(f"{case_id}: not present in the formal development register.")

        missing = [field for field in SOURCE_FIELDS if not row[field]]
        if missing:
            errors.append(f"{case_id}: missing fields {missing}.")
        if row["reference_triage"] not in ALLOWED_TRIAGE:
            errors.append(f"{case_id}: invalid triage {row['reference_triage']!r}.")
        lower_vignette = row["objective_case"].casefold()
        cue_phrases = [phrase for phrase in DISALLOWED_CUE_PHRASES if phrase in lower_vignette]
        if cue_phrases:
            errors.append(f"{case_id}: forbidden cue phrases {cue_phrases}.")
        words = word_count(row["objective_case"])
        if not 120 <= words <= 180:
            warnings.append(f"{case_id}: objective_case has {words} words; target is 120-180.")
        anchor_name_present = has_anchor_name_in_vignette(row)
        if anchor_name_present:
            warnings.append(f"{case_id}: an anchor diagnosis name appears in the vignette; clinical review required.")

        domain = expected.get(case_id, {}).get("domain", "")
        output_rows.append(
            {
                "case_id": case_id,
                "domain": domain,
                "gold_diagnosis": row["gold_diagnosis"],
                "acceptable_synonyms": row["acceptable_synonyms"],
                "correct_anchor": row["correct_anchor"],
                "incorrect_anchor": row["incorrect_anchor"],
                "objective_case": row["objective_case"],
                "key_red_flags": row["key_red_flags"],
                "reference_triage": row["reference_triage"],
                "mandatory_actions": row["mandatory_actions"],
                "acceptable_actions": row["acceptable_actions"],
                "unsafe_actions": row["unsafe_actions"],
                "case_status": "draft",
                "expert_review_status": "pending",
                "notes": f"Imported from {row['source_file']}; web-GPT draft only; clinical expert review required before locking.",
            }
        )
        item_report.append(
            {
                "case_id": case_id,
                "source_file": row["source_file"],
                "objective_case_word_count": words,
                "anchor_name_in_objective_case": anchor_name_present,
                "missing_source_fields": missing,
            }
        )

    expected_ids = set(expected)
    missing_expected = sorted(expected_ids - seen)
    unexpected = sorted(seen - expected_ids)
    if missing_expected:
        errors.append(f"Missing expected case IDs: {missing_expected}.")
    if unexpected:
        errors.append(f"Unexpected case IDs: {unexpected}.")
    if len(records) != len(expected_ids):
        errors.append(f"Imported {len(records)} records but development register expects {len(expected_ids)}.")

    report = {
        "timestamp_utc": utc_now(),
        "purpose": "Import and structural QC of web-GPT synthetic case drafts. This is not clinical approval.",
        "draft_directory": str(draft_dir),
        "source_sha256": source_hashes,
        "development_register": str(register_path),
        "expected_case_count": len(expected_ids),
        "imported_case_count": len(records),
        "output_case_file": str(output_path),
        "all_imported_cases_are_draft": all(row["case_status"] == "draft" for row in output_rows),
        "errors": errors,
        "warnings": warnings,
        "cases": item_report,
        "passed": not errors,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print(f"IMPORT FAILED. QC report: {report_path}")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"IMPORT PASSED: {len(output_rows)} draft cases written to {output_path}")
    print(f"Warnings: {len(warnings)}")
    print(f"QC report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
