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


FIELDS = [
    "case_id", "gold_diagnosis", "acceptable_synonyms", "correct_anchor", "incorrect_anchor",
    "objective_case", "key_red_flags", "reference_triage", "mandatory_actions",
    "acceptable_actions", "unsafe_actions",
]


def read_text_compatibly(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030", "gbk"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path}.")


def read_csv_compatibly(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(read_text_compatibly(path).splitlines()))


def parse_cases(path: Path) -> list[dict[str, str]]:
    text = read_text_compatibly(path)
    blocks = re.split(r"(?=^case_id:\s*)", text, flags=re.MULTILINE)
    field_pattern = re.compile(
        r"(?:^|\n|;\s*)(" + "|".join(FIELDS) + r"):\s*",
        flags=re.MULTILINE,
    )
    rows: list[dict[str, str]] = []
    for block in blocks:
        if not block.lstrip().startswith("case_id:"):
            continue
        matches = list(field_pattern.finditer(block))
        row = {field: "" for field in FIELDS}
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(block)
            row[match.group(1)] = block[match.end():end].strip().rstrip(";").strip()
        row["source_file"] = path.name
        rows.append(row)
    if rows:
        return rows

    # A compact web format may place every named field on one physical line:
    # ``case_id; ONC02; gold_diagnosis; ...``.  Parse by the known field labels
    # rather than by raw semicolon position because the vignette itself may
    # legitimately contain semicolons.
    compact_rows: list[dict[str, str]] = []
    compact_blocks = re.split(r"(?=^case_id\s*;\s*)", text, flags=re.MULTILINE)
    compact_field_pattern = re.compile(
        r"(?:^|;\s*)(" + "|".join(FIELDS) + r")(?:\s*\([^)]*\))?\s*;\s*",
        flags=re.MULTILINE,
    )
    for block in compact_blocks:
        if not block.lstrip().startswith("case_id;"):
            continue
        matches = list(compact_field_pattern.finditer(block))
        if len(matches) < len(FIELDS):
            continue
        row = {field: "" for field in FIELDS}
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(block)
            value = block[match.end():end].strip().rstrip(";").strip()
            if match.group(1) == FIELDS[-1]:
                value = re.split(r"\n\s*Human clinical review\s*:", value, maxsplit=1)[0].strip()
            row[match.group(1)] = value
        if not re.fullmatch(r"[A-Z]+\d{2}", row["case_id"]):
            continue
        row["source_file"] = path.name
        compact_rows.append(row)
    if compact_rows:
        return compact_rows

    # A second common web format places one named field on each line but uses
    # semicolons rather than colons, sometimes adding display hints such as
    # "(use |)" after the field name.
    semicolon_rows: list[dict[str, str]] = []
    semicolon_blocks = re.split(r"(?=^case_id\s*;\s*)", text, flags=re.MULTILINE)
    for block in semicolon_blocks:
        if not block.lstrip().startswith("case_id;"):
            continue
        row = {field: "" for field in FIELDS}
        for raw_line in block.splitlines():
            line = raw_line.strip()
            for field in FIELDS:
                match = re.match(rf"^{field}(?:\s*\([^)]*\))?\s*;\s*(.*)$", line)
                if match:
                    row[field] = match.group(1).strip().rstrip(";").strip()
                    break
        if not re.fullmatch(r"[A-Z]+\d{2}", row["case_id"]):
            continue
        row["source_file"] = path.name
        semicolon_rows.append(row)
    if semicolon_rows:
        return semicolon_rows

    # Some web responses follow the requested field order but omit field names,
    # yielding one semicolon-delimited record per line. Accept only the exact
    # expected column count so an unparseable response never silently shifts data.
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.casefold().startswith("case_id;"):
            continue
        raw_parts = [part.strip().rstrip(".").strip() for part in line.split(";")]
        if not re.fullmatch(r"[A-Z]+\d{2}", raw_parts[0] if raw_parts else ""):
            continue
        if len(raw_parts) < len(FIELDS):
            raise ValueError(
                f"{path.name}: positional record {raw_parts[0]!r} has {len(raw_parts)} fields; expected at least {len(FIELDS)}."
            )
        # objective_case is the sixth field and may itself contain semicolons.
        # All five trailing score-card fields are structured pipe-delimited text.
        parts = raw_parts[:5] + ["; ".join(raw_parts[5:-5])] + raw_parts[-5:]
        row = dict(zip(FIELDS, parts, strict=True))
        row["source_file"] = path.name
        rows.append(row)
    return rows


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def main() -> int:
    parser = argparse.ArgumentParser(description="Import revised web-GPT formal case drafts without overwriting originals.")
    parser.add_argument("--draft-dir", default="data/web_gpt_drafts/revised_v1")
    parser.add_argument("--original-cases", default="data/cases_formal_candidate_draft.csv")
    parser.add_argument("--reconciliation", default="ratings/formal_case_expert_review_v1/reconciliation/formal_case_review_reconciliation.csv")
    parser.add_argument("--output", default="data/cases_formal_candidate_revised_v1.csv")
    parser.add_argument("--report", default="results/formal_case_revision_import_v1/import_qc_report.json")
    parser.add_argument("--additional-held", default="", help="Comma-separated case IDs excluded from this revision round.")
    args = parser.parse_args()

    draft_dir = ROOT / args.draft_dir
    original_path = ROOT / args.original_cases
    reconciliation_path = ROOT / args.reconciliation
    output_path, report_path = ROOT / args.output, ROOT / args.report
    originals = {row["case_id"]: row for row in read_csv_compatibly(original_path)}
    reconciliation = read_csv_compatibly(reconciliation_path)
    allowed_ids = {
        row["case_id"]
        for row in reconciliation
        if row["recommended_next_action"] == "revise_then_repeat_independent_review"
    }
    held_ids = {
        row["case_id"]
        for row in reconciliation
        if row["recommended_next_action"] == "senior_clinician_decision_required_revise_or_exclude"
    }
    held_ids.update(value.strip() for value in args.additional_held.split(",") if value.strip())
    # Explicitly held cases are intentionally excluded from this revision
    # round, so they must not be counted as missing allowed revisions.
    allowed_ids.difference_update(held_ids)
    records: list[dict[str, str]] = []
    hashes: dict[str, str] = {}
    for path in sorted(draft_dir.glob("*.txt")):
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        records.extend(parse_cases(path))

    errors: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()
    output_rows: list[dict[str, str]] = []
    details: list[dict[str, object]] = []
    for row in records:
        case_id = row["case_id"]
        if not case_id:
            errors.append(f"{row['source_file']}: blank case ID.")
            continue
        if case_id in seen:
            errors.append(f"Duplicate revised case ID: {case_id}.")
        seen.add(case_id)
        if case_id in held_ids:
            errors.append(f"{case_id}: held for senior clinician decision and must not be revised yet.")
        if case_id not in allowed_ids:
            errors.append(f"{case_id}: not an allowed revision candidate.")
        missing = [field for field in FIELDS if not row[field]]
        if missing:
            errors.append(f"{case_id}: missing {missing}.")
        words = word_count(row["objective_case"])
        if not 120 <= words <= 180:
            errors.append(f"{case_id}: objective_case has {words} words, outside 120-180.")
        if row["reference_triage"] not in ALLOWED_TRIAGE:
            errors.append(f"{case_id}: invalid reference_triage {row['reference_triage']!r}.")
        lower = row["objective_case"].casefold()
        cues = [cue for cue in DISALLOWED_CUE_PHRASES if cue in lower]
        if cues:
            errors.append(f"{case_id}: objective_case contains forbidden cue phrase(s) {cues}.")
        changed = [
            field for field in ("gold_diagnosis", "correct_anchor", "incorrect_anchor", "reference_triage")
            if originals.get(case_id, {}).get(field, "").strip() != row[field].strip()
        ]
        if changed:
            warnings.append(f"{case_id}: revised {', '.join(changed)}; requires explicit clinical re-approval.")
        original = originals.get(case_id, {})
        output_rows.append({
            "case_id": case_id,
            "domain": original.get("domain", ""),
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
            "expert_review_status": "pending_revision_review",
            "notes": f"Revision v1 imported from {row['source_file']}; requires repeat independent clinical review before locking.",
        })
        details.append({"case_id": case_id, "source_file": row["source_file"], "word_count": words, "changed_design_fields": changed})

    missing_allowed = sorted(allowed_ids - seen)
    if missing_allowed:
        errors.append(f"Missing revised cases: {missing_allowed}.")
    report = {
        "timestamp_utc": utc_now(),
        "purpose": "Structural QC of revised web-GPT case drafts; not clinical approval.",
        "expected_revised_case_count": len(allowed_ids),
        "held_case_ids": sorted(held_ids),
        "imported_case_count": len(records),
        "source_sha256": hashes,
        "errors": errors,
        "warnings": warnings,
        "cases": details,
        "passed": not errors,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print(f"IMPORT FAILED. Report: {report_path}")
        return 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"IMPORT PASSED: {len(output_rows)} revised draft cases")
    print(f"Warnings: {len(warnings)}")
    print(f"Output: {output_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
