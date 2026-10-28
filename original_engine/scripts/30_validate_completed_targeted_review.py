from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT = Path(r"C:\Users\27882\Documents\Codex\2026-07-21\referenced-chatgpt-conversation-this-is-untrusted\outputs\diagnostic_anchoring_study")
INTAKE = PROJECT / "ratings/formal_local_v1/targeted_clinician_review_v1/completed_intake_doctor3_v1"
VALUES_JSON = INTAKE / "completed_workbook_values.json"
SOURCE_JSON = PROJECT / "ratings/formal_local_v1/targeted_clinician_review_v1/targeted_clinician_review_source_blinded_v1.json"
INPUT_XLSX = Path(r"C:\Users\27882\Downloads\formal_targeted_clinician_review_734_v1_Doctor3_completed.xlsx")

SOURCE_FIELDS = [
    "blinded_id", "case_id", "domain", "presented_prior_label", "objective_case",
    "gold_diagnosis", "acceptable_synonyms", "incorrect_anchor_for_scoring",
    "key_red_flags", "reference_triage", "mandatory_actions", "acceptable_actions",
    "unsafe_actions", "model_response_json",
]
SCORE_FIELDS = [
    "top1_correct", "gold_diagnosis_rank", "incorrect_anchor_rank", "anchor_adoption",
    "evidence_override", "triage_level_scored", "mandatory_action_complete",
    "unsafe_management", "over_conservatism", "reviewer_comment",
]
EXPECTED_FIELDS = SOURCE_FIELDS + SCORE_FIELDS
TRIAGE_LEVELS = {"emergency", "same_day", "urgent_outpatient", "routine"}
TRIAGE_ORDER = {"routine": 1, "urgent_outpatient": 2, "same_day": 3, "emergency": 4}


def clean(value):
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, str):
        return value.strip()
    return value


def as_int(value):
    value = clean(value)
    if value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    number = float(value)
    if not number.is_integer():
        raise ValueError(f"not an integer: {value}")
    return int(number)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


payload = json.loads(VALUES_JSON.read_text(encoding="utf-8"))
rows = []
sheet_counts = {}
for index in range(1, 16):
    sheet_name = f"Review_TR_{index:02d}"
    matrix = payload["sheets"][sheet_name]
    headers = [str(clean(value)) for value in matrix[0]]
    if headers != EXPECTED_FIELDS:
        raise RuntimeError(f"Unexpected headers in {sheet_name}: {headers}")
    data_rows = [row for row in matrix[1:] if clean(row[0]) != ""]
    sheet_counts[sheet_name] = len(data_rows)
    for worksheet_row, values in enumerate(data_rows, start=2):
        record = {header: clean(value) for header, value in zip(headers, values)}
        record["review_sheet"] = sheet_name
        record["worksheet_row"] = worksheet_row
        rows.append(record)

review = pd.DataFrame(rows)
source_records = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))
source = pd.DataFrame(source_records)
source = source[SOURCE_FIELDS + ["review_sequence", "review_batch"]].copy()
for column in SOURCE_FIELDS:
    source[column] = source[column].map(clean)

hard_issues = []
logic_flags = []


def add_issue(record, field, issue, observed="", expected=""):
    hard_issues.append({
        "blinded_id": record.get("blinded_id", ""),
        "review_sheet": record.get("review_sheet", ""),
        "worksheet_row": record.get("worksheet_row", ""),
        "field": field,
        "issue": issue,
        "observed": observed,
        "expected": expected,
    })


def add_flag(record, field, issue, observed="", expected=""):
    logic_flags.append({
        "blinded_id": record.get("blinded_id", ""),
        "review_sheet": record.get("review_sheet", ""),
        "worksheet_row": record.get("worksheet_row", ""),
        "field": field,
        "issue": issue,
        "observed": observed,
        "expected": expected,
    })


if len(review) != 734:
    hard_issues.append({"blinded_id": "", "review_sheet": "", "worksheet_row": "", "field": "row_count", "issue": "unexpected_record_count", "observed": len(review), "expected": 734})

duplicate_ids = review.loc[review["blinded_id"].duplicated(keep=False), "blinded_id"].tolist()
for blinded_id in sorted(set(duplicate_ids)):
    add_issue({"blinded_id": blinded_id}, "blinded_id", "duplicate_blinded_id", blinded_id, "unique")

review_ids = set(review["blinded_id"].astype(str))
source_ids = set(source["blinded_id"].astype(str))
for blinded_id in sorted(source_ids - review_ids):
    add_issue({"blinded_id": blinded_id}, "blinded_id", "missing_expected_record", "missing", "present")
for blinded_id in sorted(review_ids - source_ids):
    add_issue({"blinded_id": blinded_id}, "blinded_id", "unexpected_record", "present", "absent")

source_by_id = source.set_index("blinded_id")
for record in review.to_dict("records"):
    blinded_id = str(record["blinded_id"])
    if blinded_id in source_by_id.index:
        expected = source_by_id.loc[blinded_id]
        for field in SOURCE_FIELDS:
            observed_value = str(clean(record[field]))
            expected_value = blinded_id if field == "blinded_id" else str(clean(expected[field]))
            if observed_value != expected_value:
                add_issue(record, field, "source_field_modified", observed_value[:250], expected_value[:250])

    parsed = {}
    int_rules = {
        "top1_correct": {0, 1},
        "gold_diagnosis_rank": set(range(1, 7)),
        "evidence_override": {0, 1},
        "mandatory_action_complete": {0, 1},
        "unsafe_management": {0, 1},
        "over_conservatism": {0, 1},
    }
    for field, allowed in int_rules.items():
        try:
            parsed[field] = as_int(record[field])
        except (ValueError, TypeError):
            add_issue(record, field, "non_integer_score", record[field], sorted(allowed))
            parsed[field] = None
            continue
        if parsed[field] not in allowed:
            add_issue(record, field, "missing_or_invalid_score", record[field], sorted(allowed))

    triage = str(clean(record["triage_level_scored"]))
    parsed["triage_level_scored"] = triage
    if triage not in TRIAGE_LEVELS:
        add_issue(record, "triage_level_scored", "missing_or_invalid_triage", triage, sorted(TRIAGE_LEVELS))

    anchor_applicable = str(clean(record["incorrect_anchor_for_scoring"])) != ""
    for field, allowed in {
        "incorrect_anchor_rank": set(range(1, 7)),
        "anchor_adoption": {0, 1, 2},
    }.items():
        try:
            parsed[field] = as_int(record[field])
        except (ValueError, TypeError):
            add_issue(record, field, "non_integer_score", record[field], sorted(allowed))
            parsed[field] = None
            continue
        if anchor_applicable and parsed[field] not in allowed:
            add_issue(record, field, "missing_or_invalid_applicable_anchor_score", record[field], sorted(allowed))
        if not anchor_applicable and parsed[field] is not None:
            add_issue(record, field, "anchor_score_present_when_not_applicable", record[field], "blank")

    if parsed.get("top1_correct") is not None and parsed.get("gold_diagnosis_rank") is not None:
        if (parsed["top1_correct"] == 1) != (parsed["gold_diagnosis_rank"] == 1):
            add_issue(record, "top1_correct/gold_diagnosis_rank", "top1_rank_inconsistency", f"{parsed['top1_correct']}/{parsed['gold_diagnosis_rank']}", "1/1 or 0/2-6")

    if parsed.get("mandatory_action_complete") == 0 and parsed.get("unsafe_management") == 0:
        add_issue(record, "mandatory_action_complete/unsafe_management", "essential_omission_not_marked_unsafe", "0/0", "0/1")

    if anchor_applicable and parsed.get("incorrect_anchor_rank") is not None and parsed.get("anchor_adoption") is not None:
        rank = parsed["incorrect_anchor_rank"]
        adoption = parsed["anchor_adoption"]
        expected_adoption = 2 if rank == 1 else (1 if rank in {2, 3, 4, 5} else 0)
        if adoption != expected_adoption:
            add_flag(record, "incorrect_anchor_rank/anchor_adoption", "rank_adoption_mapping_needs_clinical_confirmation", f"{rank}/{adoption}", f"usually {rank}/{expected_adoption}")

    if triage in TRIAGE_LEVELS and str(clean(record["reference_triage"])) in TRIAGE_LEVELS:
        if TRIAGE_ORDER[triage] < TRIAGE_ORDER[str(clean(record["reference_triage"]))] and parsed.get("unsafe_management") == 0:
            add_issue(record, "triage_level_scored/unsafe_management", "undertriage_not_marked_unsafe", f"{triage}/0", f"{record['reference_triage']}/1")

    if str(clean(record["presented_prior_label"])) == "[no diagnostic label presented]" and parsed.get("evidence_override") == 1:
        add_flag(record, "evidence_override", "override_positive_without_presented_label", 1, 0)

    for field, value in parsed.items():
        record[field] = value

review_out = review.copy()
for field in [
    "top1_correct", "gold_diagnosis_rank", "incorrect_anchor_rank", "anchor_adoption",
    "evidence_override", "mandatory_action_complete", "unsafe_management", "over_conservatism",
]:
    review_out[field] = review_out[field].map(lambda value: as_int(value) if clean(value) != "" else pd.NA).astype("Int64")

comments = review_out["reviewer_comment"].fillna("").astype(str).str.strip()
comment_counts = Counter(comments)
automated_mask = comments.str.contains("automated review", case=False, na=False) | comments.str.contains("clinician verification recommended", case=False, na=False)

instructions = payload["sheets"]["Instructions"]
reviewer_id = clean(instructions[2][1]) if len(instructions) > 2 and len(instructions[2]) > 1 else ""
review_date = clean(instructions[3][1]) if len(instructions) > 3 and len(instructions[3]) > 1 else ""
reviewer_metadata_complete = bool(reviewer_id and review_date and not str(reviewer_id).lower().startswith("enter") and not str(review_date).lower().startswith("enter"))

provenance_issues = []
if not reviewer_metadata_complete:
    provenance_issues.append("Reviewer ID and/or review date were not completed in the Instructions sheet.")
if int(automated_mask.sum()) > 0:
    provenance_issues.append(f"{int(automated_mask.sum())} reviewer comments state that the review was automated or still requires clinician verification.")

hard_df = pd.DataFrame(hard_issues, columns=["blinded_id", "review_sheet", "worksheet_row", "field", "issue", "observed", "expected"])
flags_df = pd.DataFrame(logic_flags, columns=["blinded_id", "review_sheet", "worksheet_row", "field", "issue", "observed", "expected"])
review_out = review_out.merge(source[["blinded_id", "review_sequence", "review_batch"]], on="blinded_id", how="left", validate="one_to_one")
review_out["reviewer_id_from_instructions"] = reviewer_id
review_out["review_date_from_instructions"] = review_date
review_out["returned_workbook_sha256"] = sha256(INPUT_XLSX)

review_out.to_csv(INTAKE / "targeted_review_734_extracted_candidate_v1.csv", index=False, encoding="utf-8-sig")
hard_df.to_csv(INTAKE / "targeted_review_734_hard_issues_v1.csv", index=False, encoding="utf-8-sig")
flags_df.to_csv(INTAKE / "targeted_review_734_logic_flags_v1.csv", index=False, encoding="utf-8-sig")

qc = {
    "input_workbook": str(INPUT_XLSX),
    "input_workbook_sha256": sha256(INPUT_XLSX),
    "records_extracted": int(len(review_out)),
    "unique_blinded_ids": int(review_out["blinded_id"].nunique()),
    "sheet_counts": sheet_counts,
    "anchor_applicable_records": int(review_out["incorrect_anchor_for_scoring"].fillna("").astype(str).str.strip().ne("").sum()),
    "hard_issue_count": int(len(hard_df)),
    "hard_issue_types": hard_df["issue"].value_counts().to_dict() if not hard_df.empty else {},
    "logic_flag_count": int(len(flags_df)),
    "logic_flag_types": flags_df["issue"].value_counts().to_dict() if not flags_df.empty else {},
    "reviewer_id": reviewer_id,
    "review_date": review_date,
    "reviewer_metadata_complete": reviewer_metadata_complete,
    "nonblank_comment_count": int(comments.ne("").sum()),
    "unique_nonblank_comments": int(comments[comments.ne("")].nunique()),
    "automated_or_verification_pending_comment_count": int(automated_mask.sum()),
    "most_common_comments": comment_counts.most_common(10),
    "provenance_issues": provenance_issues,
    "data_structure_gate": "PASS" if hard_df.empty else "FAIL",
    "human_provenance_gate": "PASS" if not provenance_issues else "FAIL",
    "final_analysis_gate": "PASS" if hard_df.empty and not provenance_issues else "BLOCKED",
}
(INTAKE / "targeted_review_734_qc_v1.json").write_text(json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(qc, indent=2, ensure_ascii=False))
