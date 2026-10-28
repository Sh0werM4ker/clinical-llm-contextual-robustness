from __future__ import annotations

import re
from pathlib import Path

from .io_utils import read_csv


REQUIRED_COLUMNS = [
    "case_id",
    "domain",
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
    "case_status",
    "expert_review_status",
    "notes",
]
ALLOWED_TRIAGE = {"emergency", "same_day", "urgent_outpatient", "routine"}
ALLOWED_STATUS = {"draft", "reviewed", "locked"}
DISALLOWED_CUE_PHRASES = {
    "prior diagnosis may be wrong",
    "previous diagnosis may be wrong",
    "ignore the prior diagnosis",
    "reassess independently",
    "anchoring bias",
}


def load_cases(path: Path) -> list[dict[str, str]]:
    rows = read_csv(path)
    return [{key: (value or "").strip() for key, value in row.items()} for row in rows]


def validate_cases(path: Path) -> tuple[list[dict[str, str]], list[str], list[str]]:
    rows = load_cases(path)
    errors: list[str] = []
    warnings: list[str] = []
    if not rows:
        return rows, ["Case file contains no data rows."], warnings

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in rows[0]]
    if missing_columns:
        return rows, [f"Missing columns: {missing_columns}"], warnings

    seen: set[str] = set()
    for excel_row, row in enumerate(rows, start=2):
        case_id = row["case_id"] or f"row_{excel_row}"
        if not row["case_id"]:
            errors.append(f"Row {excel_row}: case_id is blank.")
        elif row["case_id"] in seen:
            errors.append(f"Row {excel_row}: duplicate case_id {row['case_id']}.")
        seen.add(row["case_id"])

        for column in REQUIRED_COLUMNS:
            if column not in {"acceptable_actions", "notes"} and not row[column]:
                errors.append(f"{case_id}: required field {column} is blank.")

        if row["reference_triage"] not in ALLOWED_TRIAGE:
            errors.append(
                f"{case_id}: reference_triage must be one of {sorted(ALLOWED_TRIAGE)}."
            )
        if row["case_status"] not in ALLOWED_STATUS:
            errors.append(f"{case_id}: invalid case_status {row['case_status']!r}.")
        if row["correct_anchor"].casefold() == row["incorrect_anchor"].casefold():
            errors.append(f"{case_id}: correct and incorrect anchors are identical.")

        word_count = len(re.findall(r"\b[\w-]+\b", row["objective_case"]))
        if word_count < 75 or word_count > 220:
            warnings.append(
                f"{case_id}: objective_case has {word_count} words; target is 120-180."
            )
        lower_case = row["objective_case"].casefold()
        for phrase in DISALLOWED_CUE_PHRASES:
            if phrase in lower_case:
                errors.append(f"{case_id}: objective_case contains cue phrase {phrase!r}.")

        for list_field in (
            "acceptable_synonyms",
            "key_red_flags",
            "mandatory_actions",
            "unsafe_actions",
        ):
            items = [item.strip() for item in row[list_field].split("|") if item.strip()]
            if len(items) < 2:
                warnings.append(f"{case_id}: {list_field} contains fewer than 2 items.")

        if row["case_status"] == "locked" and row["expert_review_status"] not in {
            "pilot_approved",
            "expert_approved",
        }:
            errors.append(
                f"{case_id}: locked case must have pilot_approved or expert_approved review status."
            )
    return rows, errors, warnings

