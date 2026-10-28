from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


INPUT_COLUMNS = [
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
]
REVIEW_COLUMNS = [
    "gold_diagnosis_clarity_1_to_5",
    "incorrect_anchor_plausibility_1_to_5",
    "objective_evidence_sufficiency_1_to_5",
    "triage_and_management_clarity_1_to_5",
    "anchoring_test_calibration_1_to_5",
    "recommended_disposition_include_revise_exclude",
    "required_revisions",
    "reviewer_comments",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_form(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INPUT_COLUMNS + REVIEW_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in INPUT_COLUMNS + REVIEW_COLUMNS})


def main() -> int:
    parser = argparse.ArgumentParser(description="Export two independent formal-case clinical expert review forms.")
    parser.add_argument("--input", default="data/cases_formal_candidate_draft.csv")
    parser.add_argument("--output-dir", default="ratings/formal_case_expert_review_v1")
    parser.add_argument("--expected-count", type=int, default=36)
    args = parser.parse_args()

    source = ROOT / args.input
    out_dir = ROOT / args.output_dir
    rows = read_rows(source)
    if len(rows) != args.expected_count:
        raise SystemExit(f"Expected {args.expected_count} candidate cases, found {len(rows)}.")
    if any(row.get("case_status") != "draft" for row in rows):
        raise SystemExit("All candidate cases must remain draft before independent expert review.")

    for label in ("A", "B"):
        form = out_dir / f"formal_case_expert_review_form_{label}.csv"
        write_form(rows, form)
        print(f"Wrote {form}")
    print("Use the same codebook for both reviewers. Reviewers must complete forms independently.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
