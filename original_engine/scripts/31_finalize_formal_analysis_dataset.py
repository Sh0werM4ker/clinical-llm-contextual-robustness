from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
RATINGS = PROJECT / "ratings" / "formal_local_v1"
TARGETED = RATINGS / "targeted_clinician_review_v1"

PRE_REVIEW = TARGETED / "formal_merged_pre_review_dataset_v1.csv"
HUMAN_GOLD = RATINGS / "human_gold_150_final_v2.csv"
DOCTOR3 = (
    TARGETED
    / "completed_intake_doctor3_v1"
    / "targeted_review_734_extracted_candidate_v1.csv"
)
OUTPUT = RATINGS / "formal_final_analysis_dataset_v1.csv"
QC_OUTPUT = RATINGS / "formal_final_analysis_dataset_qc_v1.json"

OUTCOMES = [
    "top1_correct",
    "gold_diagnosis_rank",
    "incorrect_anchor_rank",
    "anchor_adoption",
    "evidence_override",
    "triage_level_scored",
    "mandatory_action_complete",
    "unsafe_management",
    "over_conservatism",
]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    pre = read_csv(PRE_REVIEW)
    gold = read_csv(HUMAN_GOLD)
    doctor3 = read_csv(DOCTOR3)

    require(len(pre) == 884, f"Expected 884 master records, found {len(pre)}")
    require(len(gold) == 150, f"Expected 150 adjudicated gold records, found {len(gold)}")
    require(len(doctor3) == 734, f"Expected 734 Doctor3 records, found {len(doctor3)}")
    for name, frame in [("pre-review", pre), ("human gold", gold), ("Doctor3", doctor3)]:
        require(frame["blinded_id"].is_unique, f"Duplicate blinded_id in {name}")

    pre_ids = set(pre["blinded_id"])
    gold_ids = set(gold["blinded_id"])
    doctor3_ids = set(doctor3["blinded_id"])
    require(not (gold_ids & doctor3_ids), "Human-gold and Doctor3 ID sets overlap")
    require(gold_ids | doctor3_ids == pre_ids, "Human label sets do not exactly cover master IDs")

    gold_map = gold.set_index("blinded_id")
    doctor3_map = doctor3.set_index("blinded_id")
    final = pre.copy()

    for outcome in OUTCOMES:
        values: list[str] = []
        for blinded_id in final["blinded_id"]:
            source = gold_map if blinded_id in gold_ids else doctor3_map
            values.append(str(source.at[blinded_id, outcome]).strip())
        final[outcome] = values

    final["final_label_source"] = final["blinded_id"].map(
        lambda item: (
            "human_gold_two_clinicians_plus_adjudication_v2"
            if item in gold_ids
            else "targeted_single_clinician_doctor3_v1"
        )
    )
    final["final_label_reviewer_code"] = final["blinded_id"].map(
        lambda item: "adjudicated_human_gold" if item in gold_ids else "Doctor3"
    )
    final["final_analysis_eligible"] = "1"

    # Validate final labels.
    binary = [
        "top1_correct",
        "evidence_override",
        "mandatory_action_complete",
        "unsafe_management",
        "over_conservatism",
    ]
    for column in binary:
        invalid = set(final[column]) - {"0", "1"}
        require(not invalid, f"Invalid values in {column}: {sorted(invalid)}")
    invalid = set(final["gold_diagnosis_rank"]) - {"1", "2", "3", "4", "5", "6"}
    require(not invalid, f"Invalid gold diagnosis ranks: {sorted(invalid)}")
    invalid = set(final["triage_level_scored"]) - {
        "routine",
        "urgent_outpatient",
        "same_day",
        "emergency",
    }
    require(not invalid, f"Invalid triage values: {sorted(invalid)}")

    anchor_applicable = final["anchor_condition"].eq("incorrect_label")
    for column, allowed in {
        "incorrect_anchor_rank": {"1", "2", "3", "4", "5", "6"},
        "anchor_adoption": {"0", "1", "2"},
    }.items():
        require(
            set(final.loc[anchor_applicable, column]) <= allowed,
            f"Invalid applicable values in {column}",
        )
        require(
            set(final.loc[~anchor_applicable, column]) <= {""},
            f"Non-applicable {column} must be blank",
        )

    require(
        not ((final["top1_correct"] == "1") & (final["gold_diagnosis_rank"] != "1")).any(),
        "top1_correct=1 must imply gold_diagnosis_rank=1",
    )
    require(
        not ((final["mandatory_action_complete"] == "0") & (final["unsafe_management"] != "1")).any(),
        "mandatory_action_complete=0 must imply unsafe_management=1",
    )
    triage_score = {"routine": 1, "urgent_outpatient": 2, "same_day": 3, "emergency": 4}
    undertriage = final.apply(
        lambda row: triage_score[row["triage_level_scored"]]
        < triage_score[row["reference_triage"]],
        axis=1,
    )
    require(
        not (undertriage & (final["unsafe_management"] != "1")).any(),
        "Under-triage must imply unsafe_management=1",
    )
    require(set(final["is_mock"]) == {"0"}, "Mock records found in formal dataset")

    # Stable ordering makes future hashes and comparisons reproducible.
    final = final.sort_values("blinded_id", kind="stable").reset_index(drop=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    counts = {
        "part": final["part"].value_counts().sort_index().to_dict(),
        "anchor_condition": final["anchor_condition"].value_counts().sort_index().to_dict(),
        "intervention": final["intervention"].value_counts().sort_index().to_dict(),
        "model": final["logical_model_id"].value_counts().sort_index().to_dict(),
        "domain": final["domain"].value_counts().sort_index().to_dict(),
        "final_label_source": final["final_label_source"].value_counts().sort_index().to_dict(),
    }
    qc = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "records": len(final),
        "unique_blinded_ids": int(final["blinded_id"].nunique()),
        "anchor_applicable_records": int(anchor_applicable.sum()),
        "input_sha256": {
            str(PRE_REVIEW.relative_to(PROJECT)): sha256(PRE_REVIEW),
            str(HUMAN_GOLD.relative_to(PROJECT)): sha256(HUMAN_GOLD),
            str(DOCTOR3.relative_to(PROJECT)): sha256(DOCTOR3),
        },
        "output_sha256": sha256(OUTPUT),
        "counts": counts,
        "logic_checks": {
            "top1_rank_consistency": "PASS",
            "mandatory_action_unsafe_consistency": "PASS",
            "undertriage_unsafe_consistency": "PASS",
            "no_mock_records": "PASS",
            "human_coverage_exact": "PASS",
        },
    }
    QC_OUTPUT.write_text(json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(qc, indent=2, ensure_ascii=False))
    print(f"Final dataset: {OUTPUT}")
    print(f"QC report: {QC_OUTPUT}")


if __name__ == "__main__":
    main()
