from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RATINGS = PROJECT_ROOT / "ratings" / "formal_local_v1"
SECONDARY = PROJECT_ROOT / "outputs" / "secondary_judge_glm47" / "formal"
OUTPUT = RATINGS / "targeted_clinician_review_v1"

MASTER_PATH = RATINGS / "blinding_master.csv"
REFERENCE_PATH = RATINGS / "rater_A.csv"
PRIMARY_PATH = RATINGS / "primary_ai_judge_consolidated.csv"
SECONDARY_PATH = SECONDARY / "secondary_judge_results.csv"
HUMAN_GOLD_PATH = RATINGS / "human_gold_150_final_v2.csv"
PRIMARY_QC_PATH = RATINGS / "primary_ai_judge_consolidation_qc.json"
SECONDARY_QC_PATH = SECONDARY / "secondary_judge_qc.json"
VALIDITY_GATE_PATH = SECONDARY.parent / "validation_gate_decision_v1.json"

SCORE_FIELDS = [
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

REFERENCE_FIELDS = [
    "blinded_id",
    "case_id",
    "domain",
    "presented_prior_label",
    "objective_case",
    "gold_diagnosis",
    "acceptable_synonyms",
    "incorrect_anchor_for_scoring",
    "key_red_flags",
    "reference_triage",
    "mandatory_actions",
    "acceptable_actions",
    "unsafe_actions",
    "model_response_json",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")


def assert_unique(frame: pd.DataFrame, key: str, label: str) -> None:
    duplicates = frame.loc[frame[key].duplicated(keep=False), key].tolist()
    if duplicates:
        raise RuntimeError(f"{label} contains duplicate {key} values: {duplicates[:10]}")


def require_exact_ids(expected: set[str], frame: pd.DataFrame, label: str) -> None:
    actual = set(frame["blinded_id"].astype(str))
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise RuntimeError(
            f"{label} ID mismatch: missing={missing[:10]} ({len(missing)}), "
            f"extra={extra[:10]} ({len(extra)})"
        )


def add_pairwise_conversion_flags(frame: pd.DataFrame) -> pd.DataFrame:
    flagged = frame.copy()
    for name in [
        "part1_top1_conversion_pair",
        "part1_unsafe_conversion_pair",
        "part2_top1_change_pair",
        "part2_unsafe_change_pair",
    ]:
        flagged[name] = 0

    p1 = flagged[flagged["part"] == "part1"].copy()
    p1_keys = ["case_id", "logical_model_id", "repeat_id"]
    for outcome, flag_name in [
        ("primary_top1_correct", "part1_top1_conversion_pair"),
        ("primary_unsafe_management", "part1_unsafe_conversion_pair"),
    ]:
        no = p1[p1["anchor_condition"] == "no_label"].set_index(p1_keys)
        wrong = p1[p1["anchor_condition"] == "incorrect_label"].set_index(p1_keys)
        paired = no[["blinded_id", outcome]].join(
            wrong[["blinded_id", outcome]], lsuffix="_no", rsuffix="_wrong", how="inner"
        )
        changed = paired[paired[f"{outcome}_no"] != paired[f"{outcome}_wrong"]]
        ids = set(changed["blinded_id_no"]) | set(changed["blinded_id_wrong"])
        flagged.loc[flagged["blinded_id"].isin(ids), flag_name] = 1

    baseline = flagged[
        (flagged["part"] == "part1")
        & (flagged["anchor_condition"] == "incorrect_label")
        & (flagged["repeat_id"] == "1")
    ].copy()
    baseline_index = baseline.set_index(["case_id", "logical_model_id"])
    interventions = flagged[flagged["part"] == "part2"].copy()
    for outcome, flag_name in [
        ("primary_top1_correct", "part2_top1_change_pair"),
        ("primary_unsafe_management", "part2_unsafe_change_pair"),
    ]:
        changed_ids: set[str] = set()
        for intervention in sorted(interventions["intervention"].unique()):
            candidate = interventions[interventions["intervention"] == intervention].set_index(
                ["case_id", "logical_model_id"]
            )
            paired = baseline_index[["blinded_id", outcome]].join(
                candidate[["blinded_id", outcome]],
                lsuffix="_baseline",
                rsuffix="_intervention",
                how="inner",
            )
            changed = paired[paired[f"{outcome}_baseline"] != paired[f"{outcome}_intervention"]]
            changed_ids.update(changed["blinded_id_baseline"])
            changed_ids.update(changed["blinded_id_intervention"])
        flagged.loc[flagged["blinded_id"].isin(changed_ids), flag_name] = 1
    return flagged


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    files = [
        MASTER_PATH,
        REFERENCE_PATH,
        PRIMARY_PATH,
        SECONDARY_PATH,
        HUMAN_GOLD_PATH,
        PRIMARY_QC_PATH,
        SECONDARY_QC_PATH,
        VALIDITY_GATE_PATH,
    ]
    missing_files = [str(path) for path in files if not path.exists()]
    if missing_files:
        raise RuntimeError(f"Required files are missing: {missing_files}")

    primary_qc = json.loads(PRIMARY_QC_PATH.read_text(encoding="utf-8"))
    secondary_qc = json.loads(SECONDARY_QC_PATH.read_text(encoding="utf-8"))
    validity_gate = json.loads(VALIDITY_GATE_PATH.read_text(encoding="utf-8"))
    if int(primary_qc.get("record_count", 0)) != 884:
        raise RuntimeError("Primary judge QC does not certify 884 records")
    if not secondary_qc.get("qc_passed") or int(secondary_qc.get("successes", 0)) != 884:
        raise RuntimeError("Secondary judge QC does not certify 884 successful records")
    if not validity_gate.get("formal_secondary_run_approved"):
        raise RuntimeError("Secondary-judge validity gate is not approved")

    master = read_csv(MASTER_PATH)
    reference = read_csv(REFERENCE_PATH)[REFERENCE_FIELDS]
    primary = read_csv(PRIMARY_PATH)
    secondary = read_csv(SECONDARY_PATH)
    human = read_csv(HUMAN_GOLD_PATH)

    for label, frame in [
        ("blinding master", master),
        ("reference source", reference),
        ("primary judge", primary),
        ("secondary judge", secondary),
        ("human gold", human),
    ]:
        assert_unique(frame, "blinded_id", label)

    expected_ids = set(master["blinded_id"])
    if len(expected_ids) != 884:
        raise RuntimeError(f"Blinding master has {len(expected_ids)} unique records, expected 884")
    for label, frame in [
        ("reference source", reference),
        ("primary judge", primary),
        ("secondary judge", secondary),
    ]:
        require_exact_ids(expected_ids, frame, label)
    if len(human) != 150 or not set(human["blinded_id"]).issubset(expected_ids):
        raise RuntimeError("The finalized human-gold set is not a valid 150-record subset")

    primary_keep = ["blinded_id", "judge_confidence", "boundary_flag"] + SCORE_FIELDS
    primary = primary[primary_keep].rename(
        columns={column: f"primary_{column}" for column in primary_keep if column != "blinded_id"}
    )
    secondary_keep = ["blinded_id", "judge_confidence", "boundary_flag"] + SCORE_FIELDS
    secondary = secondary[secondary_keep].rename(
        columns={column: f"secondary_{column}" for column in secondary_keep if column != "blinded_id"}
    )
    human_keep = ["blinded_id"] + SCORE_FIELDS
    human = human[human_keep].rename(
        columns={column: f"human_{column}" for column in human_keep if column != "blinded_id"}
    )

    merged = master.merge(reference, on=["blinded_id", "case_id"], how="inner", validate="one_to_one")
    merged = merged.merge(primary, on="blinded_id", how="inner", validate="one_to_one")
    merged = merged.merge(secondary, on="blinded_id", how="inner", validate="one_to_one")
    merged = merged.merge(human, on="blinded_id", how="left", validate="one_to_one")
    merged = merged.fillna("")
    if len(merged) != 884:
        raise RuntimeError(f"Merged dataset has {len(merged)} records, expected 884")

    merged["human_gold_available"] = merged["human_top1_correct"].ne("").astype(int)
    merged["primary_secondary_top1_disagreement"] = (
        merged["primary_top1_correct"] != merged["secondary_top1_correct"]
    ).astype(int)
    merged["primary_secondary_triage_disagreement"] = (
        merged["primary_triage_level_scored"] != merged["secondary_triage_level_scored"]
    ).astype(int)
    merged["primary_secondary_anchor_rank_disagreement"] = (
        merged["incorrect_anchor_for_scoring"].ne("")
        & (merged["primary_incorrect_anchor_rank"] != merged["secondary_incorrect_anchor_rank"])
    ).astype(int)
    merged = add_pairwise_conversion_flags(merged)

    # Pre-review labels use the formal human gold standard where available and
    # otherwise preserve the primary AI-judge label. They are explicitly marked
    # provisional and must not be presented as the final safety analysis.
    for field in SCORE_FIELDS:
        human_field = f"human_{field}"
        primary_field = f"primary_{field}"
        merged[f"pre_review_{field}"] = np.where(
            merged["human_gold_available"] == 1,
            merged[human_field],
            merged[primary_field],
        )
    merged["pre_review_label_source"] = np.where(
        merged["human_gold_available"] == 1,
        "adjudicated_human_gold",
        "primary_ai_judge_pending_targeted_clinician_review",
    )
    merged["final_analysis_eligible"] = merged["human_gold_available"]

    # All non-gold records require one clinician review. This conservative rule
    # resolves the endpoint-level human-scoring gate for unsafe management,
    # mandatory-action completeness, and exact incorrect-anchor rank without
    # selecting records based on whether the study hypothesis is supported.
    candidates = merged[merged["human_gold_available"] == 0].copy()
    if len(candidates) != 734:
        raise RuntimeError(f"Expected 734 non-gold records, found {len(candidates)}")

    reason_columns = {
        "primary_unsafe_positive": candidates["primary_unsafe_management"].eq("1"),
        "primary_boundary_flag": candidates["primary_boundary_flag"].eq("1"),
        "primary_low_confidence": candidates["primary_judge_confidence"].eq("low"),
        "primary_secondary_top1_disagreement": candidates[
            "primary_secondary_top1_disagreement"
        ].eq(1),
        "primary_secondary_triage_disagreement": candidates[
            "primary_secondary_triage_disagreement"
        ].eq(1),
        "part1_top1_conversion_pair": candidates["part1_top1_conversion_pair"].eq(1),
        "part1_unsafe_conversion_pair": candidates["part1_unsafe_conversion_pair"].eq(1),
        "part2_top1_change_pair": candidates["part2_top1_change_pair"].eq(1),
        "part2_unsafe_change_pair": candidates["part2_unsafe_change_pair"].eq(1),
        "incorrect_anchor_rank_human_gate": candidates["incorrect_anchor_for_scoring"].ne(""),
    }
    for name, mask in reason_columns.items():
        candidates[name] = mask.astype(int)

    def review_reason(row: pd.Series) -> str:
        reasons = [name for name in reason_columns if int(row[name]) == 1]
        return "|".join(reasons) if reasons else "complete_non_gold_clinician_review"

    candidates["internal_review_reason"] = candidates.apply(review_reason, axis=1)
    high_mask = (
        candidates["primary_unsafe_positive"].eq(1)
        | candidates["part1_unsafe_conversion_pair"].eq(1)
        | candidates["part2_unsafe_change_pair"].eq(1)
    )
    medium_mask = (
        candidates["primary_boundary_flag"].eq(1)
        | candidates["primary_secondary_top1_disagreement"].eq(1)
        | candidates["primary_secondary_triage_disagreement"].eq(1)
        | candidates["part1_top1_conversion_pair"].eq(1)
        | candidates["part2_top1_change_pair"].eq(1)
    )
    candidates["internal_priority"] = np.select(
        [high_mask, medium_mask], ["1_high", "2_medium"], default="3_standard"
    )

    # Priority affects only the processing queue; a fixed random sequence within
    # priority prevents condition/model ordering from being visible to the rater.
    rng = np.random.default_rng(20260721)
    candidates["internal_random_order"] = rng.random(len(candidates))
    candidates = candidates.sort_values(
        ["internal_priority", "internal_random_order", "blinded_id"], kind="mergesort"
    ).reset_index(drop=True)
    candidates["review_sequence"] = np.arange(1, len(candidates) + 1)
    candidates["review_batch"] = ((candidates["review_sequence"] - 1) // 50 + 1).map(
        lambda value: f"TR_{value:02d}"
    )

    merged_path = OUTPUT / "formal_merged_pre_review_dataset_v1.csv"
    candidate_path = OUTPUT / "targeted_clinician_review_candidates_internal_v1.csv"
    public_columns = REFERENCE_FIELDS + ["review_sequence", "review_batch"]
    public_candidates = candidates[public_columns].copy()
    public_path = OUTPUT / "targeted_clinician_review_source_blinded_v1.csv"
    public_json_path = OUTPUT / "targeted_clinician_review_source_blinded_v1.json"
    merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
    candidates.to_csv(candidate_path, index=False, encoding="utf-8-sig")
    public_candidates.to_csv(public_path, index=False, encoding="utf-8-sig")
    public_json_path.write_text(
        json.dumps(public_candidates.to_dict(orient="records"), ensure_ascii=False),
        encoding="utf-8",
    )

    reason_counts = {name: int(candidates[name].sum()) for name in reason_columns}
    batch_counts = {
        str(key): int(value) for key, value in candidates["review_batch"].value_counts().sort_index().items()
    }
    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pre_targeted_review_not_final_analysis",
        "record_counts": {
            "formal_records": 884,
            "adjudicated_human_gold": 150,
            "single_clinician_review_pending": 734,
            "review_batches": len(batch_counts),
        },
        "integrity_checks": {
            "all_primary_ids_match": True,
            "all_secondary_ids_match": True,
            "all_reference_ids_match": True,
            "human_gold_is_valid_subset": True,
            "primary_qc_passed": True,
            "secondary_qc_passed": True,
        },
        "targeted_review_reason_counts_nonexclusive": reason_counts,
        "batch_counts": batch_counts,
        "label_precedence_after_return": [
            "adjudicated human-gold label",
            "single targeted clinician-review label",
            "primary AI-judge label only for fields allowed by the endpoint validity gate",
        ],
        "secondary_judge_inferential_fields": [
            item["field"] for item in validity_gate["secondary_glm_allowed_inferential_uses"]
        ],
        "secondary_judge_excluded_fields": [
            item["field"] for item in validity_gate["secondary_glm_excluded_from_inferential_claims"]
        ],
        "source_hashes": {str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256(path) for path in files},
        "output_hashes": {
            str(merged_path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256(merged_path),
            str(candidate_path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256(candidate_path),
            str(public_path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256(public_path),
            str(public_json_path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256(public_json_path),
        },
        "analysis_guardrail": (
            "Do not run or report final safety analyses until all 734 targeted clinician-review "
            "records have valid returned scores and the finalization QC passes."
        ),
    }
    audit_path = OUTPUT / "formal_analysis_merge_and_review_selection_audit_v1.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
