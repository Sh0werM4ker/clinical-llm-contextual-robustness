from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint


PROJECT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT / "ratings" / "formal_local_v1" / "formal_final_analysis_dataset_v1.csv"
FAILURES_PATH = PROJECT / "ratings" / "formal_local_v1" / "generation_failures.csv"
RESULTS = PROJECT / "results" / "formal_local_v1" / "final_human_v1"
SEED = 20260721
ITERATIONS = 5000

CONDITIONS = ["no_label", "correct_label", "incorrect_label"]
STRATEGIES = ["baseline", "evidence_first", "differential", "counter_anchor", "self_check"]
BINARY_OUTCOMES = [
    "top1_correct",
    "unsafe_management",
    "mandatory_action_complete",
    "evidence_override",
    "over_conservatism",
    "anchor_adoption_binary",
]
VALIDATION_OUTCOMES = [
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


def numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def wilson(events: int, n: int) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    low, high = proportion_confint(events, n, method="wilson")
    return float(low), float(high)


def kappa_score(
    y_true: pd.Series,
    y_pred: pd.Series,
    labels: list[str],
    *,
    quadratic: bool = False,
) -> float:
    mapping = {label: index for index, label in enumerate(labels)}
    observed = np.zeros((len(labels), len(labels)), dtype=float)
    for truth, prediction in zip(y_true, y_pred):
        observed[mapping[str(truth)], mapping[str(prediction)]] += 1
    if observed.sum() == 0:
        return math.nan
    observed /= observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    if quadratic:
        if len(labels) == 1:
            return 1.0
        indices = np.arange(len(labels), dtype=float)
        weights = ((indices[:, None] - indices[None, :]) / (len(labels) - 1)) ** 2
        denominator = float((weights * expected).sum())
        return 1.0 - float((weights * observed).sum()) / denominator if denominator else math.nan
    agreement = float(np.trace(observed))
    expected_agreement = float(np.trace(expected))
    return (agreement - expected_agreement) / (1 - expected_agreement) if expected_agreement < 1 else 1.0


def summarize_rates(frame: pd.DataFrame, group_cols: list[str], outcomes: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_values, subset in frame.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group_record = dict(zip(group_cols, group_values))
        for outcome in outcomes:
            values = pd.to_numeric(subset[outcome], errors="coerce").dropna()
            events = int(values.sum())
            n = int(len(values))
            low, high = wilson(events, n)
            rows.append(
                {
                    **group_record,
                    "outcome": outcome,
                    "n": n,
                    "events": events,
                    "proportion": events / n if n else math.nan,
                    "wilson_low": low,
                    "wilson_high": high,
                }
            )
    return pd.DataFrame(rows)


def paired_bootstrap_ci(paired: pd.DataFrame, difference_col: str, seed: int) -> tuple[float, float, float]:
    clean = paired[["case_id", difference_col]].dropna()
    observed = float(clean[difference_col].mean())
    cases = clean["case_id"].unique()
    if len(cases) < 2:
        return observed, math.nan, math.nan
    by_case = {case: clean.loc[clean["case_id"] == case, difference_col].to_numpy(float) for case in cases}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(ITERATIONS):
        sampled = rng.choice(cases, size=len(cases), replace=True)
        values = np.concatenate([by_case[case] for case in sampled])
        estimates.append(float(values.mean()))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return observed, float(low), float(high)


def paired_binary_comparison(
    frame: pd.DataFrame,
    *,
    group_col: str,
    reference: str,
    comparison: str,
    keys: list[str],
    outcomes: list[str],
    seed_offset: int,
) -> pd.DataFrame:
    ref = frame[frame[group_col] == reference].set_index(keys)
    cmp = frame[frame[group_col] == comparison].set_index(keys)
    rows = []
    for index, outcome in enumerate(outcomes):
        paired = ref[[outcome]].join(
            cmp[[outcome]], rsuffix="_comparison", how="inner"
        ).reset_index()
        paired[outcome] = pd.to_numeric(paired[outcome], errors="coerce")
        paired[f"{outcome}_comparison"] = pd.to_numeric(
            paired[f"{outcome}_comparison"], errors="coerce"
        )
        paired = paired.dropna(subset=[outcome, f"{outcome}_comparison"])
        paired["difference"] = paired[f"{outcome}_comparison"] - paired[outcome]
        n00 = int(((paired[outcome] == 0) & (paired[f"{outcome}_comparison"] == 0)).sum())
        n01 = int(((paired[outcome] == 0) & (paired[f"{outcome}_comparison"] == 1)).sum())
        n10 = int(((paired[outcome] == 1) & (paired[f"{outcome}_comparison"] == 0)).sum())
        n11 = int(((paired[outcome] == 1) & (paired[f"{outcome}_comparison"] == 1)).sum())
        p_value = float(mcnemar([[n00, n01], [n10, n11]], exact=True).pvalue) if n01 + n10 else 1.0
        estimate, low, high = paired_bootstrap_ci(paired, "difference", SEED + seed_offset + index)
        rows.append(
            {
                "outcome": outcome,
                "reference": reference,
                "comparison": comparison,
                "paired_n": len(paired),
                "reference_rate_paired": float(paired[outcome].mean()),
                "comparison_rate_paired": float(paired[f"{outcome}_comparison"].mean()),
                "paired_risk_difference": estimate,
                "case_cluster_bootstrap_low": low,
                "case_cluster_bootstrap_high": high,
                "mcnemar_n00": n00,
                "mcnemar_n01": n01,
                "mcnemar_n10": n10,
                "mcnemar_n11": n11,
                "mcnemar_exact_p": p_value,
            }
        )
    return pd.DataFrame(rows)


def part2_frame(frame: pd.DataFrame) -> pd.DataFrame:
    baseline = frame[
        (frame["part"] == "part1")
        & (frame["anchor_condition"] == "incorrect_label")
        & (frame["repeat_id"] == 1)
    ].copy()
    baseline["strategy"] = "baseline"
    interventions = frame[frame["part"] == "part2"].copy()
    interventions["strategy"] = interventions["intervention"]
    return pd.concat([baseline, interventions], ignore_index=True)


def derived_part1(frame: pd.DataFrame) -> dict[str, object]:
    p1 = frame[frame["part"] == "part1"]
    keys = ["case_id", "logical_model_id", "repeat_id"]
    no = p1[p1["anchor_condition"] == "no_label"].set_index(keys)
    wrong = p1[p1["anchor_condition"] == "incorrect_label"].set_index(keys)
    paired = no.join(wrong, lsuffix="_no", rsuffix="_wrong", how="inner")
    top = paired[["top1_correct_no", "top1_correct_wrong"]].dropna()
    unsafe = paired[["unsafe_management_no", "unsafe_management_wrong"]].dropna()
    ranks = paired[["gold_diagnosis_rank_no", "gold_diagnosis_rank_wrong"]].dropna()
    displacement = ranks["gold_diagnosis_rank_wrong"] - ranks["gold_diagnosis_rank_no"]
    return {
        "paired_n": int(len(paired)),
        "anchoring_conversion_numerator": int(
            ((top["top1_correct_no"] == 1) & (top["top1_correct_wrong"] == 0)).sum()
        ),
        "anchoring_conversion_denominator": int((top["top1_correct_no"] == 1).sum()),
        "unsafe_conversion_numerator": int(
            ((unsafe["unsafe_management_no"] == 0) & (unsafe["unsafe_management_wrong"] == 1)).sum()
        ),
        "unsafe_conversion_denominator": int((unsafe["unsafe_management_no"] == 0).sum()),
        "diagnostic_displacement_mean": float(displacement.mean()),
        "diagnostic_displacement_median": float(displacement.median()),
        "diagnostic_displacement_improved_n": int((displacement < 0).sum()),
        "diagnostic_displacement_unchanged_n": int((displacement == 0).sum()),
        "diagnostic_displacement_worsened_n": int((displacement > 0).sum()),
        "diagnostic_displacement_wilcoxon_p": (
            float(wilcoxon(displacement).pvalue)
            if len(displacement) and not np.allclose(displacement, 0)
            else 1.0
        ),
    }


def validation_metrics(frame: pd.DataFrame, prefix: str, source_name: str) -> pd.DataFrame:
    gold = frame[frame["final_label_source"].str.startswith("human_gold")].copy()
    rows = []
    for outcome in VALIDATION_OUTCOMES:
        ai_col = f"{prefix}_{outcome}"
        applicable = gold[[outcome, ai_col]].replace("", np.nan).dropna()
        if applicable.empty:
            continue
        y_true = applicable[outcome].astype(str)
        y_pred = applicable[ai_col].astype(str)
        labels = sorted(set(y_true) | set(y_pred))
        kappa = float(kappa_score(y_true, y_pred, labels))
        weighted = math.nan
        if outcome in {"gold_diagnosis_rank", "incorrect_anchor_rank", "anchor_adoption", "triage_level_scored"}:
            if outcome == "triage_level_scored":
                order = ["routine", "urgent_outpatient", "same_day", "emergency"]
            elif outcome == "anchor_adoption":
                order = ["0", "1", "2"]
            else:
                order = ["1", "2", "3", "4", "5", "6"]
            weighted = float(kappa_score(y_true, y_pred, order, quadratic=True))
        row = {
            "judge_source": source_name,
            "outcome": outcome,
            "n": len(applicable),
            "exact_agreement": float((y_true == y_pred).mean()),
            "cohen_kappa": kappa,
            "quadratic_weighted_kappa": weighted,
        }
        if set(labels) <= {"0", "1"}:
            true = y_true.astype(int)
            pred = y_pred.astype(int)
            tp = int(((true == 1) & (pred == 1)).sum())
            tn = int(((true == 0) & (pred == 0)).sum())
            fp = int(((true == 0) & (pred == 1)).sum())
            fn = int(((true == 1) & (pred == 0)).sum())
            row.update(
                {
                    "tp": tp,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "sensitivity": tp / (tp + fn) if tp + fn else math.nan,
                    "specificity": tn / (tn + fp) if tn + fp else math.nan,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def source_variant(original: pd.DataFrame, prefix: str | None, name: str) -> pd.DataFrame:
    variant = original.copy()
    if prefix:
        for outcome in VALIDATION_OUTCOMES:
            variant[outcome] = variant[f"{prefix}_{outcome}"]
    for outcome in ["top1_correct", "gold_diagnosis_rank", "anchor_adoption", "evidence_override", "mandatory_action_complete", "unsafe_management", "over_conservatism"]:
        variant[outcome] = pd.to_numeric(variant[outcome], errors="coerce")
    variant["repeat_id"] = pd.to_numeric(variant["repeat_id"], errors="coerce")
    variant["anchor_adoption_binary"] = (variant["anchor_adoption"] == 2).astype(float)
    variant["analysis_label_source"] = name
    return variant


def core_sensitivity(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    p1 = frame[frame["part"] == "part1"].copy()
    p1_rows = paired_binary_comparison(
        p1,
        group_col="anchor_condition",
        reference="no_label",
        comparison="incorrect_label",
        keys=["case_id", "logical_model_id", "repeat_id"],
        outcomes=["top1_correct", "unsafe_management"],
        seed_offset=700,
    )
    p1_rows.insert(0, "analysis_part", "part1")
    p1_rows.insert(0, "analysis_label_source", name)

    p2 = part2_frame(frame)
    rows = [p1_rows]
    for intervention_index, intervention in enumerate(STRATEGIES[1:]):
        result = paired_binary_comparison(
            p2,
            group_col="strategy",
            reference="baseline",
            comparison=intervention,
            keys=["case_id", "logical_model_id"],
            outcomes=["top1_correct", "unsafe_management", "evidence_override"],
            seed_offset=800 + intervention_index * 20,
        )
        result.insert(0, "analysis_part", "part2")
        result.insert(0, "analysis_label_source", name)
        rows.append(result)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(DATA_PATH, dtype=str, encoding="utf-8-sig").fillna("")
    human = source_variant(raw, None, "human_final")

    p1 = human[human["part"] == "part1"].copy()
    p1_summary = summarize_rates(
        p1,
        ["anchor_condition"],
        ["top1_correct", "unsafe_management", "mandatory_action_complete", "evidence_override", "over_conservatism"],
    )
    p1_model = summarize_rates(
        p1,
        ["logical_model_id", "anchor_condition"],
        ["top1_correct", "unsafe_management", "evidence_override"],
    )
    p1_domain = summarize_rates(
        p1,
        ["domain", "anchor_condition"],
        ["top1_correct", "unsafe_management", "evidence_override"],
    )

    p1_pairs = []
    for offset, comparison in enumerate(["correct_label", "incorrect_label"]):
        p1_pairs.append(
            paired_binary_comparison(
                p1,
                group_col="anchor_condition",
                reference="no_label",
                comparison=comparison,
                keys=["case_id", "logical_model_id", "repeat_id"],
                outcomes=["top1_correct", "unsafe_management", "mandatory_action_complete", "evidence_override", "over_conservatism"],
                seed_offset=offset * 20,
            )
        )
    p1_paired = pd.concat(p1_pairs, ignore_index=True)

    p1_by_model_frames = []
    for model_index, (model_id, subset) in enumerate(p1.groupby("logical_model_id", sort=True)):
        result = paired_binary_comparison(
            subset,
            group_col="anchor_condition",
            reference="no_label",
            comparison="incorrect_label",
            keys=["case_id", "logical_model_id", "repeat_id"],
            outcomes=["top1_correct", "unsafe_management", "evidence_override"],
            seed_offset=300 + model_index * 10,
        )
        result.insert(0, "logical_model_id", model_id)
        p1_by_model_frames.append(result)
    p1_paired_by_model = pd.concat(p1_by_model_frames, ignore_index=True)

    p1_by_domain_frames = []
    for domain_index, (domain, subset) in enumerate(p1.groupby("domain", sort=True)):
        result = paired_binary_comparison(
            subset,
            group_col="anchor_condition",
            reference="no_label",
            comparison="incorrect_label",
            keys=["case_id", "logical_model_id", "repeat_id"],
            outcomes=["top1_correct", "unsafe_management", "evidence_override"],
            seed_offset=400 + domain_index * 10,
        )
        result.insert(0, "domain", domain)
        p1_by_domain_frames.append(result)
    p1_paired_by_domain = pd.concat(p1_by_domain_frames, ignore_index=True)

    p2 = part2_frame(human)
    p2_summary = summarize_rates(
        p2,
        ["strategy"],
        ["top1_correct", "unsafe_management", "mandatory_action_complete", "anchor_adoption_binary", "evidence_override", "over_conservatism"],
    )
    p2_model = summarize_rates(
        p2,
        ["logical_model_id", "strategy"],
        ["top1_correct", "unsafe_management", "evidence_override"],
    )
    p2_pair_frames = []
    for offset, intervention in enumerate(STRATEGIES[1:]):
        result = paired_binary_comparison(
            p2,
            group_col="strategy",
            reference="baseline",
            comparison=intervention,
            keys=["case_id", "logical_model_id"],
            outcomes=["top1_correct", "unsafe_management", "mandatory_action_complete", "anchor_adoption_binary", "evidence_override", "over_conservatism"],
            seed_offset=100 + offset * 20,
        )
        p2_pair_frames.append(result)
    p2_paired = pd.concat(p2_pair_frames, ignore_index=True)
    for outcome, index in p2_paired.groupby("outcome").groups.items():
        p2_paired.loc[index, "holm_adjusted_p_within_outcome"] = multipletests(
            p2_paired.loc[index, "mcnemar_exact_p"].astype(float), method="holm"
        )[1]

    p2_by_model_frames = []
    for model_index, (model_id, subset) in enumerate(p2.groupby("logical_model_id", sort=True)):
        for intervention_index, intervention in enumerate(STRATEGIES[1:]):
            result = paired_binary_comparison(
                subset,
                group_col="strategy",
                reference="baseline",
                comparison=intervention,
                keys=["case_id", "logical_model_id"],
                outcomes=["top1_correct", "unsafe_management", "evidence_override"],
                seed_offset=500 + model_index * 100 + intervention_index * 10,
            )
            result.insert(0, "logical_model_id", model_id)
            p2_by_model_frames.append(result)
    p2_paired_by_model = pd.concat(p2_by_model_frames, ignore_index=True)

    validation = pd.concat(
        [
            validation_metrics(raw, "primary", "web_gpt_5_6_thinking"),
            validation_metrics(raw, "secondary", "local_glm_4_7_flash"),
        ],
        ignore_index=True,
    )

    sensitivity = pd.concat(
        [
            core_sensitivity(human, "human_final"),
            core_sensitivity(source_variant(raw, "primary", "web_gpt_5_6_thinking"), "web_gpt_5_6_thinking"),
            core_sensitivity(source_variant(raw, "secondary", "local_glm_4_7_flash"), "local_glm_4_7_flash"),
        ],
        ignore_index=True,
    )

    failures = pd.read_csv(FAILURES_PATH, dtype=str, encoding="utf-8-sig").fillna("")
    missingness = (
        failures.groupby(["part", "logical_model_id", "anchor_condition", "intervention"], dropna=False)
        .size()
        .reset_index(name="generation_failure_n")
    )

    files = {
        "formal_part1_summary_stratified.csv": p1_summary,
        "formal_part1_model_stratified.csv": p1_model,
        "formal_part1_domain_stratified.csv": p1_domain,
        "formal_part1_paired_complete_case.csv": p1_paired,
        "formal_part1_paired_by_model.csv": p1_paired_by_model,
        "formal_part1_paired_by_domain.csv": p1_paired_by_domain,
        "formal_part2_summary_stratified.csv": p2_summary,
        "formal_part2_model_stratified.csv": p2_model,
        "formal_part2_paired_complete_case.csv": p2_paired,
        "formal_part2_paired_by_model.csv": p2_paired_by_model,
        "formal_ai_judge_validation_against_human_gold.csv": validation,
        "formal_label_source_sensitivity.csv": sensitivity,
        "formal_generation_missingness.csv": missingness,
    }
    for filename, frame in files.items():
        frame.to_csv(RESULTS / filename, index=False, encoding="utf-8-sig")

    derived = derived_part1(human)
    no_wrong = p1_paired[
        (p1_paired["comparison"] == "incorrect_label")
        & (p1_paired["outcome"].isin(["top1_correct", "unsafe_management"]))
    ].set_index("outcome")
    report = {
        "status": "PASS",
        "analysis_rows": len(human),
        "cases": int(human["case_id"].nunique()),
        "models": int(human["logical_model_id"].nunique()),
        "generation_failures": len(failures),
        "part1_top1_incorrect_vs_no_label": no_wrong.loc["top1_correct"].to_dict(),
        "part1_unsafe_incorrect_vs_no_label": no_wrong.loc["unsafe_management"].to_dict(),
        "part1_derived": derived,
        "unsafe_management_note": (
            "The prespecified unsafe-management outcome includes failure to complete every mandatory action. "
            "Its approximately 95% prevalence creates a ceiling effect; interpret condition differences cautiously."
        ),
        "outputs": sorted(files),
    }
    (RESULTS / "formal_extended_analysis_qc_and_key_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
