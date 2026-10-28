from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

_MPL_CACHE = Path(__file__).resolve().parents[2] / "work" / "matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint


CONDITION_ORDER = ["no_label", "correct_label", "incorrect_label"]
INTERVENTION_ORDER = [
    "baseline",
    "evidence_first",
    "differential",
    "counter_anchor",
    "self_check",
]
TRIAGE_ORDER = {"routine": 1, "urgent_outpatient": 2, "same_day": 3, "emergency": 4}


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _wilson(successes: int, n: int) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan
    low, high = proportion_confint(successes, n, method="wilson")
    return float(low), float(high)


def proportion_summary(
    frame: pd.DataFrame, group: str, outcomes: list[str], order: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_value in order:
        subset = frame[frame[group] == group_value]
        for outcome in outcomes:
            values = _numeric(subset[outcome]).dropna()
            n = int(len(values))
            successes = int(values.sum()) if n else 0
            low, high = _wilson(successes, n)
            rows.append(
                {
                    group: group_value,
                    "outcome": outcome,
                    "n": n,
                    "events": successes,
                    "proportion": successes / n if n else math.nan,
                    "wilson_low": low,
                    "wilson_high": high,
                }
            )
    return pd.DataFrame(rows)


def cluster_bootstrap_difference(
    frame: pd.DataFrame,
    *,
    group_col: str,
    reference: str,
    comparison: str,
    outcome: str,
    seed: int,
    iterations: int = 3000,
) -> tuple[float, float, float]:
    usable = frame[frame[group_col].isin([reference, comparison])].copy()
    usable[outcome] = _numeric(usable[outcome])
    clusters = usable["case_id"].dropna().unique()
    observed = (
        usable.loc[usable[group_col] == comparison, outcome].mean()
        - usable.loc[usable[group_col] == reference, outcome].mean()
    )
    if len(clusters) < 2:
        return float(observed), math.nan, math.nan
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    aggregated = (
        usable.groupby(["case_id", group_col], dropna=False)[outcome]
        .agg(["sum", "count"])
        .reset_index()
    )

    def case_arrays(group_value: str) -> tuple[np.ndarray, np.ndarray]:
        subset = aggregated[aggregated[group_col] == group_value].set_index("case_id")
        sums = np.array([float(subset.loc[c, "sum"]) if c in subset.index else 0.0 for c in clusters])
        counts = np.array([float(subset.loc[c, "count"]) if c in subset.index else 0.0 for c in clusters])
        return sums, counts

    reference_sums, reference_counts = case_arrays(reference)
    comparison_sums, comparison_counts = case_arrays(comparison)
    for _ in range(iterations):
        indices = rng.integers(0, len(clusters), size=len(clusters))
        ref_n = reference_counts[indices].sum()
        cmp_n = comparison_counts[indices].sum()
        if ref_n and cmp_n:
            ref_mean = reference_sums[indices].sum() / ref_n
            cmp_mean = comparison_sums[indices].sum() / cmp_n
            estimates.append(float(cmp_mean - ref_mean))
    if not estimates:
        return float(observed), math.nan, math.nan
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(observed), float(low), float(high)


def paired_binary_test(
    reference_values: pd.Series, comparison_values: pd.Series
) -> tuple[int, int, int, int, float]:
    reference = _numeric(reference_values).astype(int)
    comparison = _numeric(comparison_values).astype(int)
    n00 = int(((reference == 0) & (comparison == 0)).sum())
    n01 = int(((reference == 0) & (comparison == 1)).sum())
    n10 = int(((reference == 1) & (comparison == 0)).sum())
    n11 = int(((reference == 1) & (comparison == 1)).sum())
    table = [[n00, n01], [n10, n11]]
    p_value = float(mcnemar(table, exact=True).pvalue) if n01 + n10 else 1.0
    return n00, n01, n10, n11, p_value


def prepare_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    numeric_fields = [
        "repeat_id",
        "is_mock",
        "top1_correct",
        "gold_diagnosis_rank",
        "incorrect_anchor_rank",
        "anchor_adoption",
        "evidence_override",
        "mandatory_action_complete",
        "unsafe_management",
        "over_conservatism",
    ]
    for field in numeric_fields:
        if field in frame:
            frame[field] = pd.to_numeric(frame[field], errors="coerce")
    frame["anchor_adoption_binary"] = (frame["anchor_adoption"] == 2).astype(float)
    frame["triage_score"] = frame["triage_level_scored"].map(TRIAGE_ORDER)
    frame["reference_triage_score"] = frame["reference_triage"].map(TRIAGE_ORDER)
    return frame


def part1_analysis(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    p1 = frame[frame["part"] == "part1"].copy()
    outcomes = ["top1_correct", "unsafe_management", "anchor_adoption_binary"]
    summary = proportion_summary(p1, "anchor_condition", outcomes, CONDITION_ORDER)

    keys = ["case_id", "logical_model_id", "repeat_id"]
    no_label = p1[p1["anchor_condition"] == "no_label"].set_index(keys)
    wrong = p1[p1["anchor_condition"] == "incorrect_label"].set_index(keys)
    paired = no_label.join(wrong, lsuffix="_no", rsuffix="_wrong", how="inner")
    pair_rows: list[dict[str, Any]] = []
    for index, outcome in enumerate(["top1_correct", "unsafe_management"], start=1):
        valid = paired[[f"{outcome}_no", f"{outcome}_wrong"]].dropna()
        n00, n01, n10, n11, p = paired_binary_test(
            valid[f"{outcome}_no"], valid[f"{outcome}_wrong"]
        )
        rd, low, high = cluster_bootstrap_difference(
            p1,
            group_col="anchor_condition",
            reference="no_label",
            comparison="incorrect_label",
            outcome=outcome,
            seed=seed + index,
        )
        pair_rows.append(
            {
                "outcome": outcome,
                "comparison": "incorrect_label - no_label",
                "paired_n": len(valid),
                "risk_difference": rd,
                "cluster_bootstrap_low": low,
                "cluster_bootstrap_high": high,
                "mcnemar_n00": n00,
                "mcnemar_n01": n01,
                "mcnemar_n10": n10,
                "mcnemar_n11": n11,
                "mcnemar_exact_p": p,
            }
        )

    top_valid = paired[["top1_correct_no", "top1_correct_wrong"]].dropna()
    denominator = int((top_valid["top1_correct_no"] == 1).sum())
    conversions = int(
        ((top_valid["top1_correct_no"] == 1) & (top_valid["top1_correct_wrong"] == 0)).sum()
    )
    unsafe_valid = paired[["unsafe_management_no", "unsafe_management_wrong"]].dropna()
    unsafe_denominator = int((unsafe_valid["unsafe_management_no"] == 0).sum())
    unsafe_conversions = int(
        (
            (unsafe_valid["unsafe_management_no"] == 0)
            & (unsafe_valid["unsafe_management_wrong"] == 1)
        ).sum()
    )

    rank_valid = paired[["gold_diagnosis_rank_no", "gold_diagnosis_rank_wrong"]].dropna()
    displacement = rank_valid["gold_diagnosis_rank_wrong"] - rank_valid["gold_diagnosis_rank_no"]
    if len(displacement) and not np.allclose(displacement, 0):
        rank_p = float(wilcoxon(displacement).pvalue)
    else:
        rank_p = 1.0
    derived = {
        "anchoring_conversion_numerator": conversions,
        "anchoring_conversion_denominator": denominator,
        "anchoring_conversion_rate": conversions / denominator if denominator else math.nan,
        "unsafe_conversion_numerator": unsafe_conversions,
        "unsafe_conversion_denominator": unsafe_denominator,
        "unsafe_conversion_rate": unsafe_conversions / unsafe_denominator if unsafe_denominator else math.nan,
        "diagnostic_displacement_median": float(displacement.median()) if len(displacement) else math.nan,
        "diagnostic_displacement_mean": float(displacement.mean()) if len(displacement) else math.nan,
        "diagnostic_displacement_wilcoxon_p": rank_p,
        "paired_observations": int(len(paired)),
    }
    return summary, pd.DataFrame(pair_rows), derived


def _part2_frame(frame: pd.DataFrame, baseline_repeat: int) -> pd.DataFrame:
    baseline = frame[
        (frame["part"] == "part1")
        & (frame["anchor_condition"] == "incorrect_label")
        & (frame["repeat_id"] == baseline_repeat)
    ].copy()
    baseline["strategy"] = "baseline"
    interventions = frame[frame["part"] == "part2"].copy()
    interventions["strategy"] = interventions["intervention"]
    return pd.concat([baseline, interventions], ignore_index=True)


def part2_analysis(
    frame: pd.DataFrame, seed: int, baseline_repeat: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    p2 = _part2_frame(frame, baseline_repeat)
    outcomes = [
        "top1_correct",
        "unsafe_management",
        "anchor_adoption_binary",
        "evidence_override",
        "over_conservatism",
    ]
    summary = proportion_summary(p2, "strategy", outcomes, INTERVENTION_ORDER)
    keys = ["case_id", "logical_model_id"]
    baseline = p2[p2["strategy"] == "baseline"].set_index(keys)
    rows: list[dict[str, Any]] = []
    for outcome_index, outcome in enumerate(outcomes):
        outcome_rows: list[dict[str, Any]] = []
        raw_p: list[float] = []
        for intervention_index, intervention in enumerate(INTERVENTION_ORDER[1:]):
            candidate = p2[p2["strategy"] == intervention].set_index(keys)
            paired = baseline[[outcome]].join(
                candidate[[outcome]], lsuffix="_baseline", rsuffix="_intervention", how="inner"
            ).dropna()
            n00, n01, n10, n11, p = paired_binary_test(
                paired[f"{outcome}_baseline"], paired[f"{outcome}_intervention"]
            )
            rd, low, high = cluster_bootstrap_difference(
                p2,
                group_col="strategy",
                reference="baseline",
                comparison=intervention,
                outcome=outcome,
                seed=seed + 100 + outcome_index * 10 + intervention_index,
            )
            outcome_rows.append(
                {
                    "outcome": outcome,
                    "intervention": intervention,
                    "paired_n": len(paired),
                    "risk_difference_vs_baseline": rd,
                    "cluster_bootstrap_low": low,
                    "cluster_bootstrap_high": high,
                    "mcnemar_n00": n00,
                    "mcnemar_n01": n01,
                    "mcnemar_n10": n10,
                    "mcnemar_n11": n11,
                    "mcnemar_exact_p": p,
                }
            )
            raw_p.append(p)
        adjusted = multipletests(raw_p, method="holm")[1] if raw_p else []
        for row, adjusted_p in zip(outcome_rows, adjusted):
            row["holm_adjusted_p_within_outcome"] = float(adjusted_p)
            rows.append(row)
    return summary, pd.DataFrame(rows)


def save_figures(
    frame: pd.DataFrame,
    part1_summary: pd.DataFrame,
    part2_summary: pd.DataFrame,
    output_dir: Path,
    baseline_repeat: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    def rate_chart(summary: pd.DataFrame, group_col: str, order: list[str], outcome: str, title: str, path: Path) -> None:
        data = summary[summary["outcome"] == outcome].set_index(group_col).reindex(order)
        values = data["proportion"].astype(float).to_numpy()
        lows = data["wilson_low"].astype(float).to_numpy()
        highs = data["wilson_high"].astype(float).to_numpy()
        # Numerical rounding can place a reported Wilson endpoint a fraction
        # below/above the displayed proportion; Matplotlib rejects negative
        # error-bar widths, so clip only those floating-point artefacts.
        errors = np.maximum(np.vstack([values - lows, highs - values]), 0)
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.bar(range(len(order)), values, color="#2F6B7C", width=0.65)
        ax.errorbar(range(len(order)), values, yerr=errors, fmt="none", ecolor="#1F2937", capsize=4)
        ax.set_xticks(range(len(order)), [label.replace("_", "\n") for label in order])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Proportion")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)

    rate_chart(
        part1_summary,
        "anchor_condition",
        CONDITION_ORDER,
        "top1_correct",
        "Part I: Top-1 diagnostic accuracy",
        output_dir / "figure_1_part1_accuracy.png",
    )
    rate_chart(
        part1_summary,
        "anchor_condition",
        CONDITION_ORDER,
        "unsafe_management",
        "Part I: Unsafe management rate",
        output_dir / "figure_2_part1_unsafe_management.png",
    )

    p1 = frame[frame["part"] == "part1"].copy()
    keys = ["case_id", "logical_model_id", "repeat_id"]
    no = p1[p1["anchor_condition"] == "no_label"].set_index(keys)
    wrong = p1[p1["anchor_condition"] == "incorrect_label"].set_index(keys)
    ranks = no[["gold_diagnosis_rank"]].join(
        wrong[["gold_diagnosis_rank"]], lsuffix="_no", rsuffix="_wrong", how="inner"
    ).dropna()
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for _, row in ranks.iterrows():
        ax.plot([0, 1], [row["gold_diagnosis_rank_no"], row["gold_diagnosis_rank_wrong"]], color="#94A3B8", alpha=0.65)
        ax.scatter([0, 1], [row["gold_diagnosis_rank_no"], row["gold_diagnosis_rank_wrong"]], color=["#0F766E", "#B45309"], s=28)
    ax.set_xticks([0, 1], ["No label", "Incorrect label"])
    ax.set_yticks(range(1, 7))
    ax.invert_yaxis()
    ax.set_ylabel("Gold diagnosis rank (1 is best; 6 = absent)")
    ax.set_title("Diagnostic displacement")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_3_diagnostic_displacement.png", dpi=220)
    plt.close(fig)

    rate_chart(
        part2_summary,
        "strategy",
        INTERVENTION_ORDER,
        "top1_correct",
        "Part II: Debiasing strategy accuracy",
        output_dir / "figure_4_part2_accuracy.png",
    )
    rate_chart(
        part2_summary,
        "strategy",
        INTERVENTION_ORDER,
        "unsafe_management",
        "Part II: Debiasing strategy unsafe management",
        output_dir / "figure_5_part2_unsafe_management.png",
    )
