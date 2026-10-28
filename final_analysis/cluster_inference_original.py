from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"C:\诊断锚定偏倚研究")
DATASET = (
    ROOT
    / "outputs"
    / "re_adjudication_anchor_adoption_v2"
    / "formal_final_analysis_dataset_v2_anchor_qc.csv"
)
OUTPUT_DIR = ROOT / "outputs" / "cluster_inference_revision_v3"

BOOTSTRAP_SEED = 20260721
BOOTSTRAP_REPLICATES = 10000
STRATEGIES = ["evidence_first", "differential", "counter_anchor", "self_check"]
PART1_OUTCOMES = [
    "top1_correct",
    "unsafe_management",
    "mandatory_action_complete",
    "evidence_override",
    "over_conservatism",
]
PART2_OUTCOMES = [
    "top1_correct",
    "unsafe_management",
    "mandatory_action_complete",
    "anchor_adoption_binary",
    "evidence_override",
    "over_conservatism",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_mcnemar(n01: int, n10: int) -> float:
    discordant = n01 + n10
    if discordant == 0:
        return 1.0
    tail = min(n01, n10)
    return min(
        1.0,
        2.0 * sum(math.comb(discordant, i) for i in range(tail + 1)) / (2**discordant),
    )


def holm_adjust(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    adjusted = [0.0] * len(values)
    running = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, values[index] * (m - rank)))
        adjusted[index] = running
    return adjusted


def exact_cluster_signflip(case_contributions: list[int]) -> tuple[float, int, int]:
    """Two-sided exact sign-flip P value for integer case-level contributions.

    One common sign is applied to the full contribution from each case. Dynamic
    programming evaluates the exact 2^K allocation distribution without Monte Carlo.
    """

    observed = abs(sum(case_contributions))
    distribution: Counter[int] = Counter({0: 1})
    for contribution in case_contributions:
        updated: Counter[int] = Counter()
        for total, multiplicity in distribution.items():
            updated[total + contribution] += multiplicity
            updated[total - contribution] += multiplicity
        distribution = updated
    total_allocations = 2 ** len(case_contributions)
    extreme_allocations = sum(
        multiplicity
        for total, multiplicity in distribution.items()
        if abs(total) >= observed
    )
    return extreme_allocations / total_allocations, len(distribution), total_allocations


def paired_cluster_stats(
    frame: pd.DataFrame,
    *,
    scope: str,
    outcome: str,
    reference_name: str,
    comparison_name: str,
    analysis_type: str,
) -> tuple[dict, list[dict]]:
    paired = frame.dropna(subset=["case_id", "reference", "comparison"]).copy()
    paired["reference"] = pd.to_numeric(paired["reference"], errors="raise").astype(int)
    paired["comparison"] = pd.to_numeric(paired["comparison"], errors="raise").astype(int)
    paired["difference"] = paired["comparison"] - paired["reference"]

    n = len(paired)
    if n == 0:
        raise RuntimeError(f"No pairs for {scope} {outcome} {comparison_name}")

    reference = paired["reference"].to_numpy()
    comparison = paired["comparison"].to_numpy()
    n00 = int(((reference == 0) & (comparison == 0)).sum())
    n01 = int(((reference == 0) & (comparison == 1)).sum())
    n10 = int(((reference == 1) & (comparison == 0)).sum())
    n11 = int(((reference == 1) & (comparison == 1)).sum())

    grouped = (
        paired.groupby("case_id", sort=True)["difference"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "case_difference_sum", "count": "case_pair_n"})
    )
    contributions = grouped["case_difference_sum"].astype(int).tolist()
    cluster_p, support_size, total_allocations = exact_cluster_signflip(contributions)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    cluster_sums = grouped["case_difference_sum"].to_numpy(dtype=float)
    cluster_counts = grouped["case_pair_n"].to_numpy(dtype=float)
    sampled = rng.integers(0, len(grouped), size=(BOOTSTRAP_REPLICATES, len(grouped)))
    bootstrap_rd = cluster_sums[sampled].sum(axis=1) / cluster_counts[sampled].sum(axis=1)
    ci_low, ci_high = np.quantile(bootstrap_rd, [0.025, 0.975])

    result = {
        "scope": scope,
        "analysis_type": analysis_type,
        "outcome": outcome,
        "reference": reference_name,
        "comparison": comparison_name,
        "paired_n": n,
        "case_cluster_n": len(grouped),
        "nonzero_case_cluster_n": int((grouped["case_difference_sum"] != 0).sum()),
        "reference_events": int(reference.sum()),
        "comparison_events": int(comparison.sum()),
        "reference_rate_paired": float(reference.mean()),
        "comparison_rate_paired": float(comparison.mean()),
        "paired_risk_difference": float(paired["difference"].mean()),
        "case_cluster_bootstrap_low": float(ci_low),
        "case_cluster_bootstrap_high": float(ci_high),
        "case_cluster_randomization_exact_p": float(cluster_p),
        "randomization_support_size": support_size,
        "randomization_total_sign_allocations": total_allocations,
        "mcnemar_n00": n00,
        "mcnemar_n01": n01,
        "mcnemar_n10": n10,
        "mcnemar_n11": n11,
        "mcnemar_exact_p_secondary": exact_mcnemar(n01, n10),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }

    contribution_rows = []
    for case_id, values in grouped.iterrows():
        contribution_rows.append(
            {
                "scope": scope,
                "analysis_type": analysis_type,
                "outcome": outcome,
                "reference": reference_name,
                "comparison": comparison_name,
                "case_id": case_id,
                "case_pair_n": int(values["case_pair_n"]),
                "case_difference_sum": int(values["case_difference_sum"]),
                "case_mean_difference": float(
                    values["case_difference_sum"] / values["case_pair_n"]
                ),
            }
        )
    return result, contribution_rows


def make_pair(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    *,
    keys: list[str],
    outcome: str,
    how: str = "inner",
    missing_comparison_as_zero: bool = False,
) -> pd.DataFrame:
    ref = reference[keys + [outcome]].rename(columns={outcome: "reference"})
    cmp = comparison[keys + [outcome]].rename(columns={outcome: "comparison"})
    paired = ref.merge(cmp, on=keys, how=how, validate="one_to_one")
    if missing_comparison_as_zero:
        paired["comparison"] = paired["comparison"].fillna(0)
    return paired


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATASET)
    data = data[data["final_analysis_eligible"].eq(1)].copy()
    data["anchor_adoption_binary"] = (pd.to_numeric(data["anchor_adoption"]) == 2).astype(int)
    for outcome in set(PART1_OUTCOMES + PART2_OUTCOMES):
        data[outcome] = pd.to_numeric(data[outcome], errors="coerce")

    part1 = data[data["part"].eq("part1")].copy()
    part2 = data[data["part"].eq("part2")].copy()
    baseline = part1[
        part1["anchor_condition"].eq("incorrect_label") & part1["repeat_id"].eq(1)
    ].copy()

    all_results: list[dict] = []
    all_contributions: list[dict] = []

    # Part I complete-case paired contrasts.
    no_label = part1[part1["anchor_condition"].eq("no_label")]
    for comparison_name in ["correct_label", "incorrect_label"]:
        comparison = part1[part1["anchor_condition"].eq(comparison_name)]
        for outcome in PART1_OUTCOMES:
            paired = make_pair(
                no_label,
                comparison,
                keys=["case_id", "logical_model_id", "repeat_id"],
                outcome=outcome,
            )
            result, contributions = paired_cluster_stats(
                paired,
                scope="Part I",
                outcome=outcome,
                reference_name="no_label",
                comparison_name=comparison_name,
                analysis_type="complete_case",
            )
            all_results.append(result)
            all_contributions.extend(contributions)

    # Part II complete-case paired contrasts against wrong-label repetition 1.
    part2_start = len(all_results)
    for strategy in STRATEGIES:
        comparison = part2[part2["intervention"].eq(strategy)]
        for outcome in PART2_OUTCOMES:
            paired = make_pair(
                baseline,
                comparison,
                keys=["case_id", "logical_model_id"],
                outcome=outcome,
            )
            result, contributions = paired_cluster_stats(
                paired,
                scope="Part II",
                outcome=outcome,
                reference_name="baseline",
                comparison_name=strategy,
                analysis_type="complete_case",
            )
            all_results.append(result)
            all_contributions.extend(contributions)

    part2_results = all_results[part2_start:]
    for outcome in PART2_OUTCOMES:
        selected = [result for result in part2_results if result["outcome"] == outcome]
        cluster_adjusted = holm_adjust(
            [result["case_cluster_randomization_exact_p"] for result in selected]
        )
        mcnemar_adjusted = holm_adjust(
            [result["mcnemar_exact_p_secondary"] for result in selected]
        )
        for result, cluster_p, mcnemar_p in zip(
            selected, cluster_adjusted, mcnemar_adjusted
        ):
            result["holm_adjusted_cluster_p_within_outcome"] = cluster_p
            result["holm_adjusted_mcnemar_p_secondary"] = mcnemar_p

    # Failure-inclusive top-1 sensitivity analyses.
    failure_results: list[dict] = []
    p1_wrong = part1[part1["anchor_condition"].eq("incorrect_label")]
    paired = make_pair(
        no_label,
        p1_wrong,
        keys=["case_id", "logical_model_id", "repeat_id"],
        outcome="top1_correct",
        how="left",
        missing_comparison_as_zero=True,
    )
    result, contributions = paired_cluster_stats(
        paired,
        scope="Part I",
        outcome="top1_correct",
        reference_name="no_label",
        comparison_name="incorrect_label",
        analysis_type="failure_inclusive",
    )
    result["holm_adjusted_cluster_p_within_outcome"] = result[
        "case_cluster_randomization_exact_p"
    ]
    result["holm_adjusted_mcnemar_p_secondary"] = result[
        "mcnemar_exact_p_secondary"
    ]
    failure_results.append(result)
    all_contributions.extend(contributions)

    part2_failure_results = []
    for strategy in STRATEGIES:
        comparison = part2[part2["intervention"].eq(strategy)]
        paired = make_pair(
            baseline,
            comparison,
            keys=["case_id", "logical_model_id"],
            outcome="top1_correct",
            how="left",
            missing_comparison_as_zero=True,
        )
        result, contributions = paired_cluster_stats(
            paired,
            scope="Part II",
            outcome="top1_correct",
            reference_name="baseline",
            comparison_name=strategy,
            analysis_type="failure_inclusive",
        )
        part2_failure_results.append(result)
        all_contributions.extend(contributions)
    cluster_adjusted = holm_adjust(
        [result["case_cluster_randomization_exact_p"] for result in part2_failure_results]
    )
    mcnemar_adjusted = holm_adjust(
        [result["mcnemar_exact_p_secondary"] for result in part2_failure_results]
    )
    for result, cluster_p, mcnemar_p in zip(
        part2_failure_results, cluster_adjusted, mcnemar_adjusted
    ):
        result["holm_adjusted_cluster_p_within_outcome"] = cluster_p
        result["holm_adjusted_mcnemar_p_secondary"] = mcnemar_p
        failure_results.append(result)

    # Exploratory complete-case direct comparisons versus self-check.
    direct_results: list[dict] = []
    self_check = part2[part2["intervention"].eq("self_check")]
    for outcome in ["top1_correct", "evidence_override"]:
        outcome_results = []
        for strategy in ["evidence_first", "differential", "counter_anchor"]:
            comparison = part2[part2["intervention"].eq(strategy)]
            paired = make_pair(
                self_check,
                comparison,
                keys=["case_id", "logical_model_id"],
                outcome=outcome,
            )
            result, contributions = paired_cluster_stats(
                paired,
                scope="Part II direct",
                outcome=outcome,
                reference_name="self_check",
                comparison_name=strategy,
                analysis_type="complete_case_exploratory",
            )
            outcome_results.append(result)
            all_contributions.extend(contributions)
        cluster_adjusted = holm_adjust(
            [result["case_cluster_randomization_exact_p"] for result in outcome_results]
        )
        mcnemar_adjusted = holm_adjust(
            [result["mcnemar_exact_p_secondary"] for result in outcome_results]
        )
        for result, cluster_p, mcnemar_p in zip(
            outcome_results, cluster_adjusted, mcnemar_adjusted
        ):
            result["holm_adjusted_cluster_p_within_outcome"] = cluster_p
            result["holm_adjusted_mcnemar_p_secondary"] = mcnemar_p
            direct_results.append(result)

    all_frame = pd.DataFrame(all_results)
    part1_frame = all_frame[all_frame["scope"].eq("Part I")].copy()
    part2_frame = all_frame[all_frame["scope"].eq("Part II")].copy()
    failure_frame = pd.DataFrame(failure_results)
    direct_frame = pd.DataFrame(direct_results)
    contributions_frame = pd.DataFrame(all_contributions)

    part1_frame.to_csv(OUTPUT_DIR / "part1_paired_cluster_inference_v3.csv", index=False)
    part2_frame.to_csv(OUTPUT_DIR / "part2_paired_cluster_inference_v3.csv", index=False)
    failure_frame.to_csv(
        OUTPUT_DIR / "failure_inclusive_cluster_inference_v3.csv", index=False
    )
    direct_frame.to_csv(
        OUTPUT_DIR / "direct_self_check_cluster_inference_v3.csv", index=False
    )
    contributions_frame.to_csv(
        OUTPUT_DIR / "case_cluster_contributions_v3.csv", index=False
    )

    summary = {
        "analysis_version": "cluster_inference_revision_v3",
        "source_dataset": str(DATASET),
        "source_dataset_sha256": sha256_file(DATASET),
        "independent_sampling_unit": "case_id",
        "planned_case_clusters": 30,
        "primary_p_value_method": (
            "Exact two-sided case-cluster sign-flip randomization test; one common sign "
            "per case; exact distribution evaluated by dynamic programming."
        ),
        "confidence_interval_method": (
            "Percentile case-cluster bootstrap retaining all model/repetition pairs "
            "within sampled cases."
        ),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "secondary_p_value_method": (
            "Exact McNemar test at the pair level, retained only as a sensitivity analysis."
        ),
        "multiplicity": (
            "Part II P values Holm-adjusted across the four strategies within each outcome."
        ),
        "key_top1_results": {
            "part1_complete_case_incorrect_vs_no_label": part1_frame[
                part1_frame["outcome"].eq("top1_correct")
                & part1_frame["comparison"].eq("incorrect_label")
            ].iloc[0].to_dict(),
            "part2_complete_case": part2_frame[
                part2_frame["outcome"].eq("top1_correct")
            ].to_dict(orient="records"),
            "failure_inclusive": failure_frame.to_dict(orient="records"),
        },
    }
    with (OUTPUT_DIR / "cluster_inference_summary_v3.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    def records_without_nan(frame: pd.DataFrame) -> list[dict]:
        return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")

    workbook_payload = {
        "part1": records_without_nan(part1_frame),
        "part2": records_without_nan(part2_frame),
        "failure_inclusive": records_without_nan(failure_frame),
        "direct_self_check": records_without_nan(direct_frame),
        "case_contributions": records_without_nan(contributions_frame),
    }
    with (OUTPUT_DIR / "cluster_inference_workbook_payload_v3.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(workbook_payload, handle, ensure_ascii=False, allow_nan=False)

    print("Part I top-1")
    print(
        part1_frame[part1_frame["outcome"].eq("top1_correct")][
            [
                "comparison",
                "paired_n",
                "paired_risk_difference",
                "case_cluster_bootstrap_low",
                "case_cluster_bootstrap_high",
                "case_cluster_randomization_exact_p",
                "mcnemar_exact_p_secondary",
            ]
        ].to_string(index=False)
    )
    print("\nPart II top-1")
    print(
        part2_frame[part2_frame["outcome"].eq("top1_correct")][
            [
                "comparison",
                "paired_n",
                "paired_risk_difference",
                "case_cluster_bootstrap_low",
                "case_cluster_bootstrap_high",
                "case_cluster_randomization_exact_p",
                "holm_adjusted_cluster_p_within_outcome",
                "mcnemar_exact_p_secondary",
            ]
        ].to_string(index=False)
    )
    print("\nPart II evidence override")
    print(
        part2_frame[part2_frame["outcome"].eq("evidence_override")][
            [
                "comparison",
                "paired_risk_difference",
                "case_cluster_bootstrap_low",
                "case_cluster_bootstrap_high",
                "case_cluster_randomization_exact_p",
                "holm_adjusted_cluster_p_within_outcome",
            ]
        ].to_string(index=False)
    )
    print("\nFailure-inclusive Part II top-1")
    print(
        failure_frame[failure_frame["scope"].eq("Part II")][
            [
                "comparison",
                "paired_risk_difference",
                "case_cluster_randomization_exact_p",
                "holm_adjusted_cluster_p_within_outcome",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
