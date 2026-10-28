from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchoring_study.analysis import part1_analysis, part2_analysis, prepare_data, save_figures
from anchoring_study.config import load_config, project_path


def fmt(value: float) -> str:
    return "NA" if value is None or math.isnan(value) else f"{value:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run prespecified core statistical analyses.")
    parser.add_argument("--config", default="config/pilot.toml")
    parser.add_argument("--allow-mock-for-software-test", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    data_path = project_path(config, config["paths"]["final_dataset"])
    frame = prepare_data(data_path)
    if (frame["is_mock"] == 1).any() and not args.allow_mock_for_software_test:
        print("ANALYSIS BLOCKED: mock data detected. Use live research data.")
        return 2
    if frame["case_id"].nunique() < 10:
        print("WARNING: fewer than 10 cases; estimates are software/pilot checks, not confirmatory results.")

    seed = int(config["study"]["seed"])
    baseline_repeat = int(config["design"]["part2_baseline_repeat"])
    p1_summary, p1_pairs, derived = part1_analysis(frame, seed)
    p2_summary, p2_pairs = part2_analysis(frame, seed, baseline_repeat)
    results_dir = project_path(config, config["paths"]["results_dir"])
    figures_dir = results_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    p1_summary.to_csv(results_dir / "table_part1_summary.csv", index=False, encoding="utf-8-sig")
    p1_pairs.to_csv(results_dir / "table_part1_paired_comparisons.csv", index=False, encoding="utf-8-sig")
    p2_summary.to_csv(results_dir / "table_part2_summary.csv", index=False, encoding="utf-8-sig")
    p2_pairs.to_csv(results_dir / "table_part2_paired_comparisons.csv", index=False, encoding="utf-8-sig")
    (results_dir / "part1_derived_metrics.json").write_text(
        json.dumps(derived, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_figures(frame, p1_summary, p2_summary, figures_dir, baseline_repeat)

    top1 = p1_pairs[p1_pairs["outcome"] == "top1_correct"].iloc[0]
    unsafe = p1_pairs[p1_pairs["outcome"] == "unsafe_management"].iloc[0]
    report = f"""# Core analysis report

Study ID: `{config['study']['study_id']}`  
Cases: {frame['case_id'].nunique()}  
Model configurations: {frame['logical_model_id'].nunique()}  
Rows: {len(frame)}

## Part I

- Top-1 accuracy risk difference (incorrect label minus no label): {fmt(float(top1['risk_difference']))} (case-cluster bootstrap 95% CI {fmt(float(top1['cluster_bootstrap_low']))} to {fmt(float(top1['cluster_bootstrap_high']))}).
- Unsafe-management risk difference (incorrect label minus no label): {fmt(float(unsafe['risk_difference']))} (95% CI {fmt(float(unsafe['cluster_bootstrap_low']))} to {fmt(float(unsafe['cluster_bootstrap_high']))}).
- Anchoring conversion: {derived['anchoring_conversion_numerator']}/{derived['anchoring_conversion_denominator']} = {fmt(float(derived['anchoring_conversion_rate']))}.
- Unsafe-management conversion: {derived['unsafe_conversion_numerator']}/{derived['unsafe_conversion_denominator']} = {fmt(float(derived['unsafe_conversion_rate']))}.
- Median diagnostic displacement: {fmt(float(derived['diagnostic_displacement_median']))} ranks.

## Part II

See `table_part2_paired_comparisons.csv` for intervention-versus-baseline paired risk differences, exact McNemar tests, and Holm-adjusted P values within each outcome family.

## Interpretation guardrails

- This automated report does not replace clinical interpretation or a statistician's review.
- Pilot estimates based on three cases are for workflow validation only.
- Confirmatory inference must use the locked formal case set and prespecified analysis plan.
"""
    (results_dir / "analysis_report.md").write_text(report, encoding="utf-8")
    print(f"Analysis complete. Report: {results_dir / 'analysis_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

