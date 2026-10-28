# Supplementary research code

Additional file 11 for A Controlled Diagnostic Context Perturbation Framework for Evaluating Contextual Robustness in Clinical Large Language Models.

## Contents and scope

This code-only archive contains the original research engine, the subsequent final case-cluster statistical analysis, and the submission Figure 3 plotting code. Input datasets, model responses, configuration files, manuscript documents and rendered figures are not included. Running the code requires separately supplied study inputs. This archive does not contain the later independent-validation, intermediate-evidence, new-case, or teaching-simulation projects.

- `original_engine/src/anchoring_study/`: unchanged original modules for case validation, task construction, prompting, generation, quality control, ratings and initial analysis.
- `original_engine/scripts/`: unchanged original numbered workflow scripts 00 through 32, including original R sensitivity-model code. These are historical workflow entry points. They depend on external protocol configurations, case files, ratings and, for generation, locally installed models or a separately configured API. Consult each script's command-line arguments before use. No model calls are made by the portable final-analysis entry point.
- `final_analysis/cluster_inference_original.py`: unchanged subsequent revision v3 of the final statistical analysis. Use this implementation for the final case-cluster inference. Its original hard-coded source paths are retained for provenance; the portable entry point overrides only the input and output paths.
- `run_final_analysis.py`: new portable wrapper; no statistical formulas are changed.
- `figures/figure3.R`: the R code used to draw the submitted three-panel Figure 3. R base graphics with Cairo support and Arial are required. Figures 1 and 2 are not reproduced by this archive.
- `figures/prepare_figure3_input.py`: new helper that converts final analysis CSV outputs into the plotting TSV, with risk differences and confidence limits expressed in percentage points.
- `PROVENANCE.json` and `SHA256SUMS.txt`: source identity and archive integrity records.

## Why there are two analysis versions

The original engine and its `32_formal_sensitivity_analysis.py` retain the original analyses, including pair-level exact McNemar tests. Revision v3 additionally implements the exact case-cluster sign-flip randomization test and uses it for primary inference; McNemar tests remain secondary. It also implements paired case-cluster bootstrap intervals, failure-inclusive sensitivity analyses and exploratory direct comparisons with self-check. Do not substitute the original engine's analysis outputs for revision v3 outputs.

Revision v3 uses 10,000 bootstrap replicates and seed 20260721, resampling cases while retaining their within-case pairs. Holm adjustment is applied across the four Part II strategies within each outcome, and across three exploratory direct comparisons within each outcome.

## Run the final analysis

Python 3.11 or later is required. The final-analysis dependency versions are listed separately from the historical engine requirements.

```text
python -m pip install -r requirements-final-analysis.txt
python run_final_analysis.py --dataset path/to/formal_final_analysis_dataset_v2_anchor_qc.csv --output-dir results
```

The input CSV must include `part`, `case_id`, `logical_model_id`, `anchor_condition`, `intervention`, `repeat_id`, `final_analysis_eligible`, `top1_correct`, `unsafe_management`, `mandatory_action_complete`, `anchor_adoption`, `evidence_override`, and `over_conservatism`. Eligible rows have final_analysis_eligible=1; binary endpoints are coded 0/1, and anchor_adoption_binary is derived as anchor_adoption=2. Part I conditions are no_label, correct_label and incorrect_label; Part II intervention names are evidence_first, differential, counter_anchor and self_check. Part II comparisons use Part I incorrect_label repetition 1 as baseline. The CSV must contain the final reviewed labels used in the manuscript. The code-only archive cannot reproduce study results without that input.

Outputs include Part I and Part II paired contrasts, failure-inclusive contrasts, direct self-check contrasts, per-case contributions, an analysis summary and a workbook payload, in CSV/JSON formats. The input file is not modified.

## Draw Figure 3 after running the analysis

```text
python figures/prepare_figure3_input.py --results-dir results --output results/figure3_estimates.tsv
Rscript figures/figure3.R results/figure3_estimates.tsv results/Figure3.pdf
```

The plotting helper uses unrounded estimates; the plot labels display two decimal places. R must have Cairo PDF support and Arial available for the submitted font specification.

## Verification and provenance

On 28 October 2026, all 68 Python/R/PowerShell files under src and scripts in the original D-drive research engine matched the corresponding C-drive engine copy byte for byte (SHA-256). All 65 corresponding files in the earlier desktop reproducibility package also matched. Revision v3 was found in the subsequent C-drive research-analysis directory; no same-named revision-v3 script was found under the supplied D-drive root.

All copied source files in this archive retain their original bytes. New files are limited to portable execution/plotting helpers, dependency information and documentation. Every included Python file was checked for syntax, and the new helpers' command-line help was checked. Following the author's instruction to compare code only, no study input files were compared or study analyses rerun as part of the final code comparison and packaging. This archive therefore reports source-code identity and packaging checks, not a new numerical reproduction validation.

No research data, API credentials, environment files or model weights are included. The original engine requirements are provided for its separate historical workflow; they are not required to run the final-analysis wrapper. No new software license is assigned by this packaging step.
