from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTECTED_COLUMNS = [
    "case_id", "domain", "gold_diagnosis", "acceptable_synonyms", "correct_anchor",
    "incorrect_anchor", "objective_case", "key_red_flags", "reference_triage",
    "mandatory_actions", "acceptable_actions", "unsafe_actions",
]
SCORE_COLUMNS = [
    "gold_diagnosis_clarity_1_to_5", "incorrect_anchor_plausibility_1_to_5",
    "objective_evidence_sufficiency_1_to_5", "triage_and_management_clarity_1_to_5",
    "anchoring_test_calibration_1_to_5",
]
DISPOSITION = "recommended_disposition_include_revise_exclude"


def read_csv_compatibly(path: Path) -> list[dict[str, str]]:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030", "gbk"):
        try:
            text = raw.decode(encoding)
            return list(csv.DictReader(text.splitlines()))
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path} as UTF-8 or GB18030/GBK.")


def validate(rows: list[dict[str, str]], label: str, expected_count: int) -> list[str]:
    errors: list[str] = []
    if len(rows) != expected_count:
        errors.append(f"{label}: expected {expected_count} records, found {len(rows)}.")
    ids = [row.get("case_id", "").strip() for row in rows]
    if len(set(ids)) != len(ids) or any(not value for value in ids):
        errors.append(f"{label}: case IDs are blank or non-unique.")
    for row in rows:
        case_id = row.get("case_id", "<blank>")
        for column in SCORE_COLUMNS:
            if row.get(column, "").strip() not in {"1", "2", "3", "4", "5"}:
                errors.append(f"{label} {case_id}: invalid {column}.")
        if row.get(DISPOSITION, "").strip() not in {"include", "revise", "exclude"}:
            errors.append(f"{label} {case_id}: invalid disposition.")
    return errors


def consensus_action(a: str, b: str) -> str:
    if a == b == "include":
        return "eligible_for_locking_after_final_version_check"
    if a == b == "revise":
        return "revise_then_repeat_independent_review"
    if "exclude" in {a, b}:
        return "senior_clinician_decision_required_revise_or_exclude"
    return "revise_then_repeat_independent_review"


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconcile two independent formal-case clinical review forms.")
    parser.add_argument("--rater-a", default="ratings/formal_case_expert_review_v1/formal_case_expert_review_form_A.csv")
    parser.add_argument("--rater-b", default="ratings/formal_case_expert_review_v1/formal_case_expert_review_form_B.csv")
    parser.add_argument("--output", default="ratings/formal_case_expert_review_v1/reconciliation/formal_case_review_reconciliation.csv")
    parser.add_argument("--summary", default="results/formal_case_review_v1/RECONCILIATION_SUMMARY_CN.md")
    parser.add_argument("--expected-count", type=int, default=36)
    args = parser.parse_args()

    a_path, b_path = ROOT / args.rater_a, ROOT / args.rater_b
    a_rows, b_rows = read_csv_compatibly(a_path), read_csv_compatibly(b_path)
    errors = validate(a_rows, "Rater A", args.expected_count) + validate(b_rows, "Rater B", args.expected_count)
    a_by_id = {row["case_id"].strip(): row for row in a_rows}
    b_by_id = {row["case_id"].strip(): row for row in b_rows}
    if set(a_by_id) != set(b_by_id):
        errors.append("Rater A and B do not contain the same case IDs.")
    for case_id in sorted(set(a_by_id) & set(b_by_id)):
        for column in PROTECTED_COLUMNS:
            if (a_by_id[case_id].get(column) or "").strip() != (b_by_id[case_id].get(column) or "").strip():
                errors.append(f"{case_id}: protected source field differs between reviewer forms ({column}).")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    result_rows: list[dict[str, str]] = []
    agreement: dict[str, float] = {}
    mean_abs_difference: dict[str, float] = {}
    for column in SCORE_COLUMNS:
        pairs = [(int(a_by_id[i][column]), int(b_by_id[i][column])) for i in sorted(a_by_id)]
        agreement[column] = sum(x == y for x, y in pairs) / len(pairs)
        mean_abs_difference[column] = statistics.mean(abs(x - y) for x, y in pairs)

    disposition_pairs = Counter()
    action_counts = Counter()
    for case_id in sorted(a_by_id):
        a, b = a_by_id[case_id], b_by_id[case_id]
        disposition_a, disposition_b = a[DISPOSITION].strip(), b[DISPOSITION].strip()
        action = consensus_action(disposition_a, disposition_b)
        disposition_pairs[(disposition_a, disposition_b)] += 1
        action_counts[action] += 1
        row = {column: a[column] for column in PROTECTED_COLUMNS}
        for column in SCORE_COLUMNS:
            row[f"A_{column}"] = a[column].strip()
            row[f"B_{column}"] = b[column].strip()
            row[f"absolute_difference_{column}"] = str(abs(int(a[column]) - int(b[column])))
        row.update({
            "A_disposition": disposition_a,
            "B_disposition": disposition_b,
            "recommended_next_action": action,
            "A_required_revisions": a.get("required_revisions", "").strip(),
            "B_required_revisions": b.get("required_revisions", "").strip(),
            "A_comments": a.get("reviewer_comments", "").strip(),
            "B_comments": b.get("reviewer_comments", "").strip(),
        })
        result_rows.append(row)

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(result_rows[0])
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows)

    def ids_for(action: str) -> str:
        values = [row["case_id"] for row in result_rows if row["recommended_next_action"] == action]
        return ", ".join(values) if values else "无"

    summary = ROOT / args.summary
    summary.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 正式候选病例：双专家审核整合摘要", "",
        "两份审核表均完整、病例 ID 一致，且两份表的受保护病例正文和评分卡字段完全一致。", "",
        "## 处置建议配对", "",
        f"- A revise / B revise：{disposition_pairs[('revise', 'revise')]} 例。",
        f"- A include / B revise：{disposition_pairs[('include', 'revise')]} 例。",
        f"- A revise / B exclude：{disposition_pairs[('revise', 'exclude')]} 例。",
        f"- A include / B include：{disposition_pairs[('include', 'include')]} 例。",
        "",
        "因此当前没有病例获双人同意直接纳入。所有病例都需要修订；其中存在排除意见的病例需要高年资临床医生决定“修订或排除”。", "",
        "## 下一步动作", "",
        f"- 修订后重新独立审核：{ids_for('revise_then_repeat_independent_review')}。",
        f"- 需要高年资临床医生决定修订或排除：{ids_for('senior_clinician_decision_required_revise_or_exclude')}。",
        "",
        "## 五项量表的原始一致率与平均绝对差", "",
    ]
    for column in SCORE_COLUMNS:
        lines.append(f"- {column}：原始一致率 {agreement[column]:.1%}；平均绝对差 {mean_abs_difference[column]:.2f} 分。")
    lines.extend([
        "", "## 研究保护规则", "",
        "上述意见用于修订病例库，不能直接修改已完成的 pilot 数据。修订版病例必须作为新版本保存，并在两位专家复核后才可标记为 locked。",
    ])
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"RECONCILIATION PASSED: {len(result_rows)} cases")
    print(f"Reconciliation: {output}")
    print(f"Summary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
