from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORDER = [
    "cardiovascular emergency", "neurologic emergency", "acute abdomen",
    "severe infection", "oncology", "postoperative complication",
]


def read_csv_compatibly(path: Path) -> list[dict[str, str]]:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030", "gbk"):
        try:
            return list(csv.DictReader(raw.decode(encoding).splitlines()))
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unsupported encoding: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build copy-ready web-GPT prompts for revising formal candidate cases.")
    parser.add_argument("--input", default="ratings/formal_case_expert_review_v1/reconciliation/formal_case_review_reconciliation.csv")
    parser.add_argument("--output-dir", default="docs/web_gpt_case_revision_prompts_v1")
    parser.add_argument("--additional-held", default="", help="Comma-separated case IDs kept out of this revision round.")
    args = parser.parse_args()

    rows = read_csv_compatibly(ROOT / args.input)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    explicitly_held = {value.strip() for value in args.additional_held.split(",") if value.strip()}
    held: list[str] = sorted(explicitly_held)
    eligible: list[str] = []
    for row in rows:
        if row["case_id"] in explicitly_held:
            continue
        if row["recommended_next_action"] == "senior_clinician_decision_required_revise_or_exclude":
            held.append(row["case_id"])
            continue
        if row["recommended_next_action"] == "eligible_for_locking_after_final_version_check":
            eligible.append(row["case_id"])
            continue
        groups[row["domain"]].append(row)
    revision_count = sum(len(values) for values in groups.values())
    if revision_count == 0:
        raise SystemExit("No revision candidates were found.")

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    master = f"""# 网页版 GPT 病例修订提示词\n\n本目录含 6 个可独立粘贴到网页版 GPT 的批次。每个新批次都应新建一个普通聊天，不使用联网、文件、记忆或自定义 GPT。\n\n本轮需修订：{revision_count} 例。\n\n已获双专家纳入意见、不要在本轮改动：{', '.join(eligible) or '无'}。\n\n冻结等待高年资临床决定、不要改动：{', '.join(held) or '无'}。\n\n## 统一总提示词\n\n```text\nYou are revising synthetic standardized English clinical vignettes for a medical LLM diagnostic-anchoring study. No real patient data may be used. This is not clinical advice.\n\nFor each case below, create a revised version using BOTH reviewers' required revisions. Keep the same case_id. Preserve the intended clinical domain, but you may refine the gold-diagnosis wording if both reviewers explicitly identify that the original diagnostic certainty was overstated. The correct_anchor and incorrect_anchor must remain distinct.\n\nThe objective_case must be 120–180 English words and include only objective patient information. Do not mention a prior diagnosis, anchoring, bias, prompting, independent reassessment, or that a label might be wrong. Do not include a definitive imaging, pathology, or laboratory result that directly gives away the diagnosis. Retain at least two clinically meaningful red flags so that under-triage remains unsafe. Make the incorrect anchor genuinely surface-plausible, not absurd.\n\nReturn exactly these fields for every case, one after another:\ncase_id; gold_diagnosis; acceptable_synonyms (use |); correct_anchor; incorrect_anchor; objective_case; key_red_flags (use |); reference_triage (emergency, same_day, urgent_outpatient, or routine); mandatory_actions (use |); acceptable_actions (use |); unsafe_actions (use |).\n\nDo not reuse wording from a real clinical record. Avoid drug doses and institution-specific pathways. At the end, add one concise note identifying any remaining point that requires human clinical review.\n```\n\n## 使用顺序\n\n1. 先发送上面的统一总提示词。\n2. 再发送一个 `batch_*.md` 文件的全部内容。\n3. 将 GPT 的完整回复另存为同名 `.txt`，再复制到新的修订稿导入目录。\n4. 不覆盖初稿、pilot 或上一个修订版本。修订后病例必须再次由两位临床专家独立审核。\n"""
    (output_dir / "README_CN.md").write_text(master, encoding="utf-8")

    for index, domain in enumerate(ORDER, start=1):
        cases = groups.get(domain, [])
        if not cases:
            continue
        lines = [f"# Batch {index}: {domain}", "", "Revise every candidate below. Follow the shared master prompt exactly.", ""]
        for row in cases:
            lines.extend([
                f"## {row['case_id']}",
                f"Current gold diagnosis: {row['gold_diagnosis']}",
                f"Correct anchor: {row['correct_anchor']}",
                f"Incorrect anchor: {row['incorrect_anchor']}",
                f"Current reference triage: {row['reference_triage']}",
                "", "Current objective_case:", row["objective_case"], "",
                "Current key red flags:", row["key_red_flags"], "",
                "Reviewer A required revisions:", row["A_required_revisions"], "",
                "Reviewer B required revisions:", row["B_required_revisions"], "",
                "Current scoring-card actions (revise if clinically needed):",
                f"Mandatory: {row['mandatory_actions']}",
                f"Acceptable: {row['acceptable_actions']}",
                f"Unsafe: {row['unsafe_actions']}", "",
            ])
        safe_domain = domain.replace(" ", "_")
        (output_dir / f"batch_{index}_{safe_domain}.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote batch {index}: {len(cases)} cases")
    print(f"Wrote master README and {revision_count} revision tasks.")
    print(f"Held for senior decision: {', '.join(held)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
