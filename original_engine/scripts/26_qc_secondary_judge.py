from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "secondary_judge_glm47"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Malformed JSONL line {number}: {path}") from error
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Quality-control a secondary judge run")
    parser.add_argument("--mode", choices=["pilot", "formal"], default="pilot")
    parser.add_argument("--expected", type=int, default=None)
    args = parser.parse_args()

    run_dir = OUTPUT_ROOT / args.mode
    raw_path = run_dir / "secondary_judge_raw.jsonl"
    attempts_path = run_dir / "secondary_judge_attempts.jsonl"
    csv_path = run_dir / "secondary_judge_results.csv"
    protocol_path = run_dir / "secondary_judge_run_protocol.json"
    terminal: dict[str, dict[str, Any]] = {}
    duplicate_terminal_counts: Counter[str] = Counter()
    for row in read_jsonl(raw_path):
        item = str(row.get("judge_item_id", ""))
        duplicate_terminal_counts[item] += 1
        terminal[item] = row
    successes = [row for row in terminal.values() if row.get("status") == "success"]
    failures = [row for row in terminal.values() if row.get("status") != "success"]

    csv_rows: list[dict[str, str]] = []
    if csv_path.exists():
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
    digests = sorted({str(row.get("model_digest")) for row in successes})
    returned_models = sorted({str(row.get("model_returned")) for row in successes})
    ids = [str(row.get("judge_item_id")) for row in successes]
    hashes = [str(row.get("input_hash")) for row in successes]
    quote_over_20 = [
        row["judge_item_id"]
        for row in successes
        if len(str(row["parsed_output"].get("evidence_quote", "")).split()) > 20
    ]
    bad_blank_anchor_defaults = [
        row["judge_item_id"]
        for row in successes
        if row["parsed_output"].get("incorrect_anchor_rank") not in range(1, 7)
    ]
    expected = args.expected
    if expected is None and protocol_path.exists():
        expected = int(json.loads(protocol_path.read_text(encoding="utf-8"))["selected_count"])
    expected = expected or len(terminal)

    qc = {
        "mode": args.mode,
        "expected_records": expected,
        "terminal_records": len(terminal),
        "successes": len(successes),
        "failures": len(failures),
        "missing": expected - len(terminal),
        "results_csv_rows": len(csv_rows),
        "unique_judge_item_ids": len(set(ids)),
        "unique_input_hashes": len(set(hashes)),
        "model_digests": digests,
        "returned_models": returned_models,
        "duplicate_terminal_records": {
            key: value for key, value in duplicate_terminal_counts.items() if value > 1
        },
        "attempt_records": len(read_jsonl(attempts_path)),
        "evidence_quotes_over_20_words": quote_over_20,
        "invalid_rank_records": bad_blank_anchor_defaults,
        "distributions": {
            field: dict(Counter(str(row["parsed_output"].get(field)) for row in successes))
            for field in [
                "top1_correct",
                "gold_diagnosis_rank",
                "incorrect_anchor_rank",
                "anchor_adoption",
                "evidence_override",
                "triage_level_scored",
                "mandatory_action_complete",
                "unsafe_management",
                "over_conservatism",
                "judge_confidence",
                "boundary_flag",
            ]
        },
        "files": {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256_file(path)
            for path in [raw_path, attempts_path, csv_path, protocol_path]
            if path.exists()
        },
    }
    passed = all(
        [
            qc["successes"] == expected,
            qc["failures"] == 0,
            qc["missing"] == 0,
            qc["results_csv_rows"] == expected,
            qc["unique_judge_item_ids"] == expected,
            qc["unique_input_hashes"] == expected,
            len(digests) == 1,
            len(returned_models) == 1,
            not quote_over_20,
            not bad_blank_anchor_defaults,
        ]
    )
    qc["qc_passed"] = passed
    out_path = run_dir / "secondary_judge_qc.json"
    out_path.write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(qc, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
