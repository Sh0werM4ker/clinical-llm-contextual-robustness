from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchoring_study.config import load_config, project_path
from anchoring_study.io_utils import read_csv, read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create an explicitly derived first-success review ledger."
    )
    parser.add_argument("--config", default="config/pilot.toml")
    parser.add_argument(
        "--output",
        default="outputs/pilot_advanced_local_v2/derived/canonical_first_success.jsonl",
    )
    parser.add_argument(
        "--audit",
        default="outputs/pilot_advanced_local_v2/derived/duplicate_success_audit.csv",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    raw_path = project_path(config, config["paths"]["raw_outputs"])
    records = read_jsonl(raw_path)
    task_rows = []
    for key in ("tasks_part1", "tasks_part2"):
        task_rows.extend(read_csv(project_path(config, config["paths"][key])))
    planned_ids = [row["task_id"] for row in task_rows]

    successes: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for line_number, record in enumerate(records, start=1):
        if record.get("status") == "success":
            successes[str(record["task_id"])].append((line_number, record))

    missing = [task_id for task_id in planned_ids if not successes.get(task_id)]
    if missing:
        raise RuntimeError(f"Cannot canonicalize; missing successes: {missing}")

    canonical: list[dict] = []
    audit_rows: list[dict[str, object]] = []
    for task_id in planned_ids:
        candidates = successes[task_id]
        selected_line, selected = candidates[0]
        selected_copy = dict(selected)
        selected_copy["derived_selection"] = {
            "policy": "first_success_in_append_only_ledger",
            "source_raw_path": str(raw_path),
            "source_line_number": selected_line,
            "success_candidates": len(candidates),
            "pilot_concurrency_incident": len(candidates) > 1,
        }
        canonical.append(selected_copy)
        raw_hashes = [
            hashlib.sha256(str(item[1].get("raw_output_text", "")).encode("utf-8")).hexdigest()
            for item in candidates
        ]
        audit_rows.append(
            {
                "task_id": task_id,
                "success_count": len(candidates),
                "selected_source_line": selected_line,
                "selected_run_id": selected.get("run_id", ""),
                "selected_timestamp_utc": selected.get("timestamp_utc", ""),
                "selected_raw_output_sha256": raw_hashes[0],
                "discarded_source_lines": "|".join(str(item[0]) for item in candidates[1:]),
                "discarded_run_ids": "|".join(str(item[1].get("run_id", "")) for item in candidates[1:]),
                "all_raw_outputs_identical": int(len(set(raw_hashes)) == 1),
            }
        )

    output_path = project_path(config, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in canonical:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, output_path)

    audit_path = project_path(config, args.audit)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_temp = audit_path.with_suffix(audit_path.suffix + ".tmp")
    with audit_temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    os.replace(audit_temp, audit_path)
    print(
        json.dumps(
            {
                "canonical_records": len(canonical),
                "tasks_with_duplicate_successes": sum(
                    int(row["success_count"]) > 1 for row in audit_rows
                ),
                "output": str(output_path),
                "audit": str(audit_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
