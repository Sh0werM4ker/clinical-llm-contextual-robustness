from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchoring_study.config import load_config, project_path
from anchoring_study.io_utils import read_csv, read_jsonl, write_csv_atomic


TERMINAL_STATUSES = {"success", "model_refusal", "generation_failure"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def malformed_jsonl_lines(path: Path) -> int:
    malformed = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
    return malformed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify terminal completeness and freeze formal generation outputs."
    )
    parser.add_argument("--config", default="config/formal_local_v1.toml")
    args = parser.parse_args()
    config = load_config(args.config)

    task_paths = [
        project_path(config, config["paths"]["tasks_part1"]),
        project_path(config, config["paths"]["tasks_part2"]),
    ]
    tasks: list[dict[str, str]] = []
    for path in task_paths:
        tasks.extend(read_csv(path))
    task_ids = [row["task_id"] for row in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise RuntimeError("Duplicate task_id values exist in the formal task files.")
    task_map = {row["task_id"]: row for row in tasks}

    raw_path = project_path(config, config["paths"]["raw_outputs"])
    attempt_path = project_path(config, config["paths"]["attempt_log"])
    raw_rows = read_jsonl(raw_path)
    terminal_by_task: dict[str, dict] = {}
    success_counts: Counter[str] = Counter()
    for row in raw_rows:
        task_id = str(row.get("task_id", ""))
        status = row.get("status")
        if status == "success":
            success_counts[task_id] += 1
        if status in TERMINAL_STATUSES:
            terminal_by_task[task_id] = row

    planned = set(task_ids)
    terminal = set(terminal_by_task)
    missing_terminal = sorted(planned - terminal)
    unplanned_terminal = sorted(terminal - planned)
    duplicate_successes = {
        task_id: count for task_id, count in success_counts.items() if count > 1
    }
    if missing_terminal or unplanned_terminal or duplicate_successes:
        raise RuntimeError(
            json.dumps(
                {
                    "missing_terminal": missing_terminal,
                    "unplanned_terminal": unplanned_terminal,
                    "duplicate_successes": duplicate_successes,
                },
                indent=2,
            )
        )

    status_counts = Counter(
        terminal_by_task[task_id]["status"] for task_id in task_ids
    )
    if sum(status_counts.values()) != len(tasks):
        raise RuntimeError("Terminal status counts do not reconcile to planned tasks.")
    malformed_raw = malformed_jsonl_lines(raw_path)
    malformed_attempts = malformed_jsonl_lines(attempt_path)
    if malformed_raw or malformed_attempts:
        raise RuntimeError(
            f"Malformed JSONL lines: raw={malformed_raw}, attempts={malformed_attempts}"
        )

    failure_rows: list[dict[str, object]] = []
    for task_id in task_ids:
        terminal_row = terminal_by_task[task_id]
        if terminal_row["status"] == "success":
            continue
        task = task_map[task_id]
        failure_rows.append(
            {
                "task_id": task_id,
                "terminal_status": terminal_row["status"],
                "part": task["part"],
                "case_id": task["case_id"],
                "logical_model_id": task["logical_model_id"],
                "api_model": task["api_model"],
                "anchor_condition": task["anchor_condition"],
                "intervention": task["intervention"],
                "repeat_id": task["repeat_id"],
                "attempts_used": terminal_row.get("attempts_used", ""),
                "error_type": terminal_row.get("error_type", ""),
                "error_message": terminal_row.get("error_message", ""),
                "timestamp_utc": terminal_row.get("timestamp_utc", ""),
                "prompt_hash": task["prompt_hash"],
            }
        )

    ratings_dir = ROOT / "ratings" / "formal_local_v1"
    ratings_dir.mkdir(parents=True, exist_ok=True)
    failure_path = ratings_dir / "generation_failures.csv"
    failure_fields = list(failure_rows[0]) if failure_rows else ["task_id"]
    write_csv_atomic(failure_path, failure_rows, failure_fields)

    hash_paths = {
        "config": ROOT / args.config,
        "cases": project_path(config, config["paths"]["cases"]),
        "tasks_part1": task_paths[0],
        "tasks_part2": task_paths[1],
        "task_manifest": project_path(config, config["paths"]["manifest"]),
        "raw_outputs": raw_path,
        "attempt_log": attempt_path,
        "model_registry": ROOT / "config" / "formal_model_registry_v1.json",
        "prefreeze_manifest": ROOT / "frozen" / "formal_v1" / "FREEZE_MANIFEST.json",
    }
    audit = {
        "study_id": config["study"]["study_id"],
        "generation_complete": True,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "planned_task_count": len(tasks),
        "terminal_task_count": len(terminal_by_task),
        "terminal_status_counts": dict(sorted(status_counts.items())),
        "scorable_success_count": status_counts.get("success", 0),
        "generation_failure_count": status_counts.get("generation_failure", 0),
        "model_refusal_count": status_counts.get("model_refusal", 0),
        "duplicate_success_task_count": len(duplicate_successes),
        "malformed_raw_lines": malformed_raw,
        "malformed_attempt_lines": malformed_attempts,
        "failure_register": str(failure_path.relative_to(ROOT)).replace("\\", "/"),
        "files": {
            name: {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in hash_paths.items()
        },
    }

    frozen_dir = ROOT / "frozen" / "formal_v1"
    audit_path = frozen_dir / "FORMAL_GENERATION_COMPLETION.json"
    marker_path = frozen_dir / "FORMAL_GENERATION_COMPLETED.txt"
    if audit_path.exists():
        existing = json.loads(audit_path.read_text(encoding="utf-8"))
        if existing["files"]["raw_outputs"]["sha256"] != audit["files"]["raw_outputs"]["sha256"]:
            raise RuntimeError("Frozen generation audit exists but raw output hash has changed.")
        print(json.dumps(existing, indent=2, ensure_ascii=False))
        return 0

    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    marker_path.write_text(
        "Formal generation is terminally complete.\n"
        f"Planned tasks: {len(tasks)}\n"
        f"Scorable successes: {status_counts.get('success', 0)}\n"
        f"Generation failures: {status_counts.get('generation_failure', 0)}\n"
        f"Model refusals: {status_counts.get('model_refusal', 0)}\n"
        "Terminal failures must not be regenerated beyond the prespecified retry cap.\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
