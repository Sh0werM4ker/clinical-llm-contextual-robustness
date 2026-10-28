from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchoring_study.io_utils import latest_terminal_by_task, read_csv, read_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether every task for one model in a task file has a terminal record."
    )
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--raw", default="outputs/formal_local_v1/raw/raw_outputs.jsonl"
    )
    args = parser.parse_args()

    tasks_path = ROOT / args.tasks
    raw_path = ROOT / args.raw
    tasks = [
        row for row in read_csv(tasks_path)
        if row["logical_model_id"] == args.model
    ]
    terminal = latest_terminal_by_task(read_jsonl(raw_path))
    pending = [row["task_id"] for row in tasks if row["task_id"] not in terminal]
    print({
        "model": args.model,
        "task_file": args.tasks,
        "planned": len(tasks),
        "terminal": len(tasks) - len(pending),
        "pending": len(pending),
        "first_pending": pending[:5],
    })
    return 0 if tasks and not pending else 2


if __name__ == "__main__":
    raise SystemExit(main())
