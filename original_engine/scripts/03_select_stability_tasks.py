from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchoring_study.config import load_config, project_path
from anchoring_study.io_utils import read_csv, sha256_file, write_csv_atomic
from anchoring_study.tasks import TASK_FIELDS


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select a deterministic, output-blind stability subset."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--per-part", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.per_part < 1:
        raise ValueError("--per-part must be positive")

    config = load_config(args.config)
    selected: list[dict[str, str]] = []
    for key in ("tasks_part1", "tasks_part2"):
        rows = read_csv(project_path(config, config["paths"][key]))
        if len(rows) < args.per_part:
            raise RuntimeError(f"{key} contains fewer than {args.per_part} rows")
        selected.extend(rows[: args.per_part])

    task_ids = [row["task_id"] for row in selected]
    if len(task_ids) != len(set(task_ids)):
        raise RuntimeError("The stability subset contains duplicate task IDs")
    output = project_path(config, args.output)
    write_csv_atomic(output, selected, TASK_FIELDS)
    manifest = {
        "selection_rule": "first_n_rows_from_each_deterministically_randomized_part",
        "per_part": args.per_part,
        "selected_count": len(selected),
        "task_ids": task_ids,
        "subset_csv": str(output),
        "subset_sha256": sha256_file(output),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
