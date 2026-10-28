from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchoring_study.config import load_config, project_path
from anchoring_study.qc import build_qc_report, write_qc_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit task completion and raw outputs.")
    parser.add_argument("--config", default="config/pilot.toml")
    parser.add_argument("--raw-outputs", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--tasks",
        default=None,
        help="Optional single task CSV used as the complete expected set for a subset test.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.raw_outputs:
        config["paths"]["raw_outputs"] = args.raw_outputs
    if args.tasks:
        config["paths"]["tasks_part1"] = args.tasks
        config["paths"]["tasks_part2"] = args.tasks
    report = build_qc_report(config)
    if args.output:
        path = project_path(config, args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        path = write_qc_report(config, report)
    print(json.dumps({key: report[key] for key in [
        "planned_task_count", "successful_task_count", "missing_success_count",
        "schema_problem_count", "mock_task_count", "model_refusal_count",
        "derived_review_only_count", "pilot_review_ready", "research_ready"
    ]}, ensure_ascii=False, indent=2))
    print(f"Full report: {path}")
    return 0 if report["research_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
