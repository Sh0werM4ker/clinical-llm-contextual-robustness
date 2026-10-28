from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

from anchoring_study.config import load_config, project_path
from anchoring_study.generation import LOCAL_CONFIRMATION, LIVE_CONFIRMATION, run_tasks


def main() -> int:
    parser = argparse.ArgumentParser(description="Run independent structured model requests.")
    parser.add_argument("--config", default="config/pilot.toml")
    parser.add_argument("--tasks", required=True, help="Task CSV path, relative to project root.")
    parser.add_argument("--model", default=None, help="Optional logical model filter.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mock", action="store_true", help="Generate free synthetic test outputs.")
    parser.add_argument("--confirm-live-run", default="")
    parser.add_argument("--confirm-local-run", default="")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    config = load_config(args.config)
    tasks_path = project_path(config, args.tasks)
    counters = run_tasks(
        config,
        tasks_path,
        model_filter=args.model,
        limit=args.limit,
        mock=args.mock,
        live_confirmation=args.confirm_live_run,
        local_confirmation=args.confirm_local_run,
    )
    print(json.dumps(counters, ensure_ascii=False, indent=2))
    if counters["failed"] or counters["refused"]:
        return 1
    if not args.mock:
        print("Generation completed using the provider recorded in the task file.")
    else:
        print("MOCK ONLY: these records are not research data and QC will flag them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
