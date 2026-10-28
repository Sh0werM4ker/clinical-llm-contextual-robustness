from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchoring_study.config import load_config
from anchoring_study.tasks import build_tasks, write_tasks_and_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Expand locked cases into randomized tasks.")
    parser.add_argument("--config", default="config/pilot.toml")
    args = parser.parse_args()
    config = load_config(args.config)
    part1, part2 = build_tasks(config)
    manifest = write_tasks_and_manifest(config, part1, part2)
    print(f"Locked cases: {manifest['locked_case_count']}")
    print(f"Enabled models: {manifest['model_count']}")
    print(f"Part I tasks: {manifest['part1_task_count']}")
    print(f"Part II new tasks: {manifest['part2_new_task_count']}")
    print(f"Total model outputs planned: {manifest['total_generated_outputs']}")
    print(f"Task manifest: {config['paths']['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
