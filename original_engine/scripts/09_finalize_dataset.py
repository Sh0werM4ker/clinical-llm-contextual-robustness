from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchoring_study.config import load_config, project_path
from anchoring_study.io_utils import write_csv_atomic
from anchoring_study.ratings import finalize_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate adjudication and create analysis data.")
    parser.add_argument("--config", default="config/pilot.toml")
    args = parser.parse_args()
    config = load_config(args.config)
    rows, errors = finalize_dataset(config)
    if errors:
        for error in errors[:50]:
            print(f"ERROR: {error}")
        print(f"FINALIZATION BLOCKED: {len(errors)} invalid or blank final value(s).")
        return 1
    path = project_path(config, config["paths"]["final_dataset"])
    write_csv_atomic(path, rows, list(rows[0]) if rows else [])
    print(f"FINALIZATION PASSED: {len(rows)} rows written to {path}")
    if any(row.get("is_mock") == "1" for row in rows):
        print("WARNING: dataset contains MOCK data and must not be used as research results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

