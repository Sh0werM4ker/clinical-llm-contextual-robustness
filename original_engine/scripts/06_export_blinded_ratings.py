from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchoring_study.config import load_config
from anchoring_study.ratings import export_blinded_rating_files


def main() -> int:
    parser = argparse.ArgumentParser(description="Create blinded CSVs for two independent raters.")
    parser.add_argument("--config", default="config/pilot.toml")
    parser.add_argument("--raw-outputs", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.raw_outputs:
        config["paths"]["raw_outputs"] = args.raw_outputs
    if args.output_dir:
        output_dir = args.output_dir.rstrip("/\\")
        config["paths"]["blinding_master"] = f"{output_dir}/blinding_master.csv"
        config["paths"]["rater_a"] = f"{output_dir}/rater_A.csv"
        config["paths"]["rater_b"] = f"{output_dir}/rater_B.csv"
    counts = export_blinded_rating_files(config)
    print(json.dumps(counts, indent=2))
    print("Keep ratings/blinding_master.csv away from raters.")
    print("Blinding hides model and intervention, but the presented clinical label remains visible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
