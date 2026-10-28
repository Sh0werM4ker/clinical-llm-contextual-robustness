from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchoring_study.config import load_config, project_path
from anchoring_study.io_utils import read_csv, write_csv_atomic
from anchoring_study.ratings import build_reliability_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Calculate raw agreement and Cohen's kappa.")
    parser.add_argument("--config", default="config/pilot.toml")
    args = parser.parse_args()
    config = load_config(args.config)
    a = read_csv(project_path(config, config["paths"]["rater_a"]))
    b = read_csv(project_path(config, config["paths"]["rater_b"]))
    rows = build_reliability_rows(a, b)
    path = project_path(config, config["paths"]["rater_a"]).with_name(
        "interrater_reliability.csv"
    )
    write_csv_atomic(path, rows, list(rows[0]) if rows else [])
    for row in rows:
        print(
            f"{row['field']}: n={row['n_complete_pairs']} "
            f"agreement={row['percent_agreement']} kappa={row['cohen_kappa']}"
        )
    print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
