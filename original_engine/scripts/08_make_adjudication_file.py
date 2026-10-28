from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchoring_study.config import load_config
from anchoring_study.ratings import make_adjudication_files


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare third-clinician adjudication file.")
    parser.add_argument("--config", default="config/pilot.toml")
    args = parser.parse_args()
    counts = make_adjudication_files(load_config(args.config))
    print(json.dumps(counts, indent=2))
    print(
        "A third clinician must fill every applicable blank *_final cell before finalization; "
        "non-applicable anchor fields remain blank."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
