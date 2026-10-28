from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchoring_study.cases import validate_cases
from anchoring_study.config import load_config, project_path
from anchoring_study.io_utils import utc_now


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate standardized case definitions.")
    parser.add_argument("--config", default="config/pilot.toml")
    args = parser.parse_args()
    config = load_config(args.config)
    case_path = project_path(config, config["paths"]["cases"])
    rows, errors, warnings = validate_cases(case_path)
    report = {
        "timestamp_utc": utc_now(),
        "case_file": str(case_path),
        "row_count": len(rows),
        "locked_count": sum(row.get("case_status") == "locked" for row in rows),
        "errors": errors,
        "warnings": warnings,
        "passed": not errors,
    }
    report_path = project_path(config, config["paths"]["manifest"]).with_name(
        "case_validation_report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print(f"VALIDATION FAILED: {len(errors)} error(s).")
        return 1
    print(f"VALIDATION PASSED: {len(rows)} case(s), {report['locked_count']} locked.")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
