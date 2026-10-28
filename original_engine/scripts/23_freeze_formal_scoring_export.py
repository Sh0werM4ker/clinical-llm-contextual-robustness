from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RATINGS = ROOT / "ratings" / "formal_local_v1"
JUDGE_DIR = ROOT / "outputs" / "web_judge_formal_v1" / "primary"
FROZEN = ROOT / "frozen" / "formal_v1" / "FORMAL_SCORING_EXPORT_FREEZE.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    master_path = RATINGS / "blinding_master.csv"
    rater_a_path = RATINGS / "rater_A.csv"
    rater_b_path = RATINGS / "rater_B.csv"
    failures_path = RATINGS / "generation_failures.csv"
    manifest_path = JUDGE_DIR / "web_primary_judge_manifest.json"
    registry_path = JUDGE_DIR / "FORMAL_PRIMARY_WEB_JUDGE_RUN_REGISTRY.xlsx"

    master = read_csv(master_path)
    rater_a = read_csv(rater_a_path)
    rater_b = read_csv(rater_b_path)
    failures = read_csv(failures_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if len(master) != 884 or len(rater_a) != 884 or len(rater_b) != 884:
        raise RuntimeError("Expected exactly 884 scorable rows in each blinded rating file.")
    if len(failures) != 16:
        raise RuntimeError("Expected exactly 16 terminal generation failures.")
    master_ids = [row["blinded_id"] for row in master]
    if len(set(master_ids)) != 884:
        raise RuntimeError("blinded_id values are not unique.")
    if [row["blinded_id"] for row in rater_a] != [row["blinded_id"] for row in rater_b]:
        raise RuntimeError("Rater A and B row order differs.")
    if rater_a != rater_b:
        raise RuntimeError("Rater A and B source forms are not identical before scoring.")
    forbidden = {"logical_model_id", "anchor_condition", "intervention", "repeat_id", "task_id"}
    leaked = forbidden & set(rater_a[0])
    if leaked:
        raise RuntimeError(f"Blinding leak in rating form columns: {sorted(leaked)}")

    input_files = sorted(JUDGE_DIR.glob("FORMAL_PRIMARY_WEB_JUDGE_BATCH_*.xlsx"))
    if len(input_files) != 89:
        raise RuntimeError(f"Expected 89 judge input workbooks; found {len(input_files)}.")
    if manifest["record_count"] != 884 or manifest["batch_count"] != 89:
        raise RuntimeError("Judge manifest totals are incorrect.")
    judge_ids = [row["judge_item_id"] for row in manifest["records"]]
    input_hashes = [row["input_hash"] for row in manifest["records"]]
    if len(set(judge_ids)) != 884 or len(set(input_hashes)) != 884:
        raise RuntimeError("Judge IDs or input hashes are not unique.")

    key_files = {
        "blinding_master": master_path,
        "rater_a_source": rater_a_path,
        "rater_b_source": rater_b_path,
        "generation_failures": failures_path,
        "judge_manifest": manifest_path,
        "run_registry": registry_path,
    }
    audit = {
        "study_id": "diagnostic_anchoring_local_formal_v1",
        "scoring_export_frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "scorable_record_count": 884,
        "generation_failure_count": 16,
        "primary_web_judge_batch_count": 89,
        "primary_web_judge_record_count": 884,
        "blinding_checks": {
            "unique_blinded_ids": True,
            "rater_source_forms_identical": True,
            "forbidden_metadata_columns_absent": True,
            "unique_judge_item_ids": True,
            "unique_input_hashes": True,
        },
        "key_files": {
            name: {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in key_files.items()
        },
        "judge_workbooks": [
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in input_files
        ],
    }
    if FROZEN.exists():
        existing = json.loads(FROZEN.read_text(encoding="utf-8"))
        if existing["key_files"]["judge_manifest"]["sha256"] != audit["key_files"]["judge_manifest"]["sha256"]:
            raise RuntimeError("Frozen scoring export exists but judge manifest hash changed.")
        print(json.dumps(existing, indent=2, ensure_ascii=False))
        return 0
    FROZEN.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "frozen": True,
        "scorable_records": 884,
        "generation_failures": 16,
        "judge_batches": 89,
        "judge_records": 884,
        "blinding_checks_passed": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
