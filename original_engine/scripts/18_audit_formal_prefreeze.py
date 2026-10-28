from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "frozen/formal_v1"
MANIFEST = OUTPUT_DIR / "FREEZE_MANIFEST.json"
README = OUTPUT_DIR / "README_CN.md"

FILES_TO_FREEZE = [
    ROOT / "data/cases_formal_locked_v1.csv",
    ROOT / "data/cases_formal_approved_reserve_v1.csv",
    ROOT / "config/formal_local_v1.toml",
    ROOT / "config/formal_model_registry_v1.json",
    ROOT / "src/anchoring_study/prompts.py",
    ROOT / "src/anchoring_study/schemas.py",
    ROOT / "docs/PROTOCOL_CN.md",
    ROOT / "docs/JUDGING_MANUAL_CN.md",
    ROOT / "docs/FORMAL_FREEZE_CHECKLIST_CN.md",
    ROOT / "requirements-lock.txt",
    ROOT / "outputs/formal_local_v1/tasks_part1.csv",
    ROOT / "outputs/formal_local_v1/tasks_part2.csv",
    ROOT / "outputs/formal_local_v1/task_manifest.json",
    ROOT / "outputs/formal_local_v1/case_validation_report.json",
    ROOT / "results/formal_case_lock_v1/lock_qc.json",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def query_ollama_digest(model: str) -> tuple[str | None, str | None]:
    try:
        request = urllib.request.Request(
            "http://127.0.0.1:11434/api/tags",
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.load(response)
        for item in data.get("models", []):
            if item.get("name") == model or item.get("model") == model:
                digest = item.get("digest")
                if digest:
                    return str(digest), None
                return None, f"Ollama /api/tags returned {model} without a digest."
        return None, f"Model not found in Ollama /api/tags: {model}"
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        return None, f"Ollama unavailable: {exc}"


def main() -> int:
    errors: list[str] = []
    for path in FILES_TO_FREEZE:
        if not path.exists():
            errors.append(f"Missing required freeze file: {path.relative_to(ROOT)}")

    if errors:
        raise SystemExit("\n".join(errors))

    locked = read_csv(ROOT / "data/cases_formal_locked_v1.csv")
    reserves = read_csv(ROOT / "data/cases_formal_approved_reserve_v1.csv")
    task_manifest = json.loads(
        (ROOT / "outputs/formal_local_v1/task_manifest.json").read_text(encoding="utf-8")
    )
    lock_qc = json.loads(
        (ROOT / "results/formal_case_lock_v1/lock_qc.json").read_text(encoding="utf-8")
    )
    registry = json.loads(
        (ROOT / "config/formal_model_registry_v1.json").read_text(encoding="utf-8")
    )

    material_checks = {
        "lock_qc_passed": bool(lock_qc.get("passed")),
        "locked_case_count_is_30": len(locked) == 30,
        "reserve_case_count_is_3": len(reserves) == 3,
        "all_locked_rows_marked_locked": all(row.get("case_status") == "locked" for row in locked),
        "all_locked_rows_expert_approved": all(
            row.get("expert_review_status") == "expert_approved" for row in locked
        ),
        "part1_task_count_is_540": task_manifest.get("part1_task_count") == 540,
        "part2_task_count_is_360": task_manifest.get("part2_new_task_count") == 360,
        "total_task_count_is_900": task_manifest.get("total_generated_outputs") == 900,
        "task_manifest_cases_hash_matches": task_manifest.get("cases_sha256")
        == sha256(ROOT / "data/cases_formal_locked_v1.csv"),
        "task_manifest_config_hash_matches": task_manifest.get("config_sha256")
        == sha256(ROOT / "config/formal_local_v1.toml"),
        "no_formal_raw_output_file": not (ROOT / "outputs/formal_local_v1/raw/raw_outputs.jsonl").exists(),
        "no_formal_attempt_log": not (ROOT / "outputs/formal_local_v1/raw/generation_attempts.jsonl").exists(),
    }

    task_file_checks: dict[str, bool] = {}
    for relative, expected_hash in task_manifest.get("task_files", {}).items():
        path = ROOT / relative
        task_file_checks[relative.replace("\\", "/")] = path.exists() and sha256(path) == expected_hash
    material_checks["task_file_hashes_match"] = bool(task_file_checks) and all(task_file_checks.values())

    model_digest_checks: list[dict[str, object]] = []
    for model in registry["models"]:
        current, query_error = query_ollama_digest(model["api_model"])
        expected = model["clean_pilot_digest"]
        model_digest_checks.append(
            {
                "logical_id": model["logical_id"],
                "api_model": model["api_model"],
                "expected_clean_pilot_digest": expected,
                "current_digest": current,
                "match": current == expected if current else False,
                "query_error": query_error,
            }
        )

    model_digests_verified = all(item["match"] for item in model_digest_checks)
    ethics_marker = OUTPUT_DIR / "ETHICS_OR_RESEARCH_CLASSIFICATION_CONFIRMED.txt"
    ethics_confirmed = ethics_marker.exists()
    materials_frozen = all(material_checks.values())
    formal_generation_ready = materials_frozen and model_digests_verified and ethics_confirmed

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_id": "diagnostic_anchoring_local_formal_v1",
        "status": "ready" if formal_generation_ready else "blocked_before_generation",
        "materials_frozen": materials_frozen,
        "formal_generation_ready": formal_generation_ready,
        "material_checks": material_checks,
        "task_file_checks": task_file_checks,
        "model_digest_checks": model_digest_checks,
        "ethics_or_research_classification_confirmation_present": ethics_confirmed,
        "pending_gates": [
            gate
            for gate, passed in [
                ("Start Ollama and re-run this audit so all three model digests can be verified.", model_digests_verified),
                ("Obtain and archive the institution's ethics/research-classification confirmation, then create the marker file.", ethics_confirmed),
            ]
            if not passed
        ],
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "frozen_files": {
            str(path.relative_to(ROOT)).replace("\\", "/"): {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in FILES_TO_FREEZE
        },
        "guardrail": "No formal model generation may begin unless formal_generation_ready is true.",
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    README.write_text(
        "# 正式研究冻结状态\n\n"
        f"- 病例、提示和任务材料冻结检查：{'通过' if materials_frozen else '未通过'}\n"
        f"- 正式模型生成许可：{'可以开始' if formal_generation_ready else '尚不可开始'}\n"
        f"- 本机模型 digest 复核：{'通过' if model_digests_verified else '待完成'}\n"
        f"- 伦理/研究分类书面确认归档标记：{'已存在' if ethics_confirmed else '待用户确认并归档'}\n\n"
        "详细哈希和阻断项见 `FREEZE_MANIFEST.json`。在 `formal_generation_ready=true` 以前，"
        "不得运行正式生成脚本。\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "materials_frozen": materials_frozen,
        "model_digests_verified": model_digests_verified,
        "ethics_confirmed": ethics_confirmed,
        "formal_generation_ready": formal_generation_ready,
        "manifest": str(MANIFEST),
    }, ensure_ascii=False, indent=2))
    return 0 if materials_frozen else 1


if __name__ == "__main__":
    raise SystemExit(main())
