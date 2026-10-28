from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK_FILES = [
    ROOT / "outputs/formal_local_v1/tasks_part1.csv",
    ROOT / "outputs/formal_local_v1/tasks_part2.csv",
]
RAW = ROOT / "outputs/formal_local_v1/raw/raw_outputs.jsonl"
ATTEMPTS = ROOT / "outputs/formal_local_v1/raw/generation_attempts.jsonl"
LOCK = RAW.with_suffix(RAW.suffix + ".run.lock")
OUT_DIR = ROOT / "results/formal_local_v1"
JSON_OUT = OUT_DIR / "live_status.json"
MD_OUT = OUT_DIR / "LIVE_STATUS.md"


def read_tasks() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in TASK_FILES:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def read_jsonl(path: Path) -> tuple[list[dict], int]:
    rows: list[dict] = []
    malformed = 0
    if not path.exists():
        return rows, malformed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                malformed += 1
    return rows, malformed


def main() -> int:
    tasks = read_tasks()
    raw, malformed_raw = read_jsonl(RAW)
    attempts, malformed_attempts = read_jsonl(ATTEMPTS)
    planned_ids = {row["task_id"] for row in tasks}

    successes_by_task: dict[str, list[dict]] = defaultdict(list)
    terminal_by_task: dict[str, list[dict]] = defaultdict(list)
    for row in raw:
        task_id = str(row.get("task_id", ""))
        if row.get("status") == "success":
            successes_by_task[task_id].append(row)
        if row.get("status") in {"success", "model_refusal", "generation_failure"}:
            terminal_by_task[task_id].append(row)

    successful_ids = set(successes_by_task) & planned_ids
    refusal_ids = {
        task_id for task_id, rows in terminal_by_task.items()
        if task_id in planned_ids and not successes_by_task.get(task_id)
        and any(row.get("status") == "model_refusal" for row in rows)
    }
    failure_ids = {
        task_id for task_id, rows in terminal_by_task.items()
        if task_id in planned_ids and not successes_by_task.get(task_id)
        and any(row.get("status") == "generation_failure" for row in rows)
    }
    complete_ids = successful_ids | refusal_ids | failure_ids

    task_lookup = {row["task_id"]: row for row in tasks}
    success_breakdown: Counter[str] = Counter()
    for task_id in successful_ids:
        task = task_lookup[task_id]
        success_breakdown[f"{task['part']}|{task['logical_model_id']}"] += 1

    duplicate_successes = {
        task_id: len(rows) for task_id, rows in successes_by_task.items() if len(rows) > 1
    }
    attempt_statuses = Counter(str(row.get("status", "unknown")) for row in attempts)
    lock_record = None
    if LOCK.exists():
        try:
            lock_record = json.loads(LOCK.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            lock_record = {"unreadable": True}

    status = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "planned_task_count": len(planned_ids),
        "successful_task_count": len(successful_ids),
        "refusal_task_count": len(refusal_ids),
        "generation_failure_task_count": len(failure_ids),
        "terminal_task_count": len(complete_ids),
        "remaining_task_count": len(planned_ids - complete_ids),
        "percent_successful": round(100 * len(successful_ids) / len(planned_ids), 3),
        "success_breakdown": dict(sorted(success_breakdown.items())),
        "duplicate_success_task_count": len(duplicate_successes),
        "duplicate_successes": duplicate_successes,
        "attempt_status_counts": dict(attempt_statuses),
        "malformed_raw_lines": malformed_raw,
        "malformed_attempt_lines": malformed_attempts,
        "run_lock_present": LOCK.exists(),
        "run_lock_record": lock_record,
        "research_complete": len(successful_ids) == len(planned_ids)
        and not refusal_ids and not failure_ids and not duplicate_successes,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    breakdown_lines = "\n".join(
        f"- `{key}`：{value}条" for key, value in status["success_breakdown"].items()
    ) or "- 尚无成功记录"
    MD_OUT.write_text(
        "# 正式生成实时状态\n\n"
        f"更新时间（UTC）：`{status['updated_at_utc']}`\n\n"
        f"- 计划任务：{status['planned_task_count']}条\n"
        f"- 成功：{status['successful_task_count']}条（{status['percent_successful']}%）\n"
        f"- 拒答：{status['refusal_task_count']}条\n"
        f"- 最终技术失败：{status['generation_failure_task_count']}条\n"
        f"- 剩余：{status['remaining_task_count']}条\n"
        f"- 重复成功任务：{status['duplicate_success_task_count']}个\n"
        f"- 当前运行锁：{'存在，生成中' if status['run_lock_present'] else '不存在'}\n\n"
        "## 成功记录分布\n\n"
        f"{breakdown_lines}\n\n"
        "本文件仅用于查看进度；原始JSONL和预先生成的任务表才是审计依据。\n",
        encoding="utf-8",
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
