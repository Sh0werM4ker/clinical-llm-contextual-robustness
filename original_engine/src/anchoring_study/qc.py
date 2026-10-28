from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .config import project_path
from .io_utils import read_csv, read_jsonl, utc_now
from .schemas import ClinicalResponse


def build_qc_report(config: dict[str, Any]) -> dict[str, Any]:
    task_paths = [
        project_path(config, config["paths"]["tasks_part1"]),
        project_path(config, config["paths"]["tasks_part2"]),
    ]
    tasks: list[dict[str, str]] = []
    for path in task_paths:
        if path.exists():
            tasks.extend(read_csv(path))
    planned = {task["task_id"]: task for task in tasks}
    raw_path = project_path(config, config["paths"]["raw_outputs"])
    raw_records = read_jsonl(raw_path)
    success_records = [record for record in raw_records if record.get("status") == "success"]
    refusal_records = [record for record in raw_records if record.get("status") == "model_refusal"]

    successes_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in success_records:
        successes_by_task[str(record.get("task_id"))].append(record)
    latest = {task_id: records[-1] for task_id, records in successes_by_task.items()}

    schema_problems: list[dict[str, str]] = []
    prompt_hash_mismatches: list[str] = []
    unplanned_successes: list[str] = []
    for task_id, record in latest.items():
        try:
            ClinicalResponse.model_validate(record.get("parsed_output"))
        except ValidationError as error:
            schema_problems.append({"task_id": task_id, "error": str(error)[:1000]})
        if task_id not in planned:
            unplanned_successes.append(task_id)
        elif record.get("prompt_hash") != planned[task_id].get("prompt_hash"):
            prompt_hash_mismatches.append(task_id)

    requested_to_returned: dict[str, set[str]] = defaultdict(set)
    requested_to_digests: dict[str, set[str]] = defaultdict(set)
    for record in latest.values():
        requested_to_returned[str(record.get("api_model_requested"))].add(
            str(record.get("api_model_returned"))
        )
        if record.get("model_digest"):
            requested_to_digests[str(record.get("api_model_requested"))].add(
                str(record.get("model_digest"))
            )

    missing = sorted(set(planned) - set(latest))
    duplicate_successes = {
        task_id: len(records)
        for task_id, records in successes_by_task.items()
        if len(records) > 1
    }
    mock_task_ids = sorted(
        task_id for task_id, record in latest.items() if bool(record.get("is_mock"))
    )
    derived_review_only_task_ids = sorted(
        task_id
        for task_id, record in latest.items()
        if bool((record.get("derived_selection") or {}).get("pilot_concurrency_incident"))
    )
    group_counts = Counter(
        (
            planned[task_id]["part"],
            planned[task_id]["anchor_condition"],
            planned[task_id]["intervention"],
            planned[task_id]["logical_model_id"],
        )
        for task_id in latest
        if task_id in planned
    )

    hard_failures = bool(
        missing
        or schema_problems
        or prompt_hash_mismatches
        or unplanned_successes
        or duplicate_successes
        or mock_task_ids
        or refusal_records
        or derived_review_only_task_ids
    )
    pilot_review_ready = not bool(
        missing
        or schema_problems
        or prompt_hash_mismatches
        or unplanned_successes
        or mock_task_ids
        or refusal_records
    )
    return {
        "timestamp_utc": utc_now(),
        "study_id": config["study"]["study_id"],
        "planned_task_count": len(planned),
        "successful_task_count": len(latest),
        "missing_success_count": len(missing),
        "missing_task_ids": missing,
        "duplicate_successes": duplicate_successes,
        "schema_problem_count": len(schema_problems),
        "schema_problems": schema_problems,
        "prompt_hash_mismatches": prompt_hash_mismatches,
        "unplanned_successes": unplanned_successes,
        "mock_task_count": len(mock_task_ids),
        "mock_task_ids": mock_task_ids,
        "derived_review_only_count": len(derived_review_only_task_ids),
        "derived_review_only_task_ids": derived_review_only_task_ids,
        "model_refusal_count": len(refusal_records),
        "model_refusal_task_ids": sorted(
            str(record.get("task_id")) for record in refusal_records
        ),
        "returned_model_versions": {
            key: sorted(value) for key, value in requested_to_returned.items()
        },
        "local_model_digests": {
            key: sorted(value) for key, value in requested_to_digests.items()
        },
        "group_counts": [
            {
                "part": key[0],
                "anchor_condition": key[1],
                "intervention": key[2],
                "logical_model_id": key[3],
                "n": count,
            }
            for key, count in sorted(group_counts.items())
        ],
        "research_ready": not hard_failures,
        "pilot_review_ready": pilot_review_ready,
        "note": (
            "Mock records, duplicate successes, and concurrency-incident-derived records are "
            "deliberately treated as hard failures for research readiness."
        ),
    }


def write_qc_report(config: dict[str, Any], report: dict[str, Any]) -> Path:
    results_dir = project_path(config, config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "qc_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
