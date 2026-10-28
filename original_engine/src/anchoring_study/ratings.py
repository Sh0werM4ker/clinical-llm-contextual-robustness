from __future__ import annotations

import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .config import project_path
from .io_utils import latest_success_by_task, read_csv, read_jsonl, write_csv_atomic


RATING_FIELDS = [
    "top1_correct",
    "gold_diagnosis_rank",
    "incorrect_anchor_rank",
    "anchor_adoption",
    "evidence_override",
    "triage_level_scored",
    "mandatory_action_complete",
    "unsafe_management",
    "over_conservatism",
    "rater_comment",
]

RATER_CONTEXT_FIELDS = [
    "blinded_id",
    "case_id",
    "domain",
    "presented_prior_label",
    "objective_case",
    "gold_diagnosis",
    "acceptable_synonyms",
    "incorrect_anchor_for_scoring",
    "key_red_flags",
    "reference_triage",
    "mandatory_actions",
    "acceptable_actions",
    "unsafe_actions",
    "model_response_json",
]


def export_blinded_rating_files(config: dict[str, Any]) -> dict[str, int]:
    task_paths = [
        project_path(config, config["paths"]["tasks_part1"]),
        project_path(config, config["paths"]["tasks_part2"]),
    ]
    tasks: list[dict[str, str]] = []
    for path in task_paths:
        tasks.extend(read_csv(path))
    task_map = {task["task_id"]: task for task in tasks}
    latest = latest_success_by_task(
        read_jsonl(project_path(config, config["paths"]["raw_outputs"]))
    )
    eligible_ids = sorted(set(task_map) & set(latest))
    rng = random.Random(int(config["study"]["seed"]) + 701)
    rng.shuffle(eligible_ids)

    master_rows: list[dict[str, Any]] = []
    rater_rows: list[dict[str, Any]] = []
    import json

    for sequence, task_id in enumerate(eligible_ids, start=1):
        blinded_id = f"B{sequence:05d}"
        task = task_map[task_id]
        output = latest[task_id]
        wrong_label_case = task["anchor_condition"] == "incorrect_label"
        master_rows.append(
            {
                "blinded_id": blinded_id,
                "task_id": task_id,
                "part": task["part"],
                "case_id": task["case_id"],
                "logical_model_id": task["logical_model_id"],
                "api_model_requested": task["api_model"],
                "api_model_returned": output.get("api_model_returned", ""),
                "anchor_condition": task["anchor_condition"],
                "intervention": task["intervention"],
                "repeat_id": task["repeat_id"],
                "prompt_hash": task["prompt_hash"],
                "is_mock": int(bool(output.get("is_mock"))),
            }
        )
        row = {
            "blinded_id": blinded_id,
            "case_id": task["case_id"],
            "domain": task["domain"],
            "presented_prior_label": task["presented_label"] or "[no diagnostic label presented]",
            "objective_case": task["objective_case"],
            "gold_diagnosis": task["gold_diagnosis"],
            "acceptable_synonyms": task["acceptable_synonyms"],
            "incorrect_anchor_for_scoring": task["incorrect_anchor"] if wrong_label_case else "",
            "key_red_flags": task["key_red_flags"],
            "reference_triage": task["reference_triage"],
            "mandatory_actions": task["mandatory_actions"],
            "acceptable_actions": task["acceptable_actions"],
            "unsafe_actions": task["unsafe_actions"],
            "model_response_json": json.dumps(
                output["parsed_output"], ensure_ascii=False, indent=2
            ),
        }
        row.update({field: "" for field in RATING_FIELDS})
        rater_rows.append(row)

    master_path = project_path(config, config["paths"]["blinding_master"])
    master_fields = list(master_rows[0]) if master_rows else []
    write_csv_atomic(master_path, master_rows, master_fields)
    rater_fields = RATER_CONTEXT_FIELDS + RATING_FIELDS
    write_csv_atomic(project_path(config, config["paths"]["rater_a"]), rater_rows, rater_fields)
    write_csv_atomic(project_path(config, config["paths"]["rater_b"]), rater_rows, rater_fields)
    return {"eligible_outputs": len(eligible_ids), "rater_rows": len(rater_rows)}


def cohen_kappa(values_a: Iterable[str], values_b: Iterable[str]) -> tuple[int, float, float]:
    pairs = [(str(a), str(b)) for a, b in zip(values_a, values_b) if str(a) and str(b)]
    n = len(pairs)
    if n == 0:
        return 0, math.nan, math.nan
    agreement = sum(a == b for a, b in pairs) / n
    counts_a = Counter(a for a, _ in pairs)
    counts_b = Counter(b for _, b in pairs)
    categories = set(counts_a) | set(counts_b)
    expected = sum((counts_a[c] / n) * (counts_b[c] / n) for c in categories)
    if math.isclose(expected, 1.0):
        kappa = 1.0 if math.isclose(agreement, 1.0) else math.nan
    else:
        kappa = (agreement - expected) / (1 - expected)
    return n, agreement, kappa


def build_reliability_rows(
    rater_a: list[dict[str, str]], rater_b: list[dict[str, str]]
) -> list[dict[str, Any]]:
    map_b = {row["blinded_id"]: row for row in rater_b}
    common = [row for row in rater_a if row["blinded_id"] in map_b]
    rows: list[dict[str, Any]] = []
    for field in RATING_FIELDS:
        if field == "rater_comment":
            continue
        values_a = [row[field].strip() for row in common]
        values_b = [map_b[row["blinded_id"]][field].strip() for row in common]
        n, agreement, kappa = cohen_kappa(values_a, values_b)
        rows.append(
            {
                "field": field,
                "n_complete_pairs": n,
                "percent_agreement": "" if math.isnan(agreement) else agreement,
                "cohen_kappa": "" if math.isnan(kappa) else kappa,
            }
        )
    return rows


def make_adjudication_files(config: dict[str, Any]) -> dict[str, int]:
    rater_a = read_csv(project_path(config, config["paths"]["rater_a"]))
    rater_b = read_csv(project_path(config, config["paths"]["rater_b"]))
    map_b = {row["blinded_id"]: row for row in rater_b}
    rows: list[dict[str, Any]] = []
    requiring: list[dict[str, Any]] = []
    for row_a in rater_a:
        row_b = map_b.get(row_a["blinded_id"])
        if row_b is None:
            raise ValueError(f"Missing rater B row for {row_a['blinded_id']}")
        out = {field: row_a[field] for field in RATER_CONTEXT_FIELDS}
        disagreements: list[str] = []
        for field in RATING_FIELDS:
            if field == "rater_comment":
                out["rater_A_comment"] = row_a[field]
                out["rater_B_comment"] = row_b[field]
                out["adjudicator_comment"] = ""
                continue
            a = row_a[field].strip()
            b = row_b[field].strip()
            non_applicable = (
                not row_a.get("incorrect_anchor_for_scoring", "").strip()
                and field in {"incorrect_anchor_rank", "anchor_adoption"}
            )
            out[f"{field}_rater_A"] = a
            out[f"{field}_rater_B"] = b
            if non_applicable and not a and not b:
                out[f"{field}_final"] = ""
                continue
            out[f"{field}_final"] = a if a and a == b else ""
            if not a or not b or a != b:
                disagreements.append(field)
        out["fields_requiring_adjudication"] = "|".join(disagreements)
        rows.append(out)
        if disagreements:
            requiring.append(out)

    fieldnames = list(rows[0]) if rows else []
    adjudication_path = project_path(config, config["paths"]["adjudication"])
    write_csv_atomic(adjudication_path, rows, fieldnames)
    requiring_path = adjudication_path.with_name("records_requiring_adjudication.csv")
    write_csv_atomic(requiring_path, requiring, fieldnames)
    return {"total": len(rows), "requiring_adjudication": len(requiring)}


def validate_final_values(row: dict[str, str], blinded_id: str) -> list[str]:
    errors: list[str] = []
    binary = {
        "top1_correct",
        "evidence_override",
        "mandatory_action_complete",
        "unsafe_management",
        "over_conservatism",
    }
    ranks = {"gold_diagnosis_rank", "incorrect_anchor_rank"}
    for field in binary:
        if row.get(f"{field}_final", "") not in {"0", "1"}:
            errors.append(f"{blinded_id}: {field}_final must be 0 or 1.")
    for field in ranks:
        value = row.get(f"{field}_final", "")
        if field == "incorrect_anchor_rank" and value == "":
            continue
        if value not in {"1", "2", "3", "4", "5", "6"}:
            errors.append(f"{blinded_id}: {field}_final must be 1-6 (or blank when not applicable).")
    adoption = row.get("anchor_adoption_final", "")
    if adoption not in {"", "0", "1", "2"}:
        errors.append(f"{blinded_id}: anchor_adoption_final must be 0, 1, 2, or blank.")
    triage = row.get("triage_level_scored_final", "")
    if triage not in {"emergency", "same_day", "urgent_outpatient", "routine"}:
        errors.append(f"{blinded_id}: triage_level_scored_final is invalid.")
    return errors


def finalize_dataset(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    adjudication = read_csv(project_path(config, config["paths"]["adjudication"]))
    master = read_csv(project_path(config, config["paths"]["blinding_master"]))
    master_map = {row["blinded_id"]: row for row in master}
    errors: list[str] = []
    final_rows: list[dict[str, Any]] = []
    for row in adjudication:
        blinded_id = row["blinded_id"]
        errors.extend(validate_final_values(row, blinded_id))
        if blinded_id not in master_map:
            errors.append(f"{blinded_id}: missing from blinding master.")
            continue
        metadata = master_map[blinded_id]
        final = dict(metadata)
        final.update({field: row[field] for field in RATER_CONTEXT_FIELDS if field in row})
        for field in RATING_FIELDS:
            if field == "rater_comment":
                final["adjudicator_comment"] = row.get("adjudicator_comment", "")
            else:
                final[field] = row.get(f"{field}_final", "")
        final_rows.append(final)
    return final_rows, errors
