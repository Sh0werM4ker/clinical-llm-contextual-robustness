from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .cases import validate_cases
from .config import enabled_models, project_path
from .io_utils import sha256_file, sha256_text, utc_now, write_csv_atomic
from .prompts import PROMPT_ONLY_JSON_INSTRUCTION, build_prompts


TASK_FIELDS = [
    "task_id",
    "part",
    "run_order",
    "case_id",
    "domain",
    "logical_model_id",
    "provider",
    "api_model",
    "api_base_url",
    "reasoning_effort",
    "temperature",
    "top_p",
    "num_ctx",
    "generation_seed",
    "think",
    "keep_alive",
    "ollama_format_mode",
    "max_output_tokens",
    "anchor_condition",
    "intervention",
    "repeat_id",
    "presented_label",
    "gold_diagnosis",
    "acceptable_synonyms",
    "incorrect_anchor",
    "objective_case",
    "key_red_flags",
    "reference_triage",
    "mandatory_actions",
    "acceptable_actions",
    "unsafe_actions",
    "system_prompt",
    "user_prompt",
    "prompt_hash",
    "schema_version",
]


def _make_task(
    config: dict[str, Any],
    case: dict[str, str],
    model: dict[str, Any],
    *,
    part: str,
    condition: str,
    intervention: str,
    repeat_id: int,
) -> dict[str, Any]:
    system_prompt, user_prompt, prompt_hash, presented_label = build_prompts(
        case, condition, intervention
    )
    ollama_format_mode = str(model.get("ollama_format_mode", "schema"))
    if str(model["provider"]).lower() == "ollama" and ollama_format_mode == "prompt_only":
        system_prompt = (
            f"{system_prompt}\n\nRequired JSON output contract:\n"
            f"{PROMPT_ONLY_JSON_INSTRUCTION}"
        )
        prompt_hash = sha256_text(system_prompt + "\n---USER---\n" + user_prompt)
    part_code = "P1" if part == "part1" else "P2"
    task_id = "__".join(
        [
            part_code,
            case["case_id"],
            model["logical_id"],
            condition,
            intervention,
            f"R{repeat_id}",
        ]
    )
    paired_seed_material = "|".join(
        [
            str(config["study"]["seed"]),
            case["case_id"],
            model["logical_id"],
            str(repeat_id),
        ]
    )
    generation_seed = int.from_bytes(
        paired_seed_material.encode("utf-8"), byteorder="little", signed=False
    ) % (2**31 - 1)
    generation = config["generation"]
    return {
        "task_id": task_id,
        "part": part,
        "run_order": 0,
        "case_id": case["case_id"],
        "domain": case["domain"],
        "logical_model_id": model["logical_id"],
        "provider": model["provider"],
        "api_model": model["api_model"],
        "api_base_url": model.get("api_base_url", ""),
        "reasoning_effort": model.get("reasoning_effort", ""),
        "temperature": model.get("temperature", generation.get("temperature", "")),
        "top_p": model.get("top_p", generation.get("top_p", "")),
        "num_ctx": model.get("num_ctx", generation.get("num_ctx", "")),
        "generation_seed": generation_seed,
        "think": model.get("think", False),
        "keep_alive": model.get("keep_alive", generation.get("keep_alive", "5m")),
        "ollama_format_mode": ollama_format_mode,
        "max_output_tokens": model.get(
            "max_output_tokens", generation.get("max_output_tokens", 900)
        ),
        "anchor_condition": condition,
        "intervention": intervention,
        "repeat_id": repeat_id,
        "presented_label": presented_label,
        "gold_diagnosis": case["gold_diagnosis"],
        "acceptable_synonyms": case["acceptable_synonyms"],
        "incorrect_anchor": case["incorrect_anchor"],
        "objective_case": case["objective_case"],
        "key_red_flags": case["key_red_flags"],
        "reference_triage": case["reference_triage"],
        "mandatory_actions": case["mandatory_actions"],
        "acceptable_actions": case["acceptable_actions"],
        "unsafe_actions": case["unsafe_actions"],
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "prompt_hash": prompt_hash,
        "schema_version": config["study"]["schema_version"],
    }


def build_tasks(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases_path = project_path(config, config["paths"]["cases"])
    cases, errors, _warnings = validate_cases(cases_path)
    if errors:
        raise ValueError("Case validation failed:\n- " + "\n- ".join(errors))
    locked_cases = [case for case in cases if case["case_status"] == "locked"]
    if not locked_cases:
        raise ValueError("No locked cases are available for task generation.")

    part1: list[dict[str, Any]] = []
    part2: list[dict[str, Any]] = []
    design = config["design"]
    for case in locked_cases:
        for model in enabled_models(config):
            for condition in design["part1_conditions"]:
                for repeat_id in range(1, int(design["part1_repeats"]) + 1):
                    part1.append(
                        _make_task(
                            config,
                            case,
                            model,
                            part="part1",
                            condition=condition,
                            intervention="baseline",
                            repeat_id=repeat_id,
                        )
                    )
            for intervention in design["part2_interventions"]:
                for repeat_id in range(1, int(design["part2_repeats"]) + 1):
                    part2.append(
                        _make_task(
                            config,
                            case,
                            model,
                            part="part2",
                            condition="incorrect_label",
                            intervention=intervention,
                            repeat_id=repeat_id,
                        )
                    )

    rng = random.Random(int(config["study"]["seed"]))
    rng.shuffle(part1)
    rng.shuffle(part2)
    for run_order, task in enumerate(part1, start=1):
        task["run_order"] = run_order
    for run_order, task in enumerate(part2, start=1):
        task["run_order"] = run_order
    return part1, part2


def write_tasks_and_manifest(
    config: dict[str, Any], part1: list[dict[str, Any]], part2: list[dict[str, Any]]
) -> dict[str, Any]:
    part1_path = project_path(config, config["paths"]["tasks_part1"])
    part2_path = project_path(config, config["paths"]["tasks_part2"])
    write_csv_atomic(part1_path, part1, TASK_FIELDS)
    write_csv_atomic(part2_path, part2, TASK_FIELDS)

    case_counts = Counter(task["case_id"] for task in part1)
    manifest = {
        "study_id": config["study"]["study_id"],
        "phase": config["study"]["phase"],
        "created_at_utc": utc_now(),
        "seed": config["study"]["seed"],
        "locked_case_count": len(case_counts),
        "model_count": len(enabled_models(config)),
        "part1_task_count": len(part1),
        "part2_new_task_count": len(part2),
        "total_generated_outputs": len(part1) + len(part2),
        "task_files": {
            str(part1_path.relative_to(Path(config["_project_root"]))): sha256_file(part1_path),
            str(part2_path.relative_to(Path(config["_project_root"]))): sha256_file(part2_path),
        },
        "cases_sha256": sha256_file(project_path(config, config["paths"]["cases"])),
        "config_sha256": sha256_file(Path(config["_config_path"])),
    }
    manifest_path = project_path(config, config["paths"]["manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
