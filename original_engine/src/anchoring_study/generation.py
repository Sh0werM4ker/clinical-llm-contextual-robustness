from __future__ import annotations

import json
import logging
import os
import random
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import project_path
from .io_utils import (
    append_jsonl,
    latest_terminal_by_task,
    read_csv,
    read_jsonl,
    sha256_text,
    utc_now,
)
from .schemas import ClinicalResponse


LIVE_CONFIRMATION = "I_UNDERSTAND_API_COSTS"
LOCAL_CONFIRMATION = "I_UNDERSTAND_LOCAL_MODEL_RUN"


class ModelRefusalError(RuntimeError):
    """A substantive model refusal; it must be recorded but not retried."""

    def __init__(self, message: str, metadata: dict[str, Any]):
        super().__init__(message)
        self.metadata = metadata


class GenerationRunLock:
    """Atomic cross-process guard for one append-only raw output ledger."""

    def __init__(self, path: Path):
        self.path = path
        self.file_descriptor: int | None = None

    def __enter__(self) -> "GenerationRunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.file_descriptor = os.open(
                self.path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError as error:
            owner = self.path.read_text(encoding="utf-8", errors="replace")[:1000]
            raise RuntimeError(
                "Another generation process is already using this raw-output ledger. "
                f"Lock: {self.path}. Owner record: {owner}"
            ) from error
        owner_record = json.dumps(
            {
                "pid": os.getpid(),
                "created_at_utc": utc_now(),
                "lock_path": str(self.path),
            },
            ensure_ascii=False,
        ).encode("utf-8")
        os.write(self.file_descriptor, owner_record)
        os.fsync(self.file_descriptor)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.file_descriptor is not None:
            os.close(self.file_descriptor)
            self.file_descriptor = None
        self.path.unlink(missing_ok=True)


def configure_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("anchoring_generation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _mock_response(task: dict[str, str]) -> ClinicalResponse:
    marker = int(task["prompt_hash"][:8], 16)
    anchored = task["anchor_condition"] == "incorrect_label" and marker % 4 == 0
    gold = task["gold_diagnosis"]
    wrong = task["incorrect_anchor"]
    if anchored:
        most_likely = wrong
        differential = [wrong, gold, "another plausible diagnosis"]
        triage = "routine" if task["reference_triage"] != "routine" else "routine"
        steps = ["symptomatic treatment", "routine follow-up"]
        rationale = "Synthetic mock response that intentionally follows the prior label."
    else:
        most_likely = gold
        differential = [gold, wrong, "another plausible diagnosis"]
        triage = task["reference_triage"]
        steps = [item.strip() for item in task["mandatory_actions"].split("|") if item.strip()]
        if len(steps) < 2:
            steps.append("appropriate clinical reassessment")
        rationale = "Synthetic mock response based on the listed objective red flags."
    return ClinicalResponse(
        most_likely_diagnosis=most_likely,
        ranked_differential_diagnoses=differential,
        triage_level=triage,
        recommended_next_steps=steps[:8],
        brief_rationale=rationale,
    )


def _verify_task_prompt(task: dict[str, str]) -> None:
    calculated = sha256_text(
        task["system_prompt"] + "\n---USER---\n" + task["user_prompt"]
    )
    if calculated != task["prompt_hash"]:
        raise ValueError(
            f"Prompt hash mismatch for {task['task_id']}; rebuild tasks instead of editing them."
        )


def _usage_to_dict(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return {"raw": str(usage)}


def _extract_refusal(response: Any) -> str | None:
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) == "refusal":
                return str(getattr(content, "refusal", "Model refusal"))
    return None


def _call_openai(
    client: Any, task: dict[str, str], generation: dict[str, Any]
) -> tuple[ClinicalResponse, dict[str, Any]]:
    request: dict[str, Any] = {
        "model": task["api_model"],
        "input": [
            {"role": "system", "content": task["system_prompt"]},
            {"role": "user", "content": task["user_prompt"]},
        ],
        "text_format": ClinicalResponse,
        "max_output_tokens": int(generation["max_output_tokens"]),
        "store": bool(generation.get("store_on_openai", False)),
    }
    if task.get("reasoning_effort"):
        request["reasoning"] = {"effort": task["reasoning_effort"]}
    response = client.responses.parse(**request)
    parsed = response.output_parsed
    metadata = {
        "response_id": getattr(response, "id", None),
        "request_id": getattr(response, "_request_id", None),
        "api_model_returned": getattr(response, "model", None),
        "usage": _usage_to_dict(getattr(response, "usage", None)),
        "raw_output_text": getattr(response, "output_text", ""),
        "model_digest": None,
        "model_details": None,
        "generation_options": {
            "reasoning_effort": task.get("reasoning_effort", ""),
            "max_output_tokens": int(generation["max_output_tokens"]),
        },
    }
    if parsed is None:
        refusal = _extract_refusal(response)
        if refusal:
            metadata["raw_output_text"] = refusal
            raise ModelRefusalError(refusal, metadata)
        raise RuntimeError(
            "The API returned no parsed structured output. "
            f"Raw text: {getattr(response, 'output_text', '')[:500]}"
        )
    return parsed, metadata


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _ollama_endpoint(base_url: str, endpoint: str) -> str:
    base = base_url.rstrip("/")
    if not base.endswith("/api"):
        base += "/api"
    return f"{base}/{endpoint.lstrip('/')}"


def _ollama_request(
    base_url: str,
    endpoint: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout_seconds: float = 600.0,
) -> dict[str, Any]:
    url = _ollama_endpoint(base_url, endpoint)
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        method="GET" if payload is None else "POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {error.code} from {url}: {body[:1000]}") from error
    except URLError as error:
        raise RuntimeError(
            "Cannot reach the local Ollama service. Start Ollama and verify "
            f"{url}. Details: {error.reason}"
        ) from error


def get_ollama_model_info(
    base_url: str, model_name: str, *, timeout_seconds: float = 30.0
) -> dict[str, Any]:
    registry = _ollama_request(
        base_url, "tags", timeout_seconds=timeout_seconds
    ).get("models", [])
    requested = model_name.casefold()
    for model in registry:
        candidates = {
            str(model.get("name", "")).casefold(),
            str(model.get("model", "")).casefold(),
        }
        if requested in candidates:
            return model
    installed = sorted(
        str(item.get("name") or item.get("model")) for item in registry if item
    )
    raise RuntimeError(
        f"Ollama model {model_name!r} is not installed. Installed models: "
        f"{installed or 'none'}. Run: ollama pull {model_name}"
    )


def _ollama_seed_for_attempt(
    task: dict[str, str], generation: dict[str, Any], attempt: int
) -> int:
    """Return a deterministic seed, changing only after a technical failure."""
    if attempt < 1:
        raise ValueError("attempt must be at least 1")
    base_seed = int(task["generation_seed"])
    stride = int(generation.get("retry_seed_stride", 1))
    return (base_seed + ((attempt - 1) * stride)) % (2**31 - 1)


def build_ollama_payload(
    task: dict[str, str], generation: dict[str, Any], *, attempt: int = 1
) -> dict[str, Any]:
    max_output_tokens = int(
        task.get("max_output_tokens") or generation["max_output_tokens"]
    )
    format_mode = str(task.get("ollama_format_mode") or "schema").lower()
    options: dict[str, Any] = {
        "temperature": float(task["temperature"]),
        "top_p": float(task["top_p"]),
        "num_ctx": int(task["num_ctx"]),
        "num_predict": max_output_tokens,
        "seed": _ollama_seed_for_attempt(task, generation, attempt),
    }
    payload: dict[str, Any] = {
        "model": task["api_model"],
        "messages": [
            {"role": "system", "content": task["system_prompt"]},
            {"role": "user", "content": task["user_prompt"]},
        ],
        "stream": False,
        "options": options,
        "think": _as_bool(task.get("think", False)),
        "keep_alive": task.get("keep_alive") or "5m",
    }
    if format_mode == "schema":
        payload["format"] = ClinicalResponse.model_json_schema()
    elif format_mode == "json":
        payload["format"] = "json"
    elif format_mode != "prompt_only":
        raise ValueError(f"Unsupported Ollama format mode: {format_mode!r}")
    return payload


def _call_ollama(
    task: dict[str, str],
    generation: dict[str, Any],
    model_info: dict[str, Any],
    *,
    attempt: int = 1,
) -> tuple[ClinicalResponse, dict[str, Any]]:
    payload = build_ollama_payload(task, generation, attempt=attempt)
    response = _ollama_request(
        task["api_base_url"],
        "chat",
        payload=payload,
        timeout_seconds=float(generation.get("timeout_seconds", 600.0)),
    )
    message = response.get("message") or {}
    raw_text = str(message.get("content") or "")
    if not raw_text.strip():
        raise RuntimeError(
            "Ollama returned an empty message. "
            f"done_reason={response.get('done_reason')!r}"
        )
    parsed = ClinicalResponse.model_validate_json(raw_text)
    usage = {
        "input_tokens": response.get("prompt_eval_count"),
        "output_tokens": response.get("eval_count"),
        "total_tokens": (
            int(response.get("prompt_eval_count") or 0)
            + int(response.get("eval_count") or 0)
        ),
        "prompt_eval_duration_ns": response.get("prompt_eval_duration"),
        "eval_duration_ns": response.get("eval_duration"),
        "load_duration_ns": response.get("load_duration"),
        "total_duration_ns": response.get("total_duration"),
    }
    metadata = {
        "response_id": None,
        "request_id": None,
        "api_model_returned": response.get("model"),
        "usage": usage,
        "raw_output_text": raw_text,
        "model_digest": model_info.get("digest"),
        "model_details": model_info.get("details"),
        "generation_options": payload["options"]
        | {
            "think": payload["think"],
            "keep_alive": payload["keep_alive"],
            "ollama_format_mode": task.get("ollama_format_mode") or "schema",
        },
        "created_at": response.get("created_at"),
        "done_reason": response.get("done_reason"),
    }
    return parsed, metadata


def _run_tasks_unlocked(
    config: dict[str, Any],
    tasks_path: Path,
    *,
    model_filter: str | None = None,
    limit: int | None = None,
    mock: bool = False,
    live_confirmation: str | None = None,
    local_confirmation: str | None = None,
) -> dict[str, int]:
    if mock and config["study"].get("phase") != "software_test":
        raise ValueError(
            "Mock generation is allowed only with config/software_test.toml so test records "
            "cannot contaminate pilot or formal data."
        )
    tasks = read_csv(tasks_path)
    task_ids = [task["task_id"] for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"Duplicate task IDs found in {tasks_path}.")
    for task in tasks:
        _verify_task_prompt(task)

    if model_filter:
        tasks = [task for task in tasks if task["logical_model_id"] == model_filter]
    providers = {task["provider"].lower() for task in tasks}
    if not mock and "openai" in providers:
        if live_confirmation != LIVE_CONFIRMATION:
            raise ValueError(
                "Paid OpenAI API calls are blocked. Re-run with "
                f"--confirm-live-run {LIVE_CONFIRMATION} after checking billing and task counts."
            )
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is not set. Create a local .env file first.")
    if not mock and "ollama" in providers and local_confirmation != LOCAL_CONFIRMATION:
        raise ValueError(
            "Local model generation is blocked to prevent an accidental long run. Re-run with "
            f"--confirm-local-run {LOCAL_CONFIRMATION} after checking model tags and task counts."
        )
    unsupported = providers - {"openai", "ollama"}
    if unsupported and not mock:
        raise ValueError(f"Unsupported providers in task file: {sorted(unsupported)}")
    raw_path = project_path(config, config["paths"]["raw_outputs"])
    attempts_path = project_path(config, config["paths"]["attempt_log"])
    existing_records = read_jsonl(raw_path)
    completed = latest_terminal_by_task(existing_records)
    all_pending = [task for task in tasks if task["task_id"] not in completed]
    already_complete = len(tasks) - len(all_pending)
    pending = all_pending
    if limit is not None:
        pending = pending[:limit]
    limited_out = len(all_pending) - len(pending)

    logger = configure_logging(project_path(config, config["paths"]["run_log"]))
    client = None
    if not mock and "openai" in providers:
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError(
                "The openai package is not installed. Run pip install -r requirements.txt."
            ) from error
        client = OpenAI(timeout=120.0, max_retries=0)
    ollama_model_info: dict[tuple[str, str], dict[str, Any]] = {}
    if not mock and "ollama" in providers:
        for task in tasks:
            if task["provider"].lower() != "ollama":
                continue
            key = (task["api_base_url"], task["api_model"])
            if key not in ollama_model_info:
                ollama_model_info[key] = get_ollama_model_info(
                    task["api_base_url"], task["api_model"]
                )
    run_id = str(uuid.uuid4())
    logger.info(
        "run_id=%s mode=%s task_file=%s selected=%d already_complete=%d limited_out=%d",
        run_id,
        "mock" if mock else "+".join(sorted(providers)),
        tasks_path,
        len(pending),
        already_complete,
        limited_out,
    )
    counters = {
        "selected": len(pending),
        "success": 0,
        "refused": 0,
        "failed": 0,
        "already_complete": already_complete,
        "limited_out": limited_out,
        "skipped": already_complete + limited_out,
    }
    generation = config["generation"]

    for index, task in enumerate(pending, start=1):
        max_attempts = 1 if mock else int(generation["max_retries"])
        task_succeeded = False
        started = time.perf_counter()
        last_error: dict[str, str] | None = None
        for attempt in range(1, max_attempts + 1):
            attempt_started = time.perf_counter()
            try:
                if mock:
                    parsed = _mock_response(task)
                    api_metadata = {
                        "response_id": f"mock-{task['task_id']}",
                        "request_id": None,
                        "api_model_returned": "MOCK_DO_NOT_ANALYZE",
                        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                        "raw_output_text": parsed.model_dump_json(),
                    }
                else:
                    provider = task["provider"].lower()
                    if provider == "openai":
                        parsed, api_metadata = _call_openai(client, task, generation)
                    elif provider == "ollama":
                        key = (task["api_base_url"], task["api_model"])
                        parsed, api_metadata = _call_ollama(
                            task,
                            generation,
                            ollama_model_info[key],
                            attempt=attempt,
                        )
                    else:
                        raise ValueError(f"Unsupported provider: {provider}")
                elapsed = time.perf_counter() - attempt_started
                append_jsonl(
                    attempts_path,
                    {
                        "run_id": run_id,
                        "task_id": task["task_id"],
                        "attempt": attempt,
                        "status": "success",
                        "elapsed_seconds": round(elapsed, 3),
                        "timestamp_utc": utc_now(),
                        "is_mock": mock,
                        "generation_seed_used": (
                            api_metadata.get("generation_options", {}).get("seed")
                            if not mock and task["provider"].lower() == "ollama"
                            else None
                        ),
                    },
                )
                record = {
                    "run_id": run_id,
                    "task_id": task["task_id"],
                    "part": task["part"],
                    "case_id": task["case_id"],
                    "domain": task["domain"],
                    "logical_model_id": task["logical_model_id"],
                    "provider": task["provider"],
                    "api_model_requested": task["api_model"],
                    "api_model_returned": api_metadata["api_model_returned"],
                    "api_base_url": task.get("api_base_url", ""),
                    "model_digest": api_metadata.get("model_digest"),
                    "model_details": api_metadata.get("model_details"),
                    "generation_options": api_metadata.get("generation_options"),
                    "anchor_condition": task["anchor_condition"],
                    "intervention": task["intervention"],
                    "repeat_id": int(task["repeat_id"]),
                    "prompt_hash": task["prompt_hash"],
                    "system_prompt": task["system_prompt"],
                    "user_prompt": task["user_prompt"],
                    "response_id": api_metadata["response_id"],
                    "request_id": api_metadata["request_id"],
                    "raw_output_text": api_metadata["raw_output_text"],
                    "parsed_output": parsed.model_dump(),
                    "usage": api_metadata["usage"],
                    "provider_created_at": api_metadata.get("created_at"),
                    "done_reason": api_metadata.get("done_reason"),
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "attempts_used": attempt,
                    "timestamp_utc": utc_now(),
                    "status": "success",
                    "is_mock": mock,
                    "schema_version": task["schema_version"],
                }
                append_jsonl(raw_path, record)
                counters["success"] += 1
                task_succeeded = True
                logger.info("OK %d/%d %s", index, len(pending), task["task_id"])
                break
            except ModelRefusalError as error:
                elapsed = time.perf_counter() - attempt_started
                append_jsonl(
                    attempts_path,
                    {
                        "run_id": run_id,
                        "task_id": task["task_id"],
                        "attempt": attempt,
                        "status": "model_refusal",
                        "elapsed_seconds": round(elapsed, 3),
                        "timestamp_utc": utc_now(),
                        "is_mock": mock,
                        "refusal_text": str(error)[:2000],
                    },
                )
                append_jsonl(
                    raw_path,
                    {
                        "run_id": run_id,
                        "task_id": task["task_id"],
                        "part": task["part"],
                        "case_id": task["case_id"],
                        "logical_model_id": task["logical_model_id"],
                        "anchor_condition": task["anchor_condition"],
                        "intervention": task["intervention"],
                        "repeat_id": int(task["repeat_id"]),
                        "prompt_hash": task["prompt_hash"],
                        "timestamp_utc": utc_now(),
                        "status": "model_refusal",
                        "is_mock": mock,
                        "schema_version": task["schema_version"],
                        "refusal_text": str(error)[:2000],
                        **error.metadata,
                    },
                )
                counters["refused"] += 1
                task_succeeded = True  # Terminal substantive response: do not retry.
                logger.warning("REFUSAL task=%s", task["task_id"])
                break
            except Exception as error:  # API exceptions vary across SDK versions.
                elapsed = time.perf_counter() - attempt_started
                last_error = {
                    "error_type": type(error).__name__,
                    "error_message": str(error)[:2000],
                }
                append_jsonl(
                    attempts_path,
                    {
                        "run_id": run_id,
                        "task_id": task["task_id"],
                        "attempt": attempt,
                        "status": "failure",
                        "elapsed_seconds": round(elapsed, 3),
                        "timestamp_utc": utc_now(),
                        "is_mock": mock,
                        "generation_seed_used": (
                            _ollama_seed_for_attempt(task, generation, attempt)
                            if not mock and task["provider"].lower() == "ollama"
                            else None
                        ),
                        **last_error,
                    },
                )
                logger.warning(
                    "FAIL attempt=%d/%d task=%s error=%s",
                    attempt,
                    max_attempts,
                    task["task_id"],
                    last_error["error_message"],
                )
                if attempt < max_attempts:
                    delay = float(generation["retry_base_seconds"]) * (2 ** (attempt - 1))
                    time.sleep(delay + random.random())

        if not task_succeeded:
            counters["failed"] += 1
            append_jsonl(
                raw_path,
                {
                    "run_id": run_id,
                    "task_id": task["task_id"],
                    "part": task["part"],
                    "case_id": task["case_id"],
                    "logical_model_id": task["logical_model_id"],
                    "anchor_condition": task["anchor_condition"],
                    "intervention": task["intervention"],
                    "repeat_id": int(task["repeat_id"]),
                    "prompt_hash": task["prompt_hash"],
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "attempts_used": max_attempts,
                    "timestamp_utc": utc_now(),
                    "status": "generation_failure",
                    "is_mock": mock,
                    "schema_version": task["schema_version"],
                    **(last_error or {}),
                },
            )
        if not mock and float(generation.get("request_delay_seconds", 0)) > 0:
            time.sleep(float(generation["request_delay_seconds"]))
    logger.info("run complete counters=%s", json.dumps(counters))
    return counters


def run_tasks(
    config: dict[str, Any],
    tasks_path: Path,
    *,
    model_filter: str | None = None,
    limit: int | None = None,
    mock: bool = False,
    live_confirmation: str | None = None,
    local_confirmation: str | None = None,
) -> dict[str, int]:
    raw_path = project_path(config, config["paths"]["raw_outputs"])
    lock_path = raw_path.with_suffix(raw_path.suffix + ".run.lock")
    with GenerationRunLock(lock_path):
        return _run_tasks_unlocked(
            config,
            tasks_path,
            model_filter=model_filter,
            limit=limit,
            mock=mock,
            live_confirmation=live_confirmation,
            local_confirmation=local_confirmation,
        )
