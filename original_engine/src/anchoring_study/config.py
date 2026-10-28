from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    config["_config_path"] = str(path)
    config["_project_root"] = str(PROJECT_ROOT)
    validate_config(config)
    return config


def project_path(config: dict[str, Any], configured_path: str) -> Path:
    path = Path(configured_path)
    if path.is_absolute():
        return path
    return Path(config["_project_root"]) / path


def enabled_models(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [model for model in config.get("models", []) if model.get("enabled", True)]


def validate_config(config: dict[str, Any]) -> None:
    required_sections = {"study", "paths", "design", "generation", "models"}
    missing = required_sections - set(config)
    if missing:
        raise ValueError(f"Config is missing sections: {sorted(missing)}")

    logical_ids: list[str] = []
    for model in enabled_models(config):
        for field in ("logical_id", "provider", "api_model"):
            if not model.get(field):
                raise ValueError(f"Enabled model is missing {field}: {model}")
        logical_ids.append(model["logical_id"])
        if "replace_with" in model["api_model"]:
            raise ValueError(
                f"Replace the placeholder API model for {model['logical_id']} before use."
            )
        provider = str(model["provider"]).lower()
        if provider not in {"openai", "ollama"}:
            raise ValueError(
                f"Unsupported provider {model['provider']} for {model['logical_id']}."
            )
        if provider == "ollama":
            base_url = str(
                model.get("api_base_url", "http://127.0.0.1:11434/api")
            ).rstrip("/")
            parsed = urlparse(base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError(
                    f"Invalid Ollama api_base_url for {model['logical_id']}: {base_url}"
                )
            if not bool(model.get("allow_remote_ollama", False)) and parsed.hostname not in {
                "127.0.0.1",
                "localhost",
                "::1",
            }:
                raise ValueError(
                    "Ollama is restricted to this computer by default. Set "
                    "allow_remote_ollama=true only for a documented, authorized LAN server."
                )
            if ":" not in str(model["api_model"]):
                raise ValueError(
                    f"Use an explicit Ollama model tag, not an alias: {model['api_model']}"
                )
            for numeric_field in ("temperature", "top_p", "num_ctx"):
                if model.get(numeric_field) is None:
                    raise ValueError(
                        f"Ollama model {model['logical_id']} is missing {numeric_field}."
                    )
            format_mode = str(model.get("ollama_format_mode", "schema")).lower()
            if format_mode not in {"schema", "json", "prompt_only"}:
                raise ValueError(
                    f"Invalid ollama_format_mode for {model['logical_id']}: "
                    f"{format_mode}. Use schema, json, or prompt_only."
                )
            if model.get("max_output_tokens") is not None and int(
                model["max_output_tokens"]
            ) <= 0:
                raise ValueError(
                    f"max_output_tokens must be positive for {model['logical_id']}."
                )
    if not logical_ids:
        raise ValueError("At least one model must be enabled.")
    if len(logical_ids) != len(set(logical_ids)):
        raise ValueError("Model logical_id values must be unique.")

    allowed_conditions = {"no_label", "correct_label", "incorrect_label"}
    configured_conditions = set(config["design"]["part1_conditions"])
    if configured_conditions != allowed_conditions:
        raise ValueError(
            "Part I must contain exactly no_label, correct_label, and incorrect_label."
        )

    allowed_interventions = {
        "evidence_first",
        "differential",
        "counter_anchor",
        "self_check",
    }
    configured_interventions = set(config["design"]["part2_interventions"])
    if configured_interventions != allowed_interventions:
        raise ValueError(
            "Part II must contain exactly evidence_first, differential, "
            "counter_anchor, and self_check."
        )
