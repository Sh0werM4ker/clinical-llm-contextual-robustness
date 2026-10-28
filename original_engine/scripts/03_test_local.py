from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchoring_study.config import enabled_models, load_config
from anchoring_study.generation import (
    LOCAL_CONFIRMATION,
    _call_ollama,
    get_ollama_model_info,
)
from anchoring_study.prompts import PROMPT_ONLY_JSON_INSTRUCTION


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Make one local structured Ollama request; no API key or fee."
    )
    parser.add_argument("--config", default="config/pilot_local.toml")
    parser.add_argument(
        "--model",
        default="",
        help="Optional logical model ID when the config contains multiple local models.",
    )
    parser.add_argument("--confirm-local-run", default="")
    args = parser.parse_args()
    if args.confirm_local_run != LOCAL_CONFIRMATION:
        print(
            "Local test blocked. After checking the model tag, re-run with:\n"
            "python scripts\\03_test_local.py --confirm-local-run "
            f"{LOCAL_CONFIRMATION}"
        )
        return 2

    config = load_config(args.config)
    local_models = [
        model
        for model in enabled_models(config)
        if str(model["provider"]).lower() == "ollama"
    ]
    if args.model:
        local_models = [
            model for model in local_models if str(model["logical_id"]) == args.model
        ]
    if not local_models:
        print(
            "ERROR: no enabled Ollama model was found for the requested config/model."
        )
        return 1
    model = local_models[0]
    base_url = model.get("api_base_url", "http://127.0.0.1:11434/api")
    model_info = get_ollama_model_info(base_url, model["api_model"])
    format_mode = str(model.get("ollama_format_mode", "schema"))
    system_prompt = (
        "This is a local structured-output software check. Use only the supplied "
        "information and return the requested clinical JSON fields."
    )
    if format_mode == "prompt_only":
        system_prompt = (
            f"{system_prompt}\n\nRequired JSON output contract:\n"
            f"{PROMPT_ONLY_JSON_INSTRUCTION}"
        )
    task = {
        "api_model": model["api_model"],
        "api_base_url": base_url,
        "system_prompt": system_prompt,
        "user_prompt": (
            "A healthy adult has a mild ankle inversion injury, can bear weight, and has "
            "no deformity or neurovascular deficit. Give a concise, non-emergency response."
        ),
        "temperature": str(model["temperature"]),
        "top_p": str(model["top_p"]),
        "num_ctx": str(model["num_ctx"]),
        "generation_seed": str(config["study"]["seed"]),
        "think": str(model.get("think", False)),
        "keep_alive": str(model.get("keep_alive", "10m")),
        "ollama_format_mode": format_mode,
        "max_output_tokens": str(
            model.get("max_output_tokens", config["generation"]["max_output_tokens"])
        ),
        "reasoning_effort": "",
    }
    parsed, metadata = _call_ollama(task, config["generation"], model_info)
    print("Local Ollama connection successful.")
    print(f"Requested model: {model['api_model']}")
    print(f"Returned model: {metadata['api_model_returned']}")
    print(f"Model digest: {metadata['model_digest']}")
    print(f"Model details: {metadata['model_details']}")
    print(f"Usage: {metadata['usage']}")
    print(f"Parsed diagnosis: {parsed.most_likely_diagnosis}")
    print("No OpenAI API key was used and no OpenAI API fee was incurred.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
