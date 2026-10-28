from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from anchoring_study.config import enabled_models, load_config
from anchoring_study.generation import LIVE_CONFIRMATION


class ConnectivityResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str


def main() -> int:
    parser = argparse.ArgumentParser(description="Make one tiny paid API request.")
    parser.add_argument("--config", default="config/pilot.toml")
    parser.add_argument("--confirm-live-run", default="")
    args = parser.parse_args()
    if args.confirm_live_run != LIVE_CONFIRMATION:
        print(
            "API test blocked. After enabling API billing, re-run with:\n"
            f"python scripts\\03_test_api.py --confirm-live-run {LIVE_CONFIRMATION}"
        )
        return 2

    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY was not found in .env.")
        return 1
    from openai import OpenAI

    config = load_config(args.config)
    model = enabled_models(config)[0]
    client = OpenAI(timeout=60.0, max_retries=0)
    request = {
        "model": model["api_model"],
        "input": "Return the status value api_connection_successful.",
        "text_format": ConnectivityResult,
        "max_output_tokens": 100,
        "store": False,
    }
    if model.get("reasoning_effort"):
        request["reasoning"] = {"effort": model["reasoning_effort"]}
    response = client.responses.parse(**request)
    print(f"API connection successful. Requested model: {model['api_model']}")
    print(f"Returned model: {getattr(response, 'model', 'not reported')}")
    print(f"Response ID: {getattr(response, 'id', 'not reported')}")
    print(f"Parsed status: {response.output_parsed.status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

