from __future__ import annotations

import importlib
import os
import platform
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "work" / "matplotlib"))


REQUIRED = ["openai", "pydantic", "dotenv", "pandas", "numpy", "scipy", "statsmodels", "matplotlib"]


def main() -> int:
    print(f"Project: {ROOT}")
    print(f"Python: {platform.python_version()} ({sys.executable})")
    if sys.version_info < (3, 11):
        print("ERROR: Python 3.11 or newer is required.")
        return 1

    missing: list[str] = []
    for package in REQUIRED:
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "installed")
            print(f"OK package {package}: {version}")
        except ImportError:
            print(f"MISSING package: {package}")
            missing.append(package)

    env_file = ROOT / ".env"
    print(f".env file: {'present' if env_file.exists() else 'not created yet'}")
    print(f"OPENAI_API_KEY: {'available' if os.getenv('OPENAI_API_KEY') else 'not loaded in this shell'}")
    ollama = shutil.which("ollama")
    print(f"Ollama command: {ollama if ollama else 'not installed or not on PATH'}")
    print(
        "Local Ollama route: no OPENAI_API_KEY is required; the Ollama application "
        "and an explicitly tagged local model are required."
    )
    if missing:
        print("\nRun: python -m pip install -r requirements.txt")
        return 1
    print("\nEnvironment check passed. No API request was made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
