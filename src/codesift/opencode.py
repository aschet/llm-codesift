"""Declare Ollama models in opencode's configuration.

opencode does not discover Ollama models by itself: a provider block supplies the
endpoint but not the model list, so `ollama/<model>` will not resolve unless the
models are declared explicitly or a discovery plugin supplies them.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from .config import Config, installed_models

DEFAULT_CONFIG = Path.home() / ".config" / "opencode" / "opencode.jsonc"


def strip_comments(text: str) -> str:
    """jsonc to json, leaving '//' inside string literals untouched."""
    return re.sub(r'(^|\s)//(?!/).*$', r'\1', text, flags=re.M)


def run(cfg: Config, config_path: Path | None = None, write: bool = False) -> int:
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    models = cfg.models or installed_models(cfg.host)

    data = {}
    if path.exists():
        try:
            data = json.loads(strip_comments(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            print(f"{path} is not valid JSON/JSONC: {exc}")
            return 1

    provider = data.setdefault("provider", {}).setdefault("ollama", {})
    provider.setdefault("npm", "@ai-sdk/openai-compatible")
    provider.setdefault("name", "Ollama")
    provider.setdefault("options", {})["baseURL"] = cfg.host + "/v1"
    before = set(provider.get("models") or {})
    provider["models"] = {m: {"name": m} for m in models}

    print(f"{len(models)} model(s) available; configuration lists {len(before)}")
    for m in sorted(set(models) - before):
        print(f"  + {m}")
    for m in sorted(before - set(models)):
        print(f"  - {m}")

    if not write:
        print("\nDry run. Pass --write to apply.")
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copy(path, str(path) + ".bak")
        print(f"existing configuration backed up to {path}.bak")
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    return 0
