"""Runtime configuration, assembled from CLI arguments and the environment."""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_CTX = 65536


def normalise_host(host: str) -> str:
    host = host.rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host


@dataclass
class Config:
    host: str = DEFAULT_HOST
    results_dir: Path = Path("results")
    ctx: int = DEFAULT_CTX
    models: list[str] = field(default_factory=list)
    timeout: int = 2400

    def __post_init__(self) -> None:
        self.host = normalise_host(self.host)
        self.results_dir = Path(self.results_dir)

    @property
    def is_remote(self) -> bool:
        return not any(h in self.host for h in ("localhost", "127.0.0.1", "::1"))

    def path(self, name: str) -> Path:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        return self.results_dir / name

    def resolve_models(self) -> list[str]:
        """Explicit models if given, otherwise everything installed but discarded.

        Naming a model runs it whatever the screen concluded about it; the discard
        list only governs what a sweep of the whole server picks up, so a model
        ruled out once is not measured again by accident.
        """
        if self.models:
            return list(self.models)
        from .prune import read_discarded
        dropped = set(read_discarded(self.results_dir))
        return [m for m in installed_models(self.host) if m not in dropped]


def installed_models(host: str) -> list[str]:
    with urllib.request.urlopen(normalise_host(host) + "/api/tags", timeout=30) as r:
        return sorted(m["name"] for m in json.loads(r.read())["models"])


def read_model_file(path: str | os.PathLike) -> list[str]:
    """One model per line; blank lines and # comments ignored."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def parse_kv(text: str) -> dict[str, str]:
    """Parse lines of the form key=value into a dict.

    Blank lines and lines starting with # are ignored. Whitespace around keys
    and values is stripped.  A line without '=' raises ValueError.
    """
    result: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            raise ValueError(f"no '=' found in line: {stripped!r}")
        key, value = stripped.split("=", 1)
        result[key.strip()] = value.strip()
    return result
