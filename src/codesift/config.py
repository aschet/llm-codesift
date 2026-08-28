# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""Runtime configuration, assembled from CLI arguments and the environment."""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_OUTPUT = "report.html"
# The KV cache is what pushes a model off the GPU, not the weights: at 65,536 only
# five of seventeen models measured were fully resident on a 12GB card, and an 8B
# model was carrying some 11GB of cache.
DEFAULT_CTX = 32768


def working_depth(ctx: int) -> int:
    """The prompt size the probe and the context gate measure at.

    Derived from the window rather than fixed, or a lowered window would leave a
    prompt that cannot fit in it. Three quarters fills the window as a coding
    session does while leaving room for the reply.
    """
    return ctx * 3 // 4


def data_dir(output: str | os.PathLike) -> Path:
    """Where the records for a report belong: beside it, named after it.

    A report and the measurements it was built from are one result, so they are
    kept together and two reports do not share a store. `report.html` puts them
    in `report_data`.
    """
    output = Path(output)
    return output.parent / f"{output.stem}_data"


def normalise_host(host: str) -> str:
    host = host.rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host


@dataclass
class Config:
    host: str = DEFAULT_HOST
    results_dir: Path = None  # type: ignore[assignment]
    ctx: int = DEFAULT_CTX
    models: list[str] = field(default_factory=list)
    timeout: int = 2400

    def __post_init__(self) -> None:
        self.host = normalise_host(self.host)
        self.results_dir = Path(self.results_dir if self.results_dir is not None
                                else data_dir(DEFAULT_OUTPUT))

    @property
    def is_remote(self) -> bool:
        return not any(h in self.host for h in ("localhost", "127.0.0.1", "::1"))

    def path(self, name: str) -> Path:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        return self.results_dir / name

    def resolve_models(self) -> list[str]:
        """Explicit models if given, otherwise everything installed."""
        if self.models:
            return list(self.models)
        return installed_models(self.host)


def installed_models(host: str) -> list[str]:
    with urllib.request.urlopen(normalise_host(host) + "/api/tags", timeout=30) as r:
        return sorted(m["name"] for m in json.loads(r.read())["models"])


PULL_PREFIX = "ollama pull "


def read_model_file(path: str | os.PathLike) -> list[str]:
    """One model per line; blank lines and # comments ignored.

    A leading `ollama pull ` is stripped, which is what `discover --write-models`
    writes: the same file installs the candidates and then names them.
    """
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line.startswith(PULL_PREFIX):
            line = line[len(PULL_PREFIX):].strip()
        if line:
            out.append(line)
    return out

