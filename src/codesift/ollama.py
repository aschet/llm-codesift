# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""Minimal Ollama HTTP client. Standard library only, no third-party dependencies."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any


class Ollama:
    def __init__(self, host: str, timeout: int = 2400) -> None:
        self.host = host
        self.timeout = timeout

    def _post(self, path: str, payload: dict[str, Any], timeout: int | None = None) -> dict:
        req = urllib.request.Request(
            self.host + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
            return json.loads(r.read())

    def _get(self, path: str, timeout: int = 30) -> dict:
        with urllib.request.urlopen(self.host + path, timeout=timeout) as r:
            return json.loads(r.read())

    def chat(self, model: str, prompt: str | list[dict], *, ctx: int, num_predict: int,
             tools: list | None = None, think: bool | None = False,
             keep_alive: str = "10m", timeout: int | None = None) -> dict:
        # A string is the whole conversation; a list is one already under way.
        messages = (prompt if isinstance(prompt, list)
                    else [{"role": "user", "content": prompt}])
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": keep_alive,
            "options": {"temperature": 0, "seed": 1, "num_ctx": ctx,
                        "num_predict": num_predict},
        }
        if tools:
            payload["tools"] = tools
        if think is not None:
            payload["think"] = think
        started = time.time()
        try:
            data = self._post("/api/chat", payload, timeout)
        except urllib.error.HTTPError:
            payload.pop("think", None)      # model rejects the think field
            data = self._post("/api/chat", payload, timeout)
        data["_wall"] = round(time.time() - started, 2)
        return data

    def placement(self, model: str) -> dict:
        """What a loaded model occupies, and how much of it is on the GPU.

        The total covers the weights and the KV cache the context window requires,
        and it tracks the window: one 9B model reported 5.49GB at a 4,096-token
        window and 6.58GB at 32,768.

        The two are not separated here. The size on disk is not the weights --
        one model is 6.59GB on disk and loads about 5.3GB -- so the split cannot be
        derived from it; measuring it honestly means loading the model twice, at
        two window sizes.
        """
        try:
            for m in self._get("/api/ps").get("models", []):
                if model in (m.get("name"), m.get("model")):
                    total, vram = m.get("size", 0), m.get("size_vram", 0)
                    return {
                        "total_gb": round(total / 1e9, 2),
                        "pct_gpu": round(100 * vram / total, 1) if total else None,
                    }
        except Exception:
            pass
        return {}

    def unload(self, model: str) -> None:
        try:
            self._post("/api/chat", {"model": model, "messages": [], "keep_alive": 0}, 120)
        except Exception:
            pass
