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

    def chat(self, model: str, prompt: str, *, ctx: int, num_predict: int,
             tools: list | None = None, think: bool | None = False,
             keep_alive: str = "10m", timeout: int | None = None) -> dict:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
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

    def chat_messages(self, model: str, messages: list[dict], *, ctx: int,
                      num_predict: int, keep_alive: str = "10m") -> dict:
        payload = {
            "model": model, "messages": messages, "stream": False,
            "keep_alive": keep_alive, "think": False,
            "options": {"temperature": 0, "seed": 1, "num_ctx": ctx,
                        "num_predict": num_predict},
        }
        started = time.time()
        data = self._post("/api/chat", payload)
        data["_wall"] = round(time.time() - started, 2)
        return data

    def show(self, model: str, timeout: int = 30) -> dict:
        return self._post("/api/show", {"model": model}, timeout)

    def loaded(self) -> list[str]:
        try:
            return [m.get("name") or m.get("model")
                    for m in self._get("/api/ps").get("models", [])]
        except Exception:
            return []

    def placement(self, model: str) -> dict:
        """VRAM/RAM split for a currently loaded model."""
        try:
            for m in self._get("/api/ps").get("models", []):
                if model in (m.get("name"), m.get("model")):
                    total, vram = m.get("size", 0), m.get("size_vram", 0)
                    return {
                        "total_gb": round(total / 1e9, 2),
                        "vram_gb": round(vram / 1e9, 2),
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

    def unload_all(self, deadline: float = 180.0) -> bool:
        """Unload everything and block until the GPU reports clear."""
        end = time.time() + deadline
        while time.time() < end:
            current = self.loaded()
            if not current:
                return True
            for m in current:
                self.unload(m)
            time.sleep(3)
        return False
