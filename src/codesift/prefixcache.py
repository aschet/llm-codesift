"""Prefix cache reuse across turns.

Determines how much of the cold prefill cost is paid on every turn rather than
once per session, which decides whether multi-turn use is practical.
"""
from __future__ import annotations

import json
import time

from . import gpulock
from .config import Config
from .ollama import Ollama


def _body(tag: str, blocks: int = 800) -> str:
    return "\n".join(
        f"// {tag} block {i}\n"
        f"int compute_{i}(int a, int b) {{ return a * {i % 17 + 1} + b; }}"
        for i in range(blocks))


def measure(client: Ollama, model: str, ctx: int) -> dict:
    body = _body("alpha")
    edited = "// EDITED HEADER -- this line changed near the very start\n" + body
    ask = "\n\nHow many functions are defined? Reply with one number only."

    messages = [{"role": "user", "content": f"Here is a C++ file:\n\n{body}{ask}"}]
    t1 = client.chat_messages(model, messages, ctx=ctx, num_predict=48)

    messages.append({"role": "assistant",
                     "content": (t1.get("message") or {}).get("content") or "ok"})
    messages.append({"role": "user",
                     "content": "What does compute_3 return for a=2, b=1? Number only."})
    t2 = client.chat_messages(model, messages, ctx=ctx, num_predict=48)

    messages.append({"role": "assistant",
                     "content": (t2.get("message") or {}).get("content") or "ok"})
    messages.append({"role": "user", "content": "And compute_5 for a=1, b=0? Number only."})
    t3 = client.chat_messages(model, messages, ctx=ctx, num_predict=48)

    t4 = client.chat_messages(
        model, [{"role": "user", "content": f"Here is a C++ file:\n\n{edited}{ask}"}],
        ctx=ctx, num_predict=48)

    secs = lambda r: round(r.get("prompt_eval_duration", 0) / 1e9, 2)
    rec = {
        "model": model, "ctx": ctx, "ts": time.time(),
        "t1_toks": t1.get("prompt_eval_count"), "t1_prefill_s": secs(t1),
        "t2_toks": t2.get("prompt_eval_count"), "t2_prefill_s": secs(t2),
        "t3_toks": t3.get("prompt_eval_count"), "t3_prefill_s": secs(t3),
        "t4_toks": t4.get("prompt_eval_count"), "t4_prefill_s": secs(t4),
    }
    cold = rec["t1_prefill_s"] or 0.01
    rec["append_cache_hit"] = rec["t2_prefill_s"] < cold * 0.25
    rec["edit_full_reprefill"] = rec["t4_prefill_s"] > cold * 0.75
    return rec


def run(cfg: Config, redo: bool = False) -> None:
    gpulock.acquire("prefix-cache", endpoint=cfg.host)
    client = Ollama(cfg.host, cfg.timeout)
    path = cfg.path("prefix_cache.jsonl")

    done = set()
    if path.exists() and not redo:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["model"])
            except Exception:
                pass

    for model in cfg.resolve_models():
        if model in done:
            print(f"{model}: already measured, skipping", flush=True)
            continue
        print(f"{model}: measuring prefix cache reuse", flush=True)
        rec = measure(client, model, cfg.ctx)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"  cold={rec['t1_prefill_s']}s  append={rec['t2_prefill_s']}s  "
              f"after-edit={rec['t4_prefill_s']}s  "
              f"reuse={'yes' if rec['append_cache_hit'] else 'no'}", flush=True)
        client.unload(model)
