"""Latency and context behaviour at working depth.

Measures prefill and generation speed against a deep prompt, and verifies that
the context survived: Ollama discards prompt overflow without reporting it, so
the token count processed is compared against the count submitted and a planted
fact is retrieved.
"""
from __future__ import annotations

import json
import time

from . import gpulock
from .config import Config
from .ollama import Ollama

NEEDLE_KEY = "DEPLOY_TOKEN"
NEEDLE_VALUE = "quartz-mongoose-8814"


def _filler(target_tokens: int) -> str:
    """Deterministic pseudo-code filler, sized by a rough 3.5 chars/token estimate."""
    parts, size = [], 0
    for i in range(target_tokens):
        block = (f"def handler_{i}(payload, ctx):\n"
                 f"    # step {i}: normalise then dispatch\n"
                 f"    value = payload.get('field_{i % 97}', {i})\n"
                 f"    return ctx.dispatch(value * {i % 13 + 1})\n\n")
        parts.append(block)
        size += len(block)
        if size > target_tokens * 3.5:
            break
    return "".join(parts)


def build_prompt(depth: int) -> str:
    return (f"Here is a configuration note: {NEEDLE_KEY} = {NEEDLE_VALUE}\n"
            f"Below is a large codebase excerpt. Read it, then answer the question.\n\n"
            f"{_filler(depth)}\n\n"
            f"Question: what is the value of {NEEDLE_KEY}? "
            f"Reply with only the value, nothing else.")


def _rate(count, duration_ns):
    if not count or not duration_ns:
        return None
    return round(count / (duration_ns / 1e9), 1)


def architecture(client: Ollama, model: str) -> dict:
    """What the weights are, which is what explains the timings that follow.

    A model whose experts are few and small moves a fraction of itself per token, so
    it tolerates sitting mostly outside VRAM. A dense model of the same size on disk
    moves all of itself, and does not. Recording this alongside the measurement means
    a slow result carries its own explanation.
    """
    try:
        info = (client.show(model) or {}).get("model_info") or {}
    except Exception:
        return {}
    arch = info.get("general.architecture") or ""
    if not arch:
        # Without an architecture the expert keys cannot be looked up, and reporting
        # "not a mixture of experts" on that basis would assert dense from silence.
        return {}
    experts = info.get(f"{arch}.expert_count")
    out = {"arch": arch, "expert_count": experts,
           "expert_used": info.get(f"{arch}.expert_used_count"),
           "moe": bool(experts)}
    return {k: v for k, v in out.items() if v is not None and v != ""}


def measure(client: Ollama, model: str, ctx: int, depth: int, deep: bool = True) -> dict:
    """Measure the model, optionally stopping before the expensive part.

    The shallow call costs seconds and answers whether the model generates fast
    enough to be worth anything; the deep call costs a minute or two and answers
    whether its context survives. A screen that means to reject early wants the
    first without paying for the second.
    """
    rec = {"model": model, "num_ctx": ctx, "depth_target": depth, "ts": time.time()}
    rec.update(architecture(client, model))

    try:
        shallow = client.chat(model, "Write a Python function that reverses a linked list.",
                              ctx=ctx, num_predict=256)
    except Exception as exc:
        rec["error"] = f"shallow: {type(exc).__name__}: {exc}"
        return rec
    rec["load_s"] = round(shallow.get("load_duration", 0) / 1e9, 1)
    rec["gen_toks"] = shallow.get("eval_count")
    rec["gen_tok_s"] = _rate(shallow.get("eval_count"), shallow.get("eval_duration"))
    rec["placement"] = client.placement(model)
    if not deep:
        return rec

    prompt = build_prompt(depth)
    try:
        deep = client.chat(model, prompt, ctx=ctx, num_predict=768)
    except Exception as exc:
        rec["error"] = f"deep: {type(exc).__name__}: {exc}"
        return rec
    rec["prefill_toks"] = deep.get("prompt_eval_count")
    rec["prefill_tok_s"] = _rate(deep.get("prompt_eval_count"),
                                 deep.get("prompt_eval_duration"))
    rec["prefill_s"] = round(deep.get("prompt_eval_duration", 0) / 1e9, 1)
    rec["deep_gen_tok_s"] = _rate(deep.get("eval_count"), deep.get("eval_duration"))

    message = deep.get("message") or {}
    answer = message.get("content") or ""
    thinking = message.get("thinking") or ""
    rec["prompt_sent_est"] = len(prompt) // 4
    rec["likely_truncated"] = bool(
        rec.get("prefill_toks") and rec["prefill_toks"] < rec["prompt_sent_est"] * 0.75)
    rec["retrieved"] = NEEDLE_VALUE in answer or NEEDLE_VALUE in thinking
    rec["answer"] = (answer or thinking)[-300:].strip()
    return rec


def run(cfg: Config, depth: int = 48000, redo: bool = False) -> None:
    gpulock.acquire("probe", endpoint=cfg.host)
    client = Ollama(cfg.host, cfg.timeout)
    path = cfg.path("probe.jsonl")

    done = set()
    if path.exists() and not redo:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                if not rec.get("error"):
                    done.add((rec["model"], rec.get("num_ctx")))
            except Exception:
                pass

    for model in cfg.resolve_models():
        if (model, cfg.ctx) in done:
            print(f"{model}: already measured, skipping", flush=True)
            continue
        print(f"{model}: probing at {cfg.ctx} context, {depth} token prompt", flush=True)
        rec = measure(client, model, cfg.ctx, depth)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        if rec.get("error"):
            print(f"  error: {rec['error']}", flush=True)
        else:
            gpu = (rec.get("placement") or {}).get("pct_gpu")
            print(f"  gpu={gpu}%  generation={rec.get('gen_tok_s')} tok/s  "
                  f"prefill={rec.get('prefill_tok_s')} tok/s over "
                  f"{rec.get('prefill_toks')} tokens ({rec.get('prefill_s')}s)  "
                  f"retrieved={rec.get('retrieved')}", flush=True)
        client.unload(model)
