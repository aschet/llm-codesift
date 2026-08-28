# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""Latency and context behaviour at working depth.

Measures prefill and generation speed against a deep prompt, and checks that the
context survived: Ollama discards prompt overflow without reporting it, so the
tokens it says it read are compared against the tokens it was sent.

A fact planted at the front of the prompt used to be asked for back. It was
retrieved by every model measured, and the one model that ever failed had been
sent a prompt too large for its window -- a fault the token counts name directly.
The question cost a reply of up to 768 tokens per model and answered nothing the
counting does not.
"""
from __future__ import annotations

import time

from . import gpulock, ledger, progress
from . import args
from .config import Config, working_depth
from .ollama import Ollama

# Nothing is generated at depth. The prompt is there to be read, and what is
# measured is the reading: one token is the smallest reply the server will return
# a prefill time with.
DEEP_PREDICT = 1


# Where the filler starts before a model has been measured. Every tokenizer
# reads this text differently -- 2.6 to 3.5 characters per token over the models
# measured -- so a fixed figure sizes the prompt wrong by up to a third, and the
# error runs the dangerous way: a prompt built for three quarters of the window
# arrives at the whole of it and the server discards the overflow in silence.
DEFAULT_CHARS_PER_TOKEN = 3.0

# How much of the prompt the ratio is measured from. Long enough that the chat
# template's own tokens do not distort it, short enough to cost nothing.
CALIBRATION_CHARS = 6000


def _filler(target_tokens: int, chars_per_token: float) -> str:
    """Deterministic pseudo-code filler, sized to the model's own tokenizer."""
    parts, size = [], 0
    for i in range(target_tokens):
        block = (f"def handler_{i}(payload, ctx):\n"
                 f"    # step {i}: normalise then dispatch\n"
                 f"    value = payload.get('field_{i % 97}', {i})\n"
                 f"    return ctx.dispatch(value * {i % 13 + 1})\n\n")
        parts.append(block)
        size += len(block)
        if size > target_tokens * chars_per_token:
            break
    return "".join(parts)


def build_prompt(depth: int, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> str:
    """About `depth` tokens of code for the model to read."""
    return (f"Below is a large codebase excerpt.\n\n"
            f"{_filler(depth, chars_per_token)}\n\n"
            f"Reply with the word OK.")


def calibrate(client: Ollama, model: str, ctx: int, depth: int) -> float:
    """Characters per token for this filler and this model, measured.

    Two short calls: one to learn what the chat template costs on its own, one to
    price a slice of the very text the deep prompt will be made of. The server
    reports the token count of everything it processes, so nothing is estimated.

    The slice comes from the middle of the prompt rather than from a sample of its
    own. The block index grows through the filler, and `handler_7` does not
    tokenize like `handler_431`, so pricing the first forty blocks prices text the
    prompt does not contain -- which sized every prompt about 4% too deep.

    Falls back to the default only when the server reports no count at all, and
    the fallback is deliberately low: a prompt shorter than the target measures a
    shallower depth, while one longer than the window is silently cut in half.
    """
    whole = _filler(depth, DEFAULT_CHARS_PER_TOKEN)
    half = min(len(whole), CALIBRATION_CHARS) // 2
    middle = len(whole) // 2
    sample = whole[middle - half:middle + half]
    try:
        empty = client.chat(model, "x", ctx=ctx, num_predict=1)
        priced = client.chat(model, sample, ctx=ctx, num_predict=1)
    except Exception:
        return DEFAULT_CHARS_PER_TOKEN
    overhead = empty.get("prompt_eval_count") or 0
    tokens = (priced.get("prompt_eval_count") or 0) - overhead
    return len(sample) / tokens if tokens > 0 else DEFAULT_CHARS_PER_TOKEN


def _rate(count, duration_ns):
    if not count or not duration_ns:
        return None
    return round(count / (duration_ns / 1e9), 1)


def measure(client: Ollama, model: str, ctx: int, depth: int, deep: bool = True) -> dict:
    """Measure the model, optionally stopping before the expensive part.

    The shallow call costs seconds and answers whether the model generates fast
    enough to be worth anything; the deep call costs a minute or two and answers
    whether its context survives. A screen that means to reject early wants the
    first without paying for the second.
    """
    rec = {"model": model, "num_ctx": ctx, "depth_target": depth, "ts": time.time()}

    try:
        shallow = client.chat(model, "Write a Python function that reverses a linked list.",
                              ctx=ctx, num_predict=256)
    except Exception as exc:
        rec["error"] = f"shallow: {type(exc).__name__}: {exc}"
        return rec
    rec["gen_tok_s"] = _rate(shallow.get("eval_count"), shallow.get("eval_duration"))
    rec["placement"] = client.placement(model)
    if not deep:
        return rec

    ratio = calibrate(client, model, ctx, depth)
    rec["chars_per_token"] = round(ratio, 2)
    prompt = build_prompt(depth, ratio)
    try:
        reply = client.chat(model, prompt, ctx=ctx, num_predict=DEEP_PREDICT)
    except Exception as exc:
        rec["error"] = f"deep: {type(exc).__name__}: {exc}"
        return rec
    rec["prefill_s"] = round(reply.get("prompt_eval_duration", 0) / 1e9, 1)

    # What the server says it read, which is the depth this was measured at. The
    # target is what was asked for; a prompt the window could not hold is cut
    # without a word, and the difference between the two is how that shows.
    processed = reply.get("prompt_eval_count")
    rec["depth_tokens"] = processed
    rec["likely_truncated"] = bool(processed and processed < depth * 0.9)
    return rec


LEDGER = "probe.jsonl"


def key(rec: dict) -> tuple:
    """What makes a measurement the same measurement: one model, one window."""
    return rec.get("model"), rec.get("num_ctx")


def at_depth(rec: dict) -> bool:
    """Whether this record is a measurement at depth or only the shallow one.

    The speed gate rejects a model without ever sending the deep prompt, so its
    record carries a generation rate and a placement and nothing else. Reading it
    as a full measurement reports a first token that was never waited for and a
    long prompt that was never sent.
    """
    return bool(rec and not rec.get("error") and rec.get("prefill_s") is not None
                and rec.get("depth_target"))


def stored(cfg: Config, model: str, ctx: int) -> dict | None:
    """A completed measurement of this model at this window, if one is on record.

    The deep prompt costs a minute on a large model, and triage sends the same one
    this stage does. Whichever runs first records it here and the other reads it.
    """
    found = None
    for rec in ledger.read(cfg.path(LEDGER)):
        if key(rec) == (model, ctx) and at_depth(rec):
            found = rec
    return found


def record(cfg: Config, rec: dict) -> None:
    """Store a measurement, replacing any earlier one of the same model and window."""
    ledger.replace(cfg.path(LEDGER), rec, key)


def long_prompt(rec: dict) -> dict | None:
    """What the deep prompt found, as a finding, or None where the prompt held."""
    if rec.get("likely_truncated"):
        return {"code": "context_truncated", "num_ctx": rec.get("num_ctx")}
    return None


def _report(rec: dict, depth: int) -> None:
    """What one measurement says, whether it was just taken or read back.

    Every subject is closed with a result, including a failed one: an unclosed
    subject would swallow the next model into its subtest.
    """
    if rec.get("error"):
        progress.unit("probe", "measurement", progress.FAIL, detail=rec["error"])
        progress.result("not measured")
        return
    gpu = (rec.get("placement") or {}).get("pct_gpu")
    # A rate and a wait are numbers, so they leave the status column empty; whether
    # the prompt arrived whole is a verdict, so it fills it.
    progress.unit("probe", "first token", seconds=rec.get("prefill_s"),
                  detail=f"{rec.get('gen_tok_s')} tok/s, {gpu}% on GPU")
    whole = not rec.get("likely_truncated")
    progress.unit("probe", "long prompt", progress.OK if whole else progress.FAIL,
                  detail=f"read {rec.get('depth_tokens') or 0:,} of {depth:,}")
    progress.result(f"{rec.get('prefill_s')}s to first token, "
                    f"{rec.get('gen_tok_s')} tok/s"
                    + ("" if whole else ", prompt truncated at the window"))


def run(cfg: Config, depth: int | None = None, redo: bool = False) -> None:
    depth = depth or working_depth(cfg.ctx)
    gpulock.acquire("probe", endpoint=cfg.host)
    client = Ollama(cfg.host, cfg.timeout)
    path = cfg.path(LEDGER)

    done = {}
    if not redo:
        done = {k: r for k, r in ledger.keyed(path, key).items() if at_depth(r)}

    models = cfg.resolve_models()
    for i, model in enumerate(models, 1):
        if (model, cfg.ctx) in done:
            # Measured already; the stored figures are reported unchanged.
            progress.subject(i, len(models), model)
            _report(done[(model, cfg.ctx)], depth)
            continue
        progress.subject(i, len(models), model,
                         f"{cfg.ctx} context, {depth} token prompt")
        # Triage may already have written the deep measurement; stored() returns
        # only a record that has it. --redo means measure again, so it does not
        # get to answer -- reading back the record the flag exists to replace is
        # the one thing it must not do.
        rec = (None if redo else stored(cfg, model, cfg.ctx)) \
            or measure(client, model, cfg.ctx, depth)
        record(cfg, rec)
        _report(rec, depth)
        client.unload(model)


def main(argv: list[str] | None = None) -> int:
    parser = args.stage("probe", "Latency and context behaviour at working depth.")
    parser.add_argument("--redo", action="store_true")
    a = parser.parse_args(argv)
    cfg = args.config_from(a)
    with progress.document():
        run(cfg, depth=working_depth(cfg.ctx), redo=a.redo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
