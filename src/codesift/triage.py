"""Reject a model as early and as cheaply as it can honestly be rejected.

The other stages measure a model completely, which is what a benchmark does and
what the report needs. This does the opposite: it asks the cheapest decisive
question first and stops at the first answer that ends the matter, so a model
that cannot be used costs seconds instead of the better part of an hour.

The gate order is taken from what the existing measurements cost and what they
actually rejected, not from intuition:

    gate      cost   rejected here so far
    speed      ~10s  four models generating 6 to 14 tokens a second
    tools      ~20s  one model that could not emit a parseable tool call
    quality    ~70s  the bulk of the field, below 70% on the hard set
    context    ~90s  one model that truncated at 64k and lost the needle

Nothing here invents a threshold. Each gate applies the same rule the report
applies, so a model rejected by triage is rejected for a reason the full run
would have reached anyway -- only sooner, and without the stages after it.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import gpulock, probe, screen
from .config import Config
from .ollama import Ollama
from .report import MIN_GEN_TOK_S
from .tasks import TASKSETS

LEDGER = "triage.jsonl"

# Below this the hard set is not a shortfall, it is a refusal; the report draws the
# same line, and this is that line applied earlier.
MIN_HARD_RATE = 70.0


@dataclass
class Outcome:
    model: str
    passed: bool
    gate: str = ""
    detail: str = ""
    seconds: float = 0.0
    measured: dict = field(default_factory=dict)


def _toolcall_ids(taskset: str) -> list[str]:
    return [t["id"] for t in TASKSETS[taskset] if t["kind"] == "toolcall"]


def gate_speed(cfg: Config, client: Ollama, model: str) -> tuple[bool, str, dict]:
    """Generation rate, from a shallow call. No deep prompt is paid for."""
    rec = probe.measure(client, model, cfg.ctx, 0, deep=False)
    if rec.get("error"):
        return False, rec["error"], rec
    rate = rec.get("gen_tok_s")
    if rate is None:
        return True, "generation rate unreadable; not held against it", rec
    if rate < MIN_GEN_TOK_S:
        place = (rec.get("placement") or {}).get("pct_gpu")
        shape = "dense" if rec.get("moe") is False else ""
        detail = f"generates {rate:.0f} tok/s (below {MIN_GEN_TOK_S:.0f})"
        if place and shape:
            detail += f"; {shape} and {place:.0f}% resident"
        return False, detail, rec
    return True, f"{rate:.0f} tok/s", rec


def _run_tasks(cfg: Config, client: Ollama, model: str, taskset: str,
               only: list[str] | None = None) -> list[dict]:
    """Grade a model on some tasks, recording them in the screen's own ledger.

    Not a private measurement. The tasks, the grader and the record format are the
    screen's, so a task graded here is a task the screen does not run again -- the
    cascade would otherwise ask a model the same fifteen questions the screen is
    about to ask it, and answer them at the same cost.

    Recorded as run 1, which is the run the screen would fill first.
    """
    ledger = cfg.path("screen_tasks.jsonl")
    done = screen.load_ledger(ledger)
    tasks = [t for t in TASKSETS[taskset] if not only or t["id"] in set(only)]
    return screen.measure_tasks(client, cfg, model, taskset, 1, tasks, done, ledger)


def gate_tools(cfg: Config, client: Ollama, model: str) -> tuple[bool, str, dict]:
    """A call the harness cannot parse ends a session, so it ends the screen."""
    results = _run_tasks(cfg, client, model, "basic", _toolcall_ids("basic"))
    malformed = [r for r in results if not r["format_ok"]]
    if malformed:
        return (False,
                f"{len(malformed)} of {len(results)} tool calls malformed or absent",
                {"tools": results})
    return True, f"{len(results)} tool calls well formed", {"tools": results}


def gate_quality(cfg: Config, client: Ollama, model: str) -> tuple[bool, str, dict]:
    """The hard set, which is where a model is separated from a nearly-model."""
    results = _run_tasks(cfg, client, model, "hard")
    rate = 100 * sum(1 for r in results if r["passed"]) / (len(results) or 1)
    if rate < MIN_HARD_RATE:
        return False, f"hard-set {rate:.0f}% (below {MIN_HARD_RATE:.0f}%)", {
            "hard": results, "hard_rate": rate}
    return True, f"hard-set {rate:.0f}%", {"hard": results, "hard_rate": rate}


def gate_context(cfg: Config, client: Ollama, model: str, depth: int) -> tuple[bool, str, dict]:
    """The deep prompt, paid for last because it is the most expensive gate."""
    rec = probe.measure(client, model, cfg.ctx, depth, deep=True)
    if rec.get("error"):
        return False, rec["error"], rec
    faults = []
    if rec.get("likely_truncated"):
        faults.append(f"truncates at {cfg.ctx // 1024}k")
    if rec.get("retrieved") is False:
        faults.append(f"failed the {depth // 1000}k needle")
    if faults:
        return False, "; ".join(faults), rec
    return True, "context intact at depth", rec


GATES = ("speed", "tools", "quality", "context")


def triage_model(cfg: Config, client: Ollama, model: str, depth: int,
                 stream=None) -> Outcome:
    out = stream or sys.stdout
    started, measured = time.time(), {}
    for name in GATES:
        at = time.time()
        if name == "speed":
            ok, detail, rec = gate_speed(cfg, client, model)
        elif name == "tools":
            ok, detail, rec = gate_tools(cfg, client, model)
        elif name == "quality":
            ok, detail, rec = gate_quality(cfg, client, model)
        else:
            ok, detail, rec = gate_context(cfg, client, model, depth)
        measured.update(rec)
        print(f"  {name:8} {'ok  ' if ok else 'STOP'} {time.time() - at:5.0f}s  {detail}",
              flush=True)
        if not ok:
            return Outcome(model, False, name, detail, round(time.time() - started, 1),
                           measured)
    return Outcome(model, True, "", "cleared every gate",
                   round(time.time() - started, 1), measured)


def read_ledger(cfg: Config) -> dict:
    path = Path(cfg.results_dir) / LEDGER
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        out[rec["model"]] = rec
    return out


def run(cfg: Config, depth: int = 48000, redo: bool = False, apply: bool = False,
        stream=None) -> int:
    out = stream or sys.stdout
    models = cfg.resolve_models()
    done = {} if redo else read_ledger(cfg)
    path = cfg.path(LEDGER)

    gpulock.acquire("triage", endpoint=cfg.host)
    client = Ollama(cfg.host, cfg.timeout)
    cleared, rejected = [], []

    for model in models:
        if model in done:
            prior = done[model]
            (cleared if prior["passed"] else rejected).append(model)
            print(f"{model}: already triaged, {'cleared' if prior['passed'] else 'rejected'}",
                  file=out, flush=True)
            continue
        print(f"\n{model}:", file=out, flush=True)
        result = triage_model(cfg, client, model, depth, stream=out)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(dict(
                model=result.model, passed=result.passed, gate=result.gate,
                detail=result.detail, seconds=result.seconds, ts=time.time())) + "\n")
        client.unload(model)
        (cleared if result.passed else rejected).append(model)
        print(f"  -> {'CLEARED' if result.passed else 'REJECTED at ' + result.gate} "
              f"in {result.seconds:.0f}s", file=out, flush=True)

    print(f"\n{len(cleared)} cleared, {len(rejected)} rejected", file=out)
    for model in rejected:
        rec = read_ledger(cfg).get(model, {})
        print(f"  {model:30} {rec.get('gate', ''):8} {rec.get('detail', '')[:50]}", file=out)

    if rejected and apply:
        from .prune import write_discarded
        written = write_discarded(cfg.results_dir, rejected)
        print(f"\nwrote {written}; sweeps skip these unless named explicitly", file=out)
    elif rejected:
        print("\nPass --apply to add the rejected models to the discard list.", file=out)
    return 0
