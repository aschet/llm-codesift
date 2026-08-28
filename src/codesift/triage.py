# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""Reject a model as early and as cheaply as it can honestly be rejected.

The other stages measure a model completely, which is what a benchmark does and
what the report needs. This does the opposite: it asks the cheapest decisive
question first and stops at the first answer that ends the matter, so a model
that cannot be used costs seconds instead of the better part of an hour.

The gates run in the order of what they cost: generation rate from a shallow call
at about ten seconds, tool calls at about twenty, and the deep prompt at about
ninety.

Nothing here invents a threshold. Each gate applies the same rule the report
applies, so a model rejected by triage is rejected for a reason the full run
would have reached anyway -- only sooner, and without the stages after it.

Each gate also tests something a model either can or cannot do. Quality is not of
that kind, so it is measured rather than gated: a cheap proxy does not decide
whether the expensive measurement ever happens.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import gpulock, ledger, probe, progress, screen
from . import args
from .config import Config, working_depth
from .findings import describe, sentence
from .ollama import Ollama
from .analysis import MIN_GEN_TOK_S
from .tasks import TASKS

LEDGER = "triage.jsonl"


@dataclass
class Outcome:
    model: str
    passed: bool
    gate: str = ""
    detail: str = ""
    seconds: float = 0.0
    measured: dict = field(default_factory=dict)
    findings: list = field(default_factory=list)


def _toolcall_ids() -> list[str]:
    return [t["id"] for t in TASKS if t["kind"] == "toolcall"]


def gate_speed(cfg: Config, client: Ollama, model: str) -> tuple[bool, list, dict]:
    """Generation rate, from a shallow call. No deep prompt is paid for.

    A full measurement already on record answers this without any call at all,
    which is what happens when triage is re-run over a screened field.

    What it measures goes to the probe ledger, where every other rate is. A model
    rejected here is never probed, so this is the only measurement of it there
    will be, and the finding that phrases the rejection is not a place to keep it.
    """
    rec = probe.stored(cfg, model, cfg.ctx)
    if rec is None:
        rec = probe.measure(client, model, cfg.ctx, 0, deep=False)
        if not rec.get("error"):
            probe.record(cfg, rec)
    if rec.get("error"):
        return False, [{"code": "error", "message": rec["error"]}], rec
    rate = rec.get("gen_tok_s")
    if rate is None:
        return True, [{"code": "generation_unreadable"}], rec
    if rate < MIN_GEN_TOK_S:
        return False, [{"code": "slow_generation", "tok_s": rate}], rec
    return True, [{"code": "generation_ok", "tok_s": rate}], rec


def _run_tasks(cfg: Config, client: Ollama, model: str,
               only: list[str] | None = None) -> list[dict]:
    """Grade a model on some tasks, recording them in the screen's own ledger.

    Not a private measurement. The tasks, the grader and the record format are the
    screen's, so a task graded here is a task the screen does not run again rather
    than one it asks a second time at the same cost.

    Recorded as run 1, which is the run the screen would fill first.
    """
    path = cfg.path("screen_tasks.jsonl")
    done = screen.load_ledger(path)
    tasks = [t for t in TASKS if not only or t["id"] in set(only)]
    return screen.measure_tasks(client, cfg, model, 1, tasks, done, path)


def gate_tools(cfg: Config, client: Ollama, model: str) -> tuple[bool, list, dict]:
    """A call the harness cannot parse ends a session, so it ends the screen.

    A request that failed is reported as the failure it was. It is not a call the
    harness could not parse: the model never got to make one, and a dropped
    connection is not a fact about the model.
    """
    results = _run_tasks(cfg, client, model, _toolcall_ids())
    failed = [r for r in results if not screen.measured(r)]
    if failed:
        return False, [{"code": "error", "message": failed[0]["error"]}], {"tools": results}
    malformed = [r for r in results if not r["format_ok"]]
    if malformed:
        return False, [{"code": "malformed_tool_calls", "malformed": len(malformed),
                        "total": len(results)}], {"tools": results}
    return True, [{"code": "tools_ok", "total": len(results)}], {"tools": results}


def gate_context(cfg: Config, client: Ollama, model: str, depth: int) -> tuple[bool, list, dict]:
    """The deep prompt, paid for last because it is the most expensive measurement.

    It records rather than rejects. A model that cannot use a long prompt is often
    a good model at short ones -- nemotron-cascade-2 answers 82% of the task set
    and 100% of it parseably, while repeating itself until the budget runs out at
    24k tokens -- and rejecting it here means the report cannot say either thing,
    because the screen it would have to run never happens. The finding travels to
    the report and costs the model its recommendation, not its measurement.

    This is the same measurement the probe stage takes, so it is written to the
    probe ledger and that stage skips whatever it finds already recorded.
    """
    rec = probe.stored(cfg, model, cfg.ctx)
    if rec is None:
        rec = probe.measure(client, model, cfg.ctx, depth, deep=True)
        if not rec.get("error"):
            probe.record(cfg, rec)
    if rec.get("error"):
        return False, [{"code": "error", "message": rec["error"]}], rec
    found = probe.long_prompt(rec)
    return True, [found or {"code": "context_ok"}], rec


GATES = ("speed", "tools", "context")


def triage_model(cfg: Config, client: Ollama, model: str, depth: int,
                 stream=None) -> Outcome:
    out = stream or sys.stdout
    started, measured, noted = time.time(), {}, []
    for name in GATES:
        at = time.time()
        if name == "speed":
            ok, found, rec = gate_speed(cfg, client, model)
        elif name == "tools":
            ok, found, rec = gate_tools(cfg, client, model)
        else:
            ok, found, rec = gate_context(cfg, client, model, depth)
        measured.update(rec)
        detail = "; ".join(describe(f) for f in found)
        progress.unit("triage", name, progress.OK if ok else progress.FAIL,
                      time.time() - at, detail, stream=out)
        # A gate that passes can still have found something. "context_ok" and its
        # kind say only that nothing was found, and are not worth a reader's time.
        noted.extend(f for f in found if not f["code"].endswith("_ok"))
        if not ok:
            return Outcome(model, False, name, detail, round(time.time() - started, 1),
                           measured, found)
    # What a cleared model is still worth saying: nothing, usually, but a fault
    # at depth no longer stops a model and has to reach the report somehow.
    return Outcome(model, True, "", "cleared every gate",
                   round(time.time() - started, 1), measured, noted)


def rests_on_a_failure(rec: dict) -> bool:
    """Whether an outcome was decided by a request that failed rather than by the model.

    Such an outcome is worth reporting -- the run has to say why it stopped -- but
    it is not a verdict, so running again measures the model rather than reading
    the failure back.
    """
    return any(f.get("code") == "error" for f in rec.get("findings") or [])


def read_ledger(cfg: Config) -> dict:
    """Each model's verdict, the newest of a repeated model winning."""
    return ledger.keyed(Path(cfg.results_dir) / LEDGER, lambda rec: rec["model"])


def run(cfg: Config, depth: int | None = None, redo: bool = False,
        stream=None) -> int:
    depth = depth or working_depth(cfg.ctx)
    out = stream or sys.stdout
    models = cfg.resolve_models()
    done = {} if redo else {m: r for m, r in read_ledger(cfg).items()
                            if not rests_on_a_failure(r)}
    path = cfg.path(LEDGER)

    gpulock.acquire("triage", endpoint=cfg.host)
    client = Ollama(cfg.host, cfg.timeout)
    cleared, rejected = [], []

    for i, model in enumerate(models, 1):
        if model in done:
            # Nothing to re-measure; the stored verdict is reported unchanged.
            prior = done[model]
            (cleared if prior["passed"] else rejected).append(model)
            progress.subject(i, len(models), model, stream=out)
            progress.result("cleared" if prior["passed"]
                            else f"rejected, {sentence(prior)}", stream=out)
            continue
        progress.subject(i, len(models), model, stream=out)
        result = triage_model(cfg, client, model, depth, stream=out)
        with path.open("a", encoding="utf-8") as fh:
            # The findings carry the gate that decided it and the prose every reader
            # builds from them, so neither is stored a second time.
            fh.write(json.dumps(dict(
                model=result.model, passed=result.passed,
                findings=result.findings, ts=time.time())) + "\n")
        client.unload(model)
        (cleared if result.passed else rejected).append(model)
        progress.result(f"{'cleared' if result.passed else 'rejected at ' + result.gate}"
                        f", {result.seconds:.0f}s", stream=out)

    progress.summary(f"{len(models)} models: {len(cleared)} cleared, "
                     f"{len(rejected)} rejected", stream=out)
    verdicts = read_ledger(cfg)
    for model in rejected:
        progress.note(f"{model}: {sentence(verdicts.get(model, {}))}", stream=out)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = args.stage("triage", "Reject unusable models as cheaply as possible.",
                        executes=True)
    parser.add_argument("--redo", action="store_true")
    a = parser.parse_args(argv)
    cfg = args.config_from(a)
    with progress.document():
        return run(cfg, depth=working_depth(cfg.ctx), redo=a.redo)


if __name__ == "__main__":
    raise SystemExit(main())
