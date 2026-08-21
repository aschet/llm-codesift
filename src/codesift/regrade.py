"""Re-apply the grader to replies already recorded, without touching the GPU.

A grading fix is worthless if the only way to benefit from it is to re-run every
model for hours. The reply a model gave is stored with each result, so when the
grader changes, the stored replies can simply be graded again: the measurement is
the reply, and grading it is a pure function.

Two limits are stated rather than papered over. Replies recorded before the stored
excerpt was lengthened were cut at 2000 characters, and a truncated reply cannot be
graded -- doing so would fail code that continued past the cut. Those records keep
their original verdict and are reported as unverifiable, which is the honest state:
they need a re-run, and the count says how many.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from .config import Config
from .screen import RAW_KEPT, grade
from .tasks import TASKSETS

# The excerpt length in force before RAW_KEPT was raised. A reply of exactly this
# length was almost certainly cut at it.
OLD_RAW_CAP = 2000

# Only these read the reply as code, so only these can be affected by a change to
# how code is pulled out of one. Tool calls are graded from the recorded call, and
# traces and formats from short answers that were never near the cap.
CODE_KINDS = ("codegen", "edit")

LEGACY = {"tasks": "basic", "tasks_hard": "hard"}


def task_for(rec: dict) -> dict | None:
    name = LEGACY.get(rec.get("taskset")) or rec.get("taskset") or "basic"
    for task in TASKSETS.get(name, []):
        if task["id"] == rec.get("task"):
            return task
    return None


def truncated(raw: str) -> bool:
    return len(raw) >= OLD_RAW_CAP and len(raw) in (OLD_RAW_CAP, RAW_KEPT)


def regrade(rec: dict) -> tuple[dict, str]:
    """Return (record, status) where status is one of skipped/unverifiable/same/changed."""
    if rec.get("kind") not in CODE_KINDS:
        return rec, "skipped"
    task = task_for(rec)
    raw = rec.get("raw") or ""
    if task is None or not raw:
        return rec, "skipped"
    if truncated(raw):
        return rec, "unverifiable"
    passed, format_ok, detail, score = grade(
        task, {"message": {"content": raw, "tool_calls": rec.get("tool_calls")}})
    if (passed, format_ok, score) == (rec.get("passed"), rec.get("format_ok"),
                                      rec.get("score")):
        return rec, "same"
    out = dict(rec, passed=passed, format_ok=format_ok, detail=detail, score=score,
               regraded=True)
    return out, "changed"


def _summarise(rows: list[dict]) -> None:
    """Recompute a summary's rates from the task records it carries."""
    for row in rows:
        tasks = row.get("tasks") or []
        if not tasks:
            continue
        from .screen import _score
        n = len(tasks)
        row["n"] = n
        row["passed"] = sum(1 for t in tasks if t.get("passed"))
        row["pass_rate"] = round(100 * sum(_score(t) for t in tasks) / n, 1)
        row["fully_passed_rate"] = round(100 * row["passed"] / n, 1)
        row["format_ok_rate"] = round(
            100 * sum(1 for t in tasks if t.get("format_ok")) / n, 1)
        row["hit_cap_n"] = sum(1 for t in tasks if t.get("hit_cap"))


def run(cfg: Config, apply: bool = False, stream=None) -> int:
    out = stream or sys.stdout
    tally = {"changed": 0, "same": 0, "unverifiable": 0, "skipped": 0}
    flips, unsure = [], {}

    ledger = Path(cfg.results_dir) / "screen_tasks.jsonl"
    summary = Path(cfg.results_dir) / "screen.jsonl"
    if not ledger.exists():
        print("no screen_tasks.jsonl; nothing to regrade", file=out)
        return 0

    def note(rec, before, status):
        tally[status] += 1
        # Only a changed verdict is a flip. A record that merely gained a score it
        # never carried has not been re-judged, and reporting it as one buried two
        # real corrections under three hundred and sixty lines of noise.
        if status == "changed" and before != rec["passed"]:
            flips.append((rec["model"], rec.get("taskset") or "basic", rec.get("run"),
                          rec["task"], before, rec["passed"]))
        elif status == "unverifiable":
            unsure[rec["model"]] = unsure.get(rec["model"], 0) + 1

    new_ledger = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            new_ledger.append(line)          # not ours to touch
            continue
        before = rec.get("passed")
        rec, status = regrade(rec)
        note(rec, before, status)
        new_ledger.append(json.dumps(rec))

    new_summary = []
    if summary.exists():
        for line in summary.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except Exception:
                new_summary.append(line)
                continue
            row["tasks"] = [regrade(t)[0] for t in (row.get("tasks") or [])]
            new_summary.append(row)
        _summarise([r for r in new_summary if isinstance(r, dict)])

    print(f"{len(flips)} verdict(s) change, {tally['changed'] - len(flips)} rescored, "
          f"{tally['same']} unchanged, {tally['unverifiable']} unverifiable, "
          f"{tally['skipped']} not code tasks", file=out)
    for model, taskset, run_i, task, before, after in flips:
        print(f"  {'FAIL->PASS' if after else 'PASS->FAIL'}  {model:26} "
              f"{taskset:6} run{run_i} {task}", file=out)
    if unsure:
        print("\nreplies cut at the old 2000-character store, which keep their "
              "original result and need a re-run to settle:", file=out)
        for model, n in sorted(unsure.items(), key=lambda kv: -kv[1]):
            print(f"  {model:30} {n}", file=out)

    if not apply:
        print("\nDry run. Pass --apply to rewrite the records.", file=out)
        return 0

    for path, lines in ((ledger, new_ledger), (summary, new_summary)):
        if not path.exists():
            continue
        shutil.copy(path, str(path) + ".bak")
        path.write_text("".join(
            (line if isinstance(line, str) else json.dumps(line)) + "\n"
            for line in lines), encoding="utf-8")
    print(f"\nrewrote {ledger.name} and {summary.name}; originals kept as .bak",
          file=out)
    return 0
