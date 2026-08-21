"""Drop models the screen has ruled out, and the records they produced.

Two separate things happen here. The records are deleted, which is what keeps a
report from being crowded by models nobody will run again. And the names are
written to a discard list, which is what keeps the next sweep from measuring them
all over again: the stages resolve "every model on the server" through that list,
so a discarded model returns only when it is named explicitly.

Nothing is deleted without a backup beside the file it came from, and nothing is
deleted at all unless --apply is passed.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from .config import Config
from .report import analyse

# Every ledger keyed by model.
LEDGERS = ("screen.jsonl", "screen_tasks.jsonl", "probe.jsonl", "prefix_cache.jsonl",
           "agentic.jsonl")
DISCARDED = "discarded.txt"


def read_discarded(results_dir: Path) -> list[str]:
    path = Path(results_dir) / DISCARDED
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def write_discarded(results_dir: Path, names) -> Path:
    path = Path(results_dir) / DISCARDED
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = sorted(set(read_discarded(results_dir)) | set(names))
    path.write_text(
        "# Models the screen ruled out. Sweeps that resolve every installed model\n"
        "# skip these; naming one explicitly still runs it.\n"
        + "".join(f"{n}\n" for n in merged), encoding="utf-8")
    return path


def count_records(results_dir: Path, names) -> dict:
    """How many records each ledger holds for these models."""
    wanted, found = set(names), {}
    for ledger in LEDGERS:
        path = Path(results_dir) / ledger
        if not path.exists():
            continue
        n = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                if json.loads(line).get("model") in wanted:
                    n += 1
            except Exception:
                pass
        if n:
            found[ledger] = n
    return found


def strip_records(results_dir: Path, names) -> dict:
    """Rewrite each ledger without these models, keeping a backup of the original.

    A line that cannot be parsed is kept rather than dropped: it belongs to no
    model that can be identified, so discarding it would lose data this was not
    asked to touch.
    """
    wanted, removed = set(names), {}
    for ledger in LEDGERS:
        path = Path(results_dir) / ledger
        if not path.exists():
            continue
        keep, dropped = [], 0
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                if json.loads(line).get("model") in wanted:
                    dropped += 1
                    continue
            except Exception:
                pass
            keep.append(line)
        if not dropped:
            continue
        shutil.copy(path, str(path) + ".bak")
        path.write_text("".join(f"{line}\n" for line in keep), encoding="utf-8")
        removed[ledger] = dropped
    return removed


def _slug(model: str) -> str:
    return "".join(c if c.isalnum() or c in "-." else "_" for c in model)


def retained_apps(results_dir: Path, names) -> list[Path]:
    """Applications kept from the agent stage, which are not records but are output."""
    root = Path(results_dir) / "agent_apps"
    if not root.is_dir():
        return []
    slugs = {_slug(n) for n in names}
    return sorted(p for p in root.iterdir()
                  if p.is_dir() and p.name.split("__")[0] in slugs)


def run(cfg: Config, verdicts=("unsuitable",), models=None, apply=False,
        forget=False, keep_records=False, stream=None) -> int:
    out = stream or sys.stdout

    if forget:
        path = Path(cfg.results_dir) / DISCARDED
        had = read_discarded(cfg.results_dir)
        if not had:
            print("nothing has been discarded", file=out)
            return 0
        if not apply:
            print(f"would restore {len(had)} model(s) to future sweeps:", file=out)
            for name in had:
                print(f"  {name}", file=out)
            print("\nDry run. Pass --apply to act.", file=out)
            return 0
        path.unlink()
        print(f"restored {len(had)} model(s); {path} removed", file=out)
        return 0

    if models:
        names, reasons = list(models), {}
    else:
        A = analyse(cfg, [m for m in _known(cfg.results_dir)])
        want = {v.strip().lower() for v in verdicts}
        names, reasons = [], {}
        for m in A.models:
            sev, why = A.verdict(m)
            if sev in want:
                names.append(m)
                reasons[m] = "; ".join(why) or sev

    if not names:
        print("no model matches; nothing to discard", file=out)
        return 0

    counts = count_records(cfg.results_dir, names)
    apps = retained_apps(cfg.results_dir, names)
    total = sum(counts.values())

    print(f"{len(names)} model(s) to discard:", file=out)
    for name in names:
        print(f"  {name:34} {reasons.get(name, '')[:70]}", file=out)
    print(f"\n{total} record(s) across {len(counts)} ledger(s):", file=out)
    for ledger, n in sorted(counts.items()):
        print(f"  {ledger:22} {n}", file=out)
    for app in apps:
        print(f"  {app}", file=out)
    if keep_records:
        print("\nRecords kept (--keep-records); only the discard list changes.", file=out)

    if not apply:
        print("\nDry run. Pass --apply to act.", file=out)
        return 0

    if not keep_records:
        removed = strip_records(cfg.results_dir, names)
        print(f"\nremoved {sum(removed.values())} record(s); "
              f"originals kept alongside as .bak", file=out)
        for app in apps:
            shutil.rmtree(app, ignore_errors=True)
            print(f"removed {app}", file=out)
    path = write_discarded(cfg.results_dir, names)
    print(f"wrote {path}; future sweeps skip these unless named explicitly", file=out)
    return 0


def _known(results_dir: Path) -> list[str]:
    """Every model any ledger has a record for."""
    seen = set()
    for ledger in LEDGERS:
        path = Path(results_dir) / ledger
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                name = json.loads(line).get("model")
            except Exception:
                continue
            if name:
                seen.add(name)
    return sorted(seen)
