# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""JSON Lines stores, which is how every stage keeps what it measured.

One record per line, written as it is taken, so an interrupted run keeps what it
has already paid for. A line that will not parse is skipped rather than fatal: a
run killed mid-write leaves a partial line behind, and every measurement before
it is still good.

A record is identified by a key its own stage chooses -- a model and a window for
the probe, a model and a task for the screen -- and writing one replaces whatever
was stored under that key, so a re-measurement supersedes rather than accumulates.
"""
from __future__ import annotations

import json
from pathlib import Path


def read(path: str | Path) -> list[dict]:
    """Every record on file, in order."""
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def keyed(path: str | Path, key) -> dict:
    """Records by `key(rec)`, the last of a repeated key winning.

    A record the key cannot be taken from is skipped, the same as one that will
    not parse: it is not the record being asked for either way.
    """
    out = {}
    for rec in read(path):
        try:
            out[key(rec)] = rec
        except Exception:
            pass
    return out


def replace(path: str | Path, rec: dict, key) -> None:
    """Store `rec`, dropping whatever was held under the same key."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mine = key(rec)
    except Exception:
        mine = object()                  # keyless: kept, replacing nothing
    kept = []
    for old in read(path):
        try:
            if key(old) == mine:
                continue
        except Exception:
            pass
        kept.append(old)
    kept.append(rec)
    write(path, kept)


def write(path: str | Path, records) -> None:
    """Replace the file with these records."""
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in records),
                          encoding="utf-8")
