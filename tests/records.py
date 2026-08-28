# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""The records the stages write, as fixtures.

Three test modules build the same ledgers -- the analysis that reads them, the
report that renders them, and the regrade that rewrites them -- so they are
described once.
"""
import json
import tempfile
import unittest
from pathlib import Path

from codesift.config import Config

# Refused immediately rather than timing out, so tests stay fast offline.
OFFLINE = "http://127.0.0.1:1"


# The screen stores one record per task and nothing else; every figure the report
# shows is computed from them. Fixtures build the same records the screen writes.
TASKS_PER_MODEL = 12


def screen_record(model, run=1, rate=100.0, tools_ok=True, ctx=65536,
                  n=TASKS_PER_MODEL, capped=0, silent=0):
    """One model's task records, scoring `rate` over `n` tasks.

    A low pass rate does not rule a model out; a fixture that needs one ruled out
    gives it an unparseable tool call, which does.
    """
    out = [dict(model=model, run=run, ctx=ctx, task="t_tool", kind="toolcall",
                passed=tools_ok, score=float(tools_ok), format_ok=tools_ok,
                detail="ok", wall=0.5, hit_cap=False, raw="call", ts=1.0)]
    rest = n - 1
    winners = round(rate / 100 * rest)
    for i in range(rest):
        ok = i < winners
        # A reply that reached the budget having written something is cut off; one
        # that reached it having written nothing never began answering.
        hit = i < capped + silent
        out.append(dict(model=model, run=run, ctx=ctx, task=f"t{i}", kind="codegen",
                        passed=ok, score=1.0 if ok else 0.0, format_ok=True,
                        detail="ok", wall=1.0, hit_cap=hit,
                        raw="" if i < silent else "def f(): pass", ts=1.0))
    return out


def probe_record(model, prefill=60.0, gen=45.0, truncated=False,
                 pct_gpu=40.0, ctx=65536, weights=5.35, cache=11.31):
    # `weights` and `cache` only set the total a fixture wants; the record carries
    # the total, which is all the server reports.
    place = {"pct_gpu": pct_gpu, "total_gb": round(weights + cache, 2), "vram_gb": 10.5}
    return dict(model=model, num_ctx=ctx, depth_target=ctx * 3 // 4,
                gen_tok_s=gen, prefill_tok_s=800.0,
                prefill_s=prefill, prefill_toks=ctx * 3 // 4, likely_truncated=truncated,
                placement=place)


class RecordsCase(unittest.TestCase):
    """A results directory and the means to fill it."""

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.cfg = Config(host=OFFLINE, results_dir=self.tmp)

    def write(self, name, records):
        with (self.tmp / name).open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")

    def write_tasks(self, groups):
        """Task records for one or more models, as the screen would have left them."""
        flat = []
        for g in groups:
            flat.extend(g if isinstance(g, list) else [g])
        self.write("screen_tasks.jsonl", flat)
