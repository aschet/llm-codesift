# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""Every judgement the records support, with nothing about how they are shown.

`analyse` returns an `Assessment`: which models are eligible, what verdict each
one earns, the finding behind it, and the pass rate and speed it is ranked on.
The report renders that; triage reads the floor it gates on.
"""
from __future__ import annotations

from collections import defaultdict

from . import ledger
from .config import Config
from .probe import at_depth, long_prompt
from .screen import _score

# Speed is judged in two places. Triage rejects a model that generates too slowly to
# be used at all; the report compares what is left on a modelled coding session, built
# from both measured rates.
#
# The floor is set on what a reply costs to wait for, not fitted to the field. At 20
# tokens a second a typical 600-token answer takes half a minute and the longest of
# them two minutes, which has stopped being interactive whatever the prefill does.
MIN_GEN_TOK_S = 20.0

# The session the speed score is a claim about. Coding is not one turn: the context is
# read once, and after that each exchange re-reads only what it added, because the
# server keeps the processed prefix between turns.
#
# ANSWER_TOKENS is about the mean reply length over the passing code and edit tasks;
# USER_TOKENS is a short instruction. The ordering is not sensitive to either: between
# 4 and 16 turns no model moves more than two places, and raising what each turn adds
# from 700 tokens to 5000 moves none more than three.
SESSION_TURNS = 8
ANSWER_TOKENS = 600
USER_TOKENS = 100


def _aggregate(by_model: dict) -> dict:
    """Each model's task records reduced to the figures the page reports."""
    out = {}
    for m, v in by_model.items():
        # A model measured on a handful of tasks has no comparable rate; the
        # figure would move with which tasks happened to be run.
        if len(v) < 10:
            continue
        kind = defaultdict(lambda: [0, 0])
        # A task whose output could not be parsed at all is counted separately from
        # one that parsed and was wrong: for tool calls the two mean very different
        # things to a harness, and only the first ends a session.
        malformed = defaultdict(int)
        for t in v:
            kind[t["kind"]][1] += 1
            kind[t["kind"]][0] += bool(t["passed"])
            if not t.get("format_ok", True):
                malformed[t["kind"]] += 1
        walls = sorted(t["wall"] for t in v if t.get("wall"))
        out[m] = dict(rate=round(100 * sum(_score(t) for t in v) / len(v), 1),
                      n=len(v), kind=dict(kind), malformed=dict(malformed),
                      capped=sum(1 for t in v if t.get("hit_cap")
                                 and (t.get("raw") or "")),
                      # A reply that reached the budget having written nothing is
                      # a different fault from one that was writing and got cut:
                      # the first says the model never finished thinking, the
                      # second that it needed more room.
                      silent=sum(1 for t in v if t.get("hit_cap")
                                 and not (t.get("raw") or "")),
                      med=walls[len(walls) // 2] if walls else 0,
                      fmt=round(100 * sum(1 for t in v if t.get("format_ok")) / len(v), 1))
    return out


class Assessment:
    """Every judgement the records support about one field of models.

    Built from the ledgers once, then asked: what each model's verdict is, the
    findings behind it, the session it would cost, and the order they rank in.
    The report renders it; other stages ask it which models are worth their time.
    """

    # What keeps a model out of the ranking altogether, as opposed to what merely
    # counts against it. Read off the verdict rather than tested a second time, or
    # the two answer differently about the same model.
    #
    # Behaviour at depth is not among them. A model that cannot use a long prompt
    # is frequently a good model at short ones, and a report that hides its pass
    # rate and its speed cannot say so: the finding belongs beside the figures,
    # not instead of them.
    DISQUALIFYING = {"malformed_tool_calls", "slow_generation",
                     "not_screened", "not_probed"}

    def __init__(self, cfg: Config, models: list[str]):
        self.models = list(models)
        wanted = set(self.models)
        self.screen = [r for r in ledger.read(cfg.results_dir / "screen_tasks.jsonl")
                       if r["model"] in wanted]
        self.probe = [r for r in ledger.read(cfg.results_dir / "probe.jsonl")
                      if r["model"] in wanted]
        tasks_by_model = defaultdict(list)
        for r in self.screen:
            tasks_by_model[r["model"]].append(r)
        self.agg = _aggregate(tasks_by_model)
        self.prb = {r["model"]: r for r in self.probe if not r.get("error")}
        self.sc, self.sc_rank = self._scores()

    def tool_stats(self, m: str) -> tuple:
        """Tool-call results: (passed, total, malformed)."""
        a = self.agg.get(m, {})
        k = a.get("kind", {}).get("toolcall")
        passed, total = (k[0], k[1]) if k and k[1] else (0, 0)
        return passed, total, a.get("malformed", {}).get("toolcall", 0)

    def session_time(self, m: str) -> float | None:
        """Seconds for a coding session at working depth.

        The context is read once, then each of SESSION_TURNS exchanges re-reads what
        the conversation added to itself and writes an answer. Prompt tokens go at
        the measured prefill rate, written tokens at the measured generation rate.

        Both rates are measured; only the session they are spent on is chosen. The
        wait for the first token alone would not do: it correlates with the share of
        the model on the GPU at -0.89, so it measures residency more than speed.
        """
        d = self.prb.get(m, {})
        prefill, gen = d.get("prefill_s"), d.get("gen_tok_s")
        # The depth the server read, not the one that was asked for: the prefill
        # was paid on the first and dividing it by the second is a rate nobody
        # measured.
        depth = d.get("depth_tokens") or d.get("depth_target")
        if not prefill or not gen or not depth:
            return None
        read_rate = depth / prefill
        added = SESSION_TURNS * (ANSWER_TOKENS + USER_TOKENS)
        return (depth + added) / read_rate + SESSION_TURNS * ANSWER_TOKENS / gen

    def unmeasured(self, m: str) -> list[dict]:
        """Which of the two records a verdict rests on are missing.

        Every gate is phrased as "if the measurement says so", which passes silently
        when there is no measurement, so absence is stated here once instead of
        reading as a good result at every gate it has no data for.
        """
        gaps = []
        if m not in self.agg:
            gaps.append({"code": "not_screened"})
        # A shallow record is not a probe: it holds a rate and a placement, and
        # says nothing about the two things the deep prompt asks.
        if not at_depth(self.prb.get(m) or {}):
            gaps.append({"code": "not_probed"})
        return gaps

    def verdict(self, m: str) -> tuple[str, list[dict]]:
        """suitable / limited / unsuitable, and the findings behind it.

        Findings rather than sentences, in the shape the gates record: `findings`
        phrases them, so a fault a gate rejected a model for and the same fault
        read back off its records cannot be worded two ways on one page.
        """
        why, sev = [], "suitable"
        passed, total, malformed = self.tool_stats(m)
        if malformed:
            why.append({"code": "malformed_tool_calls", "malformed": malformed,
                        "total": total})
            sev = "unsuitable"
        elif total and passed < total:
            why.append({"code": "wrong_tool", "wrong": total - passed, "total": total})
            sev = "limited"
        # Absence of a bad result is not a good result: a model missing either record
        # cannot be called suitable on the strength of a measurement never taken.
        gaps = self.unmeasured(m)
        if gaps:
            why.extend(gaps)
            sev = "limited" if sev == "suitable" else sev   # only ever a downgrade
        d = self.prb.get(m, {})
        # Window and depth come from the record: both are the reader's to choose.
        # A fault at depth limits a model rather than disqualifying it -- it says
        # what the model cannot be asked to do, not that it cannot be used.
        at_length = long_prompt(d)
        if at_length:
            why.append(at_length)
            sev = "limited" if sev == "suitable" else sev
        gen = d.get("gen_tok_s")
        if gen and gen < MIN_GEN_TOK_S:
            why.append({"code": "slow_generation", "tok_s": gen})
            sev = "unsuitable"
        # Split by what was written: a reply that reached the budget having emitted
        # no answer spent all of it reasoning, which is a different fault from one
        # that was writing and got cut.
        agg = self.agg.get(m, {})
        for code, count in (("replies_silent", agg.get("silent", 0)),
                            ("replies_capped", agg.get("capped", 0))):
            if count:
                why.append({"code": code, "count": count, "total": agg.get("n", 0)})
        return sev, why

    def eligible(self, m: str) -> list[dict]:
        """The findings that bar a model from being ranked or recommended."""
        return [f for f in self.verdict(m)[1] if f["code"] in self.DISQUALIFYING]

    def _scores(self) -> tuple[dict, list]:
        """Pass rate and speed for each model that cleared the gates, and their order.

        The two are reported side by side rather than combined into one number: a
        weight would decide a trade the reader is better placed to make. The picks
        name the ends of that trade so what lies between them is visible.
        """
        elig = [m for m in self.models if m in self.agg and not self.eligible(m)]
        if not elig:
            return {}, []
        # Every task, over the five capabilities.
        q = {m: self.agg.get(m, {}).get("rate", 0.0) for m in elig}
        pf = {m: (self.prb.get(m, {}).get("prefill_s") or 0) for m in elig}
        gn = {m: (self.prb.get(m, {}).get("gen_tok_s") or 0) for m in elig}
        sess = {m: (self.session_time(m) or 0.0) for m in elig}
        # The ratio of the fastest modelled session to the model's own, so a model
        # taking twice as long scores half. The reference is the quickest model
        # measured whatever its verdict, since how fast a model is does not depend on
        # how well anything else scores; a model without probe data scores zero.
        quickest = [sess[m] for m in elig if sess[m] > 0]
        best_sess = min(quickest) if quickest else 0
        speed = {m: (100 * best_sess / sess[m] if sess[m] > 0 else 0.0) for m in elig}
        out = {m: dict(quality=q[m], speed=round(speed[m], 1),
                       prefill=pf[m], gen=gn[m], session=round(sess[m], 1)) for m in elig}
        # Verdict, then quality, then speed, applied in sequence.
        rank = {"suitable": 0, "limited": 1, "unsuitable": 2}
        return out, sorted(elig, key=lambda m: (rank.get(self.verdict(m)[0], 3),
                                                -q[m], -speed[m]))


def analyse(cfg: Config, models: list[str]) -> Assessment:
    """Load the records for these models and derive every judgement made about them."""
    return Assessment(cfg, models)
