"""Renders the measurement records into a single self-contained HTML report."""
from __future__ import annotations

import datetime as dt
import json
import os
import statistics as st
import subprocess
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .tasks import AGENT_TASKS

# A model is slow when it is slow to use, not when something else is faster. Anchoring
# the label on the fastest model in the run made it depend on which models happened to
# be measured together: adding one quick model relabelled the whole field. These are
# absolute, in seconds, and describe the machine the measurement ran on.
#
# The quantity is the modelled time to finish one representative task from a cold
# context. Past two minutes a coding assistant has stopped being interactive and become
# a batch job; past five it is not usable for the work at all, which is where a model
# too large for the available VRAM lands once most of its weights sit behind the bus.
SLOW_TASK_S = 120.0
UNUSABLE_TASK_S = 300.0

# Generation rate is the sharper of the two latency measures, because it is the one
# governed by how much of the model moves per token. A mixture of experts moves a few
# billion parameters whatever fraction of it sits outside VRAM; a dense model of the
# same size on disk moves all of them, and pays the bus for every one. In the field
# this was written against the two populations separate by 30 tokens a second with
# nothing in between, while their modelled task times sit only 45 seconds apart.
#
# The floor is not fitted to that gap. Below roughly 20 tokens a second a reply of the
# length these tasks produce takes over a minute to appear, which is slower than it can
# be read; the model has stopped being interactive whatever its prefill does.
MIN_GEN_TOK_S = 20.0

# Quality and speed shares of the composite. Quality carries the larger share because
# a slow model that writes working code can still be used, while a fast one that does
# not cannot be used at all; speed only orders models that already clear the floor.
W_QUALITY = 0.7
W_SPEED = 0.3


def _load(results_dir: Path, name: str) -> list[dict]:
    path = results_dir / name
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


@dataclass
class Assessment:
    """Everything derived from the records, independent of how it is presented.

    The report renders it; other stages ask it which models are worth their time.
    """

    models: list[str]
    screen: list
    probe: list
    cache: list
    agentic: list
    by_set: dict
    t1: dict
    t2: dict
    prb: dict
    cch: dict
    ag: dict
    task_tokens: float
    task_turns: float
    params_measured: bool
    tool_stats: object
    task_time: object
    eligible: object
    verdict: object
    sc: dict
    sc_rank: list
    qfloor: float

    def by_verdict(self, wanted):
        """Models whose verdict is one of `wanted`, in the order given to the run."""
        want = {w.strip().lower() for w in wanted}
        return [m for m in self.models
                if (m in self.t1 or m in self.t2)
                and self.verdict(m, with_agent=False)[0] in want]


def analyse(cfg: Config, models: list[str]) -> Assessment:
    """Load the records for these models and derive every judgement made about them."""
    SHORTLIST = list(models)
    screen = _load(cfg.results_dir, "screen.jsonl")
    probe = _load(cfg.results_dir, "probe.jsonl")
    cache = _load(cfg.results_dir, "prefix_cache.jsonl")
    agentic = _load(cfg.results_dir, "agentic.jsonl")

    SHORT = set(SHORTLIST)
    screen = [r for r in screen if r["model"] in SHORT]
    probe = [r for r in probe if r["model"] in SHORT]
    cache = [r for r in cache if r["model"] in SHORT]
    agentic = [r for r in agentic if r["model"] in SHORT]

    # "tasks"/"tasks_hard" are the names used before the suites were renamed;
    # accept them so reports can still be produced from older records.
    LEGACY = {"tasks": "basic", "tasks_hard": "hard"}
    by_set = defaultdict(lambda: defaultdict(list))
    for r in screen:
        if r.get("n", 0) >= 10:
            name = r.get("taskset") or "basic"
            by_set[LEGACY.get(name, name)][r["model"]].append(r)
    prb = {r["model"]: r for r in probe if not r.get("error")}
    cch = {r["model"]: r for r in cache}
    # Every attempt is kept, grouped by task. Repeats are samples, not
    # supersessions: this stage is visibly variable -- one model answered the same
    # task by writing nothing in two turns, then by making four edits in ten, then
    # by writing nothing again -- so which attempt happened to be last says very
    # little, and how often it succeeded says a great deal.
    # Only the screen's own tasks. Records left by a separate tool sharing this
    # ledger are its business to report, not the screen's.
    MINE = {t["id"] for t in AGENT_TASKS}
    ag = defaultdict(list)
    attempts = defaultdict(list)
    for r in agentic:
        if r["task"] in MINE:
            attempts[(r["model"], r["task"])].append(r)
    for (model, _), tries in sorted(attempts.items()):
        best = max(tries, key=lambda r: (bool(r["passed"]), r.get("ts") or 0))
        ag[model].append(dict(best, attempts=len(tries),
                              completed=sum(1 for r in tries if r["passed"])))


    def agg(runs_):
        out = {}
        for m, v in runs_.items():
            rates = [r["pass_rate"] for r in v]
            kind = defaultdict(lambda: [0, 0])
            # A task whose output could not be parsed at all is counted separately from
            # one that parsed and was wrong: for tool calls the two mean very different
            # things to a harness, and only the first ends a session.
            malformed = defaultdict(int)
            for r in v:
                for t in r["tasks"]:
                    kind[t["kind"]][1] += 1
                    kind[t["kind"]][0] += bool(t["passed"])
                    if not t.get("format_ok", True):
                        malformed[t["kind"]] += 1
            out[m] = dict(rate=st.mean(rates), spread=max(rates) - min(rates) if len(rates) > 1 else 0,
                          runs=len(v), kind=dict(kind), malformed=dict(malformed),
                          capped=sum(r.get("hit_cap_n", 0) for r in v),
                          med=st.mean([r["median_wall"] for r in v if r.get("median_wall")] or [0]),
                          fmt=st.mean([r["format_ok_rate"] for r in v]))
        return out


    t1, t2 = agg(by_set["basic"]), agg(by_set["hard"])


    def session_params():
        """Parameters for the session model, taken from the agentic runs where available.

        Falls back to stated defaults when no agentic data exists, so the report still
        renders; the values used are printed in the section text either way.
        """
        out = [(r.get("tokens") or {}).get("output") or 0 for r in agentic]
        out = [o for o in out if o]
        turns = [r.get("turns") or 0 for r in agentic]
        turns = [t for t in turns if t]
        return (st.median(out) if out else 768.0,
                st.median(turns) if turns else 5.0,
                bool(out))


    TASK_TOKENS, TASK_TURNS, PARAMS_MEASURED = session_params()


    def tool_stats(m):
        """Tool-call results across both task sets: (passed, total, malformed).

        Both sets are counted because the basic set offers only two tool tasks, so a
        single miss there reads as 50% and overstates what was observed.
        """
        passed = total = malformed = 0
        for agg_ in (t1, t2):
            k = agg_.get(m, {}).get("kind", {}).get("toolcall")
            if k and k[1]:
                passed += k[0]
                total += k[1]
                malformed += agg_.get(m, {}).get("malformed", {}).get("toolcall", 0)
        return passed, total, malformed


    def task_time(m):
        """Modelled seconds to finish one representative task from a cold context.

        The prefill is paid once, then the measured volume of output is generated at
        the measured rate. Prefill and throughput are therefore weighed against each
        other by the work itself rather than by a chosen ratio, and the result is a
        wall-clock figure a reader can judge directly.
        """
        d = prb.get(m, {})
        pf, gn = d.get("prefill_s"), d.get("gen_tok_s")
        if not pf or not gn:
            return None
        return pf + TASK_TOKENS / gn


    def unmeasured(m):
        """Which of the two records a verdict rests on are missing.

        Every gate is phrased as "if the measurement says so", which passes silently
        when there is no measurement. A model screened but never probed would clear
        the truncation, retrieval, generation-rate and task-time gates by having no
        data for any of them, so absence is stated here once rather than being
        rediscovered as a good result four times over.
        """
        gaps = []
        if m not in t2:
            gaps.append("no hard-set result")
        if not prb.get(m):
            gaps.append("no probe result")
        return gaps


    def eligible(m):
        """Hard gates: a model failing any of these cannot be recommended at all."""
        # Ranking a model on the basic set against models ranked on the hard set
        # compares two different measurements, and a model with no probe has no
        # latency figure at all, so either gap sets it aside rather than scoring it
        # on a proxy.
        fails = list(unmeasured(m))
        passed, total, malformed = tool_stats(m)
        if malformed:
            fails.append(f"{malformed} of {total} tool calls malformed or absent")
        d = prb.get(m, {})
        if d.get("likely_truncated"):
            fails.append("truncates the context")
        if d.get("retrieved") is False:
            fails.append("failed retrieval at depth")
        gen = prb.get(m, {}).get("gen_tok_s")
        if gen and gen < MIN_GEN_TOK_S:
            fails.append(f"generates {gen:.0f} tok/s, below {MIN_GEN_TOK_S:.0f}")
        ts = task_time(m)
        if ts and ts > UNUSABLE_TASK_S:
            fails.append(f"{ts / 60:.0f} minutes per task on this machine")
        return fails


    def nothing_it_wrote_ran(m):
        """Whether a graded task produced nothing that would even load.

        Each graded task names the one check that everything else rests on -- the
        application starting, the module importing -- and failing that is not a
        shortfall but a refusal: what the model wrote does not run.

        Judged on the artefact rather than on the session, which is why one attempt
        is enough here while the pass-or-fail tasks below need a repeat. A model
        that gave up after two turns may simply have had a bad draw; code that does
        not load is a property of what it wrote, and re-running cannot make code it
        has already written begin to load.
        """
        pivots = {t["id"]: t["pivot"] for t in AGENT_TASKS if t.get("pivot")}
        for record in (ag.get(m) or []):
            pivot = pivots.get(record.get("task"))
            if not pivot or not record.get("checks"):
                continue
            attempts = [r for r in (ag.get(m) or []) if r.get("task") == record["task"]]
            if any(c["name"] == pivot and c["passed"]
                   for r in attempts for c in (r.get("checks") or [])):
                continue
            return True, record["task"]
        return False, ""


    def agent_outcome(m):
        """(tasks never completed, tasks tried, whether a failure was repeated).

        A task is only held against a model once it has failed every attempt made
        at it. The screen will not report a rate from one run either, and this
        stage is more variable than the screen, not less.
        """
        runs = ag.get(m) or []
        if not runs:
            return 0, 0, False
        never = [r for r in runs if not r["passed"]]
        repeated = any(r.get("attempts", 1) > 1 for r in never)
        return len(never), len(runs), repeated


    def verdict(m, with_agent=True):
        """suitable / limited / unsuitable, with the reason that drove it.

        `with_agent` is off when the result decides who gets an agent run. Letting
        the agent stage's own results choose its participants would be circular: a
        model with no agent record would be downgraded for having none, dropped from
        the stage on that basis, and so never acquire one.
        """
        why, sev = [], "suitable"
        passed, total, malformed = tool_stats(m)
        if malformed:
            why.append(f"{malformed} of {total} tool calls malformed or absent")
            sev = "unsuitable"
        elif total and passed < total:
            # The call was well-formed and the wrong one. A harness returns the wrong
            # result and the model gets another turn, so this degrades a session rather
            # than ending it.
            why.append(f"tool choice {100 * passed / total:.0f}% (calls well-formed)")
            sev = "limited"
        # Absence of a bad result is not a good result: a model missing either record
        # cannot be called suitable on the strength of a measurement never taken.
        gaps = unmeasured(m)
        if gaps:
            why.extend(gaps)
            # Only ever a downgrade. A missing record cannot rehabilitate a model
            # already disqualified by one that is present.
            sev = "limited" if sev == "suitable" else sev
        r2 = t2.get(m, {}).get("rate")
        if r2 is not None and r2 < 70:
            why.append(f"hard-set {r2:.0f}% (below 70%)")
            sev = "unsuitable"
        elif r2 is not None and r2 < 85 and sev != "unsuitable":
            why.append(f"hard-set {r2:.0f}% (below 85%)")
            sev = "limited"
        d = prb.get(m, {})
        if d.get("likely_truncated"):
            why.append("truncates at 64k"); sev = "unsuitable"
        if d.get("retrieved") is False:
            why.append("failed 48k needle"); sev = "unsuitable"
        gen = d.get("gen_tok_s")
        if gen and gen < MIN_GEN_TOK_S:
            place = (d.get("placement") or {}).get("pct_gpu")
            shape = "dense" if d.get("moe") is False else ""
            detail = f"generates {gen:.0f} tok/s (below {MIN_GEN_TOK_S:.0f})"
            if place and shape:
                detail += f"; {shape} and {place:.0f}% resident"
            elif place:
                detail += f"; {place:.0f}% resident"
            why.append(detail)
            sev = "unsuitable"
        ts = task_time(m)
        if ts and ts > UNUSABLE_TASK_S:
            why.append(f"task time {ts:.0f}s (over {UNUSABLE_TASK_S:.0f}s)")
            sev = "unsuitable"
        elif ts and ts > SLOW_TASK_S:
            why.append(f"task time {ts:.0f}s (over {SLOW_TASK_S:.0f}s)")
            sev = "limited" if sev == "suitable" else sev
        sp = t1.get(m, {}).get("spread", 0)
        if sp >= 10:
            why.append(f"run-to-run spread ±{sp:.0f} points")
        # A reply cut off at the output budget was not graded on what the model would
        # have said. It counts as a failure, but the score is a floor rather than a
        # measurement, so the reader is told how much of it rests on truncated replies.
        capped = sum(agg_.get(m, {}).get("capped", 0) for agg_ in (t1, t2))
        if capped:
            why.append(f"{capped} repl{'y' if capped == 1 else 'ies'} cut at the output budget")

        # Finishing a task unattended is a different question from answering a
        # prompt well, and a model can be good at one and not the other. It only
        # ever lowers a verdict here: the stage runs on a subset of the field, so
        # rewarding it would rank a model above another for having been measured.
        if with_agent:
            dead, which = nothing_it_wrote_ran(m)
            if dead:
                # Whatever it scores on a prompt, it cannot be driven through work
                # by an agent harness, which is what this screen is for.
                why.append(f"what it wrote for {which.replace('ag_', '')} never ran")
                sev = "unsuitable"
            failed, tried, repeated = agent_outcome(m)
            if failed:
                noun = "task" if failed == 1 else "tasks"
                if repeated:
                    why.append(f"failed {failed} agent {noun} on every attempt")
                    if sev == "suitable":
                        sev = "limited"
                elif sev != "unsuitable":
                    # Said, not counted. One attempt is where the screen would ask
                    # for another run before believing it, and so does this.
                    why.append(f"failed {failed} agent {noun} once, not retried")
        return sev, why


    def scores():
        """Composite of quality and speed, over models that cleared the gates.

        The quality floor is the point of this function. A model that fails most of
        its tasks finishes them quickly, so raw latency credit rewards exactly the
        models that are not doing the work; below the floor, speed earns nothing and
        the composite is quality alone. Since the floor sits 20 points under the best
        quality seen, the most a below-floor model can score is W_QUALITY x
        (floor - 0.1), which is below the W_QUALITY x floor every above-floor model is
        guaranteed. One can therefore never outrank the other.
        """
        elig = [m for m in SHORTLIST if (m in t2 or m in t1) and not eligible(m)]
        if not elig:
            return {}, [], 0
        q = {m: (t2[m]["rate"] if m in t2 else t1.get(m, {}).get("rate", 0)) for m in elig}
        pf = {m: (prb.get(m, {}).get("prefill_s") or 0) for m in elig}
        gn = {m: (prb.get(m, {}).get("gen_tok_s") or 0) for m in elig}

        sess = {m: (task_time(m) or 0.0) for m in elig}
        # Speed is the ratio of the fastest modelled time to the model's own, so a model
        # taking twice as long scores half. Rescaling the field between its own extremes
        # would instead award the slowest model zero however narrow the spread, turning a
        # small latency difference into the maximum penalty; a model without probe data
        # scores zero rather than being read as instantaneous.
        floor = max(q.values()) - 20
        over = {m: q[m] >= floor for m in elig}
        # The fastest time is taken from models above the quality floor only. A model
        # that fails most of its tasks returns quickly because it is not doing them,
        # and letting it set the denominator compresses every real contender into
        # single digits. The floor's own model is always above it, so this is never
        # empty. It also means removing a failing model does not move anyone's score.
        _fast = [sess[m] for m in elig if over[m] and sess[m] > 0]
        best_sess = min(_fast) if _fast else 0
        # Left uncapped: a below-floor model scoring over 100 is one that beat every
        # usable model on the clock and still earns nothing for it, which is the point.
        speed = {m: (100 * best_sess / sess[m] if sess[m] > 0 else 0.0) for m in elig}
        comp = {m: round(W_QUALITY * q[m] + (W_SPEED * speed[m] if over[m] else 0.0), 1)
                for m in elig}
        out = {m: dict(quality=q[m], speed=round(speed[m], 1), composite=comp[m],
                       prefill=pf[m], gen=gn[m], session=round(sess[m], 1),
                       above_floor=over[m]) for m in elig}
        # The floor already makes the ordering hold arithmetically; sorting on it as
        # well means a later change to the weights cannot quietly undo that.
        return out, sorted(elig, key=lambda m: (not over[m], -comp[m])), floor


    SC, SC_RANK, QFLOOR = scores()


    return Assessment(models=SHORTLIST, screen=screen, probe=probe, cache=cache,
                      agentic=agentic, by_set=by_set, t1=t1, t2=t2, prb=prb, cch=cch,
                      ag=ag, task_tokens=TASK_TOKENS, task_turns=TASK_TURNS,
                      params_measured=PARAMS_MEASURED, tool_stats=tool_stats,
                      task_time=task_time, eligible=eligible, verdict=verdict,
                      sc=SC, sc_rank=SC_RANK, qfloor=QFLOOR)


def build(cfg: Config, models: list[str]) -> str:
    """Return the complete HTML document for the records in cfg.results_dir."""
    OLLAMA = cfg.host
    REMOTE = cfg.is_remote
    SHORTLIST = list(models)

    def gpus():
        """Local GPUs. Meaningless when the server is remote, so skipped in that case."""
        if REMOTE:
            return []
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=20).stdout.strip()
            found = []
            for line in out.splitlines():
                if "," in line:
                    name, mem = line.split(",", 1)
                    mb = int("".join(ch for ch in mem if ch.isdigit()) or 0)
                    found.append(f"{name.strip()} {mb/1024:.0f}GB")
            return found
        except Exception:
            return []


    def quants(models):
        """Quantisation level actually present in each measured model.

        Short timeouts and an early exit: when the server is unreachable the report
        should still render, just without this detail.
        """
        seen = {}
        failures = 0
        for m in models:
            if failures >= 2:          # server is not answering; stop retrying
                break
            try:
                req = urllib.request.Request(
                    OLLAMA + "/api/show",
                    data=json.dumps({"model": m}).encode(),
                    headers={"Content-Type": "application/json"})
                mi = json.loads(urllib.request.urlopen(req, timeout=5).read()).get("model_info") or {}
                q = mi.get("general.file_type_str") or mi.get("general.quantization")
                if not q:
                    d = json.loads(urllib.request.urlopen(req, timeout=5).read())
                    q = (d.get("details") or {}).get("quantization_level")
                if q:
                    seen[m] = str(q)
            except Exception:
                failures += 1
        return seen


    def load(n):
        p = os.path.join(B, n)
        if not os.path.exists(p):
            return []
        r = []
        for line in open(p):
            try:
                r.append(json.loads(line))
            except Exception:
                pass
        return r



    def rejected():
        """Models triage ruled out, which no longer appear anywhere else.

        Pruning a discarded model removes its measurements, so it vanishes from a
        report built from those measurements. A reader then cannot tell whether a
        model is missing because it failed, because it was never run, or because it
        was never installed -- and the first of those is a finding. The triage
        ledger outlives the pruning, so the rejection is still on record.
        """
        out = {}
        for rec in _load(cfg.results_dir, "triage.jsonl"):
            if rec.get("passed") or not rec.get("model"):
                continue
            # The first gate to reject a model is the one that decided it; a later
            # gate never ran.
            if rec["model"] not in out or rec["ts"] < out[rec["model"]]["ts"]:
                out[rec["model"]] = rec
        return [out[m] for m in sorted(out)]


    A = analyse(cfg, SHORTLIST)
    screen, probe, cache, agentic = A.screen, A.probe, A.cache, A.agentic
    by_set, t1, t2 = A.by_set, A.t1, A.t2
    prb, cch, ag = A.prb, A.cch, A.ag
    tool_stats, task_time, eligible, verdict = A.tool_stats, A.task_time, A.eligible, A.verdict
    TASK_TOKENS, TASK_TURNS, PARAMS_MEASURED = A.task_tokens, A.task_turns, A.params_measured


    SC, SC_RANK, QFLOOR = A.sc, A.sc_rank, A.qfloor
    _g = gpus()
    _qmap = quants([m for m in SHORTLIST if m in t1 or m in t2])
    _q = sorted(set(_qmap.values()))

    ranked = sorted([m for m in SHORTLIST if m in t2 or m in t1],
                    key=lambda m: (-(t2.get(m, {}).get("rate", -1)), -(t1.get(m, {}).get("rate", 0))))

    E = lambda s: (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    def num(v, suf="", d=1):
        return f"{v:.{d}f}{suf}" if isinstance(v, (int, float)) else "&mdash;"


    def bar(pct, sev):
        w = max(0, min(100, pct or 0))
        return (f'<div class="bar"><span style="width:{w:.0f}%" class="s-{sev}"></span></div>'
                f'<span class="figure">{pct:.0f}<span class="pc">%</span></span>')


    def sev_of(pct):
        return "good" if pct >= 90 else "warn" if pct >= 75 else "bad"


    _rej = rejected()
    _rejrows = "".join(
        f'<tr class="r-bad"><th>{E(r["model"])}</th>'
        f'<td><span class="pill p-unsuitable">{E(r.get("gate") or "")}</span></td>'
        f'<td class="detail">{E(r.get("detail") or "")}</td>'
        f'<td class="figure">{r.get("seconds") or 0:.0f}s</td></tr>' for r in _rej)
    rejsec = (f"""<section>
      <h2>Ruled Out Before Scoring</h2>
      <div class="lede"><p>Triage asks the cheapest decisive question first and stops at the
    first answer that ends the matter, so these models were rejected without paying for the
    full run. Each gate applies a rule the report applies anyway, so a model rejected here is
    one the full run would have rejected. Their measurements were discarded with them, which is
    why they appear in no other table: without this one, nothing would distinguish a model that
    failed from a model that was never run.</p></div>
      <div class="scroll"><table><thead><tr>
        <th>Model</th><th>Gate</th><th>Why</th><th class="figure">Cost</th>
      </tr></thead><tbody>{_rejrows}</tbody></table></div>
    </section>""" if _rej else "")

    rows_v = []
    for m in ranked:
        sv, why = verdict(m)
        r1 = t1.get(m, {}).get("rate")
        r2 = t2.get(m, {}).get("rate")
        d = prb.get(m, {})
        # The same figures the recommendation cards carry, so a model can be read
        # here without being looked up there. Speed score exists only for models
        # that reached scoring; task time needs a probe. Either may be absent, and
        # an absent figure is shown as one rather than as a zero.
        rows_v.append(f'''<article class="card c-{sv}">
      <header><h3>{E(m)}</h3><span class="pill p-{sv}">{sv}</span></header>
      <dl>
        <div><dt>Basic set</dt><dd>{num(r1,"%",0)}</dd></div>
        <div><dt>Hard set</dt><dd>{num(r2,"%",0)}</dd></div>
        <div><dt>Task time</dt><dd>{num(task_time(m),"s",0)}</dd></div>
        <div><dt>Speed score</dt><dd>{num(SC.get(m,{}).get("speed"),"",0)}</dd></div>
        <div><dt>Prefill @48k</dt><dd>{num(d.get("prefill_s"),"s",0)}</dd></div>
        <div><dt>Generation</dt><dd>{num(d.get("gen_tok_s")," t/s",0)}</dd></div>
      </dl>
      <p class="why">{E("; ".join(why)) if why else "No disqualifying signal."}</p>
    </article>''')

    # A model triage rejected belongs here too: it was assessed, and the verdict is
    # unsuitable. Its figures are the ones the gate that stopped it measured and no
    # others, because triage stops at the first answer that ends the matter -- so the
    # card states what was measured and says plainly that the rest was not, rather
    # than showing a dash that reads as a missing measurement.
    for r in _rej:
        rows_v.append(f'''<article class="card c-unsuitable">
      <header><h3>{E(r["model"])}</h3>
        <span class="pill p-unsuitable">unsuitable</span></header>
      <p class="role">Rejected at the {E(r.get("gate") or "")} gate</p>
      <dl>
        <div><dt>Gate</dt><dd>{E(r.get("gate") or "&mdash;")}</dd></div>
        <div><dt>Cost</dt><dd>{r.get("seconds") or 0:.0f}<span class="pc">s</span></dd></div>
      </dl>
      <p class="why">{E(r.get("detail") or "")}. Nothing beyond this gate was
      measured, and its records were discarded with it.</p>
    </article>''')


    def screen_table(agg_, label, total):
        if not agg_:
            return ""
        body = ""
        for m in sorted(agg_, key=lambda x: -agg_[x]["rate"]):
            a = agg_[m]
            sev = sev_of(a["rate"])
            ks = " ".join(f'<span class="kv"><i>{k[:4]}</i>{v[0]}/{v[1]}</span>'
                          for k, v in sorted(a["kind"].items()))
            sp = f' <span class="spread">±{a["spread"]:.0f}</span>' if a["spread"] else ""
            body += (f'<tr class="r-{sev}"><th>{E(m)}</th>'
                     f'<td class="barcell">{bar(a["rate"], sev)}{sp}</td>'
                     f'<td class="figure">{a["med"]:.1f}s</td>'
                     f'<td class="figure">{a["runs"]}</td><td class="kinds">{ks}</td></tr>')
        return f'''<section><h2>{label}</h2>
    <div class="lede"><p>Reports pass rates over {total} tasks per run, grouped by failure
    mode so that an aggregate score does not conceal which capability is deficient: code covers
    generation from a specification, edit the repair of existing source, form adherence to output
    constraints a harness must parse, tool the emission of valid calls with correctly typed arguments,
    and trac the prediction of program output, which isolates comprehension from generation. All runs use
    temperature 0 with a fixed seed, and results are graded by executing the produced code against
    assertions. Where a model was measured repeatedly, the spread between its best and worst run appears
    beside the mean and bounds how far apart two scores must lie to be meaningful.</p></div>
    <div class="scroll"><table><thead><tr><th>Model</th><th>Pass rate</th><th>Median/task</th>
    <th>Runs</th><th>By category</th></tr></thead><tbody>{body}</tbody></table></div></section>'''


    speed = ""
    if prb:
        rs = ""
        for m in sorted(prb, key=lambda x: (prb[x].get("prefill_s") or 9e9)):
            d = prb[m]
            pf = d.get("prefill_s") or 0
            ts = task_time(m)
            sev = ("good" if ts is None or ts <= SLOW_TASK_S
                   else "warn" if ts <= UNUSABLE_TASK_S else "bad")
            trunc = d.get("likely_truncated")
            ok = ('<span class="pill p-unsuitable">truncated</span>' if trunc
                  else '<span class="pill p-suitable">64k ok</span>' if d.get("retrieved")
                  else '<span class="pill p-limited">needle failed</span>')
            rs += (f'<tr class="r-{sev}"><th>{E(m)}</th>'
                   f'<td class="figure">{E(_qmap[m]) if _qmap.get(m) else "&mdash;"}</td>'
                   f'<td class="figure">{num(d.get("gen_tok_s"),"",1)}</td>'
                   f'<td class="figure">{num(d.get("prefill_tok_s"),"",0)}</td>'
                   f'<td class="figure strong">{num(pf,"s",1)}</td>'
                   f'<td class="figure">{num((d.get("placement") or {}).get("pct_gpu"),"%",0)}</td>'
                   f'<td>{ok}</td></tr>')
        speed = f'''<section><h2>Speed at Working Depth</h2>
    <div class="lede"><p>Measures latency at operational context depth using a single prompt
    of approximately 48,000 tokens at <code>num_ctx=65536</code>, since short benchmark prompts do not
    reproduce the conditions under which a coding harness operates. Prefill is the interval before the
    first output token and dominates total response time whenever a model exceeds available VRAM;
    generation throughput applies only thereafter. The final column reports a correctness check rather
    than a performance figure: Ollama discards prompt overflow without error, so tokens processed are
    compared against tokens submitted and a planted fact is retrieved to confirm the context was
    retained. Truncation removes the earliest context first, which in a harness comprises the system
    prompt and tool definitions.</p></div>
    <div class="scroll"><table><thead><tr><th>Model</th><th>Quantisation</th><th>Gen tok/s</th>
    <th>Prefill tok/s</th><th>Prefill time</th><th>On GPU</th><th>64k behaviour</th></tr></thead>
    <tbody>{rs}</tbody></table></div></section>'''

    cachesec = ""
    if cch:
        rs = ""
        for m in sorted(cch):
            d = cch[m]
            hit = d.get("append_cache_hit")
            rs += (f'<tr class="r-{"good" if hit else "bad"}"><th>{E(m)}</th>'
                   f'<td class="figure">{num(d.get("t1_prefill_s"),"s")}</td>'
                   f'<td class="figure strong">{num(d.get("t2_prefill_s"),"s",2)}</td>'
                   f'<td class="figure">{num(d.get("t4_prefill_s"),"s")}</td>'
                   f'<td><span class="pill p-{"suitable" if hit else "unsuitable"}">'
                   f'{"reused" if hit else "no reuse"}</span></td></tr>')
        cachesec = f'''<section><h2>Prefix Cache Reuse</h2>
    <div class="lede"><p>Determines what proportion of the prefill cost reported above is
    incurred on every turn rather than once per session, which decides whether multi-turn agent use is
    practical at these latencies. Turn 1 submits deep context cold, turn 2 appends a follow-up leaving
    the prefix unchanged, and turn 4 modifies text near the start, invalidating the cache from that
    point. Reported values are measured prefill duration; token counts include cached tokens and do not
    indicate reuse. The outcome is a property of the harness rather than the model, as prompts that grow
    by appending retain the cached prefix while prompts re-rendered each turn do not.</p></div>
    <div class="scroll"><table><thead><tr><th>Model</th><th>Turn 1 (cold)</th><th>Turn 2 (append)</th>
    <th>Turn 4 (early edit)</th><th>Verdict</th></tr></thead><tbody>{rs}</tbody></table></div></section>'''

    agentsec = ""
    if ag:
        rs = ""
        for m, v in sorted(ag.items(), key=lambda kv: -sum(r["passed"] for r in kv[1])):
            passed = sum(1 for r in v if r["passed"])
            sev = "good" if passed == len(v) else "bad" if passed == 0 else "warn"

            def chip(r):
                # A graded task carries its fraction on the chip: "8/18" and "0/18"
                # are both failures, and collapsing them would discard the difference.
                label = r["task"].replace("ag_", "")
                checks = r.get("checks") or []
                if checks:
                    label += f" {sum(1 for c in checks if c['passed'])}/{len(checks)}"
                tries = r.get("attempts", 1)
                if tries > 1:
                    # A task attempted more than once is the only one whose result
                    # carries weight in the verdict, so the count is shown.
                    label += f" [{r.get('completed', 0)}/{tries}]"
                cls = ("ok" if r["passed"]
                       else "part" if (r.get("score") or 0) >= 50 else "no")
                title = r.get("detail") or ""
                if tries > 1:
                    title = f"{r.get('completed', 0)} of {tries} attempts completed. {title}"
                return (f'''<span class="tchip {cls}" title="{E(title)}">'''
                        f'''{E(label)} {r.get("wall_s") or 0:.0f}s</span>''')

            chips = "".join(chip(r) for r in sorted(v, key=lambda r: r["task"]))
            wall = sum(r.get("wall_s") or 0 for r in v)
            tools = sum(r.get("tool_calls") or 0 for r in v)
            turns = sum(r.get("turns") or 0 for r in v)
            rs += (f'''<tr class="r-{sev}"><th>{E(m)}</th>
                   <td class="figure strong">{passed}/{len(v)}</td>
                   <td><div class="tasks">{chips}</div></td>
                   <td class="figure">{wall:.0f}s</td>
                   <td class="figure">{tools}</td>
                   <td class="figure">{turns}</td></tr>''')
        agentsec = f'''<section><h2>Agent Task Completion</h2>
    <div class="lede"><p>Evaluates task completion under a real agent harness, which
    single-turn scoring cannot assess. Each task seeds an isolated repository containing a defect, and
    opencode drives the model through it with tool access until it terminates or times out. Grading
    inspects the resulting files and executes the repository's own checks; claims of success in the
    model's output are disregarded, and files a task designates immutable are hashed so that modifying
    them in place of the source fails the task outright. Results indicate whether a model completes work
    unattended, which may diverge from its single-prompt scores.</p>
    <p>The application task is scored per requirement rather than pass or fail: it asks for an
    installable package with a storage layer, an eleven-route API, a web page and the model's own test
    suite, and a single verdict over that much surface would report only that something was missing.
    Its chip carries the fraction of requirements met, and hovering any chip names the ones that were
    not. A task attempted more than once shows how many of those attempts it completed, in square
    brackets, and only a task that failed every attempt made at it counts against the verdict: this
    stage is more variable than the screen, and the screen does not report a rate from a single run
    either. A single failure is reported without lowering anything. The finished applications are kept under <code>results/agent_apps/</code>, with a screenshot
    of each interface, so they can be read rather than only counted.</p></div>
    <div class="scroll"><table><thead><tr><th>Model</th><th>Passed</th><th>Tasks</th>
    <th>Total time</th><th>Tool calls</th><th>Turns</th></tr></thead>
    <tbody>{rs}</tbody></table></div></section>'''

    keep = [m for m in ranked if verdict(m)[0] == "suitable"]
    def build_rec():
        QUAL = [m for m in SC_RANK if SC[m]["above_floor"]]
        if not SC_RANK:
            return ("<section><h2>Recommendation</h2><div class=\"lede\"><p>No model cleared the "
                    "eligibility gates, so no recommendation can be derived.</p></div></section>")
        pool = QUAL or SC_RANK
        best_overall = pool[0]
        best_quality = max(pool, key=lambda m: (SC[m]["quality"], -SC[m]["prefill"]))
        fastest = min(pool, key=lambda m: SC[m]["session"])
        picks = []
        for role, m, basis in (
                ("Best balance of quality and speed", best_overall,
                 f"composite {SC[best_overall]['composite']}"),
                ("Highest quality", best_quality,
                 f"{SC[best_quality]['quality']:.0f}% on the hard set"),
                ("Shortest modelled task time at acceptable quality", fastest,
                 f"{SC[fastest]['session']:.0f}s per task at {SC[fastest]['quality']:.0f}% quality")):
            picks.append(f'''<article class="card c-suitable">
          <header><h3>{E(m)}</h3><span class="pill p-suitable">pick</span></header>
          <p class="role">{role}</p>
          <dl>
            <div><dt>Quality</dt><dd>{SC[m]["quality"]:.0f}<span class="pc">%</span></dd></div>
            <div><dt>Speed score</dt><dd>{SC[m]["speed"]:.0f}</dd></div>
            <div><dt>Task time</dt><dd>{SC[m]["session"]:.0f}<span class="pc">s</span></dd></div>
            <div><dt>Generation</dt><dd>{SC[m]["gen"]:.0f}<span class="pc">t/s</span></dd></div>
          </dl>
          <p class="why">Selected on {basis}.</p>
        </article>''')
        rows = ""
        for i, m in enumerate(SC_RANK, 1):
            v = SC[m]
            flag = "" if v["above_floor"] else ' <span class="pill p-unsuitable">below floor</span>'
            rows += (f'''<tr class="r-{"good" if i == 1 and v["above_floor"] else "warn" if v["above_floor"] else "bad"}">
                <td class="figure">{i}</td><th>{E(m)}{flag}</th>
                <td class="barcell">{bar(v["composite"], "good" if i == 1 else "warn")}</td>
                <td class="figure">{v["quality"]:.0f}%</td>
                <td class="figure">{v["speed"]:.0f}</td>
                <td class="figure strong">{v["session"]:.0f}s</td>
                <td class="figure">{v["prefill"]:.0f}s</td>
                <td class="figure">{v["gen"]:.0f} t/s</td></tr>''')
        excluded = ""
        for m in SHORTLIST:
            f = eligible(m)
            if f and (m in t1 or m in t2):
                excluded += f'''<tr class="r-bad"><th>{E(m)}</th>
                    <td class="detail">{E("; ".join(f))}</td></tr>'''
        exsec = (f'''<h3 class="sub-h">Excluded Before Scoring</h3>
          <div class="scroll"><table><thead><tr><th>Model</th><th>Gate not met</th></tr></thead>
          <tbody>{excluded}</tbody></table></div>''' if excluded else "")
        return f'''<section>
      <h2>Recommendation</h2>
      <div class="lede"><p>Ranks eligible models by a composite of quality and speed. Models that
    fail to emit a parseable tool call, truncate context, fail retrieval at depth, or take longer than
    {UNUSABLE_TASK_S:.0f}s per task are excluded before scoring, as those failures end a session rather
    than degrade it. Emitting a well-formed call to the wrong tool is graded instead: the harness returns
    the wrong result and the model gets another turn. The remainder score {W_QUALITY*100:.0f}% quality, taken
    as the hard-set pass rate, and {W_SPEED*100:.0f}% speed. Speed is the modelled time to complete one representative
    task of {TASK_TOKENS:.0f} output tokens at the model's generation rate, expressed as a ratio to the
    fastest model, so one taking twice as long scores half; that token volume and the {TASK_TURNS:.0f}
    turns it spans are the medians observed in the agent runs below, so the assumed session size is
    stated rather than implied by a weighting. Speed earns nothing below a quality floor of {QFLOOR:.0f}%,
    set 20 points below the highest quality measured, so below it the composite is quality alone. That
    floor is not a nicety: a model that fails most of its tasks finishes them quickly, and unconditional
    latency credit would rank it above models that actually produce working code. A model below the
    floor therefore cannot outrank one above it, whatever its speed, and its speed column is still shown
    so the trade being refused is visible. A higher composite indicates a better overall trade rather than superiority
    on any single axis. {SATNOTE}</p></div>
      <div class="cards">{"".join(picks)}</div>
      <h3 class="sub-h">Full Ranking</h3>
      <div class="scroll"><table><thead><tr><th>#</th><th>Model</th><th>Composite</th><th>Quality</th>
      <th>Speed</th><th>Task time</th><th>Prefill</th><th>Generation</th></tr></thead><tbody>{rows}</tbody></table></div>
      {exsec}
    </section>'''


    gen = dt.datetime.now().strftime("%d %B %Y, %H:%M")

    ENVLINE = " · ".join(filter(None, [
        "Ollama",
        (f"remote host {OLLAMA}" if REMOTE else
         (", ".join(_g) if _g else "GPU not detected"))]))
    CTXVALS = sorted({r.get("num_ctx") for r in probe if r.get("num_ctx")} |
                     {r.get("ctx") for r in screen if r.get("ctx")})
    CTXSTR = ", ".join(f"{c:,}" for c in CTXVALS) if CTXVALS else "not recorded"
    CTXMAX = f"{max(CTXVALS):,}-token" if CTXVALS else "the configured"
    NTASKS = sorted({r["n"] for r in screen if r.get("n", 0) >= 10})
    NRUNS = sorted({v["runs"] for v in list(t1.values()) + list(t2.values())}) or [0]
    HWNOTE = (f"measured on {', '.join(_g)}" if _g
              else (f"measured against the Ollama server at {OLLAMA}" if REMOTE
                    else "measured on the host's GPU"))
    QNOTE = (f"{', '.join(_q)} builds" if _q else "the locally installed builds")

    _sat_t2 = sum(1 for v in t2.values() if v["rate"] >= 100)
    _sat_t1 = sum(1 for v in t1.values() if v["rate"] >= 100)
    _maxspread = max([v["spread"] for v in list(t1.values()) + list(t2.values())] or [0])
    SATNOTE = (
        f"{_sat_t2} models reached the maximum hard-set score, so the suite does not resolve their "
        f"relative standing; separating them requires a deeper benchmark than this screen provides."
        if _sat_t2 > 1 else
        f"Run-to-run variation reached {_maxspread:.0f} points, which bounds the smallest difference "
        f"this screen can resolve." if _maxspread else
        "Differences smaller than the run-to-run variation shown in the tables are not resolved here.")

    RECSEC = build_rec()

    html = f'''<title>Local Coding Model Evaluation</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500&display=swap">
    <style>
    :root {{
      --paper:#f6f7f9; --raised:#ffffff; --ink:#0f1319; --muted:#5b6472; --line:#dfe3ea;
      --accent:#3a6ea5; --accent-soft:#e8eef6;
      --good:#2f7d4f; --warn:#b07d1a; --bad:#b4453a;
      --good-bg:#e9f3ec; --warn-bg:#f8f1e0; --bad-bg:#f8ebe9;
      --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
      --sans:"IBM Plex Sans",system-ui,-apple-system,sans-serif;
      --display:"Archivo","IBM Plex Sans",system-ui,sans-serif;
    }}
    @media (prefers-color-scheme:dark) {{
      :root:not([data-theme="light"]) {{
        --paper:#0f1319; --raised:#161b23; --ink:#e6eaf0; --muted:#8d97a6; --line:#262d38;
        --accent:#7aa9dc; --accent-soft:#1a2531;
        --good:#63b585; --warn:#d6a548; --bad:#e0776b;
        --good-bg:#15251c; --warn-bg:#2a2113; --bad-bg:#2b1917;
      }}
    }}
    :root[data-theme="dark"] {{
      --paper:#0f1319; --raised:#161b23; --ink:#e6eaf0; --muted:#8d97a6; --line:#262d38;
      --accent:#7aa9dc; --accent-soft:#1a2531;
      --good:#63b585; --warn:#d6a548; --bad:#e0776b;
      --good-bg:#15251c; --warn-bg:#2a2113; --bad-bg:#2b1917;
    }}
    * {{ box-sizing:border-box; }}
    body {{ background:var(--paper); color:var(--ink); font-family:var(--sans);
      line-height:1.6; margin:0; padding:clamp(1.5rem,4vw,4rem) clamp(1rem,4vw,2rem); }}
    .wrap {{ max-width:74rem; margin:0 auto; display:flex; flex-direction:column; gap:3.5rem; }}
    h1,h2,h3 {{ font-family:var(--display); text-wrap:balance; margin:0; letter-spacing:-.015em; }}
    h1 {{ font-size:clamp(2rem,5vw,3rem); font-weight:700; line-height:1.05; }}
    h2 {{ font-size:1.4rem; font-weight:600; padding-bottom:.6rem; border-bottom:2px solid var(--accent);
      margin-bottom:.25rem; }}
    h3 {{ font-size:1rem; font-weight:600; font-family:var(--mono); letter-spacing:-.02em; }}
    .eyebrow {{ font-family:var(--mono); font-size:.7rem; text-transform:uppercase;
      letter-spacing:.16em; color:var(--accent); font-weight:600; margin:0 0 .9rem; }}
    .lede {{ color:var(--muted); margin:.35rem 0 1.25rem; }}
    .lede p {{ margin:0 0 .7rem; }}
    .lede p:last-child {{ margin-bottom:0; }}
    .tasks {{ display:flex; gap:.35rem; flex-wrap:wrap; }}
    .tchip {{ font-family:var(--mono); font-size:.68rem; padding:.14rem .42rem; border-radius:2px;
      white-space:nowrap; }}
    .tchip.ok {{ background:var(--good-bg); color:var(--good); }}
    .tchip.no {{ background:var(--bad-bg); color:var(--bad); }}
    .tchip.part {{ background:var(--warn-bg); color:var(--warn); }}
    header.top p.sub {{ color:var(--muted); max-width:60ch; font-size:1.05rem; margin:1rem 0 0; }}
    .meta {{ display:flex; flex-wrap:wrap; gap:.5rem 1.75rem; font-family:var(--mono);
      font-size:.76rem; color:var(--muted); margin-top:1.5rem; padding-top:1.25rem;
      border-top:1px solid var(--line); }}
    .meta b {{ color:var(--ink); font-weight:600; }}
    section {{ display:flex; flex-direction:column; }}
    .cards {{ display:grid; gap:1rem; grid-template-columns:repeat(auto-fill,minmax(15.5rem,1fr)); }}
    .card {{ background:var(--raised); border:1px solid var(--line); border-radius:3px;
      border-left:4px solid var(--muted); padding:1.1rem 1.2rem; display:flex;
      flex-direction:column; gap:.85rem; }}
    .card.c-suitable {{ border-left-color:var(--good); }}
    .card.c-limited {{ border-left-color:var(--warn); }}
    .card.c-unsuitable {{ border-left-color:var(--bad); }}
    .card header {{ display:flex; align-items:center; justify-content:space-between; gap:.5rem; }}
    .card dl {{ margin:0; display:grid; grid-template-columns:1fr 1fr; gap:.6rem .75rem; }}
    .card dl div {{ display:flex; flex-direction:column; gap:.1rem; }}
    dt {{ font-size:.68rem; text-transform:uppercase; letter-spacing:.09em; color:var(--muted); }}
    dd {{ margin:0; font-family:var(--mono); font-size:1.15rem; font-weight:500;
      font-variant-numeric:tabular-nums; }}
    .role {{ margin:0; font-size:.85rem; color:var(--ink); font-weight:500; line-height:1.35; }}
    .sub-h {{ font-family:var(--sans); font-size:.78rem; font-weight:600; text-transform:uppercase;
      letter-spacing:.09em; color:var(--muted); margin:2.25rem 0 .8rem; }}
    .why {{ margin:0; font-size:.8rem; color:var(--muted); border-top:1px solid var(--line);
      padding-top:.7rem; }}
    .pill {{ font-family:var(--mono); font-size:.66rem; text-transform:uppercase; letter-spacing:.08em;
      padding:.2rem .5rem; border-radius:2px; font-weight:600; white-space:nowrap; }}
    .p-suitable {{ background:var(--good-bg); color:var(--good); }}
    .p-limited {{ background:var(--warn-bg); color:var(--warn); }}
    .p-unsuitable {{ background:var(--bad-bg); color:var(--bad); }}
    .scroll {{ overflow-x:auto; border:1px solid var(--line); border-radius:3px;
      background:var(--raised); }}
    table {{ border-collapse:collapse; width:100%; font-size:.85rem; }}
    th,td {{ text-align:left; padding:.62rem .85rem; border-bottom:1px solid var(--line);
      white-space:nowrap; }}
    tbody tr:first-child th, tbody tr:first-child td {{ padding-top:.75rem; }}
    thead th {{ font-family:var(--sans); font-size:.74rem; text-transform:uppercase;
      letter-spacing:.07em; color:var(--ink); font-weight:600; background:var(--accent-soft);
      padding:.85rem .85rem .7rem; border-bottom:2px solid var(--accent);
      vertical-align:bottom; }}
    tbody th {{ font-family:var(--mono); font-weight:500; font-size:.82rem; }}
    tbody tr:last-child td, tbody tr:last-child th {{ border-bottom:none; }}
    tbody tr {{ border-left:3px solid transparent; }}
    tr.r-good th {{ box-shadow:inset 3px 0 0 var(--good); }}
    tr.r-warn th {{ box-shadow:inset 3px 0 0 var(--warn); }}
    tr.r-bad th {{ box-shadow:inset 3px 0 0 var(--bad); }}
    .figure {{ font-family:var(--mono); font-variant-numeric:tabular-nums; }}
    .strong {{ font-weight:600; }}
    .pc {{ font-size:.72em; color:var(--muted); }}
    .barcell {{ display:flex; align-items:center; gap:.6rem; min-width:11rem; }}
    .bar {{ flex:1; height:6px; background:var(--accent-soft); border-radius:1px; overflow:hidden;
      min-width:5rem; }}
    .bar span {{ display:block; height:100%; }}
    .s-good {{ background:var(--good); }} .s-warn {{ background:var(--warn); }}
    .s-bad {{ background:var(--bad); }}
    .spread {{ font-family:var(--mono); font-size:.72rem; color:var(--muted); }}
    .kinds {{ display:flex; gap:.4rem; flex-wrap:wrap; }}
    .kv {{ font-family:var(--mono); font-size:.7rem; color:var(--muted);
      background:var(--accent-soft); padding:.12rem .4rem; border-radius:2px; }}
    .kv i {{ font-style:normal; color:var(--accent); font-weight:600; margin-right:.3rem; }}
    .detail {{ font-family:var(--mono); font-size:.72rem; color:var(--muted);
      white-space:normal; max-width:22rem; }}
    code {{ font-family:var(--mono); font-size:.88em; background:var(--accent-soft);
      padding:.1rem .3rem; border-radius:2px; }}
    .verdict {{ background:var(--raised); border:1px solid var(--line); border-radius:3px;
      padding:1.4rem 1.6rem; }}
    .verdict ul {{ margin:.6rem 0 0; padding-left:1.1rem; }}
    .verdict li {{ margin-bottom:.45rem; }}
    .caveats {{ color:var(--muted); font-size:.87rem; }}
    .caveats li {{ margin-bottom:.5rem; }}
    footer {{ border-top:1px solid var(--line); padding-top:1.25rem; color:var(--muted);
      font-family:var(--mono); font-size:.72rem; }}
    @media (max-width:34rem) {{ .card dl {{ grid-template-columns:1fr; }} }}
    </style>

    <div class="wrap">
    <header class="top">
      <p class="eyebrow">{E(ENVLINE)}</p>
      <h1>Local Coding Model Evaluation</h1>
      <p class="sub">A screening evaluation of {len(ranked)} models covering code correctness,
      behaviour at a {CTXMAX} context, and response latency. Its purpose is to establish which models
      merit a deeper benchmark and to exclude those that fail outright. It is not a ranking of the
      leading candidates.</p>
      <div class="meta">
        <span><b>{len(ranked)}</b> models</span>
        <span><b>{sum(len(v) for v in by_set["basic"].values())}</b> basic runs</span>
        <span><b>{sum(len(v) for v in by_set["hard"].values())}</b> hard runs</span>
        <span>context <b>{E(CTXSTR)}</b></span>
        <span>temperature <b>0</b>, seed <b>1</b></span>
        <span>tasks in <b>Python</b></span>
        <span>server <b>{E(OLLAMA)}</b></span>
        <span>generated <b>{gen}</b></span>
      </div>
    </header>

    {RECSEC}

    <section>
      <h2>Assessment</h2>
      <div class="lede"><p>Classifies each model by the nature of its shortcomings rather than
    by score alone. Unsuitable denotes a failure that ends the model's usefulness outright: a tool call
    that could not be parsed or was never emitted, context truncation, failed retrieval at depth, or a
    generation rate below
    {MIN_GEN_TOK_S:.0f} tok/s, or a modelled task time above {UNUSABLE_TASK_S:.0f}s. The generation floor is
    what a model too large for available VRAM trips: a mixture of experts moves a few billion parameters
    per token however little of it is resident, while a dense model moves all of them across the bus, and
    the two separate by a wide margin on this measure. Limited denotes
    a model that clears those but falls below a graded threshold, either under 85% on the hard set or a
    task time above {SLOW_TASK_S:.0f}s. Both latency thresholds are absolute rather than relative to the
    fastest model measured, so a verdict describes how a model behaves on this machine and does not
    change when a quicker model joins the comparison. Each card states the threshold breached and the
    value that breached it.</p></div>
      <div class="cards">{"".join(rows_v)}</div>
    </section>

    {rejsec}

    {screen_table(t2, "Hard Set", 15)}
    {screen_table(t1, "Basic Set", 14)}
    {speed}
    {cachesec}
    {agentsec}



    <footer>Generated from results/*.jsonl &middot; codesift &middot; {gen}</footer>
    </div>
    '''

    return html


def run(cfg: Config, output: Path, models: list[str] | None = None) -> Path:
    if models is None:
        seen = []
        for name in ("screen.jsonl", "probe.jsonl", "prefix_cache.jsonl", "agentic.jsonl"):
            for rec in _load(cfg.results_dir, name):
                if rec.get("model") and rec["model"] not in seen:
                    seen.append(rec["model"])
        models = seen
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # build() lives inside a function, so its template carries one indent level
    html = "\n".join(line[4:] if line.startswith("    ") else line
                     for line in build(cfg, models).splitlines())
    output.write_text(html, encoding="utf-8")
    return output
