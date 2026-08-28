# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""Renders an assessment as one self-contained HTML page.

The page carries its own stylesheet and no asset but the font stylesheet, so the
markup and the CSS are constants here and each section is a function of the
assessment. Nothing decides anything: what a model is worth was settled in
`analysis`, and this says it.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

from . import args, ledger
from .analysis import ANSWER_TOKENS, MIN_GEN_TOK_S, SESSION_TURNS, analyse
from .config import Config, DEFAULT_HOST
from .findings import describe, sentence


class Markup(str):
    """Text that is already HTML, and must not be escaped a second time."""


def esc(value) -> Markup:
    """Whatever is put on the page, safe to interpolate.

    Markup built here passes through; anything else -- a model name off the
    server, a finding, a task id -- is escaped. Everything the page shows goes
    through this, so forgetting to escape is not one of the ways to be wrong.
    """
    if isinstance(value, Markup):
        return value
    return Markup(str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def num(v, suf="", d=1) -> Markup:
    """A figure, or a dash where there is no measurement to show."""
    return Markup(f"{v:.{d}f}{suf}" if isinstance(v, (int, float)) else "&mdash;")


def _vram(place: dict) -> Markup:
    """Memory the loaded model occupies, weights and context cache together."""
    total = place.get("total_gb")
    return Markup("&mdash;" if total is None else f"{total:.1f}")


def bar(pct, sev, unit="%", extra="") -> Markup:
    """A filled bar and its figure.

    The fill is neutral: the verdict has its own pill and its own row stripe,
    and tinting the numbers would read as a judgement on the figure.
    """
    w = max(0, min(100, pct or 0))
    suffix = f'<span class="pc">{unit}</span>' if unit else ""
    return Markup(f'<div class="barwrap">'
                  f'<div class="bar"><span style="width:{w:.0f}%" class="s-{sev}"></span></div>'
                  f'<span class="figure">{pct:.0f}{suffix}</span>{extra}</div>')


# The order the breakdown is read in, fixed so a column of figures lines up down
# the table whether or not a model has tasks of that kind.
KIND_ORDER = ("codegen", "edit", "format", "toolcall", "trace")


STYLE = """:root {
  --paper:#f6f7f9; --raised:#ffffff; --ink:#0f1319; --muted:#5b6472; --line:#dfe3ea;
  --accent:#3a6ea5; --accent-soft:#e8eef6;
  --good:#2f7d4f; --warn:#b07d1a; --bad:#b4453a;
  --good-bg:#e9f3ec; --warn-bg:#f8f1e0; --bad-bg:#f8ebe9;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  --sans:"IBM Plex Sans",system-ui,-apple-system,sans-serif;
  --display:"Archivo","IBM Plex Sans",system-ui,sans-serif;
}
@media (prefers-color-scheme:dark) {
  :root:not([data-theme="light"]) {
    --paper:#0f1319; --raised:#161b23; --ink:#e6eaf0; --muted:#8d97a6; --line:#262d38;
    --accent:#7aa9dc; --accent-soft:#1a2531;
    --good:#63b585; --warn:#d6a548; --bad:#e0776b;
    --good-bg:#15251c; --warn-bg:#2a2113; --bad-bg:#2b1917;
  }
}
:root[data-theme="dark"] {
  --paper:#0f1319; --raised:#161b23; --ink:#e6eaf0; --muted:#8d97a6; --line:#262d38;
  --accent:#7aa9dc; --accent-soft:#1a2531;
  --good:#63b585; --warn:#d6a548; --bad:#e0776b;
  --good-bg:#15251c; --warn-bg:#2a2113; --bad-bg:#2b1917;
}
* { box-sizing:border-box; }
body { background:var(--paper); color:var(--ink); font-family:var(--sans);
  line-height:1.6; margin:0; padding:clamp(1.5rem,4vw,4rem) clamp(1rem,4vw,2rem); }
.wrap { max-width:74rem; margin:0 auto; display:flex; flex-direction:column; gap:3.5rem; }
h1,h2,h3 { font-family:var(--display); text-wrap:balance; margin:0; letter-spacing:-.015em; }
h1 { font-size:clamp(2rem,5vw,3rem); font-weight:700; line-height:1.05; }
h2 { font-size:1.4rem; font-weight:600; padding-bottom:.6rem; border-bottom:2px solid var(--accent);
  margin-bottom:.25rem; }
h3 { font-size:1rem; font-weight:600; font-family:var(--mono); letter-spacing:-.02em; }
.eyebrow { font-family:var(--mono); font-size:.7rem; text-transform:uppercase;
  letter-spacing:.16em; color:var(--accent); font-weight:600; margin:0 0 .9rem; }
.lede { color:var(--muted); margin:.35rem 0 1.25rem; }
.lede p { margin:0 0 .7rem; }
.lede p:last-child { margin-bottom:0; }
.tasks { display:flex; gap:.35rem; flex-wrap:wrap; }
.tchip { font-family:var(--mono); font-size:.68rem; padding:.14rem .42rem; border-radius:2px;
  white-space:nowrap; }
.tchip.ok { background:var(--good-bg); color:var(--good); }
.tchip.no { background:var(--bad-bg); color:var(--bad); }
.tchip.part { background:var(--warn-bg); color:var(--warn); }
header.top p.sub { color:var(--muted); font-size:1.05rem; margin:1rem 0 0; }
.meta { display:flex; flex-wrap:wrap; gap:.5rem 1.75rem; font-family:var(--mono);
  font-size:.76rem; color:var(--muted); margin-top:1.5rem; padding-top:1.25rem;
  border-top:1px solid var(--line); }
.meta b { color:var(--ink); font-weight:600; }
section { display:flex; flex-direction:column; }
.cards { display:grid; gap:1rem; grid-template-columns:repeat(auto-fill,minmax(15.5rem,1fr)); }
.card { background:var(--raised); border:1px solid var(--line); border-radius:3px;
  border-left:4px solid var(--muted); padding:1.1rem 1.2rem; display:flex;
  flex-direction:column; gap:.85rem; }
.card.c-suitable { border-left-color:var(--good); }
.card.c-limited { border-left-color:var(--warn); }
.card.c-unsuitable { border-left-color:var(--bad); }
.card header { display:flex; align-items:center; justify-content:space-between; gap:.5rem; }
.card dl { margin:0; display:grid; grid-template-columns:1fr 1fr; gap:.6rem .75rem; }
.card dl div { display:flex; flex-direction:column; gap:.1rem; }
dt { font-size:.68rem; text-transform:uppercase; letter-spacing:.09em; color:var(--muted); }
dd { margin:0; font-family:var(--mono); font-size:1.15rem; font-weight:500;
  font-variant-numeric:tabular-nums; }
.role { margin:0; font-size:.85rem; color:var(--ink); font-weight:500; line-height:1.35; }
.why { margin:0; font-size:.8rem; color:var(--muted); border-top:1px solid var(--line);
  padding-top:.7rem; }
.pill { font-family:var(--mono); font-size:.66rem; text-transform:uppercase; letter-spacing:.08em;
  padding:.2rem .5rem; border-radius:2px; font-weight:600; white-space:nowrap; }
.p-suitable { background:var(--good-bg); color:var(--good); }
.p-limited { background:var(--warn-bg); color:var(--warn); }
.p-unsuitable { background:var(--bad-bg); color:var(--bad); }
.p-unmeasured { background:var(--accent-soft); color:var(--muted); }
.scroll { overflow-x:auto; border:1px solid var(--line); border-radius:3px;
  background:var(--raised); }
table { border-collapse:collapse; width:100%; font-size:.85rem; }
th,td { text-align:left; padding:.62rem .85rem; border-bottom:1px solid var(--line);
  white-space:nowrap; }
tbody tr:first-child th, tbody tr:first-child td { padding-top:.75rem; }
thead th { font-family:var(--sans); font-size:.74rem; text-transform:uppercase;
  letter-spacing:.07em; color:var(--ink); font-weight:600; background:var(--accent-soft);
  padding:.85rem .85rem .7rem; border-bottom:2px solid var(--accent);
  vertical-align:bottom; }
tbody th { font-family:var(--mono); font-weight:500; font-size:.82rem; }
tbody tr:last-child td, tbody tr:last-child th { border-bottom:none; }
tbody tr { border-left:3px solid transparent; }
tr.r-good th { box-shadow:inset 3px 0 0 var(--good); }
tr.r-warn th { box-shadow:inset 3px 0 0 var(--warn); }
tr.r-bad th { box-shadow:inset 3px 0 0 var(--bad); }
.figure { font-family:var(--mono); font-variant-numeric:tabular-nums; }
.strong { font-weight:600; }
.pc { font-size:.72em; color:var(--muted); }
/* The unit belongs to the column, so it is stated once in the header rather than
   repeated in every cell. */
.unit { font-weight:400; text-transform:none; letter-spacing:0; color:var(--muted); }
/* The flex box is inside the cell, never the cell itself: `display:flex` on a
   <td> takes it out of table layout, and two such cells side by side are
   wrapped into one anonymous cell and stack. */
.barwrap { display:flex; align-items:center; gap:.6rem; }
/* Everything beside the bar reserves its width, so the bar gets the same share
   of every cell in the column and all the bars match. The bar itself is not
   pinned, which would make the column unshrinkable. */
.barwrap .figure, .barwrap .spread { flex:none; text-align:right; }
.barwrap .figure { min-width:3.4rem; }
.barwrap .spread { min-width:2.2rem; }
.bar { flex:1 1 auto; min-width:2rem; max-width:7rem; height:.4rem;
  background:var(--accent-soft); border-radius:1px; overflow:hidden; }
.bar span { display:block; height:100%; }
.s-good { background:var(--good); } .s-warn { background:var(--warn); }
.s-bad { background:var(--bad); }
.s-plain { background:var(--accent); }
.spread { font-family:var(--mono); font-size:.72rem; color:var(--muted); }
.kindwrap { display:flex; gap:.4rem; }
.kv { font-family:var(--mono); font-size:.7rem; color:var(--muted);
  background:var(--accent-soft); padding:.12rem .4rem; border-radius:2px;
  font-variant-numeric:tabular-nums; }
.kv i { font-style:normal; color:var(--accent); font-weight:600; margin-right:.3rem; }
.kv b { display:inline-block; min-width:3ch; text-align:right; font-weight:inherit; }
.detail { font-family:var(--mono); font-size:.72rem; color:var(--muted);
  white-space:normal; max-width:18rem; }
code { font-family:var(--mono); font-size:.88em; background:var(--accent-soft);
  padding:.1rem .3rem; border-radius:2px; }
.verdict { background:var(--raised); border:1px solid var(--line); border-radius:3px;
  padding:1.4rem 1.6rem; }
.verdict ul { margin:.6rem 0 0; padding-left:1.1rem; }
.verdict li { margin-bottom:.45rem; }
.caveats { color:var(--muted); font-size:.87rem; }
.caveats li { margin-bottom:.5rem; }
@media (max-width:34rem) { .card dl { grid-template-columns:1fr; } }
"""


PAGE = """<title>Local Coding Model Evaluation</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500&display=swap">
<style>
{style}</style>

<div class="wrap">
<header class="top">
  <p class="eyebrow">{envline}</p>
  <h1>Local Coding Model Evaluation</h1>
  <p class="sub">A screening evaluation covering code correctness, behaviour at
  {ctxphrase}, and the speed of a coding session. Its purpose is to establish which models
  merit a deeper benchmark.</p>
  <div class="meta">
    <span>generated <b>{generated}</b></span>
  </div>
</header>

{recommendation}

{pass_rate}
{speed}
</div>
"""


def _rejected(cfg, models):
    """Models triage ruled out, which the stages after it never measured.

    A reader cannot otherwise tell whether a model is missing because it
    failed, because it was never run, or because it was never installed -- and
    the first of those is a finding.

    The newest record for a model is its verdict: the ledger outlives the rules
    that wrote it, and a model rejected last week and cleared today is cleared.
    Bound by the same shortlist as every other section, since one page describes
    one field.
    """
    wanted = set(models)
    latest = ledger.keyed(cfg.results_dir / "triage.jsonl", lambda rec: rec["model"])
    return [rec for m, rec in sorted(latest.items())
            if m in wanted and not rec.get("passed")]

def _pass_rate(A):
    """One table over every task, whichever capability it exercises."""
    models = sorted(A.agg, key=lambda m: -A.sc.get(m, {}).get("quality", 0))
    if not models:
        return ""
    body = ""
    for m in models:
        a = A.agg[m]
        rate, kind, med = a["rate"], a["kind"], a["med"]
        # Percentages, since the categories hold different numbers of tasks;
        # the counts behind each figure stay available on hover. Every category
        # is rendered at a fixed width and in a fixed order, including one a
        # model has no tasks for, so a column of them can be read down.
        ks = ""
        for k in KIND_ORDER:
            v = kind.get(k)
            have = bool(v and v["n"])
            title = (f"{v['score']:.1f} of {v['n']} tasks met" if have
                     else "no tasks of this kind")
            figure = f"{100 * v['score'] / v['n']:.0f}" if have else "&mdash;"
            ks += (f'<span class="kv" title="{title}">'
                   f'<i>{k[:4]}</i><b>{figure}</b></span>')
        body += (f'<tr class="r-plain"><th>{esc(m)}</th>'
                 f'<td class="barcell">{bar(rate, "plain", unit="")}</td>'
                 f'<td class="figure">{med:.1f}</td>'
                 f'<td class="kinds"><div class="kindwrap">{ks}</div></td></tr>')
    return f'''<section><h2>Pass Rate</h2>
<div class="lede"><p>Python coding tasks, highest first. Each answer is executed against
assertions rather than compared to a reference, at temperature 0, once per task, and scores
the share of them it meets. The categories: <code>code</code> writes from a specification,
<code>edit</code> repairs existing source, <code>form</code> obeys output constraints,
<code>tool</code> emits calls with correctly typed arguments, <code>trac</code> predicts what
a program prints. What each category figure is drawn from is on hover.</p></div>
<div class="scroll"><table><thead><tr><th>Model</th><th>Pass rate <span class="unit">%</span></th>
<th>Median per task <span class="unit">s</span></th>
<th>By category <span class="unit">%</span></th></tr></thead>
<tbody>{body}</tbody></table></div></section>'''

def _speed(A, probe_depth):
    """Every model that was measured at all, quickest session first."""
    speed = ""
    if A.prb:
        rs = ""
        for m in sorted(A.prb, key=lambda x: (A.session_time(x) or 9e9)):
            d = A.prb[m]
            # Rates, and the session they add up to. What the model did with a long
            # prompt is a finding, and the ranking carries it; a column of it read
            # OK on every row but one. A model stopped before the deep prompt has
            # no session and no wait, and the empty cells say so.
            rs += (f'<tr class="r-plain"><th>{esc(m)}</th>'
                   f'<td class="figure strong">{num(A.session_time(m),"",0)}</td>'
                   f'<td class="figure">{num(d.get("prefill_s"),"",1)}</td>'
                   f'<td class="figure">{num(d.get("gen_tok_s"),"",1)}</td>'
                   f'<td class="figure">{num((d.get("placement") or {}).get("pct_gpu"),"",0)}</td>'
                   f'<td class="figure">{_vram(d.get("placement") or {})}</td></tr>')
        speed = f'''<section><h2>Speed</h2>
<div class="lede"><p>A coding harness session simulated from the measured rates, quickest first:
{SESSION_TURNS} exchanges over a context of about {probe_depth} tokens, each writing a
{ANSWER_TOKENS}-token answer. The context is read once, since the server keeps the processed
prompt between turns. A model measured only at a short prompt has no session or first token
figure.</p></div>
<div class="scroll"><table><thead><tr><th>Model</th><th>Session <span class="unit">s</span></th>
<th>First token <span class="unit">s</span></th>
<th>Generation <span class="unit">tok/s</span></th><th>On GPU <span class="unit">%</span></th>
<th>Memory <span class="unit">GB</span></th>
</tr></thead>
<tbody>{rs}</tbody></table></div></section>'''
    return speed

def _excluded(A, rejected):
    """Every model the run covered that earned no figures, and why.

    A model stopped before it was measured, one whose records rule it out, and
    one a sweep never reached are all named with what is known about them. A
    model the report names nowhere else is the one case a reader cannot tell
    from a model that was never asked for.
    """
    out = {m: "; ".join(describe(f) for f in A.verdict(m)[1]) for m in A.models
           if A.eligible(m)}
    # A rejection wins over whatever the records left behind: the run stopped at
    # the first decisive finding, and that is why the model has no figures. Its
    # gaps follow from the rejection and are not separate facts about it.
    out.update({r["model"]: sentence(r) for r in rejected})
    return dict(sorted(out.items(), key=lambda kv: kv[0].lower()))

# Coloured by verdict, the same as the cards and the pills. A model with nothing
# measured takes no colour: there is no finding against it to colour.
SEV = {"suitable": "good", "limited": "warn", "unsuitable": "bad",
       "unmeasured": "plain"}


def _rows(A, excluded: dict) -> str:
    """Every model the run covered: the ranked ones, then the ones with no figures.

    A model stopped before it was measured has no pass rate and no speed, and a
    dash is what that is. It is still on the page, in the order the reader reads,
    with the finding that stopped it.
    """
    rows = ""
    for i, m in enumerate(A.sc_rank, 1):
        v = A.sc[m]
        sv, why = A.verdict(m)
        # Last, so prose of varying length takes the width left over instead
        # of widening a column the figures have to share.
        rows += f'''<tr class="r-{SEV.get(sv, "warn")}">
            <td class="figure">{i}</td><th>{esc(m)}</th>
            <td><span class="pill p-{sv}">{sv}</span></td>
            <td class="barcell">{bar(v["quality"], "plain", unit="")}</td>
            <td class="barcell">{bar(v["speed"], "plain", unit="")}</td>
            <td class="detail">{esc("; ".join(describe(f) for f in why))}</td></tr>'''
    for m, reason in excluded.items():
        # Its own verdict, not a blanket one: a model stopped for generating too
        # slowly is unsuitable, and one that simply was not screened is not. With
        # nothing measured at all there is no verdict to report, and saying so is
        # the honest row -- a sweep that never reached a model has found nothing
        # against it.
        sv = "unmeasured" if not (m in A.agg or m in A.prb) else A.verdict(m)[0]
        rows += f'''<tr class="r-{SEV.get(sv, "warn")}">
            <td class="figure">&mdash;</td><th>{esc(m)}</th>
            <td><span class="pill p-{sv}">{sv}</span></td>
            <td class="figure">&mdash;</td>
            <td class="figure">&mdash;</td>
            <td class="detail">{esc(reason)}</td></tr>'''
    return rows


def _recommendation(A, excluded):
    if not A.sc_rank:
        return ("<section><h2>Recommendation</h2><div class=\"lede\"><p>No model can be "
                "recommended: none of them was measured on everything a recommendation "
                "rests on, or each was ruled out by what was measured.</p></div></section>"
                "<section><h2>Ranking</h2>"
                '''<div class="scroll"><table><thead><tr><th>#</th><th>Model</th>
  <th>Verdict</th><th>Pass rate <span class="unit">%</span></th>
  <th>Speed <span class="unit">%</span></th><th>Finding</th></tr></thead>'''
                f"<tbody>{_rows(A, excluded)}</tbody></table></div></section>")
    pool = [m for m in A.sc_rank if A.verdict(m)[0] == "suitable"] or A.sc_rank
    best_quality = max(pool, key=lambda m: (A.sc[m]["quality"], -A.sc[m]["prefill"]))
    fastest = min(pool, key=lambda m: A.sc[m]["session"])
    # The middle ground: the quickest of the stronger half. Half is a share of
    # whatever was measured rather than a threshold on the measurement, so it
    # needs no number nobody derived.
    stronger = sorted(pool, key=lambda m: -A.sc[m]["quality"])[:max(1, len(pool) // 2)]
    balanced = min(stronger, key=lambda m: A.sc[m]["session"])
    # The role says why the card is here; the two figures beside it say by how
    # much, and each is named for what the pick is chosen on.
    picks = []
    for role, m in (("Highest pass rate", best_quality),
                    ("Best balance", balanced),
                    ("Fastest session", fastest)):
        picks.append(f'''<article class="card c-suitable">
      <header><h3>{esc(m)}</h3></header>
      <p class="role">{role}</p>
      <dl>
        <div><dt>Pass rate</dt><dd>{A.sc[m]["quality"]:.0f}<span class="pc">%</span></dd></div>
        <div><dt>Speed</dt><dd>{A.sc[m]["speed"]:.0f}<span class="pc">%</span></dd></div>
      </dl>
    </article>''')
    rows = _rows(A, excluded)
    return f'''<section>
  <h2>Recommendation</h2>
  <div class="lede"><p>Candidates for a deeper benchmark, taken from the models marked
suitable: the highest pass rate, the fastest session, and the quickest of the stronger
half.</p></div>
  <div class="cards">{"".join(picks)}</div>
</section>

<section>
  <h2>Ranking</h2>
  <div class="lede"><p>Ordered by verdict, then pass rate, then speed. Unsuitable: a tool
call that could not be parsed or was never emitted, or generation below
{MIN_GEN_TOK_S:.0f} tok/s. Limited: a
well-formed tool call sent to the wrong tool, or a long prompt the model could not use. A reply
that reaches the output limit stops mid-answer and is graded as written, so it fails for length
rather than for being wrong; raise the budget with <code>--num-predict</code>. A model with no
figures was stopped before it was measured, and the finding says what stopped it.</p></div>
  <div class="scroll"><table><thead><tr><th>#</th><th>Model</th><th>Verdict</th>
  <th>Pass rate <span class="unit">%</span></th><th>Speed <span class="unit">%</span></th>
  <th>Finding</th></tr></thead><tbody>{rows}</tbody></table></div>
</section>'''

def build(cfg: Config, models: list[str]) -> str:
    """Return the complete HTML document for the records in cfg.results_dir."""
    A = analyse(cfg, list(models))
    windows = sorted({r.get("num_ctx") for r in A.probe if r.get("num_ctx")} |
                     {r.get("ctx") for r in A.screen if r.get("ctx")})
    depths = sorted({r.get("depth_target") for r in A.probe if r.get("depth_target")})
    probe_depth = ", ".join(f"{d:,}" for d in depths) if depths else "the configured depth"

    # The tool that produced the page, not the server it talked to. The host appears
    # only when it is remote, where it explains why there are no GPU details.
    envline = " \u00b7 ".join(filter(None, [
        "codesift", "Ollama", (f"remote host {cfg.host}" if cfg.is_remote else "")]))
    return PAGE.format(
        style=STYLE,
        envline=esc(envline),
        ctxphrase=_windows(windows),
        generated=dt.datetime.now().strftime("%d %B %Y, %H:%M"),
        recommendation=_recommendation(A, _excluded(A, _rejected(cfg, A.models))),
        pass_rate=_pass_rate(A),
        speed=_speed(A, probe_depth))


def _windows(windows: list) -> str:
    """The context windows the records were taken at, as a phrase.

    Every one of them, not just the largest: measurements from two windows are
    not interchangeable, and this is the only place the reader is told which were
    used. The article and the plural come with it, since one window reads "a
    32,768-token context" and two do not.
    """
    if not windows:
        return "the configured context"
    if len(windows) == 1:
        return f"a {windows[0]:,}-token context"
    return "{} and {:,}-token contexts".format(
        ", ".join(f"{c:,}" for c in windows[:-1]), windows[-1])


def run(cfg: Config, output: Path, models: list[str] | None = None) -> Path:
    if models is None:
        # Every model anything was recorded about, triage included: a model
        # rejected at a gate is on record nowhere else.
        seen = []
        for name in ("screen_tasks.jsonl", "probe.jsonl", "triage.jsonl"):
            for rec in ledger.read(cfg.results_dir / name):
                if rec.get("model") and rec["model"] not in seen:
                    seen.append(rec["model"])
        models = seen
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build(cfg, models), encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = args.stage("report", "Render the HTML report.", measuring=False)
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help="named on the page when the measurement was taken elsewhere")
    args.add_output(parser)
    a = parser.parse_args(argv)
    cfg = args.config_from(a)
    print(f"wrote {run(cfg, Path(a.output), cfg.models or None)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
