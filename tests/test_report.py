# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""The report must render from whatever records exist, including none.

A reporting step that crashes on sparse data loses the measurements that were
successfully collected, so the degenerate cases are exercised explicitly.
"""
import json
import re
import tempfile
import unittest
from pathlib import Path

from codesift import analysis, report
from codesift.config import Config
from codesift.findings import sentence
from tests.records import OFFLINE, RecordsCase, probe_record, screen_record






class ReportCase(RecordsCase):
    """Records, plus the page built from them."""

    def render(self, models=None):
        out = report.run(self.cfg, self.tmp / "report.html", models)
        return out.read_text(encoding="utf-8")

    def assessed(self, html):
        """Where a verdict and its finding live: the one table over the field."""
        return html.split("<h2>Ranking</h2>", 1)[1].split("<h2>", 1)[0]

    def row(self, section, model):
        """One model's row, from <th>model</th> to the end of that row."""
        start = section.index(f"<th>{model}</th>")
        return section[section.rindex("<tr", 0, start):section.index("</tr>", start)]

    def picks(self, html):
        """The recommendation cards, which only measured models can appear in."""
        return html.split("<h2>Recommendation</h2>", 1)[1].split("</section>", 1)[0]


class TestDegenerateInputs(ReportCase):
    def test_no_records_at_all(self):
        html = self.render()
        self.assertIn("<title>", html)
        self.assertIn("</div>", html)

    def test_single_model_with_full_data(self):
        self.write_tasks([screen_record("solo")])
        self.write("probe.jsonl", [probe_record("solo")])
        html = self.render(["solo"])
        self.assertIn("solo", html)
        self.assertIn("<h2>Speed</h2>", html)

    def test_screen_only_without_probe(self):
        self.write_tasks([screen_record("a"), screen_record("b", rate=60.0)])
        html = self.render(["a", "b"])
        self.assertIn("<h2>Pass Rate</h2>", html)
        self.assertNotIn("<h2>Speed</h2>", html)
        self.assertNotIn("Speed below", html)


class TestContent(ReportCase):
    def setUp(self):
        super().setUp()
        self.write_tasks([
            screen_record("good"), screen_record("good"),
            screen_record("weak", rate=50.0),
            screen_record("weak", rate=50.0)])
        self.write("probe.jsonl", [probe_record("good"), probe_record("weak", prefill=200.0)])

    def test_models_not_requested_are_excluded(self):
        html = self.render(["good"])
        self.assertIn("good", html)
        self.assertNotIn(">weak<", html)

    def test_truncating_model_is_marked_unsuitable(self):
        self.write("probe.jsonl", [probe_record("good"),
                                   probe_record("weak", truncated=True)])
        html = self.render(["good", "weak"])
        self.assertIn("unsuitable", html)

    def test_both_themes_are_defined(self):
        html = self.render(["good", "weak"])
        self.assertIn("prefers-color-scheme:dark", html)
        self.assertIn('[data-theme="dark"]', html)
        self.assertIn("--paper", html)

    def test_explanations_carry_no_bold(self):
        import re
        html = self.render(["good", "weak"])
        for block in re.findall(r'<div class="lede">.*?</div>', html, re.S):
            self.assertNotIn("<b>", block)

    def test_remote_host_suppresses_local_gpu_details(self):
        cfg = Config(host="http://192.168.1.50:11434", results_dir=self.tmp)
        html = report.run(cfg, self.tmp / "remote.html", ["good"]).read_text(encoding="utf-8")
        self.assertIn("remote host", html)


class TestFailingModels(ReportCase):
    """Degenerate measurements must still produce a report.

    A screen exists to expose failure, so the failing cases are the ones that
    must not crash the reporting step.
    """

    def test_single_model_failing_everything(self):
        self.write_tasks([screen_record("bad", rate=0.0),
                                    screen_record("bad", rate=0.0)])
        self.write("probe.jsonl", [probe_record("bad", prefill=500.0, gen=1.0,
                                                truncated=True)])
        html = self.render(["bad"])
        self.assertIn("bad", html)
        self.assertIn("unsuitable", html)
        self.assertIn("No model can be recommended", html,
                      "with nothing eligible the recommendation must say so")

    def test_every_model_excluded(self):
        # Unparseable tool calls: a harness cannot proceed at all, which is what
        # it takes to be excluded rather than merely limited.
        self.write_tasks([screen_record(m, rate=0.0, tools_ok=False)
                          for m in ("a", "b")])
        self.write("probe.jsonl", [probe_record("a"), probe_record("b")])
        html = self.render(["a", "b"])
        self.assertIn("No model can be recommended", html)

    def test_single_model_passing_everything(self):
        """Normalisation across one model divides by a zero range."""
        self.write_tasks([screen_record("solo")])
        self.write("probe.jsonl", [probe_record("solo")])
        html = self.render(["solo"])
        self.assertIn("<h2>Ranking</h2>", html)
        self.assertIn("solo", html)

    def test_zero_generation_rate(self):
        """A model that produced no measurable output must not divide by zero."""
        self.write_tasks([screen_record("stalled"),
                                    screen_record("stalled")])
        self.write("probe.jsonl", [probe_record("stalled", gen=0.0)])
        html = self.render(["stalled"])
        self.assertIn("stalled", html)

    def test_probe_without_any_screen_records(self):
        self.write("probe.jsonl", [probe_record("only-probed")])
        html = self.render(["only-probed"])
        self.assertIn("<h2>Speed</h2>", html)

    def test_records_for_unknown_models_are_ignored(self):
        self.write_tasks([screen_record("ghost")])
        html = self.render(["someone-else"])
        self.assertNotIn("ghost", html)

    def test_malformed_lines_are_skipped(self):
        (self.tmp / "screen.jsonl").write_text(
            "not json at all\n" + __import__("json").dumps(screen_record("ok")) + "\n",
            encoding="utf-8")
        html = self.render(["ok"])
        self.assertIn("ok", html)



class TestRejectedAreNamed(unittest.TestCase):
    """Pruning a discarded model removes its measurements, so a report built from
    those measurements loses it entirely. Silence cannot distinguish a model that
    failed from one that was never run, and the first is a finding."""

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.cfg = Config(results_dir=self.tmp, models=[])

    def write(self, name, rows):
        with (self.tmp / name).open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

    def render(self, models):
        out = self.tmp / "r.html"
        report.run(Config(results_dir=self.tmp, models=models), out, models)
        return out.read_text(encoding="utf-8")

    def assessed(self, html):
        return html.split("<h2>Ranking</h2>", 1)[1].split("<h2>", 1)[0]

    def row(self, section, model):
        start = section.index(f"<th>{model}</th>")
        return section[section.rindex("<tr", 0, start):section.index("</tr>", start)]

    def test_a_rejected_model_is_named_with_what_it_was_rejected_for(self):
        self.write("triage.jsonl", [
            dict(model="slow:1", passed=False, findings=[{"code": "slow_generation", "tok_s": 9.0}], ts=1.0),
        ])
        page = self.render(["slow:1"])
        self.assertIn("slow:1", page)
        self.assertIn("9 tok/s", page)

    def test_the_newest_verdict_is_the_one_shown(self):
        # One run writes one record holding every finding it reached, so two
        # records mean two runs and the later one is what is true now.
        self.write("triage.jsonl", [
            dict(model="m:1", passed=False,
                 findings=[{"code": "slow_generation", "tok_s": 9.0}], ts=1.0),
            dict(model="m:1", passed=False,
                 findings=[{"code": "malformed_tool_calls", "malformed": 2, "total": 4}],
                 ts=2.0),
        ])
        page = self.render(["m:1"])
        self.assertIn("50% of tool calls unparseable", page)
        self.assertNotIn("9 tok/s", page)

    def test_a_rejected_model_is_listed_with_what_stopped_it(self):
        # Leaving it out makes the ranking look like the whole field when it is not.
        # It joins the models the gates excluded: both are out of the ranking, and
        # both a reader wants named with the reason.
        self.write("triage.jsonl", [
            dict(model="slow:1", passed=False, findings=[{"code": "slow_generation", "tok_s": 9.0}], ts=1.0),
        ])
        page = self.render(["slow:1"])
        assessment = self.assessed(page)
        self.assertIn("slow:1", assessment)
        self.assertIn("9 tok/s", assessment)
        self.assertNotIn("barcell", assessment, "nothing was measured to draw")

    def test_no_figure_is_invented_for_a_model_that_was_stopped(self):
        self.write("triage.jsonl", [
            dict(model="slow:1", passed=False, findings=[{"code": "slow_generation", "tok_s": 9.0}], ts=1.0),
        ])
        # Triage stops at the first decisive finding, so nothing after it was
        # measured and nothing after it is shown -- not even as a dash, which
        # reads as a measurement that came back empty.
        assessment = self.assessed(self.render(["slow:1"]))
        self.assertNotIn("<dt>", assessment)
        for absent in ("Prefill", "Speed score"):
            self.assertNotIn(absent, assessment, absent)

    def test_a_model_triaged_again_is_judged_by_the_newer_record(self):
        # The ledger outlives the rules that wrote it. A model rejected under an
        # older rule and cleared under the current one is cleared, or the page
        # both ranks it and reports it as stopped.
        self.write("triage.jsonl", [
            dict(model="m:1", passed=False, ts=1.0,
                 findings=[{"code": "needle_missed", "depth": 24576}]),
            dict(model="m:1", passed=True, ts=2.0, findings=[]),
        ])
        page = self.render(["m:1"])
        self.assertNotIn("needle", page, "the older rejection is not the verdict")
        row = self.row(self.assessed(page), "m:1")
        self.assertIn("unmeasured", row, "named on the page, so its state is stated")
        self.assertIn("not screened", row)

    def test_rejections_outside_the_shortlist_are_left_out(self):
        # The triage ledger outlives a run. A report narrowed to some models must
        # not describe two fields at once.
        self.write("triage.jsonl", [
            dict(model="asked:1", passed=False, findings=[{"code": "slow_generation", "tok_s": 9.0}], ts=1.0),
            dict(model="other:1", passed=False,
                 findings=[{"code": "context_truncated", "num_ctx": 32768}], ts=2.0),
        ])
        page = self.render(["asked:1"])
        self.assertIn("asked:1", page)
        self.assertNotIn("other:1", page)

    def test_the_default_field_includes_models_only_triage_saw(self):
        # Their measurements were pruned, so the triage ledger is the only place
        # they survive; a report of "everything stored" has to look there.
        self.write("triage.jsonl", [
            dict(model="pruned:1", passed=False, findings=[{"code": "slow_generation", "tok_s": 9.0}], ts=1.0),
        ])
        out = self.tmp / "all.html"
        report.run(Config(results_dir=self.tmp), out, None)
        self.assertIn("pruned:1", out.read_text(encoding="utf-8"))

    def test_a_model_that_cleared_triage_is_not_called_rejected(self):
        self.write("triage.jsonl", [
            dict(model="m:1", passed=True, findings=[], ts=1.0),
        ])
        self.assertNotIn('p-unsuitable">unsuitable', self.render(["m:1"]))

    def test_it_survives_no_triage_ledger_at_all(self):
        self.assertNotIn('p-unsuitable">unsuitable', self.render([]))

class TestNothingReachesThePageUnescaped(ReportCase):
    """Text on the page is escaped once, and markup built here is not escaped at all.

    A model name comes off the server, so it is not ours to trust; a dash or a bar
    is markup this module made, and escaping it again would print the tags.
    """

    def test_a_model_name_carrying_markup_is_escaped(self):
        name = "<script>alert(1)</script>:8b"
        self.write_tasks([screen_record(name)])
        self.write("probe.jsonl", [probe_record(name)])
        page = self.render([name])
        self.assertNotIn("<script>", page)
        self.assertIn("&lt;script&gt;", page)

    def test_markup_this_module_built_is_left_alone(self):
        from codesift.report import bar, esc, num
        self.assertEqual(esc(num(None)), "&mdash;", "a dash must not be escaped twice")
        self.assertIn("<div class=\"bar\">", esc(bar(50, "plain")))

    def test_escaping_is_idempotent(self):
        from codesift.report import esc
        once = esc("a & b")
        self.assertEqual(esc(once), once)


class TestThePageSaysWhatItMeasuredAndStops(ReportCase):
    """The copy states the rule and the figure. It does not argue for either.

    Both of these have gone wrong in edits rather than in design, so they are
    checked here instead of being remembered.
    """

    INTERNAL = ("gate", "ledger", "needle", "jsonl", "regrade", "triage")

    def page(self):
        self.write_tasks([screen_record("m")])
        self.write("probe.jsonl", [probe_record("m")])
        return self.render(["m"])

    def ledes(self, page):
        return re.findall(r'<div class="lede"><p>(.*?)</p>', page, re.S)

    def test_no_internal_vocabulary_reaches_the_reader(self):
        page = self.page().lower()
        body = page[page.index('<div class="wrap">'):]
        for word in self.INTERNAL:
            with self.subTest(word=word):
                self.assertNotIn(word, body, f"{word!r} is how the code talks, not the page")

    def test_no_lede_appends_an_aside(self):
        # A dash in a lede has every time been an interpretation bolted onto a
        # fact: what a verdict is supposed to mean to the reader, what the figure
        # is really saying. The fact is the whole job. A flag named in <code> is
        # not that, so it is taken out before looking.
        for lede in self.ledes(self.page()):
            prose = re.sub(r"<code>.*?</code>", "", lede, flags=re.S)
            with self.subTest(lede=lede[:40]):
                self.assertNotIn("--", prose)
                self.assertNotIn("\u2014", prose)


if __name__ == "__main__":
    unittest.main()


class TestToolCallGate(ReportCase):
    """A call that never arrives and a call to the wrong tool are different failures.

    A harness cannot proceed without a parseable call, so that ends a session. A
    well-formed call to the wrong tool returns the wrong result and the model gets
    another turn, which degrades a session instead.
    """

    def field(self, model, *, passed, format_ok):
        """Two tool tasks; only one of them fails, as the real cases did."""
        recs = screen_record(model)
        recs.append(dict(recs[0], task="t_tool2", passed=True, score=1.0,
                         format_ok=True))
        recs[0].update(passed=passed, score=float(passed), format_ok=format_ok)
        self.write_tasks([recs, screen_record("clean")])
        self.write("probe.jsonl", [probe_record(model), probe_record("clean")])
        return self.render([model, "clean"])

    def verdict(self, html, model):
        import re
        m = re.search(r"<th>" + re.escape(model) + r'(?:<span[^>]*>.*?</span>)?</th>\s*'
                      r'<td><span class="pill p-(\w+)">', self.assessed(html), re.S)
        return m.group(1) if m else None

    def test_an_unparseable_call_excludes_the_model(self):
        html = self.field("mute", passed=False, format_ok=False)
        self.assertEqual(self.verdict(html, "mute"), "unsuitable")
        self.assertIn("unparseable", self.assessed(html))
        self.assertNotIn("mute", self.picks(html), "it must not be recommended")

    def test_a_well_formed_call_to_the_wrong_tool_is_graded(self):
        html = self.field("misdirected", passed=False, format_ok=True)
        self.assertEqual(self.verdict(html, "misdirected"), "limited")
        self.assertIn("to the wrong tool", html)
        ranking = html.split("<h2>Ranking</h2>")[1].split("</table>")[0]
        self.assertIn("misdirected", ranking)

    def test_every_tool_task_counts_toward_the_figure(self):
        # One miss out of two tool tasks. Dropping any of them would let the same
        # miss read as a much larger share.
        html = self.field("misdirected", passed=False, format_ok=True)
        self.assertIn("50% of tool calls to the wrong tool", html)




class TestGenerationFloor(ReportCase):
    """The gate keys on generation rate, which is where the physics is.

    How much of a model sits outside VRAM matters far less than how much of it moves
    per token. A mixture of experts moves a few billion parameters whatever its
    placement; a dense model moves all of them across the bus. The two separate on
    generation rate by a much wider margin than on any wall-clock total.
    """

    def render_one(self, **probe):
        self.write_tasks([screen_record("m")])
        self.write("probe.jsonl", [probe_record("m", **probe)])
        return self.render(["m"])

    def verdict(self, html):
        import re
        m = re.search(r'<th>m(?:<span[^>]*>.*?</span>)?</th>\s*'
                      r'<td><span class="pill p-(\w+)">', self.assessed(html), re.S)
        return m.group(1) if m else None

    def excluded(self, html):
        """A model with no figures is in the table with dashes and a finding."""
        return self.assessed(html)

    def test_a_model_slower_than_reading_speed_is_excluded(self):
        html = self.render_one(gen=6.0, pct_gpu=36.0)
        self.assertEqual(self.verdict(html), "unsuitable")
        self.assertIn("too slow to use", self.assessed(html))
        self.assertIn("No model can be recommended", html)
        self.assertNotIn("barcell", self.assessed(html), "it has no figures to show")

    def test_poor_placement_alone_does_not_condemn_a_model(self):
        # The sharpest case in the measured field: a mixture of experts at a third
        # resident, generating faster than anything else on the card.
        html = self.render_one(gen=60.1, pct_gpu=33.0)
        self.assertEqual(self.verdict(html), "suitable")

    def test_the_reason_carries_its_own_explanation(self):
        html = self.render_one(gen=13.0, pct_gpu=47.0)
        self.assertIn("too slow to use: 13 tok/s", html)

    def test_an_older_record_without_architecture_still_reports_the_rate(self):
        rec = probe_record("m", gen=6.0)
        self.write_tasks([screen_record("m")])
        self.write("probe.jsonl", [rec])
        html = self.render(["m"])
        self.assertIn("too slow to use: 6 tok/s", self.excluded(html))
        # Where the model sat is the Speed table's column, not the finding's job.
        self.assertIn("40", html.split("<h2>Speed</h2>")[1])
        self.assertNotIn("dense", html)


class TestTheReportNamesTheWindowThatWasUsed(ReportCase):
    """A card must not cite numbers the run did not use.

    Both strings were literals reading `64k` and `48k`. Anyone screening at a
    smaller window got a verdict describing a measurement nobody took, and the
    figure they were being judged on appeared nowhere on the page.
    """

    def field(self, ctx, **probe_kwargs):
        self.cfg = Config(host=OFFLINE, results_dir=self.tmp, ctx=ctx)
        self.write_tasks([screen_record("m", ctx=ctx),
                                    screen_record("m", ctx=ctx)])
        self.write("probe.jsonl", [probe_record("m", ctx=ctx, **probe_kwargs)])
        return analysis.analyse(self.cfg, ["m"])

    def why(self, ctx, **probe_kwargs):
        return sentence({"findings": self.field(ctx, **probe_kwargs).verdict("m")[1]})

    def test_truncation_names_the_window_at_32k(self):
        self.assertIn("32k window", self.why(32768, truncated=True))

    def test_truncation_names_the_window_at_64k(self):
        self.assertIn("64k window", self.why(65536, truncated=True))

    def test_a_truncated_prompt_names_the_window_it_was_cut_at(self):
        self.assertIn("32k window", self.why(32768, truncated=True))
        self.assertIn("64k window", self.why(65536, truncated=True))

    def test_a_32k_run_never_mentions_the_default_window(self):
        self.field(32768)
        html = self.render(["m"])
        self.assertIn("32,768", html)
        self.assertIn("24,576", html)
        self.assertNotIn("65,536", html)
        self.assertNotIn("48,000", html)

    def test_the_window_is_stated_once(self):
        # It was in the opening sentence and again in the header strip below it.
        self.field(32768)
        html = self.render(["m"])
        self.assertEqual(html.count("32,768"), 1)

    def test_records_from_two_windows_name_both(self):
        # Not a defect: measurements taken at different windows are not
        # interchangeable, and hiding one of them would present them as if
        # they were.
        self.cfg = Config(host=OFFLINE, results_dir=self.tmp, ctx=32768)
        self.write_tasks([screen_record("m", ctx=32768),
                                    screen_record("m", ctx=32768),
                                    screen_record("n", ctx=65536),
                                    screen_record("n", ctx=65536)])
        self.write("probe.jsonl", [probe_record("m", ctx=32768), probe_record("n", ctx=65536)])
        html = self.render(["m", "n"])
        self.assertIn("32,768", html)
        self.assertIn("65,536", html)


class TestRejectionTextIsRebuiltNotReplayed(ReportCase):
    """A rejection is stored as fields, and phrased when the page is built.

    Wording frozen into the ledger at measurement time could only be changed by
    measuring the model again.
    """

    def rejection(self, rec):
        self.write("triage.jsonl", [dict(model="dud", passed=False, **rec)])
        row = self.render(["dud"])
        row = row[row.index("<th>dud</th>"):]
        return re.search(r'<td class="detail">(.*?)</td>', row, re.S).group(1)

    def test_the_sentence_comes_from_the_fields(self):
        text = self.rejection(dict(findings=[{"code": "context_truncated",
                                              "num_ctx": 32768}]))
        self.assertIn("32k window", text)

    def test_every_finding_emitted_anywhere_has_a_sentence(self):
        # Both the gates and the records report through findings, so both are
        # read here: a code with no sentence renders as its own name on the page.
        from codesift import findings
        emitted = set()
        for module in ("triage", "analysis"):
            emitted |= set(re.findall(r'"code": "(\w+)"',
                                      Path(f"src/codesift/{module}.py").read_text()))
        self.assertTrue(emitted, "no findings found in the gates")
        for code in emitted:
            with self.subTest(code=code):
                said = findings.describe({"code": code, "tok_s": 1.0, "wrong": 1,
                                          "malformed": 1, "total": 2, "num_ctx": 32768,
                                          "depth": 24576, "count": 1, "message": "x"})
                self.assertNotEqual(said, code, "renders as its own code, not a sentence")

    def test_a_gate_and_the_records_word_the_same_fault_the_same_way(self):
        # The one thing two paths must never do is disagree on one page: the
        # ranking row is phrased from the records, the excluded row from the gate.
        from codesift import findings
        for finding in ({"code": "context_truncated", "num_ctx": 32768},
                        {"code": "slow_generation", "tok_s": 9.0}):
            with self.subTest(code=finding["code"]):
                self.assertEqual(findings.describe(finding),
                                 findings.sentence({"findings": [finding]}))


class TestHeadingsAndBars(ReportCase):
    """Section headings are title case, and a measurement is not colour-graded.

    The pass-rate bar and the stripe down the left of its row were coloured by
    thresholds at 90 and 75 -- the same bands that were taken out of the verdicts
    for turning a measurement into a cliff. Colouring them amber said the report
    judged the figure after all.
    """

    SMALL = {"a", "an", "and", "at", "by", "for", "in", "of", "on", "or", "the", "to"}

    def page(self):
        self.write_tasks([screen_record("m", rate=83.0),
                                    screen_record("m", rate=76.0)])
        self.write("probe.jsonl", [probe_record("m")])
        return self.render(["m"])

    def test_every_section_heading_is_title_case(self):
        for head in re.findall(r"<h2>([^<]+)</h2>", self.page()):
            for i, word in enumerate(head.split()):
                if i and word.lower() in self.SMALL:
                    continue
                with self.subTest(heading=head, word=word):
                    self.assertTrue(word[0].isupper(), f"{head!r} is not title case")

    def test_the_pass_rate_bar_is_not_graded_by_colour(self):
        page = self.page()
        table = page[page.index("<h2>Pass Rate</h2>"):]
        table = table[:table.index("</section>")]
        for cls in ("s-good", "s-warn", "s-bad"):
            self.assertNotIn(cls, table, "the bar is coloured by a hidden threshold")

    def test_the_pass_rate_row_carries_no_severity_stripe(self):
        page = self.page()
        table = page[page.index("<h2>Pass Rate</h2>"):]
        table = table[:table.index("</section>")]
        for cls in ('r-good', 'r-warn', 'r-bad'):
            self.assertNotIn(cls, table, "the row handle is coloured by a hidden threshold")


class TestBarCellsStayTableCells(ReportCase):
    """A bar cell must remain a table cell, whatever sits beside it.

    `display:flex` on a <td> takes it out of table layout, and two such cells side
    by side are wrapped into one anonymous cell, which stacks the speed bar under
    the quality bar instead of beside it.
    """

    def page(self):
        self.write_tasks([screen_record("m")])
        self.write("probe.jsonl", [probe_record("m")])
        return self.render(["m"])

    def test_the_cell_is_never_the_flex_container(self):
        html = self.page()
        for rule in re.findall(r"\.barcell[^{]*\{([^}]*)\}", html):
            self.assertNotIn("display:flex", rule.replace(" ", ""))

    def test_every_bar_cell_holds_a_flex_wrapper(self):
        for cell in re.findall(r'<td class="barcell">(.*?)</td>', self.page()):
            self.assertTrue(cell.startswith('<div class="barwrap">'), cell[:60])

    def test_the_ranking_puts_the_two_bars_in_separate_cells(self):
        row = re.search(r'<tbody>(<tr.*?</tr>)', self.page(), re.S).group(1)
        self.assertEqual(row.count('class="barcell"'), 2)

    def test_nothing_of_varying_width_sits_beside_a_bar(self):
        # Anything beside the bar that is present on some rows and absent on others
        # takes its width out of the bar, so no two bars in the column match.
        for cell in re.findall(r'<td class="barcell">(.*?)</td>', self.page()):
            self.assertNotIn('class="spread"', cell)


class TestTheHeader(ReportCase):
    """It names the tool and the server, and nothing measured from this machine."""

    def eyebrow(self, *probes):
        self.write_tasks([screen_record("m")])
        self.write("probe.jsonl", list(probes))
        html = self.render(["m"])
        return re.search(r'<p class="eyebrow">(.*?)</p>', html, re.S).group(1)

    def test_it_names_the_tool_and_the_server(self):
        self.assertEqual(self.eyebrow(probe_record("m")).strip(), "codesift \u00b7 Ollama")

    def test_it_states_no_memory_figure(self):
        # The GPU total was read off the local machine, which is the wrong one when
        # the server is elsewhere; the figure that replaced it was Ollama's placement,
        # which said more about the largest model measured than about the page.
        self.assertNotIn("GB", self.eyebrow(probe_record("m")))

    def test_a_remote_host_is_named(self):
        self.cfg = Config(host="http://box.lan:11434", results_dir=self.tmp)
        self.assertIn("box.lan", self.eyebrow(probe_record("m")))


class TestWhatMustBeResident(ReportCase):
    """Memory is reported as the one figure the server actually gives.

    It was briefly split into weights and cache, with the weights taken from the
    size on disk. That is not the loaded weight size: one 9B model is 6.59GB on
    disk and loads about 5.3GB, and an E4B sub-model ships 9.61GB to load 3.26GB.
    The subtraction produced a cache of zero and a total smaller than its own
    stated weights.
    """

    def row(self, **probe_kwargs):
        self.write_tasks([screen_record("m")])
        self.write("probe.jsonl", [probe_record("m", **probe_kwargs)])
        return self.render(["m"])

    def test_the_total_is_what_is_shown(self):
        html = self.row(weights=5.35, cache=11.31)
        self.assertIn("16.7", html)          # the figure that decides whether it fits

    def test_no_figure_is_derived_from_the_size_on_disk(self):
        # 3.26 loaded against 9.61 on disk: any subtraction here is fiction.
        html = self.row(weights=9.61, cache=-6.35)
        self.assertNotIn("9.6", html)

    def test_the_column_names_memory_rather_than_an_internal(self):
        html = self.row()
        self.assertIn('Memory <span class="unit">GB</span>', html)
        self.assertNotIn("Weights + context", html)

    def test_a_record_without_the_split_falls_back_to_the_total(self):
        # Records written before the cache was measured carry only a total, and
        # must still render rather than showing a blank column.
        self.write_tasks([screen_record("m")])
        rec = probe_record("m")
        rec["placement"] = {"pct_gpu": 40.0, "total_gb": 16.7}
        self.write("probe.jsonl", [rec])
        self.assertIn("16.7", self.render(["m"]))

    def test_no_placement_at_all_renders_a_dash(self):
        self.write_tasks([screen_record("m")])
        rec = probe_record("m")
        rec["placement"] = {}
        self.write("probe.jsonl", [rec])
        self.assertIn("&mdash;", self.render(["m"]))


class TestMissingFigures(ReportCase):
    """A figure that was never measured shows a dash, and the dash is a dash.

    Escaping the placeholder a second time put the literal text `&mdash;` in the
    table, which reads as a broken template rather than as an absent measurement.
    """

    def test_an_absent_figure_renders_as_an_unescaped_dash(self):
        # Nothing placed the model, so the memory column has no figure to print.
        self.write_tasks([screen_record("solo")])
        rec = probe_record("solo")
        rec["placement"] = {}
        self.write("probe.jsonl", [rec])
        html = self.render(["solo"])
        self.assertNotIn("&amp;mdash;", html)
        self.assertIn('<td class="figure">&mdash;</td>', html)


class TestSpeedScore(ReportCase):
    """Latency should cost points in proportion to the time actually lost."""

    def test_slowest_model_is_not_scored_zero(self):
        recs, probes = [], []
        for name, prefill in (("quick", 40.0), ("slow", 70.0)):
            recs.append(screen_record(name))
            probes.append(probe_record(name, prefill=prefill, gen=100.0))
        self.write_tasks(recs)
        self.write("probe.jsonl", probes)
        html = self.render(["quick", "slow"])
        table = html.split("<h2>Ranking</h2>")[1].split("</table>")[0]
        # Found by header rather than by position, so inserting a column does not
        # silently make this assert about a different number.
        head = [re.sub(r"<[^>]+>", "", h).strip()
                for h in re.findall(r"<th[^>]*>(.*?)</th>", table.split("</thead>")[0], re.S)]
        col = next(i for i, h in enumerate(head) if h.startswith("Speed"))
        rows = re.findall(r"<tr class=\"r-\w+\">.*?</tr>", table, re.S)
        cells = [[re.sub(r"<[^>]+>", "", c).strip()
                  for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", r, re.S)] for r in rows]
        speeds = {c[1]: float(re.sub(r"[^\d.]", "", c[col]) or 0) for c in cells if len(c) > col}
        self.assertEqual(speeds["quick"], 100.0)
        # 40s against 70s to read the context, generating at the same rate: slower,
        # but far from a zero score.
        self.assertGreater(speeds["slow"], 50.0)
        self.assertLess(speeds["slow"], 90.0)


class TestTruncatedReplies(ReportCase):
    """A reply cut at the output budget is a floor, not a measurement.

    The budget covers a model's reasoning as well as its answer. When it binds, the
    grader sees an unfinished reply and records a failure — which is indistinguishable
    in the score from a model that answered wrongly. The report says so instead of
    letting the number stand unqualified.
    """

    def render_with(self, capped=0, silent=0):
        self.write_tasks([screen_record("m", capped=capped, silent=silent)])
        self.write("probe.jsonl", [probe_record("m")])
        import re
        assessment = self.assessed(self.render(["m"]))
        found = re.findall(r'<td class="detail">(.*?)</td>', assessment, re.S)
        return " ".join(found)

    def test_truncation_is_reported_beside_the_model(self):
        self.assertIn("17% of replies cut off at the output limit", self.render_with(2))

    def test_a_single_truncation_reads_correctly(self):
        self.assertIn("8% of replies cut off at the output limit", self.render_with(1))

    def test_a_clean_run_says_nothing_about_the_budget(self):
        self.assertNotIn("output limit", self.render_with(0))

    def test_a_reply_that_never_started_is_not_called_cut_off(self):
        # One model reached the budget on five tasks having written nothing at all,
        # and three did the same at nearly three times the budget. Calling that cut
        # off points at the budget, which is the wrong thing to change.
        said = self.render_with(silent=2)
        self.assertIn("produced no answer within the output limit", said)
        self.assertNotIn("cut off", said)


class TestQualityFloor(ReportCase):
    """A fast model that does not do the work must not lead the ranking.

    An 8B model that fails most of its tasks returns in a fraction of the time
    precisely because it is not doing them, and unconditional latency credit put
    it at rank 1 above a model scoring 100%. A relative quality floor used to
    prevent that; the verdict ordering does it now, and decides the reference time
    for the speed figure as well.
    """

    def field(self):
        # "quick" is twelve times faster and answers under half its tasks; "solid"
        # gets everything right and takes its time over it.
        self.write_tasks([
            screen_record("solid", rate=100.0),
            screen_record("solid", rate=100.0),
            screen_record("quick", rate=57.1),
            screen_record("quick", rate=46.7),
        ])
        self.write("probe.jsonl", [probe_record("solid", prefill=80.0, gen=47.0),
                                   probe_record("quick", prefill=20.0, gen=279.0)])
        return analysis.analyse(self.cfg, ["solid", "quick"])

    def test_one_table_covers_every_task(self):
        # Two tables with the same columns and the same text gave a reader nothing
        # to say which to believe.
        A = self.field()
        html = report.build(self.cfg, A.models)
        self.assertEqual(html.count("<h2>Pass Rate</h2>"), 1)
        for gone in ("Task Set A", "Task Set B", "Hard Set", "Basic Set"):
            self.assertNotIn(gone, html, gone)

    def test_the_low_quality_model_is_ranked_below(self):
        A = self.field()
        self.assertEqual(A.sc_rank[0], "solid")
        self.assertLess(A.sc_rank.index("solid"), A.sc_rank.index("quick"))

    def test_no_single_score_combines_quality_and_speed(self):
        # A weighted score decided the trade on the reader's behalf, and could not
        # be compared across verdicts, so the number and the position disagreed.
        A = self.field()
        self.assertNotIn("composite", A.sc["solid"])
        self.assertFalse(hasattr(report, "W_QUALITY"))

    def test_the_ranking_never_contradicts_the_verdict(self):
        # A composite alone put an unsuitable model above four suitable ones, and
        # 80% quality above 100% because the slower model took three times as long.
        A = self.field()
        order = {"suitable": 0, "limited": 1, "unsuitable": 2}
        seen = [order[A.verdict(m)[0]] for m in A.sc_rank]
        self.assertEqual(seen, sorted(seen),
                         "a worse verdict was ranked above a better one")

    def test_the_speed_figure_is_a_share_of_the_quickest_measured(self):
        # Nothing exceeds 100%: the quickest model measured is the reference,
        # whatever its verdict.
        A = self.field()
        html = report.build(self.cfg, A.models)
        table = html[html.index("<h2>Ranking</h2>"):]
        row = table[table.index("<th>quick</th>"):]
        row = row[:row.index("</tr>")]
        # The second bar in the row is the speed score; the first is quality.
        speed_cell = row.split('class="barcell"')[2]
        head = html[html.index("<h2>Ranking</h2>"):]
        head = head[:head.index("</thead>")]
        self.assertIn('Speed <span class="unit">%</span>', head,
                      "a share of the fastest measured model, so the column is a percentage")
        self.assertRegex(speed_cell, r'class="figure">\d+</span>',
                         "the unit belongs to the header, not to every cell")

    def test_the_ranking_bars_are_not_coloured_by_verdict(self):
        # A 93% quality bar tinted amber reads as a judgement on the figure. The
        # verdict has its own pill and its own row stripe.
        A = self.field()
        html = report.build(self.cfg, A.models)
        table = html[html.index("<h2>Ranking</h2>"):]
        body = table[table.index("<tbody>"):table.index("</tbody>")]
        for cell in body.split('class="barcell"')[1:]:
            fill = cell[:cell.index("</div>")]
            self.assertIn("s-plain", fill)
            for sev in ("s-good", "s-warn", "s-bad"):
                self.assertNotIn(sev, fill)

    def test_the_ranking_is_coloured_by_verdict(self):
        # It encoded rank position and floor status, so a limited model showed red
        # while an unsuitable one showed amber.
        A = self.field()
        html = report.build(self.cfg, A.models)
        table = html[html.index("<h2>Ranking</h2>"):]
        sev = {"suitable": "good", "limited": "warn", "unsuitable": "bad"}
        for m in A.sc_rank:
            row = table[table.index(f"<th>{m}"):]
            row = table[:table.index(f"<th>{m}")].rsplit("<tr", 1)[1]
            self.assertIn(f'r-{sev[A.verdict(m)[0]]}"', row, m)

    def test_speed_does_not_depend_on_anyone_else_s_verdict(self):
        # How fast a model is is a property of that model. Taking the reference from
        # the suitable models made a quicker one read as over 100%, and would have
        # moved every figure when another model's quality verdict changed.
        A = self.field()
        self.assertEqual(A.sc["quick"]["speed"], 100.0,
                         "the quickest measured model sets the reference")
        self.assertLessEqual(A.sc["solid"]["speed"], 100.0)
        self.assertEqual(A.verdict("quick")[0], "suitable",
                         "a low score is not a disqualification")

    def test_a_failing_sprinter_does_not_compress_the_field(self):
        self.write_tasks([
            screen_record("good", rate=100.0), screen_record("good", rate=100.0),
            screen_record("mid", rate=90.0), screen_record("mid", rate=90.0),
            screen_record("bail", rate=47.0, tools_ok=False),
            screen_record("bail", rate=47.0, tools_ok=False),
        ])
        self.write("probe.jsonl", [probe_record("good", gen=45.0), probe_record("mid", gen=90.0),
                                   probe_record("bail", gen=600.0)])
        A = analysis.analyse(self.cfg, ["good", "mid", "bail"])
        self.assertEqual(A.sc["mid"]["speed"], 100.0,
                         "the fastest suitable model sets the reference")
        self.assertGreater(A.sc["good"]["speed"], 20.0,
                           "a real contender is not compressed into single digits by a "
                           "model that is quick because it fails")

    def test_quality_does_not_move_when_the_field_changes(self):
        # Speed is relative and moves with the field by construction; quality is
        # absolute and must not.
        A = self.field()
        before = A.sc["solid"]["quality"]
        B = analysis.analyse(self.cfg, ["solid"])
        self.assertEqual(B.sc["solid"]["quality"], before)

    def test_a_gap_does_not_rescue_a_disqualified_model(self):
        broken = screen_record("broken", rate=100.0)
        broken[0].update(format_ok=False, passed=False, score=0.0)  # unparseable call
        self.write_tasks([broken])          # and no probe
        A = analysis.analyse(self.cfg, ["broken"])
        sev, why = A.verdict("broken")
        self.assertEqual(sev, "unsuitable")
        self.assertEqual([f["code"] for f in why],
                         ["malformed_tool_calls", "not_probed"])






class TestEveryModelIsAccountedFor(ReportCase):
    """A model that was measured appears somewhere, with its figures or its reason.

    A model that cleared the gates is in the ranking with its finding beside it;
    one that did not is in the table naming what excluded it.
    """

    def field(self, models):
        screen, probe = [], []
        for name, rate, gen in models:
            screen.append(screen_record(name, rate=rate))
            probe.append(probe_record(name, gen=gen))
        self.write_tasks(screen)
        self.write("probe.jsonl", probe)
        return self.render([m[0] for m in models])

    def test_a_scored_model_carries_both_figures_in_one_row(self):
        section = self.assessed(self.field([("good", 100.0, 60.0)]))
        row = section[section.index("<th>good"):]
        row = row[:row.index("</tr>")]
        self.assertIn('p-suitable', row)
        self.assertEqual(row.count('class="barcell"'), 2,
                         "pass rate and speed both belong to the row")

    def test_a_model_kept_out_of_the_scoring_is_named_with_its_reason(self):
        # One table over the whole field: a model with nothing measured is in it
        # with dashes where the figures would be, and the finding that stopped it.
        html = self.field([("good", 100.0, 60.0), ("slow", 100.0, 5.0)])
        row = self.row(self.assessed(html), "slow")
        self.assertIn("&mdash;", row)
        self.assertIn("too slow to use", row)
        self.assertNotIn("barcell", row, "there is no figure to draw a bar from")

    def test_its_measurements_are_still_on_the_page(self):
        # Excluded from the ranking, not from the report: what was measured is
        # still measured, and the detail tables state it.
        html = self.field([("good", 100.0, 60.0), ("slow", 100.0, 5.0)])
        self.assertIn("slow", html.split("<h2>Pass Rate</h2>")[1])

    def test_a_clean_model_says_nothing_rather_than_nothing_found(self):
        section = self.assessed(self.field([("good", 100.0, 60.0)]))
        self.assertIn('<td class="detail"></td>', section)
