"""The report must render from whatever records exist, including none.

A reporting step that crashes on sparse data loses the measurements that were
successfully collected, so the degenerate cases are exercised explicitly.
"""
import io
import json
import re
import tempfile
import unittest
from pathlib import Path

from codesift import report
from codesift.config import Config

# Refused immediately rather than timing out, so tests stay fast offline.
OFFLINE = "http://127.0.0.1:1"


def screen_record(model, taskset="basic", run=1, rate=100.0):
    tasks = [
        dict(task="t1", kind="codegen", passed=rate >= 50, format_ok=True,
             detail="ok", wall=1.0),
        dict(task="t2", kind="toolcall", passed=True, format_ok=True,
             detail="ok", wall=0.5),
    ]
    return dict(model=model, run=run, taskset=taskset, ctx=65536, n=14,
                passed=int(rate / 100 * 14), pass_rate=rate, format_ok_rate=100.0,
                hit_cap_n=0, median_wall=1.0, total_s=10.0, tasks=tasks)


def probe_record(model, prefill=60.0, gen=45.0, truncated=False, retrieved=True,
                 moe=True, pct_gpu=40.0):
    return dict(model=model, num_ctx=65536, gen_tok_s=gen, prefill_tok_s=800.0,
                prefill_s=prefill, prefill_toks=48000, likely_truncated=truncated,
                retrieved=retrieved, placement={"pct_gpu": pct_gpu}, moe=moe)


class ReportCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.cfg = Config(host=OFFLINE, results_dir=self.tmp)

    def write(self, name, records):
        with (self.tmp / name).open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")

    def render(self, models=None):
        out = report.run(self.cfg, self.tmp / "report.html", models)
        return out.read_text(encoding="utf-8")


class TestDegenerateInputs(ReportCase):
    def test_no_records_at_all(self):
        html = self.render()
        self.assertIn("<title>", html)
        self.assertIn("</div>", html)

    def test_single_model_with_full_data(self):
        self.write("screen.jsonl", [screen_record("solo", "basic"),
                                    screen_record("solo", "hard")])
        self.write("probe.jsonl", [probe_record("solo")])
        html = self.render(["solo"])
        self.assertIn("solo", html)
        self.assertIn("Speed at Working Depth", html)

    def test_screen_only_without_probe(self):
        self.write("screen.jsonl", [screen_record("a"), screen_record("b", rate=60.0)])
        html = self.render(["a", "b"])
        self.assertIn("Basic Set", html)
        self.assertNotIn("Speed at Working Depth", html)


class TestContent(ReportCase):
    def setUp(self):
        super().setUp()
        self.write("screen.jsonl", [
            screen_record("good", "basic"), screen_record("good", "hard"),
            screen_record("weak", "basic", rate=50.0),
            screen_record("weak", "hard", rate=50.0)])
        self.write("probe.jsonl", [probe_record("good"), probe_record("weak", prefill=200.0)])

    def test_models_not_requested_are_excluded(self):
        html = self.render(["good"])
        self.assertIn("good", html)
        self.assertNotIn(">weak<", html)

    def test_truncating_model_is_marked_unsuitable(self):
        self.write("probe.jsonl", [probe_record("good"),
                                   probe_record("weak", truncated=True, retrieved=False)])
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
        self.write("screen.jsonl", [screen_record("bad", "basic", rate=0.0),
                                    screen_record("bad", "hard", rate=0.0)])
        self.write("probe.jsonl", [probe_record("bad", prefill=500.0, gen=1.0,
                                                truncated=True, retrieved=False)])
        html = self.render(["bad"])
        self.assertIn("bad", html)
        self.assertIn("unsuitable", html)
        self.assertIn("No model cleared", html,
                      "with nothing eligible the recommendation must say so")

    def test_every_model_excluded_by_gates(self):
        self.write("screen.jsonl", [screen_record(m, ts, rate=0.0)
                                    for m in ("a", "b") for ts in ("basic", "hard")])
        self.write("probe.jsonl", [probe_record("a", truncated=True, retrieved=False),
                                   probe_record("b", truncated=True, retrieved=False)])
        html = self.render(["a", "b"])
        self.assertIn("No model cleared", html)

    def test_single_model_passing_everything(self):
        """Normalisation across one model divides by a zero range."""
        self.write("screen.jsonl", [screen_record("solo", "basic"),
                                    screen_record("solo", "hard")])
        self.write("probe.jsonl", [probe_record("solo")])
        html = self.render(["solo"])
        self.assertIn("Full Ranking", html)
        self.assertIn("solo", html)

    def test_zero_generation_rate(self):
        """A model that produced no measurable output must not divide by zero."""
        self.write("screen.jsonl", [screen_record("stalled", "basic"),
                                    screen_record("stalled", "hard")])
        self.write("probe.jsonl", [probe_record("stalled", gen=0.0)])
        html = self.render(["stalled"])
        self.assertIn("stalled", html)

    def test_probe_without_any_screen_records(self):
        self.write("probe.jsonl", [probe_record("only-probed")])
        html = self.render(["only-probed"])
        self.assertIn("Speed at Working Depth", html)

    def test_records_for_unknown_models_are_ignored(self):
        self.write("screen.jsonl", [screen_record("ghost", "basic")])
        html = self.render(["someone-else"])
        self.assertNotIn("ghost", html)

    def test_malformed_lines_are_skipped(self):
        (self.tmp / "screen.jsonl").write_text(
            "not json at all\n" + __import__("json").dumps(screen_record("ok")) + "\n",
            encoding="utf-8")
        html = self.render(["ok"])
        self.assertIn("ok", html)


if __name__ == "__main__":
    unittest.main()


class TestToolCallGate(ReportCase):
    """A call that never arrives and a call to the wrong tool are different failures.

    A harness cannot proceed without a parseable call, so that ends a session. A
    well-formed call to the wrong tool returns the wrong result and the model gets
    another turn, which degrades a session instead.
    """

    def field(self, model, *, passed, format_ok):
        """One tool task per set; only the basic one fails, as the real cases did."""
        rec = screen_record(model, "basic")
        rec["tasks"][1].update(passed=passed, format_ok=format_ok)
        hard = screen_record(model, "hard")
        self.write("screen.jsonl", [rec, hard,
                                    screen_record("clean", "basic"),
                                    screen_record("clean", "hard")])
        self.write("probe.jsonl", [probe_record(model), probe_record("clean")])
        return self.render([model, "clean"])

    def verdict(self, html, model):
        import re
        m = re.search(r"<h3>" + re.escape(model) + r'</h3><span class="pill p-(\w+)">',
                      html.split("<h2>Assessment</h2>")[1])
        return m.group(1) if m else None

    def test_an_unparseable_call_excludes_the_model(self):
        html = self.field("mute", passed=False, format_ok=False)
        self.assertEqual(self.verdict(html, "mute"), "unsuitable")
        self.assertIn("malformed or absent", html)
        ranking = html.split("Full Ranking")[1].split("</table>")[0]
        self.assertNotIn("mute", ranking)

    def test_a_well_formed_call_to_the_wrong_tool_is_graded(self):
        html = self.field("misdirected", passed=False, format_ok=True)
        self.assertEqual(self.verdict(html, "misdirected"), "limited")
        self.assertIn("calls well-formed", html)
        ranking = html.split("Full Ranking")[1].split("</table>")[0]
        self.assertIn("misdirected", ranking)

    def test_both_task_sets_count_toward_the_tool_figure(self):
        # One miss out of two tool tasks across both sets is 50%. Counting the basic set
        # alone, which holds a single tool task here, would report 0% for the same miss.
        html = self.field("misdirected", passed=False, format_ok=True)
        self.assertIn("tool choice 50%", html)


class TestLatencyVerdict(ReportCase):
    """Latency is judged in seconds on this machine, not against the other models.

    An earlier version set the bar at a multiple of the fastest model measured, which
    meant a verdict changed when a quicker model joined the run: two models at 100% on
    the hard set were relabelled limited purely because something faster appeared
    beside them.
    """

    def field(self, *models):
        """models as (name, prefill, gen) -> screen and probe records."""
        screen, probe = [], []
        for name, prefill, gen in models:
            screen += [screen_record(name, "basic"), screen_record(name, "hard")]
            probe.append(probe_record(name, prefill=prefill, gen=gen))
        self.write("screen.jsonl", screen)
        self.write("probe.jsonl", probe)
        return [m[0] for m in models]

    def verdict(self, html, model):
        import re
        m = re.search(r"<h3>" + re.escape(model) + r'</h3><span class="pill p-(\w+)">',
                      html.split("<h2>Assessment</h2>")[1])
        return m.group(1) if m else None

    def test_a_faster_model_joining_does_not_relabel_the_field(self):
        # 60s prefill and 45 t/s: about 77s per task, comfortably inside the bar.
        alone = self.render(self.field(("steady", 60.0, 45.0)))
        self.assertEqual(self.verdict(alone, "steady"), "suitable")
        with_quick = self.render(self.field(("steady", 60.0, 45.0), ("quick", 5.0, 90.0)))
        self.assertEqual(self.verdict(with_quick, "steady"), "suitable")
        self.assertEqual(self.verdict(with_quick, "quick"), "suitable")

    def test_a_slow_model_is_limited_on_its_own_numbers(self):
        # 130s prefill and 40 t/s: about 149s per task, past the two-minute mark.
        html = self.render(self.field(("crawler", 130.0, 40.0)))
        self.assertEqual(self.verdict(html, "crawler"), "limited")
        self.assertIn("task time", html)

    def test_a_model_too_slow_to_use_is_excluded_before_scoring(self):
        # The case a dense model spilling out of VRAM produces: prefill collapses and
        # every token is generated across the bus.
        html = self.render(self.field(("offloaded", 320.0, 5.0), ("steady", 60.0, 45.0)))
        self.assertEqual(self.verdict(html, "offloaded"), "unsuitable")
        self.assertIn("minutes per task", html)
        ranking = html.split("Full Ranking")[1].split("</table>")[0]
        self.assertNotIn("offloaded", ranking)

    def test_the_thresholds_are_stated_in_the_report(self):
        html = self.render(self.field(("steady", 60.0, 45.0)))
        self.assertIn(f"{report.SLOW_TASK_S:.0f}s", html)
        self.assertIn(f"{report.UNUSABLE_TASK_S:.0f}s", html)


class TestGenerationFloor(ReportCase):
    """The gate keys on generation rate, which is where the physics is.

    How much of a model sits outside VRAM matters far less than how much of it moves
    per token. A mixture of experts moves a few billion parameters whatever its
    placement; a dense model moves all of them across the bus. The two separate on
    generation rate by a much wider margin than on any wall-clock total.
    """

    def render_one(self, **probe):
        self.write("screen.jsonl", [screen_record("m", "basic"), screen_record("m", "hard")])
        self.write("probe.jsonl", [probe_record("m", **probe)])
        return self.render(["m"])

    def verdict(self, html):
        import re
        m = re.search(r'<h3>m</h3><span class="pill p-(\w+)">',
                      html.split("<h2>Assessment</h2>")[1])
        return m.group(1) if m else None

    def test_a_model_slower_than_reading_speed_is_excluded(self):
        html = self.render_one(gen=6.0, moe=False, pct_gpu=36.0)
        self.assertEqual(self.verdict(html), "unsuitable")
        self.assertIn("below 20", html)
        # Excluded before scoring, so there is no ranking for it to appear in.
        self.assertIn("No model cleared", html)
        self.assertNotIn("Full Ranking", html)

    def test_poor_placement_alone_does_not_condemn_a_model(self):
        # The sharpest case in the measured field: a mixture of experts at a third
        # resident, generating faster than anything else on the card.
        html = self.render_one(gen=60.1, moe=True, pct_gpu=33.0)
        self.assertEqual(self.verdict(html), "suitable")

    def test_the_reason_carries_its_own_explanation(self):
        html = self.render_one(gen=13.0, moe=False, pct_gpu=47.0)
        self.assertIn("dense and 47% resident", html)

    def test_an_older_record_without_architecture_still_reports_the_rate(self):
        rec = probe_record("m", gen=6.0)
        del rec["moe"]
        self.write("screen.jsonl", [screen_record("m", "basic"), screen_record("m", "hard")])
        self.write("probe.jsonl", [rec])
        html = self.render(["m"])
        self.assertEqual(self.verdict(html), "unsuitable")
        self.assertIn("40% resident", html)
        self.assertNotIn("dense and", html)


class TestQuantisationColumn(ReportCase):
    def test_missing_quantisation_renders_as_a_dash(self):
        # The server is unreachable here, so no quantisation can be resolved and
        # every row falls back to the placeholder.
        self.write("screen.jsonl", [screen_record("solo", "basic"),
                                    screen_record("solo", "hard")])
        self.write("probe.jsonl", [probe_record("solo")])
        html = self.render(["solo"])
        self.assertNotIn("&amp;mdash;", html)
        self.assertIn('<td class="figure">&mdash;</td>', html)


class TestSpeedScore(ReportCase):
    """Latency should cost points in proportion to the time actually lost."""

    def test_slowest_model_is_not_scored_zero(self):
        recs, probes = [], []
        for name, prefill in (("quick", 40.0), ("slow", 80.0)):
            recs += [screen_record(name, "basic"), screen_record(name, "hard")]
            probes.append(probe_record(name, prefill=prefill, gen=100.0))
        self.write("screen.jsonl", recs)
        self.write("probe.jsonl", probes)
        html = self.render(["quick", "slow"])
        table = html.split("Full Ranking")[1].split("</table>")[0]
        rows = re.findall(r"<tr class=\"r-\w+\">.*?</tr>", table, re.S)
        cells = [[re.sub(r"<[^>]+>", "", c).strip()
                  for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", r, re.S)] for r in rows]
        speeds = {c[1]: float(c[4]) for c in cells if len(c) > 4}
        self.assertEqual(speeds["quick"], 100.0)
        # 47.7s against 87.7s modelled: slower, but far from a zero score.
        self.assertGreater(speeds["slow"], 50.0)
        self.assertLess(speeds["slow"], 60.0)


class TestUnknownArchitecture(ReportCase):
    """Absence of evidence is not evidence of density.

    A probe record predating the architecture field, or one whose model was gone
    from the server when it was written, carries no `moe` key. Rendering that as
    "dense" would attach the wrong cause to a slow result — and to a fast one it
    would misdescribe the model outright.
    """

    def why(self, html):
        """The card's reason text, not the section's explanatory prose."""
        import re
        return re.search(r'<p class="why">(.*?)</p>', html, re.S).group(1)

    def test_a_record_without_the_field_is_not_called_dense(self):
        rec = probe_record("m", gen=6.0)
        del rec["moe"]
        self.write("screen.jsonl", [screen_record("m", "basic"), screen_record("m", "hard")])
        self.write("probe.jsonl", [rec])
        why = self.why(self.render(["m"]))
        self.assertIn("generates 6 tok/s", why)
        self.assertIn("40% resident", why)
        self.assertNotIn("dense", why)

    def test_an_explicit_false_is_reported_as_dense(self):
        self.write("screen.jsonl", [screen_record("m", "basic"), screen_record("m", "hard")])
        self.write("probe.jsonl", [probe_record("m", gen=6.0, moe=False)])
        self.assertIn("dense and", self.why(self.render(["m"])))


class TestTruncatedReplies(ReportCase):
    """A reply cut at the output budget is a floor, not a measurement.

    The budget covers a model's reasoning as well as its answer. When it binds, the
    grader sees an unfinished reply and records a failure — which is indistinguishable
    in the score from a model that answered wrongly. The report says so instead of
    letting the number stand unqualified.
    """

    def render_with(self, capped):
        basic = screen_record("m", "basic")
        basic["hit_cap_n"] = capped
        self.write("screen.jsonl", [basic, screen_record("m", "hard")])
        self.write("probe.jsonl", [probe_record("m")])
        import re
        assessment = self.render(["m"]).split("<h2>Assessment</h2>")[1]
        return re.search(r'<p class="why">(.*?)</p>', assessment, re.S).group(1)

    def test_truncation_is_reported_on_the_card(self):
        self.assertIn("2 replies cut at the output budget", self.render_with(2))

    def test_a_single_truncation_reads_correctly(self):
        self.assertIn("1 reply cut at the output budget", self.render_with(1))

    def test_a_clean_run_says_nothing_about_the_budget(self):
        self.assertNotIn("output budget", self.render_with(0))


class TestQualityFloor(ReportCase):
    """A fast model that does not do the work must not lead the ranking.

    This is the failure the floor exists to prevent: an 8B model that fails most
    of its tasks returns in a fraction of the time precisely because it is not
    doing them, and unconditional latency credit put it at rank 1 above a model
    scoring 100%.
    """

    def field(self):
        # "quick" is twelve times faster and answers under half its tasks; "solid"
        # gets everything right and takes its time over it.
        self.write("screen.jsonl", [
            screen_record("solid", "basic", rate=100.0),
            screen_record("solid", "hard", rate=100.0),
            screen_record("quick", "basic", rate=57.1),
            screen_record("quick", "hard", rate=46.7),
        ])
        self.write("probe.jsonl", [probe_record("solid", gen=47.0),
                                   probe_record("quick", gen=279.0)])
        return report.analyse(self.cfg, ["solid", "quick"])

    def test_the_low_quality_model_is_ranked_below(self):
        A = self.field()
        self.assertEqual(A.sc_rank[0], "solid")
        self.assertLess(A.sc["quick"]["composite"], A.sc["solid"]["composite"])

    def test_speed_earns_nothing_below_the_floor(self):
        A = self.field()
        self.assertFalse(A.sc["quick"]["above_floor"])
        self.assertEqual(A.sc["quick"]["composite"], round(0.6 * 46.7, 1),
                         "below the floor the composite is quality alone")
        self.assertGreater(A.sc["quick"]["speed"], 100.0,
                           "a below-floor model faster than every usable one still "
                           "reports the speed it earns nothing for")

    def test_speed_still_separates_models_above_the_floor(self):
        self.write("screen.jsonl", [
            screen_record("slow", "basic", rate=90.0), screen_record("slow", "hard", rate=90.0),
            screen_record("fast", "basic", rate=90.0), screen_record("fast", "hard", rate=90.0),
        ])
        self.write("probe.jsonl", [probe_record("slow", gen=20.0),
                                   probe_record("fast", gen=80.0)])
        A = report.analyse(self.cfg, ["slow", "fast"])
        self.assertTrue(all(A.sc[m]["above_floor"] for m in ("slow", "fast")))
        self.assertEqual(A.sc_rank[0], "fast")

    def test_no_below_floor_model_outranks_an_above_floor_one(self):
        A = self.field()
        over = [m for m in A.sc_rank if A.sc[m]["above_floor"]]
        under = [m for m in A.sc_rank if not A.sc[m]["above_floor"]]
        self.assertTrue(over and under, "the field must contain both to be a test")
        self.assertLess(max(A.sc_rank.index(m) for m in over),
                        min(A.sc_rank.index(m) for m in under))

    def test_the_rendered_ranking_leads_with_a_qualifying_model(self):
        self.field()
        html = self.render(["solid", "quick"])
        rows = re.findall(r'<tr class="r-\w+">\s*<td[^>]*>\d+</td>\s*<th>(.*?)</th>',
                          html, re.S)
        self.assertTrue(rows, "the ranking table did not render")
        self.assertIn("solid", rows[0])
        self.assertNotIn("below floor", rows[0])


class TestUnmeasuredOnTheHardSet(ReportCase):
    """A model missing a record is unmeasured, not good.

    Every gate is written as "if the measurement says so", so a missing record
    clears all of them at once. A model with one basic run and no hard run was
    labelled suitable on the strength of a measurement never taken, and a model
    that was screened but never probed cleared the truncation, retrieval,
    generation-rate and task-time gates the same way.
    """

    def only_basic(self):
        self.write("screen.jsonl", [screen_record("partial", "basic", rate=64.3)])
        self.write("probe.jsonl", [probe_record("partial")])
        return report.analyse(self.cfg, ["partial"])

    def test_it_is_not_called_suitable(self):
        sev, why = self.only_basic().verdict("partial")
        self.assertNotEqual(sev, "suitable")
        self.assertIn("no hard-set result", why)

    def test_it_is_set_aside_before_scoring(self):
        A = self.only_basic()
        self.assertIn("no hard-set result", A.eligible("partial"))
        self.assertNotIn("partial", A.sc_rank)

    def test_a_model_that_was_never_probed_is_not_suitable_either(self):
        self.write("screen.jsonl", [screen_record("noprobe", "basic", rate=100.0),
                                    screen_record("noprobe", "hard", rate=100.0)])
        self.write("probe.jsonl", [])
        A = report.analyse(self.cfg, ["noprobe"])
        sev, why = A.verdict("noprobe")
        self.assertNotEqual(sev, "suitable")
        self.assertIn("no probe result", why)
        self.assertIn("no probe result", A.eligible("noprobe"))
        self.assertNotIn("noprobe", A.sc_rank)

    def test_a_null_taskset_still_reads_as_a_basic_result(self):
        # Records written before the sets were named carry no taskset at all; one
        # written with an explicit null must not vanish into a third bucket.
        rec = screen_record("legacy", "basic", rate=80.0)
        rec["taskset"] = None
        self.write("screen.jsonl", [rec])
        A = report.analyse(self.cfg, ["legacy"])
        self.assertIn("legacy", A.t1)
        self.assertEqual(A.t1["legacy"]["rate"], 80.0)

    def test_a_measured_model_is_unaffected(self):
        self.write("screen.jsonl", [screen_record("full", "basic", rate=100.0),
                                    screen_record("full", "hard", rate=100.0)])
        self.write("probe.jsonl", [probe_record("full")])
        A = report.analyse(self.cfg, ["full"])
        self.assertEqual(A.verdict("full")[0], "suitable")
        self.assertEqual(A.eligible("full"), [])


class TestSpeedDenominator(ReportCase):
    """The fastest time is taken from models that pass, not models that give up."""

    def test_a_failing_sprinter_does_not_compress_the_field(self):
        self.write("screen.jsonl", [
            screen_record("good", "basic", rate=100.0), screen_record("good", "hard", rate=100.0),
            screen_record("mid", "basic", rate=90.0), screen_record("mid", "hard", rate=90.0),
            screen_record("bail", "basic", rate=47.0), screen_record("bail", "hard", rate=47.0),
        ])
        self.write("probe.jsonl", [probe_record("good", gen=45.0), probe_record("mid", gen=90.0),
                                   probe_record("bail", gen=600.0)])
        A = report.analyse(self.cfg, ["good", "mid", "bail"])
        self.assertEqual(A.sc["mid"]["speed"], 100.0,
                         "the fastest model above the floor sets the reference")
        self.assertGreater(A.sc["good"]["speed"], 20.0,
                           "a real contender is not compressed into single digits by a "
                           "model that is quick because it fails")

    def test_removing_a_below_floor_model_leaves_the_others_untouched(self):
        # Pruning failures must not silently restate everyone else's numbers.
        records = [
            screen_record("good", "basic", rate=100.0), screen_record("good", "hard", rate=100.0),
            screen_record("mid", "basic", rate=90.0), screen_record("mid", "hard", rate=90.0),
        ]
        probes = [probe_record("good", gen=45.0), probe_record("mid", gen=90.0)]
        self.write("screen.jsonl", records + [screen_record("bail", "basic", rate=47.0),
                                              screen_record("bail", "hard", rate=47.0)])
        self.write("probe.jsonl", probes + [probe_record("bail", gen=600.0)])
        before = report.analyse(self.cfg, ["good", "mid", "bail"]).sc

        self.write("screen.jsonl", records)
        self.write("probe.jsonl", probes)
        after = report.analyse(self.cfg, ["good", "mid"]).sc
        for m in ("good", "mid"):
            self.assertEqual(before[m]["composite"], after[m]["composite"], m)


class TestVerdictNeverImproves(ReportCase):
    """No check may raise a verdict that an earlier check lowered.

    The gates are applied in sequence over one severity variable, so a later
    assignment can undo an earlier disqualification. A missing record in
    particular must never rehabilitate a model that a record it does have
    already ruled out.
    """

    def test_a_gap_does_not_rescue_a_disqualified_model(self):
        broken = screen_record("broken", "basic", rate=100.0)
        broken["tasks"][1]["format_ok"] = False       # a tool call the harness cannot parse
        broken["tasks"][1]["passed"] = False
        self.write("screen.jsonl", [broken])          # and no hard set, and no probe
        A = report.analyse(self.cfg, ["broken"])
        sev, why = A.verdict("broken")
        self.assertEqual(sev, "unsuitable")
        self.assertTrue(any("malformed" in w for w in why))
        self.assertIn("no hard-set result", why)


class TestPartialDataIsNeverSuitable(ReportCase):
    """Every combination of missing records, rather than the ones already hit twice.

    Both bugs found in this area were the same shape: a gate phrased as "if the
    measurement says so" passing because there was no measurement. This asserts
    the property over the whole space instead of one case at a time.
    """

    COMBINATIONS = [(basic, hard, probe)
                    for basic in (True, False)
                    for hard in (True, False)
                    for probe in (True, False)]

    def test_only_a_fully_measured_model_can_be_suitable(self):
        for has_basic, has_hard, has_probe in self.COMBINATIONS:
            with self.subTest(basic=has_basic, hard=has_hard, probe=has_probe):
                screen = []
                if has_basic:
                    screen.append(screen_record("m", "basic", rate=100.0))
                if has_hard:
                    screen.append(screen_record("m", "hard", rate=100.0))
                self.write("screen.jsonl", screen)
                self.write("probe.jsonl", [probe_record("m")] if has_probe else [])
                A = report.analyse(self.cfg, ["m"])
                complete = has_hard and has_probe
                sev = A.verdict("m")[0]
                if complete:
                    self.assertEqual(sev, "suitable")
                    self.assertEqual(A.eligible("m"), [])
                    self.assertIn("m", A.sc_rank)
                else:
                    self.assertNotEqual(sev, "suitable",
                                        "a perfect score on records that exist is not a "
                                        "verdict on records that do not")
                    self.assertTrue(A.eligible("m"))
                    self.assertNotIn("m", A.sc_rank)


def agent_record(model, task="ag_module", passed=True, score=None):
    return dict(model=model, task=task, passed=passed, score=score, detail="",
                wall_s=300.0, timed_out=False, returncode=0, tool_calls=10,
                tools=[], turns=8, errors=[], tokens={}, peak_input_tokens=0)


class TestAgentResultsInTheVerdict(ReportCase):
    """Finishing work unattended is a different question from answering a prompt.

    The agent stage runs on a subset of the field, so its results can only lower a
    verdict. Rewarding them would rank a model above another for the accident of
    having been measured, which is the shape of two bugs already found here.
    """

    def field(self, agentic):
        self.write("screen.jsonl", [screen_record("m", "basic", rate=100.0),
                                    screen_record("m", "hard", rate=100.0)])
        self.write("probe.jsonl", [probe_record("m")])
        self.write("agentic.jsonl", agentic)
        return report.analyse(self.cfg, ["m"])

    def test_no_agent_data_leaves_the_verdict_alone(self):
        self.assertEqual(self.field([]).verdict("m")[0], "suitable")

    def test_completing_every_task_does_not_promote(self):
        A = self.field([agent_record("m", passed=True, score=100.0)])
        sev, why = A.verdict("m")
        self.assertEqual(sev, "suitable")
        self.assertEqual(why, [], "a clean sweep needs no remark")

    def test_one_failure_is_reported_but_does_not_lower_the_verdict(self):
        # This stage is variable: the same model answered one task by writing
        # nothing in two turns, then by making four edits in ten, then by writing
        # nothing again. The screen would not report a rate from a single run, and
        # neither should a stage with more spread than the screen.
        A = self.field([dict(agent_record("m", task="ag_feature", passed=False), ts=1.0)])
        sev, why = A.verdict("m")
        self.assertEqual(sev, "suitable")
        self.assertTrue(any("once, not retried" in w for w in why),
                        "the reader is still told it happened")

    def test_a_failure_that_survives_a_retry_does_lower_it(self):
        A = self.field([dict(agent_record("m", task="ag_feature", passed=False), ts=1.0),
                        dict(agent_record("m", task="ag_feature", passed=False), ts=2.0)])
        sev, why = A.verdict("m")
        self.assertEqual(sev, "limited")
        self.assertTrue(any("on every attempt" in w for w in why))

    def test_a_retry_that_succeeds_clears_the_task(self):
        A = self.field([dict(agent_record("m", task="ag_feature", passed=False), ts=1.0),
                        dict(agent_record("m", task="ag_feature", passed=True), ts=2.0)])
        self.assertEqual(A.verdict("m"), ("suitable", []))

    def test_a_retry_that_succeeds_first_still_clears_the_task(self):
        # Order in the ledger must not decide it; a task the model has completed is
        # a task it can complete.
        A = self.field([dict(agent_record("m", task="ag_feature", passed=True), ts=1.0),
                        dict(agent_record("m", task="ag_feature", passed=False), ts=2.0)])
        self.assertEqual(A.verdict("m")[0], "suitable")

    def test_it_cannot_rescue_a_model_the_screen_ruled_out(self):
        self.write("screen.jsonl", [screen_record("m", "basic", rate=100.0),
                                    screen_record("m", "hard", rate=47.0)])
        self.write("probe.jsonl", [probe_record("m")])
        self.write("agentic.jsonl", [agent_record("m", passed=True, score=100.0)])
        self.assertEqual(report.analyse(self.cfg, ["m"]).verdict("m")[0], "unsuitable")

    def test_selection_ignores_agent_results_so_it_cannot_be_circular(self):
        # A model downgraded for having no agent record would be dropped from the
        # stage on that basis, and so could never acquire one.
        A = self.field([agent_record("m", passed=False, score=0.0)])
        self.assertEqual(A.verdict("m", with_agent=False)[0], "suitable")
        self.assertEqual(A.by_verdict(["suitable"]), ["m"],
                         "the stage must still pick up a model its own results failed")


class TestAgentAttemptsAreGrouped(ReportCase):
    """The agent ledger appends, so a task measured again leaves several records.

    They are samples of the same task rather than one superseding another, so
    they collapse to a single row that carries how many attempts were made and
    how many succeeded.
    """

    def field(self, agentic):
        self.write("screen.jsonl", [screen_record("m", "basic", rate=100.0),
                                    screen_record("m", "hard", rate=100.0)])
        self.write("probe.jsonl", [probe_record("m")])
        self.write("agentic.jsonl", agentic)
        return report.analyse(self.cfg, ["m"])

    def test_repeats_of_one_task_become_one_row_carrying_the_count(self):
        A = self.field([dict(agent_record("m", task="ag_feature", passed=False), ts=1.0),
                        dict(agent_record("m", task="ag_feature", passed=False), ts=2.0),
                        dict(agent_record("m", task="ag_feature", passed=True), ts=3.0)])
        self.assertEqual(len(A.ag["m"]), 1, "one row per task, not per attempt")
        row = A.ag["m"][0]
        self.assertEqual((row["attempts"], row["completed"]), (3, 1))

    def test_different_tasks_are_all_kept(self):
        A = self.field([dict(agent_record("m", task="ag_feature"), ts=1.0),
                        dict(agent_record("m", task="ag_module"), ts=2.0)])
        self.assertEqual({r["task"] for r in A.ag["m"]}, {"ag_feature", "ag_module"})


def served(passed=True):
    """An 18-check result whose `serves` outcome is the thing under test."""
    names = ["package_layout", "pyproject", "importable", "serves", "seed_three",
             "task_shape", "create", "read_one", "update", "toggle_done", "delete",
             "filter_done", "filter_text", "reorder", "persist", "ui_page",
             "ui_drag", "own_tests"]
    return [dict(name=n, passed=(passed if n in ("package_layout", "pyproject",
                                                 "importable", "serves") else passed),
                 detail="") for n in names]


class TestForeignRecordsAreIgnored(ReportCase):
    """Another tool may share this results directory; its records are not ours.

    The application task now lives in appsift with its own ledger, but records
    left by an earlier version of this project, or by any other tool writing
    here, must not appear in the screen's section or in a verdict.
    """

    def field(self, agentic):
        self.write("screen.jsonl", [screen_record("m", "basic", rate=100.0),
                                    screen_record("m", "hard", rate=100.0)])
        self.write("probe.jsonl", [probe_record("m")])
        self.write("agentic.jsonl", agentic)
        return report.analyse(self.cfg, ["m"])

    def stale(self):
        checks = [dict(name=n, passed=False, detail="")
                  for n in ("package_layout", "serves", "persist")]
        return dict(agent_record("m", task="ag_todoapp", passed=False),
                    checks=checks, ts=1.0, score=0.0)

    def test_the_screen_no_longer_owns_that_task(self):
        self.assertNotIn("ag_todoapp", [t["id"] for t in report.AGENT_TASKS])

    def test_a_stale_record_is_not_shown(self):
        self.assertEqual(self.field([self.stale()]).ag["m"], [])

    def test_a_stale_record_does_not_reach_the_verdict(self):
        A = self.field([self.stale()])
        self.assertEqual(A.verdict("m"), ("suitable", []),
                         "another tool's record must not grade a model in silence")

    def test_our_own_records_beside_it_are_unaffected(self):
        A = self.field([self.stale(),
                        dict(agent_record("m", task="ag_fixbug", passed=True), ts=2.0)])
        self.assertEqual([r["task"] for r in A.ag["m"]], ["ag_fixbug"])


class TestThePivotIsPerTask(ReportCase):
    """Each graded task names the check everything else rests on.

    Hard-coding the application's `serves` meant the rule would go dead the
    moment that task changed or was dropped, and would never apply to the module
    task, whose equivalent is `importable`.
    """

    def field(self, agentic):
        self.write("screen.jsonl", [screen_record("m", "basic", rate=100.0),
                                    screen_record("m", "hard", rate=100.0)])
        self.write("probe.jsonl", [probe_record("m")])
        self.write("agentic.jsonl", agentic)
        return report.analyse(self.cfg, ["m"])

    def module_record(self, importable, ts=1.0):
        checks = [dict(name="layout", passed=True, detail=""),
                  dict(name="importable", passed=importable, detail=""),
                  dict(name="add", passed=importable, detail="")]
        return dict(agent_record("m", task="ag_module", passed=False),
                    checks=checks, ts=ts, score=33.3)

    def test_every_graded_task_declares_one(self):
        graded = [t for t in report.AGENT_TASKS if t.get("graded")]
        self.assertTrue(graded)
        for task in graded:
            with self.subTest(task=task["id"]):
                self.assertTrue(task.get("pivot"),
                                "a graded task must name the check others rest on")

    def test_a_module_that_does_not_import_is_unsuitable(self):
        A = self.field([self.module_record(importable=False)])
        sev, why = A.verdict("m")
        self.assertEqual(sev, "unsuitable")
        self.assertTrue(any("module never ran" in w for w in why), why)

    def test_a_module_that_imports_is_not_condemned_for_scoring_badly(self):
        A = self.field([self.module_record(importable=True)])
        self.assertNotEqual(A.verdict("m")[0], "unsuitable")

    def test_importing_on_any_attempt_clears_it(self):
        A = self.field([self.module_record(importable=False, ts=1.0),
                        self.module_record(importable=True, ts=2.0)])
        self.assertNotEqual(A.verdict("m")[0], "unsuitable")


class TestRegradeReportsFlipsOnly(ReportCase):
    """A record that gains a score has not been re-judged.

    Reporting it as a flip labelled every rescored record `FAIL->PASS`, including
    tasks the model had passed all along, and buried the two genuine corrections
    under three hundred and sixty lines of noise.
    """

    def records(self, **overrides):
        from tests import reference
        answer = "```python\n" + reference.HARD["h_cg_roman"] + "\n```"
        base = dict(model="m", run=1, taskset="hard", task="h_cg_roman",
                    kind="codegen", passed=True, format_ok=True, detail="ok",
                    wall=1.0, raw=answer)
        return [dict(base, **overrides)]

    def run_regrade(self, records):
        from codesift import regrade
        self.write("screen_tasks.jsonl", records)
        out = io.StringIO()
        regrade.run(self.cfg, stream=out)
        return out.getvalue()

    def test_a_record_that_only_gains_a_score_is_not_a_flip(self):
        text = self.run_regrade(self.records())      # no "score" key at all
        self.assertIn("0 verdict(s) change", text)
        self.assertNotIn("FAIL->PASS", text)
        self.assertIn("rescored", text)

    def test_a_real_change_of_verdict_is_still_reported(self):
        text = self.run_regrade(self.records(passed=False, score=0.0))
        self.assertIn("1 verdict(s) change", text)
        self.assertIn("FAIL->PASS", text)


class TestAssessmentCardsCarryTheSameFigures(ReportCase):
    """A model should be readable on its own card, without cross-referencing.

    The assessment card showed the task sets and the probe; the recommendation
    card showed task time and speed score. Comparing two models meant reading two
    sections, and a model excluded before scoring appeared only in one of them.
    """

    FIELDS = ("Basic set", "Hard set", "Task time", "Speed score",
              "Prefill @48k", "Generation")

    def field(self, models):
        screen, probe = [], []
        for name, rate, gen in models:
            screen += [screen_record(name, "basic", rate=rate),
                       screen_record(name, "hard", rate=rate)]
            probe.append(probe_record(name, gen=gen))
        self.write("screen.jsonl", screen)
        self.write("probe.jsonl", probe)
        return self.render([m[0] for m in models])

    def card(self, html, model):
        """The assessment card, not the recommendation one, which comes first."""
        section = html[html.index("<h2>Assessment</h2>"):]
        start = section.index(f"<h3>{model}</h3>")
        return section[start:section.index("</article>", start)]

    def test_every_figure_from_the_recommendation_card_is_present(self):
        html = self.field([("good", 100.0, 60.0)])
        card = self.card(html, "good")
        for label in self.FIELDS:
            with self.subTest(field=label):
                self.assertIn(f"<dt>{label}</dt>", card)

    def test_a_model_excluded_before_scoring_still_shows_what_it_has(self):
        # It has no speed score, since scoring never reached it; the figures that
        # were measured must still appear, and the missing one as absent.
        html = self.field([("good", 100.0, 60.0), ("slow", 100.0, 5.0)])
        card = self.card(html, "slow")
        self.assertIn("<dt>Speed score</dt><dd>&mdash;</dd>", card)
        self.assertIn("<dt>Generation</dt><dd>5 t/s</dd>", card)
        self.assertIn("<dt>Hard set</dt>", card)

    def test_a_missing_figure_is_never_shown_as_a_zero(self):
        self.write("screen.jsonl", [screen_record("noprobe", "basic", rate=90.0),
                                    screen_record("noprobe", "hard", rate=90.0)])
        self.write("probe.jsonl", [])
        card = self.card(self.render(["noprobe"]), "noprobe")
        self.assertIn("<dt>Task time</dt><dd>&mdash;</dd>", card)
        self.assertIn("<dt>Generation</dt><dd>&mdash;</dd>", card)
        self.assertNotIn("<dd>0s</dd>", card)
