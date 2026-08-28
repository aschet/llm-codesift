# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""What the records support, asserted without rendering anything.

These read an `Assessment` directly. A verdict asserted by grepping the HTML it
produced was the shape that let two gates pass silently on missing data.
"""
import unittest

from codesift import analysis
from codesift.findings import sentence
from tests.records import RecordsCase, probe_record, screen_record


class TestLatencyVerdict(RecordsCase):
    """Latency is judged in two places, and neither is a fixed line in the report.

    A model too slow to use at all is rejected at triage on its generation rate,
    which is where a dense model spilling out of VRAM is caught. What survives is
    compared on the wait before its first token, relative to the quickest.
    """

    def field(self, *models):
        screen, probe = [], []
        for name, prefill, gen in models:
            screen.append(screen_record(name, rate=95.0))
            probe.append(probe_record(name, gen=gen, prefill=prefill))
        self.write_tasks(screen)
        self.write("probe.jsonl", probe)
        return analysis.analyse(self.cfg, [m[0] for m in models])

    def test_a_slow_prefill_is_not_a_verdict(self):
        # It is reported and compared, not judged against a line nobody derived.
        A = self.field(("quick", 20.0, 60.0), ("crawler", 130.0, 60.0))
        self.assertEqual(A.verdict("crawler")[0], "suitable")
        self.assertNotIn("modelled", sentence({"findings": A.verdict("crawler")[1]}))

    def test_the_slower_model_scores_lower_on_speed(self):
        A = self.field(("quick", 20.0, 60.0), ("crawler", 130.0, 60.0))
        self.assertEqual(A.sc["quick"]["speed"], 100.0)
        # 6.5 times slower to read the context, but generating at the same rate, so
        # a session separates them by less than the first token does.
        self.assertLess(A.sc["crawler"]["speed"], 50.0)

    def session_of(self, prefill, gen, depth):
        """The model the score states: read once, then re-read what was added."""
        rate = depth / prefill
        added = analysis.SESSION_TURNS * (analysis.ANSWER_TOKENS + analysis.USER_TOKENS)
        return ((depth + added) / rate
                + analysis.SESSION_TURNS * analysis.ANSWER_TOKENS / gen)

    def test_the_session_reads_once_and_writes_every_turn(self):
        # Coding is not one turn, and the server keeps the processed prefix between
        # them, so the context is read once and answers are written throughout.
        A = self.field(("quick", 20.0, 60.0), ("crawler", 130.0, 60.0))
        want = self.session_of(130.0, 60.0, 65536 * 3 // 4)
        self.assertAlmostEqual(A.session_time("crawler"), want, places=6)
        self.assertEqual(A.sc["crawler"]["session"], round(want, 1))

    def test_generation_rate_moves_the_score_on_its_own(self):
        # Two models that read the prompt at the same speed and write at different
        # ones are not equally fast, which scoring on first token alone said.
        A = self.field(("brisk", 20.0, 120.0), ("dawdler", 20.0, 30.0))
        self.assertEqual(A.sc["brisk"]["speed"], 100.0)
        want = (self.session_of(20.0, 120.0, 65536 * 3 // 4)
                / self.session_of(20.0, 30.0, 65536 * 3 // 4))
        self.assertAlmostEqual(A.sc["dawdler"]["speed"], round(100 * want, 1), places=1)
        self.assertLess(A.sc["dawdler"]["speed"], 90.0)

    def test_generation_below_the_floor_is_still_a_hard_gate(self):
        # The backstop for anything that reached the report without triage.
        A = self.field(("quick", 20.0, 60.0), ("bus", 20.0, 5.0))
        self.assertNotIn("bus", A.sc_rank)

class TestPartialDataIsNeverSuitable(RecordsCase):
    """Every combination of missing records, rather than the ones already hit twice.

    Both bugs found in this area were the same shape: a gate phrased as "if the
    measurement says so" passing because there was no measurement. This asserts
    the property over the whole space instead of one case at a time.
    """

    COMBINATIONS = [(tasks, probe)
                    for tasks in (True, False)
                    for probe in (True, False)]

    def test_only_a_fully_measured_model_can_be_suitable(self):
        for has_tasks, has_probe in self.COMBINATIONS:
            with self.subTest(tasks=has_tasks, probe=has_probe):
                self.write("screen_tasks.jsonl",
                           screen_record("m", rate=100.0) if has_tasks else [])
                self.write("probe.jsonl", [probe_record("m")] if has_probe else [])
                A = analysis.analyse(self.cfg, ["m"])
                complete = has_tasks and has_probe
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


if __name__ == "__main__":
    unittest.main()
