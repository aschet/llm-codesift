# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""Re-grading stored replies reports what changed, and only what changed."""
import io
import unittest

from tests.records import RecordsCase


class TestRegradeReportsFlipsOnly(RecordsCase):
    """A record that gains a score has not been re-judged.

    Reporting it as a flip labelled every rescored record `FAIL->PASS`, including
    tasks the model had passed all along, and buried the two genuine corrections
    under three hundred and sixty lines of noise.
    """

    def records(self, **overrides):
        from tests import reference
        answer = "```python\n" + reference.SOLUTIONS["cg_roman"] + "\n```"
        base = dict(model="m", run=1, task="cg_roman",
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


if __name__ == "__main__":
    unittest.main()
