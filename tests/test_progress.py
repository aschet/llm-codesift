# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""The output is TAP 14, and a consumer is entitled to hold it to the grammar.

https://testanything.org/tap-version-14-specification.html

Only the constraints the specification states are tested here. What the stages
choose to say is theirs, and no other test reads this output -- a gate's verdict
is asserted from its record, not from its printout.
"""
import io
import re
import unittest

from codesift import progress


def written(build) -> str:
    out = io.StringIO()
    with progress.document(out):
        build(out)
    return out.getvalue()


class TestTheDocument(unittest.TestCase):
    def setUp(self):
        progress.reset()

    def test_the_version_line_comes_first(self):
        # "To indicate that this is TAP14 the first line must be TAP version 14".
        text = written(lambda out: progress.note("anything", out))
        self.assertEqual(text.splitlines()[0], "TAP version 14")

    def test_the_plan_counts_the_points_at_its_own_level(self):
        def build(out):
            for i, name in enumerate(("a:1", "b:1"), 1):
                progress.subject(i, 2, name, stream=out)
                progress.result("done", stream=out)
        self.assertEqual(written(build).splitlines()[-1], "1..2")

    def test_a_document_with_nothing_in_it_still_plans(self):
        self.assertEqual(written(lambda out: None).splitlines()[-1],
                         "1..0 # no test points")

    def test_the_reporter_can_write_a_second_document(self):
        written(lambda out: progress.note("first", out))
        self.assertEqual(written(lambda out: None).splitlines()[0], "TAP version 14")


class TestSubtests(unittest.TestCase):
    def setUp(self):
        progress.reset()

    def build(self, out, failing=False):
        progress.subject(1, 1, "gemma4:e4b", stream=out)
        progress.unit("triage", "speed", progress.OK, 11.8, "115 tok/s", stream=out)
        progress.unit("triage", "context",
                      progress.FAIL if failing else progress.OK, 88.1, "held",
                      stream=out)
        progress.result("cleared", stream=out)

    def test_a_named_subtest_is_terminated_by_a_matching_description(self):
        # "A Commented Subtest with a Subtest Name must be terminated by a Test
        # Point with a matching Description."
        lines = written(self.build).splitlines()
        [name] = [l.split(": ", 1)[1] for l in lines if l.strip().startswith("# Subtest:")]
        closing = [l for l in lines if re.fullmatch(rf"(not )?ok \d+ - {re.escape(name)}", l)]
        self.assertTrue(closing, f"nothing terminates the subtest named {name}")

    def test_a_subtest_is_indented_four_spaces_and_its_yaml_two_further(self):
        lines = written(self.build).splitlines()
        point = next(l for l in lines if l.lstrip().startswith("ok 1 - triage speed"))
        self.assertEqual(len(point) - len(point.lstrip()), 4)
        opener = lines[lines.index(point) + 1]
        self.assertEqual(opener, " " * 6 + "---")

    def test_a_failing_point_fails_the_subtest_that_holds_it(self):
        # Not stated as a rule, but every consumer assumes it, and a passing
        # parent over a failing child is what a reader cannot make sense of.
        lines = written(lambda out: self.build(out, failing=True)).splitlines()
        self.assertIn("not ok 1 - gemma4:e4b", lines)

    def test_a_subject_with_no_units_is_a_point_not_an_empty_subtest(self):
        def build(out):
            progress.subject(1, 1, "m:1", stream=out)
            progress.result("cleared", stream=out)
        text = written(build)
        self.assertNotIn("# Subtest", text)
        self.assertIn("ok 1 - m:1", text)


class TestGrammar(unittest.TestCase):
    def setUp(self):
        progress.reset()

    def test_a_hash_in_a_description_is_escaped(self):
        # The first unescaped # opens the Directive slot, where only TODO and
        # SKIP mean anything.
        def build(out):
            progress.subject(1, 1, "m:1", stream=out)
            progress.unit("screen", "task #3", progress.OK, stream=out)
            progress.result("", stream=out)
        line = next(l for l in written(build).splitlines() if " - screen" in l)
        self.assertIn(r"\#", line)
        self.assertNotIn(" # ", line)

    def test_diagnostics_are_yaml_rather_than_a_directive(self):
        def build(out):
            progress.subject(1, 1, "m:1", stream=out)
            progress.unit("probe", "first token", seconds=6.5,
                          detail="115 tok/s", stream=out)
            progress.result("", stream=out)
        lines = written(build).splitlines()
        point = next(l for l in lines if "first token" in l)
        self.assertNotIn("#", point)
        self.assertIn("      duration_s: 6.5", lines)

    def test_every_line_is_something_the_grammar_names(self):
        def build(out):
            progress.subject(1, 1, "m:1", stream=out)
            progress.unit("screen", "cg_roman", progress.OK, 2.8, "ok", stream=out)
            progress.result("24 of 29 passed", stream=out)
            progress.summary("1 model", stream=out)
        allowed = re.compile(
            r"^\s*(TAP version 14|#.*|(not )?ok \d+ - .*|1\.\.\d+( # .*)?|"
            r"---|\.\.\.|[a-z_]+: .*|Bail out!.*)$")
        for line in written(build).splitlines():
            with self.subTest(line=line):
                self.assertRegex(line, allowed)

    def test_a_bail_out_is_written_at_the_margin(self):
        # "Bail out!" ends the document wherever it appears, so it is not indented
        # into whatever subtest happened to be open.
        out = io.StringIO()
        with progress.document(out):
            progress.subject(1, 1, "m:1", stream=out)
            progress.unit("probe", "first token", progress.FAIL, stream=out)
            progress.bail("the server stopped answering", stream=out)
        self.assertIn("Bail out! the server stopped answering", out.getvalue().splitlines())

    def test_an_abandoned_run_bails_instead_of_planning(self):
        # A plan would claim a count for test points that will never be written.
        out = io.StringIO()
        with self.assertRaises(SystemExit):
            with progress.document(out):
                raise SystemExit("unknown task id(s): cg_roman")
        lines = out.getvalue().splitlines()
        self.assertEqual(lines[-1], "Bail out! unknown task id(s): cg_roman")


if __name__ == "__main__":
    unittest.main()
