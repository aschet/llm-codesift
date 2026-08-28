# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""Every task must be well formed, solvable, and not already satisfied.

A task whose reference solution fails is unwinnable; a defect task whose starting
code already passes measures nothing. Both are silent in a measurement run, so
they are caught here.
"""
import contextlib
import io
import unittest

from codesift.tasks import TASKS

from . import reference

VALID_KINDS = {"codegen", "edit", "format", "toolcall", "trace"}


def _exec(source, extra=""):
    namespace = {}
    exec(source, namespace)
    if extra:
        exec(extra, namespace)
    return namespace


class TestTaskShape(unittest.TestCase):
    def test_ids_are_unique(self):
        ids = [t["id"] for t in TASKS]
        self.assertEqual(len(ids), len(set(ids)), "task ids must be unique")

    def test_required_fields(self):
        for task in TASKS:
            with self.subTest(task=task["id"]):
                self.assertIn(task["kind"], VALID_KINDS)
                self.assertTrue(task["prompt"].strip())
                if task["kind"] in ("codegen", "edit"):
                    self.assertTrue(task["tests"].strip())
                if task["kind"] == "edit":
                    self.assertTrue(task["code"].strip())
                if task["kind"] == "trace":
                    self.assertTrue(task["expect"].strip())
                if task["kind"] == "format":
                    self.assertIn(task["check"],
                                  {"json_exact", "bare_code", "one_word", "no_comments"})


class TestTraceExpectations(unittest.TestCase):
    """The stated answer must equal what the snippet actually prints."""

    def test_expected_output_matches_execution(self):
        for task in (t for t in TASKS if t["kind"] == "trace"):
            with self.subTest(task=task["id"]):
                snippet = task["prompt"].split("\n\n", 1)[1]
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    exec(snippet, {})
                self.assertEqual(buffer.getvalue().strip(), task["expect"])


class TestSolvability(unittest.TestCase):
    def test_reference_solutions_pass(self):
        solutions = reference.SOLUTIONS
        for task in TASKS:
            if task["kind"] not in ("codegen", "edit"):
                continue
            with self.subTest(task=task["id"]):
                self.assertIn(task["id"], solutions, "no reference solution")
                _exec(solutions[task["id"]], task["tests"])

    def test_defective_code_actually_fails(self):
        for task in TASKS:
            if task["kind"] != "edit":
                continue
            with self.subTest(task=task["id"]):
                with self.assertRaises(Exception, msg="starting code already passes"):
                    _exec(task["code"], task["tests"])


if __name__ == "__main__":
    unittest.main()
