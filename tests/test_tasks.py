"""Every task must be well formed, solvable, and not already satisfied.

A task whose reference solution fails is unwinnable; a defect task whose starting
code already passes measures nothing. Both are silent in a measurement run, so
they are caught here.
"""
import contextlib
import io
import shutil
import unittest
from pathlib import Path

from codesift import agent
from codesift.tasks import AGENT_TASKS, TASKSETS

from . import reference

VALID_KINDS = {"codegen", "edit", "format", "toolcall", "trace"}


def _exec(source, extra=""):
    namespace = {}
    exec(source, namespace)
    if extra:
        exec(extra, namespace)
    return namespace


class TestTaskShape(unittest.TestCase):
    def test_ids_unique_across_suites(self):
        ids = [t["id"] for suite in TASKSETS.values() for t in suite]
        ids += [t["id"] for t in AGENT_TASKS]
        self.assertEqual(len(ids), len(set(ids)), "task ids must be unique")

    def test_required_fields(self):
        for name, suite in TASKSETS.items():
            for task in suite:
                with self.subTest(taskset=name, task=task["id"]):
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
        for name, suite in TASKSETS.items():
            for task in (t for t in suite if t["kind"] == "trace"):
                with self.subTest(taskset=name, task=task["id"]):
                    snippet = task["prompt"].split("\n\n", 1)[1]
                    buffer = io.StringIO()
                    with contextlib.redirect_stdout(buffer):
                        exec(snippet, {})
                    self.assertEqual(buffer.getvalue().strip(), task["expect"])


class TestSolvability(unittest.TestCase):
    def test_reference_solutions_pass(self):
        for suite_name, solutions in (("basic", reference.BASIC), ("hard", reference.HARD)):
            for task in TASKSETS[suite_name]:
                if task["kind"] not in ("codegen", "edit"):
                    continue
                with self.subTest(taskset=suite_name, task=task["id"]):
                    self.assertIn(task["id"], solutions, "no reference solution")
                    _exec(solutions[task["id"]], task["tests"])

    def test_defective_code_actually_fails(self):
        for suite_name in ("basic", "hard"):
            for task in TASKSETS[suite_name]:
                if task["kind"] != "edit":
                    continue
                with self.subTest(taskset=suite_name, task=task["id"]):
                    with self.assertRaises(Exception,
                                           msg="starting code already passes"):
                        _exec(task["code"], task["tests"])


class TestAgentTasks(unittest.TestCase):
    def setUp(self):
        self.workdir = Path(self.enterContext(__import__("tempfile").TemporaryDirectory()))

    def test_seeded_repository_fails_verification(self):
        for task in AGENT_TASKS:
            with self.subTest(task=task["id"]):
                repo = agent.seed(task, self.workdir)
                passed, _, _ = agent.verify(task, repo)
                self.assertFalse(passed, "seeded repository already passes")
                shutil.rmtree(repo, ignore_errors=True)

    def test_reference_edit_satisfies_verification(self):
        for task in AGENT_TASKS:
            with self.subTest(task=task["id"]):
                repo = agent.seed(task, self.workdir)
                for rel, content in reference.AGENT[task["id"]].items():
                    (repo / rel).parent.mkdir(parents=True, exist_ok=True)
                    (repo / rel).write_text(content, encoding="utf-8")
                passed, detail, _ = agent.verify(task, repo)
                self.assertTrue(passed, f"reference solution rejected: {detail}")
                shutil.rmtree(repo, ignore_errors=True)

    def test_immutable_files_are_declared_and_present(self):
        for task in AGENT_TASKS:
            for rel in task.get("immutable", []):
                with self.subTest(task=task["id"], file=rel):
                    self.assertIn(rel, task["files"],
                                  "immutable file is not part of the seed")


if __name__ == "__main__":
    unittest.main()
