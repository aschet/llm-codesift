"""Agent stage helpers: event parsing, verification, and tamper detection."""
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from codesift import agent
from codesift.tasks import AGENT_TASKS

TASKS = {t["id"]: t for t in AGENT_TASKS}


def event(kind, part=None, **extra):
    return json.dumps({"type": kind, "part": part or {}, **extra})


class TestParseEvents(unittest.TestCase):
    def test_empty_stream(self):
        tools, steps, errors, tokens, peak, said = agent.parse_events("")
        self.assertEqual((tools, steps, errors, peak, said), ([], 0, [], 0, ""))
        self.assertEqual(tokens["output"], 0)

    def test_non_json_lines_are_ignored(self):
        tools, steps, *_ = agent.parse_events("hello\n\n" + event("step_start"))
        self.assertEqual(steps, 1)
        self.assertEqual(tools, [])

    def test_counts_steps_tools_and_tokens(self):
        stream = "\n".join([
            event("step_start"),
            event("tool", {"type": "tool", "tool": "read"}),
            event("tool", {"type": "tool", "tool": "edit"}),
            event("step_finish", {"tokens": {"input": 7000, "output": 120, "total": 7120}}),
            event("step_start"),
            event("step_finish", {"tokens": {"input": 9000, "output": 80, "total": 9080}}),
        ])
        tools, steps, errors, tokens, peak, _ = agent.parse_events(stream)
        self.assertEqual(tools, ["read", "edit"])
        self.assertEqual(steps, 2)
        self.assertEqual(errors, [])
        self.assertEqual(tokens["output"], 200)
        self.assertEqual(peak, 9000, "peak input should be the largest single step")

    def test_what_the_model_said_is_kept(self):
        # A session that ends after two turns having written nothing is only
        # diagnosable from what the model produced instead of a tool call.
        stream = "\n".join([
            event("step_start"),
            event("message", {"type": "text", "text": "Here is the plan."}),
            event("message", {"type": "text", "text": "I will start now."}),
        ])
        *_, said = agent.parse_events(stream)
        self.assertIn("Here is the plan.", said)
        self.assertIn("I will start now.", said)

    def test_reasoning_is_kept_too(self):
        # The turn worth explaining is one that produced no tool call and no
        # answer. A model that spends it reasoning leaves nothing else behind.
        stream = event("message", {"type": "reasoning", "text": "I will start soon."})
        *_, said = agent.parse_events(stream)
        self.assertIn("I will start soon.", said)

    def test_the_harness_does_not_read_opencode_s_own_storage(self):
        # opencode keeps every message in a SQLite database, which was where this
        # was diagnosed. Its location differs by platform, so the harness reads the
        # stream it is given on stdout instead. `opencode debug paths` reports the
        # directory portably if a person needs to look.
        src = (Path(__file__).parent.parent / "src" / "codesift"
               / "agent.py").read_text(encoding="utf-8")
        for path in (".local/share", "Application Support", "APPDATA", "opencode.db"):
            self.assertNotIn(path, src)

    def test_errors_are_captured(self):
        stream = json.dumps({"type": "error", "error": {"name": "UnknownError"}})
        _, _, errors, _, _, _ = agent.parse_events(stream)
        self.assertEqual(errors, ["UnknownError"])


class TestVerification(unittest.TestCase):
    def setUp(self):
        self.workdir = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_argv_style_check(self):
        task = TASKS["ag_fixbug"]
        repo = agent.seed(task, self.workdir)
        passed, detail, checks = agent.verify(task, repo)
        self.assertFalse(passed)
        self.assertIn("AssertionError", detail)
        self.assertEqual(checks, [], "a pass-or-fail task reports no per-check detail")

    def test_script_style_check(self):
        """ag_feature verifies via an embedded script rather than an argv list."""
        task = TASKS["ag_feature"]
        self.assertIn("verify_src", task)
        repo = agent.seed(task, self.workdir)
        passed, _, _ = agent.verify(task, repo)
        self.assertFalse(passed)

    def test_no_shell_metacharacters_in_checks(self):
        """Checks must not rely on a shell, which is not portable."""
        for task in AGENT_TASKS:
            with self.subTest(task=task["id"]):
                if "verify" in task:
                    self.assertIsInstance(task["verify"], list)
                    for arg in task["verify"]:
                        self.assertNotIn("<<", arg)
                        self.assertNotIn("|", arg)

    def test_seed_is_isolated_per_call(self):
        task = TASKS["ag_fixbug"]
        first = agent.seed(task, self.workdir)
        second = agent.seed(task, self.workdir)
        self.assertNotEqual(first, second)
        (first / "src" / "stats.py").write_text("# changed", encoding="utf-8")
        self.assertNotIn("# changed",
                         (second / "src" / "stats.py").read_text(encoding="utf-8"))
        for repo in (first, second):
            shutil.rmtree(repo, ignore_errors=True)


class TestPreflight(unittest.TestCase):
    def test_missing_opencode_is_reported(self):
        original = shutil.which
        shutil.which = lambda name: None
        try:
            message = agent.preflight(["any:model"])
        finally:
            shutil.which = original
        self.assertIsNotNone(message)
        self.assertIn("opencode", message)


if __name__ == "__main__":
    unittest.main()


class TestCheckParsing(unittest.TestCase):
    def test_reads_name_verdict_and_detail(self):
        out = ("noise before\n"
               "CHECK serves PASS ok\n"
               "CHECK reorder FAIL the listing did not follow the requested order\n"
               "CHECK ui_drag PASS\n"
               "CHECKS 2/3\n")
        checks = agent.parse_checks(out)
        self.assertEqual([c["name"] for c in checks], ["serves", "reorder", "ui_drag"])
        self.assertEqual([c["passed"] for c in checks], [True, False, True])
        self.assertEqual(checks[1]["detail"],
                         "the listing did not follow the requested order")
        self.assertEqual(checks[2]["detail"], "", "a detail is optional")

    def test_lines_that_are_not_checks_are_ignored(self):
        self.assertEqual(agent.parse_checks("CHECK malformed\nCHECK a MAYBE b\n"), [])


class TestDeniedTools(unittest.TestCase):
    """A configuration that refuses a tool fails every task for the wrong reason."""

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def write(self, payload):
        path = self.tmp / "opencode.jsonc"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_silence_when_nothing_is_denied(self):
        self.assertIsNone(agent.denied_tools(self.write({"permission": {"bash": "allow"}})))
        self.assertIsNone(agent.denied_tools(self.write({})))
        self.assertIsNone(agent.denied_tools(self.tmp / "absent.jsonc"))

    def test_a_denial_is_named(self):
        msg = agent.denied_tools(self.write({"permission": {"bash": "deny", "edit": "ask"}}))
        named = msg.split("denies: ")[1].split(".")[0]
        self.assertEqual(named, "bash", "only an outright denial is worth reporting")

    def test_a_nested_denial_is_found(self):
        msg = agent.denied_tools(self.write({"permission": {"bash": {"*": "deny"}}}))
        self.assertIn("bash", msg)


class TestRetention(unittest.TestCase):
    def test_a_retained_task_is_built_where_it_can_be_found(self):
        # A whole application is worth opening afterwards, so it must not land in
        # a temporary directory named after nothing.
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        target = tmp / "apps" / "gemma4_12b__ag_module"
        repo = agent.seed(TASKS["ag_module"], tmp / "work", at=target)
        self.assertEqual(repo, target)
        self.assertTrue((repo / "SPEC.md").exists())

        (repo / "leftover.txt").write_text("from an earlier run", encoding="utf-8")
        again = agent.seed(TASKS["ag_module"], tmp / "work", at=target)
        self.assertFalse((again / "leftover.txt").exists(),
                         "a rerun must not grade the previous attempt's files")

    def test_a_task_that_writes_from_scratch_asks_for_more_time(self):
        self.assertGreater(TASKS["ag_module"]["min_timeout"], 600)


class TestRepositoryPaths(unittest.TestCase):
    """Everything downstream runs with the repository as its working directory.

    A relative path handed to one of those resolves against the repository rather
    than against the caller. A retained application seeded at a relative path
    produced a doubled path, and the grader could not open its own check script,
    reporting a model that had built the whole application as having built none
    of it.
    """

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.cwd = Path.cwd()
        os.chdir(self.tmp)
        self.addCleanup(os.chdir, self.cwd)

    def test_a_seeded_repository_is_absolute(self):
        for at in (None, Path("apps") / "m__ag_module"):
            with self.subTest(at=str(at)):
                repo = agent.seed(TASKS["ag_fixbug"], Path("work"), at=at)
                self.assertTrue(repo.is_absolute(), f"{repo} is relative")
                self.assertTrue((repo / "src" / "stats.py").exists())

    def test_a_check_script_is_found_from_a_relative_seed(self):
        repo = agent.seed(TASKS["ag_feature"], Path("work"),
                          at=Path("apps") / "m__ag_feature")
        passed, detail, _ = agent.verify(TASKS["ag_feature"], repo)
        self.assertFalse(passed)
        self.assertNotIn("can't open file", detail,
                         "the grader must be able to find its own script")


class TestModuleGrading(unittest.TestCase):
    """The module task, and the property the application task lacks.

    There, thirteen of eighteen checks could not be attempted unless the
    application started, so one mis-named file scored the same as writing
    nothing. Here only the import cascades: a model that gets most of the
    functions right must score most of the points.
    """

    REFERENCE = Path(__file__).parent / "tasklist_reference"

    def setUp(self):
        self.task = TASKS["ag_module"]
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def grade(self, damage=None):
        repo = agent.seed(self.task, self.tmp)
        shutil.copytree(self.REFERENCE, repo, dirs_exist_ok=True)
        if damage:
            damage(repo)
        passed, detail, checks = agent.verify(self.task, repo)
        return passed, detail, {c["name"]: c for c in checks}

    def edit(self, repo, old, new):
        path = repo / "src" / "tasklist.py"
        text = path.read_text(encoding="utf-8")
        self.assertIn(old, text)
        path.write_text(text.replace(old, new), encoding="utf-8")

    def test_one_broken_function_costs_one_point(self):
        _, _, checks = self.grade(
            lambda repo: self.edit(repo, "if q:", "if False:"))
        self.assertFalse(checks["filter_text"]["passed"])
        for still in ("add", "get", "update", "delete", "filter_done", "reorder",
                      "persist", "toggle_done"):
            self.assertTrue(checks[still]["passed"], f"{still} should be unaffected")

    def test_a_second_broken_function_costs_a_second_point(self):
        _, _, checks = self.grade(
            lambda repo: self.edit(repo, "    except FileNotFoundError:\n        return new_store()",
                                   "    except FileNotFoundError:\n        raise"))
        self.assertFalse(checks["load_missing"]["passed"])
        self.assertTrue(checks["persist"]["passed"])

    def test_a_missing_guard_costs_only_its_own_check(self):
        _, _, checks = self.grade(
            lambda repo: self.edit(repo, 'if not title:\n        raise ValueError("title is required")',
                                   "if False:\n        pass"))
        self.assertFalse(checks["empty_title"]["passed"])
        self.assertTrue(checks["add"]["passed"])

    def test_only_the_import_cascades(self):
        def damage(repo):
            (repo / "src" / "tasklist.py").write_text("import nonexistent_module\n",
                                                      encoding="utf-8")
        passed, _, checks = self.grade(damage)
        self.assertFalse(passed)
        self.assertFalse(checks["importable"]["passed"])
        self.assertTrue(checks["layout"]["passed"], "the file is still there")

    def test_missing_tests_do_not_touch_the_functions(self):
        _, _, checks = self.grade(lambda repo: shutil.rmtree(repo / "tests"))
        self.assertFalse(checks["own_tests"]["passed"])
        self.assertFalse(checks["layout"]["passed"])
        self.assertTrue(checks["add"]["passed"])

    def test_a_module_from_outside_the_repository_is_refused(self):
        src = (Path(__file__).parent.parent / "src" / "codesift" / "tasks"
               / "checks" / "module_check.py").read_text(encoding="utf-8")
        self.assertIn("imported from outside the repository", src)

    def test_it_costs_seconds_not_an_hour(self):
        self.assertLessEqual(self.task["verify_timeout"], 300)
        self.assertEqual(self.task.get("tier", 1), 1)


class TestCheckDetailsMatchTheirOutcome(unittest.TestCase):
    """A passing check must not print the reason it would have failed.

    One did: `load_missing` returned its failure message unconditionally, so a
    correct implementation was reported as passing while the line beside it read
    like a fault.
    """

    REFERENCE = Path(__file__).parent / "tasklist_reference"

    def test_every_passing_check_reads_as_a_pass(self):
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        repo = agent.seed(TASKS["ag_module"], tmp)
        shutil.copytree(self.REFERENCE, repo, dirs_exist_ok=True)
        _, _, checks = agent.verify(TASKS["ag_module"], repo)
        self.assertTrue(checks)
        for check in checks:
            with self.subTest(check=check["name"]):
                self.assertTrue(check["passed"], "the reference must pass everything")
                for wrong in ("did not", "no such", "was accepted", "still present"):
                    self.assertNotIn(wrong, check["detail"],
                                     "a passing check is describing a failure")


class TestOneMissingFunctionCostsOnlyItsChecks(unittest.TestCase):
    """The gradient has to survive an incomplete answer, not just a wrong one.

    Observed: a model wrote eight of the nine functions, the module imported
    cleanly, and it scored zero of sixteen because the grader refused the whole
    module over the one that was absent. That is the cliff this task exists to
    avoid, and the mutation tests missed it because they only ever broke a
    function's behaviour, never removed one.
    """

    REFERENCE = Path(__file__).parent / "tasklist_reference"

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def grade_without(self, function):
        repo = agent.seed(TASKS["ag_module"], self.tmp)
        shutil.copytree(self.REFERENCE, repo, dirs_exist_ok=True)
        path = repo / "src" / "tasklist.py"
        text = path.read_text(encoding="utf-8")
        start = text.index(f"def {function}(")
        end = text.find("\ndef ", start)
        path.write_text(text[:start] + (text[end + 1:] if end > 0 else ""),
                        encoding="utf-8")
        _, _, checks = agent.verify(TASKS["ag_module"], repo)
        return {c["name"]: c for c in checks}

    def test_a_missing_reorder_costs_the_reorder_checks_and_no_others(self):
        checks = self.grade_without("reorder")
        self.assertFalse(checks["reorder"]["passed"])
        self.assertFalse(checks["reorder_rejects_unknown"]["passed"])
        for still in ("add", "get", "update", "delete", "toggle_done",
                      "filter_done", "filter_text", "load_missing", "empty_title"):
            self.assertTrue(checks[still]["passed"], f"{still} should be unaffected")

    def test_the_module_is_still_reported_as_importable(self):
        checks = self.grade_without("reorder")
        self.assertTrue(checks["importable"]["passed"],
                        "it imported; saying otherwise makes the verdict call it dead")
        self.assertIn("reorder", checks["importable"]["detail"])

    def test_a_missing_save_costs_persistence_alone(self):
        checks = self.grade_without("save")
        self.assertFalse(checks["persist"]["passed"])
        self.assertTrue(checks["load_missing"]["passed"])
        self.assertTrue(checks["add"]["passed"])


class TestTheGradientHoldsUnderEveryOmission(unittest.TestCase):
    """Remove each function in turn and check the score degrades, not collapses.

    The hand-picked mutations elsewhere all break a function's behaviour, which
    leaves a module with every function present -- the one case where a grader
    that refuses an incomplete module looks correct. One did, and a model that
    wrote eight of the nine functions scored zero. Sweeping the whole space is
    what makes the claim in this task's docstring testable rather than asserted.
    """

    REFERENCE = Path(__file__).parent / "tasklist_reference"
    API = ("new_store", "add", "get", "update", "delete", "tasks", "reorder",
           "save", "load")
    # Every check builds a store and puts tasks in it, so these two are load
    # bearing for the whole suite and are expected to take it down with them.
    FOUNDATIONAL = {"new_store", "add"}

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def grade_without(self, function):
        repo = agent.seed(TASKS["ag_module"], self.tmp / function)
        shutil.copytree(self.REFERENCE, repo, dirs_exist_ok=True)
        path = repo / "src" / "tasklist.py"
        text = path.read_text(encoding="utf-8")
        start = text.index(f"def {function}(")
        end = text.find("\ndef ", start)
        path.write_text(text[:start] + (text[end + 1:] if end > 0 else ""),
                        encoding="utf-8")
        _, _, checks = agent.verify(TASKS["ag_module"], repo)
        return checks

    def test_removing_any_one_function_leaves_the_rest_scoring(self):
        for function in self.API:
            with self.subTest(removed=function):
                checks = self.grade_without(function)
                met = sum(1 for c in checks if c["passed"])
                if function in self.FOUNDATIONAL:
                    continue
                self.assertGreaterEqual(
                    met, len(checks) // 2,
                    f"removing {function} cost {len(checks) - met} of {len(checks)} "
                    f"checks; a partial answer must score partially")

    def test_the_module_still_reads_as_importable_whichever_is_absent(self):
        # The verdict treats a failed import as "nothing it wrote ran", which is
        # far too strong a thing to say about a module missing one function.
        for function in self.API:
            with self.subTest(removed=function):
                checks = {c["name"]: c for c in self.grade_without(function)}
                self.assertTrue(checks["importable"]["passed"])
                self.assertIn(function, checks["importable"]["detail"])
