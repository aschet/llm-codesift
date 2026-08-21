"""The cascade must reject early, and must never reject for the wrong reason.

Two properties matter. A model that fails a cheap gate must not go on to pay for
the expensive ones, which is the entire point. And a gate must apply the same
rule the report applies, so that triage and the full run cannot disagree about
the same model.
"""
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from codesift import screen, triage
from codesift.tasks import TASKSETS
from tests import reference
from codesift.config import Config
from codesift.report import MIN_GEN_TOK_S

OFFLINE = "http://127.0.0.1:1"


# A prompt is all the fake sees, so answers are looked up by it. Each answer is
# derived from the task's own definition -- the reference solution, the expected
# trace output, the wanted tool -- so a model that "knows everything" is graded by
# the real grader rather than around it. If this stops producing 100%, the tasks
# and the grader have drifted apart, which is worth failing a test over.
# Answers come from tests/reference.perfect_answer, which derives them from each
# task's own definition, so the gates are exercised through the real grader.
ANSWERS = {task["prompt"]: reference.perfect_answer(task)
           for name in TASKSETS for task in TASKSETS[name]}


class FakeClient:
    """A model whose behaviour at each gate is dictated by the test."""

    def __init__(self, gen=60.0, tool_ok=True, hard_pass=True,
                 truncated=False, retrieved=True):
        self.gen, self.tool_ok = gen, tool_ok
        self.hard_pass, self.truncated, self.retrieved = hard_pass, truncated, retrieved
        self.deep_calls = 0
        self.task_calls = 0

    def chat(self, model, prompt, ctx=None, num_predict=None, tools=None):
        if "DEPLOY_TOKEN" in prompt:                      # the deep probe
            self.deep_calls += 1
            return {"message": {"content": "quartz-mongoose-8814" if self.retrieved else "no"},
                    "_wall": 1.0, "prompt_eval_count": 100 if self.truncated else 10 ** 6,
                    "prompt_eval_duration": 10 ** 9, "eval_count": 10,
                    "eval_duration": 10 ** 9}
        if tools:
            self.task_calls += 1
            answer = ANSWERS.get(prompt, {"content": ""})
            msg = answer if self.tool_ok else {"content": "I would search the files."}
            return {"message": msg, "_wall": 0.1, "eval_count": 5, "eval_duration": 10 ** 8}
        if prompt == "Write a Python function that reverses a linked list.":
            return {"message": {"content": "def r(h): return h"}, "_wall": 0.1,
                    "eval_count": 5, "eval_duration": int(5 / self.gen * 1e9),
                    "load_duration": 10 ** 9}
        self.task_calls += 1
        answer = ANSWERS.get(prompt.split("\n\n```python")[0])
        msg = answer if (self.hard_pass and answer) else {"content": "nope"}
        return {"message": msg, "_wall": 0.1, "eval_count": 5, "eval_duration": 10 ** 8}

    def placement(self, model):
        return {"pct_gpu": 50.0}

    def show(self, model):
        return {"model_info": {}}

    def unload(self, model):
        pass


class TriageCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.cfg = Config(host=OFFLINE, results_dir=self.tmp, models=["m:1"], ctx=65536)
        self.enterContext(mock.patch.object(triage.gpulock, "acquire"))
        self.out = io.StringIO()

    def go(self, client, **kw):
        self.enterContext(mock.patch.object(triage, "Ollama", lambda *a, **k: client))
        triage.run(self.cfg, depth=1000, stream=self.out, **kw)
        return self.out.getvalue()

    def ledger(self):
        return triage.read_ledger(self.cfg)


class TestEarlyExit(TriageCase):
    def test_a_slow_model_never_pays_for_the_deep_probe(self):
        client = FakeClient(gen=MIN_GEN_TOK_S - 5)
        text = self.go(client)
        self.assertEqual(client.deep_calls, 0, "the expensive gate must not be reached")
        self.assertEqual(client.task_calls, 0, "no task should have been graded")
        self.assertIn("REJECTED at speed", text)
        self.assertFalse(self.ledger()["m:1"]["passed"])

    def test_a_model_that_cannot_call_a_tool_never_reaches_the_hard_set(self):
        client = FakeClient(tool_ok=False)
        text = self.go(client)
        self.assertIn("REJECTED at tools", text)
        self.assertEqual(client.deep_calls, 0)
        # only the two tool tasks were graded, not the fifteen hard ones
        self.assertLessEqual(client.task_calls, 4)

    def test_a_weak_model_never_pays_for_the_deep_probe_either(self):
        client = FakeClient(hard_pass=False)
        text = self.go(client)
        self.assertIn("REJECTED at quality", text)
        self.assertEqual(client.deep_calls, 0)

    def test_context_is_checked_last_and_only_for_survivors(self):
        client = FakeClient(truncated=True)
        text = self.go(client)
        self.assertIn("REJECTED at context", text)
        self.assertEqual(client.deep_calls, 1, "reached, but only after the cheap gates")

    def test_a_good_model_clears_every_gate(self):
        client = FakeClient()
        text = self.go(client)
        self.assertIn("CLEARED", text)
        self.assertTrue(self.ledger()["m:1"]["passed"])
        self.assertEqual(client.deep_calls, 1)


class TestGateOrder(TriageCase):
    def test_the_order_runs_cheapest_first(self):
        # Taken from measured cost: a shallow call is seconds, the hard set is a
        # minute or so, the deep prompt is longer still.
        self.assertEqual(triage.GATES, ("speed", "tools", "quality", "context"))

    def test_the_thresholds_are_the_report_s_own(self):
        # A model triage rejects must be one the full run would also reject, or the
        # two disagree about the same model and neither can be trusted.
        from codesift import report
        self.assertEqual(triage.MIN_GEN_TOK_S, report.MIN_GEN_TOK_S)
        self.assertEqual(triage.MIN_HARD_RATE, 70.0)


class TestResumption(TriageCase):
    def test_a_second_run_measures_nothing_again(self):
        client = FakeClient(gen=1.0)
        self.go(client)
        before = client.task_calls, client.deep_calls
        again = io.StringIO()
        triage.run(self.cfg, depth=1000, stream=again)
        self.assertIn("already triaged, rejected", again.getvalue())
        self.assertEqual((client.task_calls, client.deep_calls), before)

    def test_redo_measures_again(self):
        client = FakeClient(gen=1.0)
        self.go(client)
        again = io.StringIO()
        triage.run(self.cfg, depth=1000, redo=True, stream=again)
        self.assertIn("REJECTED at speed", again.getvalue())

    def test_apply_adds_the_rejected_to_the_discard_list(self):
        from codesift import prune
        client = FakeClient(gen=1.0)
        self.go(client, apply=True)
        self.assertEqual(prune.read_discarded(self.tmp), ["m:1"])

    def test_without_apply_nothing_is_discarded(self):
        from codesift import prune
        client = FakeClient(gen=1.0)
        text = self.go(client)
        self.assertEqual(prune.read_discarded(self.tmp), [])
        self.assertIn("Pass --apply", text)


class TestLockIsReentrant(unittest.TestCase):
    def test_acquiring_twice_does_not_release_the_lock(self):
        # A cascade runs several stages in one process. On POSIX, closing any
        # descriptor to a file drops that process's lock on it, so reopening would
        # have silently released the lock the first call took.
        from codesift import gpulock
        first = gpulock.acquire("a", endpoint="http://triage-test")
        second = gpulock.acquire("b", endpoint="http://triage-test")
        self.assertIs(first, second)
        self.assertFalse(first.closed)

    def test_a_different_endpoint_is_a_different_lock(self):
        # The lock is per endpoint so runs against separate servers do not block
        # each other; reentrancy must not collapse them into one.
        from codesift import gpulock
        one = gpulock.acquire("a", endpoint="http://triage-test-a")
        two = gpulock.acquire("b", endpoint="http://triage-test-b")
        self.assertIsNot(one, two)


if __name__ == "__main__":
    unittest.main()


class TestTheAnswerBookIsComplete(unittest.TestCase):
    """The fake model must be able to score full marks, or the tests prove nothing.

    If a task and its grader drift apart, a perfect answer stops passing and every
    early-exit test below quietly starts exercising the wrong branch.
    """

    def test_a_perfect_answer_exists_for_every_task(self):
        for name in ("basic", "hard"):
            for task in TASKSETS[name]:
                with self.subTest(task=task["id"]):
                    self.assertIn(task["prompt"], ANSWERS)

    def test_the_perfect_answers_actually_pass_the_real_grader(self):
        for name in ("basic", "hard"):
            for task in TASKSETS[name]:
                with self.subTest(task=task["id"]):
                    passed, _, detail, _ = screen.grade(
                        task, {"message": ANSWERS[task["prompt"]]})
                    self.assertTrue(passed, f"{task['id']}: {detail}")


class TestNoDuplicatedWork(TriageCase):
    """Triage and the screen must not ask a model the same question twice.

    The quality gate grades the whole hard set, which is exactly what the screen
    is about to do. Measuring it twice costs the model's slowest task set over
    again for no further information.
    """

    def ledger(self):
        return [json.loads(l) for l in
                (self.tmp / "screen_tasks.jsonl").read_text().splitlines() if l.strip()]

    def test_the_gates_record_into_the_screen_s_own_ledger(self):
        self.go(FakeClient())
        recorded = self.ledger()
        self.assertTrue(recorded, "triage recorded nothing the screen could reuse")
        sets = {r["taskset"] for r in recorded}
        self.assertEqual(sets, {"basic", "hard"})
        self.assertTrue(all(r["run"] == 1 for r in recorded),
                        "recorded as run 1, the run the screen fills first")

    def test_the_screen_then_measures_nothing_again(self):
        client = FakeClient()
        self.go(client)
        graded_by_triage = client.task_calls

        with mock.patch.object(screen, "Ollama", lambda *a, **k: client), \
             mock.patch.object(screen.gpulock, "acquire"), \
             mock.patch("sys.stdout", new=io.StringIO()) as out:
            screen.run(self.cfg, taskset="hard", runs=1)
        self.assertEqual(client.task_calls, graded_by_triage,
                         "the screen re-ran tasks triage had already graded")
        self.assertIn("complete", out.getvalue())

    def test_the_screen_still_writes_a_summary_for_a_set_triage_completed(self):
        # Triage records tasks without a summary. Without this the hard set would
        # be missing from the report despite having been measured in full.
        client = FakeClient()
        self.go(client)
        with mock.patch.object(screen, "Ollama", lambda *a, **k: client), \
             mock.patch.object(screen.gpulock, "acquire"), \
             mock.patch("sys.stdout", new=io.StringIO()):
            screen.run(self.cfg, taskset="hard", runs=1)
        rows = [json.loads(l) for l in
                (self.tmp / "screen.jsonl").read_text().splitlines() if l.strip()]
        hard = [r for r in rows if r["taskset"] == "hard"]
        self.assertEqual(len(hard), 1)
        self.assertEqual(hard[0]["n"], len(TASKSETS["hard"]))
        graded = [r for r in self.ledger() if r["taskset"] == "hard"]
        expected = round(100 * sum(1 for r in graded if r["passed"]) / len(graded), 1)
        self.assertEqual(hard[0]["pass_rate"], expected,
                         "the summary must agree with the records triage wrote")

    def test_a_rejected_model_leaves_behind_what_it_did_answer(self):
        # The cheap gates ran; their results are measurements and should survive.
        self.go(FakeClient(hard_pass=False))
        sets = {r["taskset"] for r in self.ledger()}
        self.assertIn("basic", sets, "the tool-call results were thrown away")
