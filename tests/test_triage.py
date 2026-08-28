# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
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

from codesift import progress, screen, triage
from codesift.tasks import TASKS
from tests import reference
from codesift.config import Config
from codesift.analysis import MIN_GEN_TOK_S

OFFLINE = "http://127.0.0.1:1"


# A prompt is all the fake sees, so answers are looked up by it. Each answer is
# derived from the task's own definition -- the reference solution, the expected
# trace output, the wanted tool -- so a model that "knows everything" is graded by
# the real grader rather than around it. If this stops producing 100%, the tasks
# and the grader have drifted apart, which is worth failing a test over.
# Answers come from tests/reference.perfect_answer, which derives them from each
# task's own definition, so the gates are exercised through the real grader.
ANSWERS = {task["prompt"]: reference.perfect_answer(task)
           for task in TASKS}


class FakeClient:
    """A model whose behaviour at each gate is dictated by the test."""

    def __init__(self, gen=60.0, tool_ok=True, answers_well=True, truncated=False,
                 drops=0):
        self.gen, self.tool_ok = gen, tool_ok
        self.answers_well, self.truncated = answers_well, truncated
        self.deep_calls = 0
        self.task_calls = 0
        self.drops = drops          # requests to fail before answering any

    def chat(self, model, prompt, ctx=None, num_predict=None, tools=None):
        if tools and self.drops:
            self.drops -= 1
            raise OSError("Connection reset by peer")
        if "codebase excerpt" in prompt:                  # the deep probe
            self.deep_calls += 1
            return {"message": {"content": "OK"}, "_wall": 1.0,
                    "prompt_eval_count": 100 if self.truncated else 10 ** 6,
                    "prompt_eval_duration": 10 ** 9, "eval_count": 1,
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
        msg = answer if (self.answers_well and answer) else {"content": "nope"}
        return {"message": msg, "_wall": 0.1, "eval_count": 5, "eval_duration": 10 ** 8}

    def placement(self, model):
        return {"pct_gpu": 50.0}

    def show(self, model):
        return {"model_info": {}}

    def unload(self, model):
        pass


class TriageCase(unittest.TestCase):
    def setUp(self):
        progress.reset()
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

    def codes(self, model="m:1"):
        """What the gates found, from the record rather than from the printout.

        A gate's verdict is what these tests are about; how it is displayed is
        progress.py's business and changes without any of this changing.
        """
        return [f["code"] for f in self.ledger()[model]["findings"]]


class TestARequestThatFailedIsNotAnAnswer(TriageCase):
    """A dropped request measures nothing, and must not be read as a bad reply.

    The two arrive at the same place -- no usable answer -- and mean opposite
    things. A model that emits a call the harness cannot parse is unusable; one
    whose connection dropped has not been asked yet.
    """

    def test_a_dropped_request_is_not_reported_as_an_unparseable_tool_call(self):
        self.go(FakeClient(drops=1))
        self.assertEqual(self.codes(), ["error"])

    def test_the_next_run_asks_again_rather_than_reading_the_failure_back(self):
        first = FakeClient(drops=1)
        self.go(first)
        second = FakeClient()
        self.go(second)
        self.assertGreater(second.task_calls, 0, "the model was never asked again")
        self.assertEqual(self.codes(), [], "nothing was found against it")
        self.assertTrue(self.ledger()["m:1"]["passed"])

    def test_the_task_that_failed_is_not_left_on_file_as_a_measurement(self):
        self.go(FakeClient(drops=1))
        graded = [json.loads(l) for l in
                  (self.tmp / "screen_tasks.jsonl").read_text().splitlines()]
        failed = [r for r in graded if not screen.measured(r)]
        self.assertTrue(failed, "the failure itself must be on record")
        for rec in failed:
            self.assertNotIn("passed", rec)
            self.assertNotIn("format_ok", rec)


class TestEarlyExit(TriageCase):
    def test_a_slow_model_never_pays_for_the_deep_probe(self):
        client = FakeClient(gen=MIN_GEN_TOK_S - 5)
        self.go(client)
        self.assertEqual(client.deep_calls, 0, "the expensive gate must not be reached")
        self.assertEqual(client.task_calls, 0, "no task should have been graded")
        self.assertEqual(self.codes(), ["slow_generation"])
        self.assertFalse(self.ledger()["m:1"]["passed"])

    def test_a_model_that_cannot_call_a_tool_is_not_graded_further(self):
        client = FakeClient(tool_ok=False)
        self.go(client)
        self.assertEqual(self.codes(), ["malformed_tool_calls"])
        self.assertEqual(client.deep_calls, 0)
        # only the tool tasks were graded, not the rest of the set
        self.assertLessEqual(client.task_calls, 4)

    def test_a_model_that_answers_poorly_is_no_longer_rejected_here(self):
        # Quality gated access to the expensive measurement and predicted it badly:
        # one model rejected at 67% went on to meet seventeen of
        # eighteen requirements building an application.
        client = FakeClient(answers_well=False)
        self.go(client)
        self.assertTrue(self.ledger()["m:1"]["passed"])
        self.assertNotIn("quality", " ".join(self.codes()))

    def test_context_is_checked_last_and_only_for_survivors(self):
        client = FakeClient(truncated=True)
        self.go(client)
        self.assertEqual(self.codes(), ["context_truncated"])
        self.assertEqual(client.deep_calls, 1, "reached, but only after the cheap gates")

    def test_a_fault_at_depth_is_recorded_but_does_not_stop_the_model(self):
        # It is frequently a good model at short prompts, and rejecting it here
        # means the screen never runs and the report can say neither thing.
        self.go(FakeClient(truncated=True))
        self.assertTrue(self.ledger()["m:1"]["passed"], "it must go on to be screened")
        self.assertEqual(self.codes(), ["context_truncated"],
                         "and the finding must survive")

    def test_a_good_model_clears_every_gate(self):
        client = FakeClient()
        self.go(client)
        # A cleared model records no finding: `passed` says it, and every such row
        # carried one reading "cleared every gate" for no reader.
        self.assertEqual(self.codes(), [])
        self.assertTrue(self.ledger()["m:1"]["passed"])
        self.assertEqual(client.deep_calls, 1)


class TestGateOrder(TriageCase):
    def test_the_order_runs_cheapest_first(self):
        # Taken from measured cost: a shallow call is seconds, the deep prompt is
        # a minute or more.
        self.assertEqual(triage.GATES, ("speed", "tools", "context"))

    def test_every_gate_tests_something_a_model_can_or_cannot_do(self):
        # Not how well it does it. A gate that scores quality decides whether the
        # expensive measurement ever happens, on a proxy that predicts it poorly.
        self.assertNotIn("quality", triage.GATES)
        self.assertFalse(hasattr(triage, "MIN_HARD_RATE"))

    def test_the_thresholds_are_the_report_s_own(self):
        # A model triage rejects must be one the full run would also reject, or the
        # two disagree about the same model and neither can be trusted.
        from codesift import analysis
        self.assertEqual(triage.MIN_GEN_TOK_S, analysis.MIN_GEN_TOK_S)


class TestResumption(TriageCase):
    def test_a_second_run_measures_nothing_again(self):
        client = FakeClient(gen=1.0)
        self.go(client)
        before = client.task_calls, client.deep_calls
        again = io.StringIO()
        triage.run(self.cfg, depth=1000, stream=again)
        # The verdict stands from the first run; what must not happen is the model
        # being measured for it a second time.
        self.assertFalse(self.ledger()["m:1"]["passed"])
        self.assertEqual((client.task_calls, client.deep_calls), before)

    def test_redo_measures_again(self):
        client = FakeClient(gen=1.0)
        self.go(client)
        again = io.StringIO()
        triage.run(self.cfg, depth=1000, redo=True, stream=again)
        self.assertEqual(self.codes(), ["slow_generation"])

    def test_a_rejection_is_recorded_for_the_report(self):
        client = FakeClient(gen=1.0)
        self.go(client)
        [rec] = [json.loads(l) for l in
                 (self.tmp / "triage.jsonl").read_text().splitlines() if l.strip()]
        self.assertFalse(rec["passed"])
        self.assertEqual([f["code"] for f in rec["findings"]], ["slow_generation"])


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


class TestTheSpeedGateStoresWhatItMeasured(TriageCase):
    """A model rejected on speed is never probed, so the gate's rate is all there is.

    Kept only inside the finding that phrases the rejection, the report is left
    making a claim about a model whose figure it cannot show beside any other.
    """

    def records(self):
        path = self.tmp / "probe.jsonl"
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
                if l.strip()]

    def test_the_rate_reaches_the_probe_ledger(self):
        ok, found, _ = triage.gate_speed(self.cfg, FakeClient(gen=9.0), "slow:1")
        self.assertFalse(ok)
        rec = self.records()
        self.assertEqual([r["model"] for r in rec], ["slow:1"])
        self.assertEqual(rec[0]["gen_tok_s"], 9.0)
        self.assertEqual([f["code"] for f in found], ["slow_generation"])

    def test_it_is_not_taken_for_a_deep_measurement(self):
        from codesift import probe
        triage.gate_speed(self.cfg, FakeClient(gen=9.0), "slow:1")
        self.assertFalse(probe.at_depth(self.records()[0]),
                         "no deep prompt was ever sent")
        self.assertIsNone(probe.stored(self.cfg, "slow:1", self.cfg.ctx))

    def test_a_model_that_clears_is_recorded_too(self):
        # The rate is a measurement whatever the verdict, and the probe stage
        # replaces this record with its own when it measures at depth.
        ok, _, _ = triage.gate_speed(self.cfg, FakeClient(gen=60.0), "fast:1")
        self.assertTrue(ok)
        self.assertEqual(self.records()[0]["gen_tok_s"], 60.0)


if __name__ == "__main__":
    unittest.main()


class TestTheAnswerBookIsComplete(unittest.TestCase):
    """The fake model must be able to score full marks, or the tests prove nothing.

    If a task and its grader drift apart, a perfect answer stops passing and every
    early-exit test below quietly starts exercising the wrong branch.
    """

    def test_a_perfect_answer_exists_for_every_task(self):
        for task in TASKS:
            with self.subTest(task=task["id"]):
                self.assertIn(task["prompt"], ANSWERS)

    def test_the_perfect_answers_actually_pass_the_real_grader(self):
        for task in TASKS:
            with self.subTest(task=task["id"]):
                passed, _, detail, _ = screen.grade(
                    task, {"message": ANSWERS[task["prompt"]]})
                self.assertTrue(passed, f"{task['id']}: {detail}")


class TestNoDuplicatedWork(TriageCase):
    """Triage and the screen must not ask a model the same question twice.

    The tools gate grades tasks the screen is about to grade, so what it measures
    is recorded where the screen will find it rather than measured again.
    """

    def ledger(self):
        return [json.loads(l) for l in
                (self.tmp / "screen_tasks.jsonl").read_text().splitlines() if l.strip()]

    def test_the_gates_record_into_the_screen_s_own_ledger(self):
        self.go(FakeClient())
        recorded = self.ledger()
        self.assertTrue(recorded, "triage recorded nothing the screen could reuse")
        self.assertTrue(recorded, "triage recorded nothing for the screen to reuse")
        self.assertTrue(all(r["run"] == 1 for r in recorded),
                        "recorded as run 1, the run the screen fills first")

    def test_the_screen_then_measures_nothing_again(self):
        client = FakeClient()
        self.go(client)
        graded_by_triage = client.task_calls

        with mock.patch.object(screen, "Ollama", lambda *a, **k: client), \
             mock.patch.object(screen.gpulock, "acquire"), \
             mock.patch("sys.stdout", new=io.StringIO()):
            screen.run(self.cfg)
        self.assertLess(client.task_calls - graded_by_triage, len(TASKS),
                        "the screen re-ran tasks triage had already graded")

    def test_the_screen_completes_the_set_triage_started(self):
        # Triage grades the tool tasks into the screen's own ledger; the screen
        # measures the rest and every task ends up on record exactly once.
        client = FakeClient()
        self.go(client)
        with mock.patch.object(screen, "Ollama", lambda *a, **k: client), \
             mock.patch.object(screen.gpulock, "acquire"), \
             mock.patch("sys.stdout", new=io.StringIO()):
            screen.run(self.cfg)
        graded = self.ledger()
        self.assertEqual(len(graded), len(TASKS))
        self.assertEqual(len({r["task"] for r in graded}), len(TASKS))

    def test_a_rejected_model_leaves_behind_what_it_did_answer(self):
        # The cheap gates ran; their results are measurements and should survive.
        self.go(FakeClient(answers_well=False))
        graded = {r["task"] for r in self.ledger()}
        self.assertTrue(graded & {t["id"] for t in TASKS if t["kind"] == "toolcall"},
                        "the tool-call results were thrown away")
