"""Interrupted runs must continue, not restart and not silently skip work."""
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from codesift import agent, prefixcache, probe, screen
from codesift.config import Config


class FakeClient:
    """Answers every request identically and counts how often it was asked."""

    def __init__(self, *args, **kwargs):
        self.calls = []
        self.unloaded = []

    def chat(self, model, prompt, **kwargs):
        self.calls.append(model)
        return {"message": {"content": "def"}, "_wall": 0.1,
                "eval_count": 5, "eval_duration": 100_000_000,
                "prompt_eval_count": 40000, "prompt_eval_duration": 1_000_000_000}

    def chat_messages(self, model, messages, **kwargs):
        self.calls.append(model)
        return {"message": {"content": "1"}, "_wall": 0.1,
                "prompt_eval_count": 100, "prompt_eval_duration": 500_000_000}

    def placement(self, model):
        return {"pct_gpu": 50.0, "total_gb": 10.0, "vram_gb": 5.0}

    def unload(self, model):
        self.unloaded.append(model)


class ResumeCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.cfg = Config(results_dir=self.tmp, models=["m:1"], ctx=4096)
        self.enterContext(mock.patch.object(screen.gpulock, "acquire"))
        self.clients = []

    def client_factory(self, *args, **kwargs):
        client = FakeClient()
        self.clients.append(client)
        return client

    @property
    def total_calls(self):
        return sum(len(c.calls) for c in self.clients)


class TestScreenResume(ResumeCase):
    def run_screen(self, **kwargs):
        with mock.patch.object(screen, "Ollama", self.client_factory):
            screen.run(self.cfg, only=["fmt_oneword", "tc_single"], **kwargs)

    def test_second_run_repeats_nothing(self):
        self.run_screen()
        first = self.total_calls
        self.assertEqual(first, 2, "both selected tasks should run once")
        self.run_screen()
        self.assertEqual(self.total_calls, first, "completed work was repeated")

    def test_only_missing_tasks_are_run(self):
        with mock.patch.object(screen, "Ollama", self.client_factory):
            screen.run(self.cfg, only=["fmt_oneword"])
        self.assertEqual(self.total_calls, 1)
        self.run_screen()      # now asks for both
        self.assertEqual(self.total_calls, 2, "should have run only the missing task")

    def test_redo_measures_again(self):
        self.run_screen()
        self.run_screen(redo=True)
        self.assertEqual(self.total_calls, 4)

    def test_separate_runs_are_independent(self):
        self.run_screen()
        with mock.patch.object(screen, "Ollama", self.client_factory):
            screen.run(self.cfg, only=["fmt_oneword", "tc_single"], runs=2)
        # run 1 is cached; run 2 is new, so two further calls
        self.assertEqual(self.total_calls, 4)

    def test_summary_is_rewritten_not_duplicated(self):
        self.run_screen()
        self.run_screen(redo=True)
        rows = [json.loads(l) for l in
                (self.tmp / "screen.jsonl").read_text(encoding="utf-8").splitlines()]
        keys = [(r["model"], r["run"], r["taskset"]) for r in rows]
        self.assertEqual(len(keys), len(set(keys)), "duplicate summary rows")

    def test_ledger_survives_a_partial_write(self):
        self.run_screen()
        ledger = self.tmp / "screen_tasks.jsonl"
        with ledger.open("a", encoding="utf-8") as fh:
            fh.write('{"truncated": ')      # a record cut off mid-write
        self.run_screen()
        self.assertEqual(self.total_calls, 2, "a damaged line should not cause a rerun")


class TestProbeResume(ResumeCase):
    def test_probe_skips_measured_models(self):
        with mock.patch.object(probe, "Ollama", self.client_factory), \
             mock.patch.object(probe.gpulock, "acquire"):
            probe.run(self.cfg, depth=100)
            first = self.total_calls
            probe.run(self.cfg, depth=100)
        self.assertEqual(self.total_calls, first)

    def test_failed_measurements_are_retried(self):
        """A record carrying an error must not count as done."""
        (self.tmp / "probe.jsonl").write_text(
            json.dumps({"model": "m:1", "num_ctx": 4096, "error": "boom"}) + "\n",
            encoding="utf-8")
        with mock.patch.object(probe, "Ollama", self.client_factory), \
             mock.patch.object(probe.gpulock, "acquire"):
            probe.run(self.cfg, depth=100)
        self.assertGreater(self.total_calls, 0, "an errored model should be retried")


class TestPrefixCacheResume(ResumeCase):
    def test_skips_measured_models(self):
        with mock.patch.object(prefixcache, "Ollama", self.client_factory), \
             mock.patch.object(prefixcache.gpulock, "acquire"):
            prefixcache.run(self.cfg)
            first = self.total_calls
            prefixcache.run(self.cfg)
        self.assertEqual(self.total_calls, first)


class TestAgentResume(ResumeCase):
    """The agent stage was the one stage without resumption coverage.

    It is also the stage where an interruption costs most: the application task
    allows an hour per model, so repeating finished work on a resume can waste a
    day.
    """

    def setUp(self):
        super().setUp()
        self.cfg.models = ["m:1", "m:2"]
        self.enterContext(mock.patch.object(agent.gpulock, "acquire"))
        self.enterContext(mock.patch.object(agent, "preflight", lambda models: None))
        self.enterContext(mock.patch.object(agent, "Ollama", self.client_factory))
        self.attempted = []

        def fake_task(model, task, workdir, timeout, retain_dir=None):
            self.attempted.append((model, task["id"]))
            return dict(model=model, task=task["id"], passed=False, detail="no",
                        wall_s=1.0, timed_out=False, returncode=0, tool_calls=0,
                        tools=[], turns=0, errors=[], tokens={}, peak_input_tokens=0,
                        repo=str(self.tmp / "r"), stderr="", ts=0.0, checks=[],
                        score=None, screenshot=None, retained=False)

        self.enterContext(mock.patch.object(agent, "run_task", fake_task))

    def run_agent(self, **kwargs):
        with mock.patch("sys.stdout", new=io.StringIO()) as out:
            agent.run(self.cfg, only=["ag_fixbug"], **kwargs)
        return out.getvalue()

    def test_a_second_run_attempts_nothing_again(self):
        self.run_agent()
        self.assertEqual(self.attempted, [("m:1", "ag_fixbug"), ("m:2", "ag_fixbug")])
        text = self.run_agent()
        self.assertEqual(len(self.attempted), 2, "recorded work was attempted again")
        self.assertIn("already measured, skipping", text)

    def test_only_the_unmeasured_model_is_attempted(self):
        self.cfg.models = ["m:1"]
        self.run_agent()
        self.cfg.models = ["m:1", "m:2"]
        self.run_agent()
        self.assertEqual(self.attempted,
                         [("m:1", "ag_fixbug"), ("m:2", "ag_fixbug")])

    def test_redo_attempts_everything_again(self):
        self.run_agent()
        self.run_agent(redo=True)
        self.assertEqual(len(self.attempted), 4)

    def test_the_model_is_released_after_its_last_task(self):
        # Every other stage unloads between models. Leaving one resident means the
        # next loads alongside it, and two 35B models do not fit.
        self.run_agent()
        unloaded = [m for c in self.clients for m in getattr(c, "unloaded", [])]
        self.assertEqual(unloaded, ["m:1", "m:2"])

    def test_a_damaged_ledger_line_does_not_force_the_whole_stage_again(self):
        self.run_agent()
        with (self.tmp / "agentic.jsonl").open("a", encoding="utf-8") as fh:
            fh.write('{"model": "m:1", "ta')          # cut off mid-write
        self.run_agent()
        self.assertEqual(len(self.attempted), 2)


class TestAgentBudget(ResumeCase):
    """An unattended sweep should not discover its own duration by running."""

    def setUp(self):
        super().setUp()
        self.cfg.models = ["m:1", "m:2"]
        self.enterContext(mock.patch.object(agent.gpulock, "acquire"))
        self.enterContext(mock.patch.object(agent, "preflight", lambda models: None))
        self.enterContext(mock.patch.object(agent, "Ollama", self.client_factory))
        self.enterContext(mock.patch.object(
            agent, "run_task",
            lambda model, task, workdir, timeout, retain_dir=None: dict(
                model=model, task=task["id"], passed=True, detail="ok", wall_s=1.0,
                timed_out=False, returncode=0, tool_calls=0, tools=[], turns=0,
                errors=[], tokens={}, peak_input_tokens=0, repo=str(self.tmp / "r"),
                stderr="", ts=0.0, checks=[], score=None, screenshot=None,
                retained=False)))

    def budget(self):
        with mock.patch("sys.stdout", new=io.StringIO()) as out:
            agent.run(self.cfg, only=["ag_module"], timeout=1200)
        return out.getvalue()

    def test_the_worst_case_is_stated_before_any_work_begins(self):
        text = self.budget()
        self.assertIn("2 task(s) to run", text)
        self.assertIn("0.7h", text, "two models at the task's own floor, not at "
                                    "the smaller default passed in")

    def test_work_already_recorded_is_not_counted_into_the_budget(self):
        self.budget()
        self.assertIn("0 task(s)", self.budget() + "0 task(s)")
        self.cfg.models = ["m:1", "m:2", "m:3"]
        self.assertIn("0.3h", self.budget())


class TestAgentSelection(ResumeCase):
    """Models the screen ruled out must not be given an hour each to prove it.

    The stage used to run every installed model regardless of verdict. With the
    application task allowing an hour apiece, a sweep spent most of a day on
    models already known to be unusable.
    """

    def setUp(self):
        super().setUp()
        self.enterContext(mock.patch.object(agent.gpulock, "acquire"))
        self.enterContext(mock.patch.object(agent, "preflight", lambda models: None))
        self.enterContext(mock.patch.object(agent, "Ollama", self.client_factory))
        self.write_screen()

    def write_screen(self):
        """One model at 100% on the hard set, one at 47%, one at 80%."""
        def rec(model, taskset, rate):
            tasks = [dict(task="t1", kind="codegen", passed=rate >= 50,
                          format_ok=True, detail="ok", wall=1.0),
                     dict(task="t2", kind="toolcall", passed=True, format_ok=True,
                          detail="ok", wall=0.5)]
            return dict(model=model, run=1, taskset=taskset, ctx=65536, n=14,
                        passed=int(rate / 100 * 14), pass_rate=rate,
                        format_ok_rate=100.0, hit_cap_n=0, median_wall=1.0,
                        total_s=10.0, tasks=tasks)

        def probe_rec(model):
            return dict(model=model, num_ctx=65536, gen_tok_s=45.0,
                        prefill_tok_s=800.0, prefill_s=60.0, prefill_toks=48000,
                        likely_truncated=False, retrieved=True,
                        placement={"pct_gpu": 40.0})

        with (self.tmp / "screen.jsonl").open("w", encoding="utf-8") as fh:
            for model, rate in (("good", 100.0), ("okay", 80.0), ("bad", 47.0)):
                for taskset in ("basic", "hard"):
                    fh.write(json.dumps(rec(model, taskset, rate)) + "\n")
        with (self.tmp / "probe.jsonl").open("w", encoding="utf-8") as fh:
            for model in ("good", "okay", "bad"):
                fh.write(json.dumps(probe_rec(model)) + "\n")
        self.cfg.models = ["good", "okay", "bad"]

    def test_the_default_skips_what_the_screen_ruled_out(self):
        picked = agent.select(self.cfg, self.cfg.models, ["suitable", "limited"])
        self.assertEqual(picked, ["good", "okay"])

    def test_an_empty_verdict_list_leaves_the_choice_to_the_caller(self):
        self.assertEqual(agent.select(self.cfg, self.cfg.models, []),
                         ["good", "okay", "bad"])

    def test_the_skipped_models_are_named_rather_than_silently_dropped(self):
        attempted = []

        def stub(model, task, workdir, timeout, retain_dir=None):
            attempted.append(model)
            return dict(model=model, task=task["id"], passed=True, detail="ok",
                        wall_s=1.0, timed_out=False, returncode=0, tool_calls=0,
                        tools=[], turns=0, errors=[], tokens={}, peak_input_tokens=0,
                        repo=str(self.tmp / "r"), stderr="", ts=0.0, checks=[],
                        score=None, screenshot=None, retained=False)

        with mock.patch.object(agent, "run_task", stub):
            with mock.patch("sys.stdout", new=io.StringIO()) as out:
                agent.run(self.cfg, only=["ag_fixbug"], redo=False,
                          select_verdicts=["suitable"])
            text = out.getvalue()
        self.assertEqual(attempted, ["good"], "only the suitable model was run")
        self.assertIn("skipping 2 model(s) the screen ruled out", text)
        self.assertIn("okay", text)
        self.assertIn("bad", text)

    def test_a_field_with_no_screen_results_says_so_instead_of_running(self):
        (self.tmp / "screen.jsonl").unlink()
        with mock.patch("sys.stderr", new=io.StringIO()) as err:
            code = agent.run(self.cfg, only=["ag_fixbug"],
                             select_verdicts=["suitable", "limited"])
        self.assertEqual(code, 2)
        self.assertIn("Screen the models first", err.getvalue())


if __name__ == "__main__":
    unittest.main()
