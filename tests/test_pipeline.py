# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""The whole pipeline, run for real, with only the model faked.

Every other test of `run` mocks each stage, which proves the stages are called
and nothing about what happens inside them. Two of the worst faults this project
has had were invisible to that: a results path resolved against the wrong
directory, and an edit placed in the wrong function left a name undefined on a
line no unit test reached. Both would have failed here on the first run.

So nothing internal is patched. Only the Ollama client is replaced, because there
is no server -- everything else is the real thing: grading, ledgers, resumption,
the verdicts and the rendered report. The working directory is a temporary one and
the results directory is given relatively, since that is the shape that broke.
"""
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from codesift import cli, progress
from codesift.tasks import TASKS
from tests import reference

# The fake model only sees a prompt, so its answers are looked up by one.
TASK_BY_PROMPT = {t["prompt"]: t for t in TASKS}


class FakeOllama:
    """Answers every request plausibly, and records which models were asked."""

    def __init__(self, *args, **kwargs):
        self.asked, self.unloaded = [], []

    def chat(self, model, prompt, ctx=None, num_predict=None, tools=None):
        self.asked.append(model)
        common = {"_wall": 0.1, "eval_count": 20, "eval_duration": 10 ** 8,
                  "load_duration": 10 ** 9, "prompt_eval_count": 10 ** 6,
                  "prompt_eval_duration": 10 ** 9}
        if "DEPLOY_TOKEN" in prompt:                       # the deep probe
            return dict(common, message={"content": "quartz-mongoose-8814"})
        task = TASK_BY_PROMPT.get(prompt.split("\n\n```python")[0])
        if task is None:
            return dict(common, message={"content": "def f():\n    return 1"})
        return dict(common, message=reference.perfect_answer(task))

    def placement(self, model):
        return {"pct_gpu": 60.0, "total_gb": 10.0, "vram_gb": 6.0}

    def unload(self, model):
        self.unloaded.append(model)


class TestTheWholePipeline(unittest.TestCase):
    def setUp(self):
        progress.reset()
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.cwd = Path.cwd()
        os.chdir(self.tmp)
        self.addCleanup(os.chdir, self.cwd)
        self.client = FakeOllama()

        for module in (cli.screen, cli.probe, cli.triage):
            self.enterContext(mock.patch.object(module, "Ollama",
                                                lambda *a, **k: self.client))
            if hasattr(module, "gpulock"):
                self.enterContext(mock.patch.object(module.gpulock, "acquire"))

    def run_pipeline(self, *extra, models=("m:1",)):
        out = io.StringIO()
        with mock.patch("sys.stdout", new=out):
            code = cli.main(["run", "--models", *models, "--results-dir", "results",
                             "-o", "report.html", *extra])
        return code, out.getvalue()

    def ledger(self, name):
        path = self.tmp / "results" / name
        if not path.exists():
            return []
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

    def test_every_stage_records_something(self):
        code, _ = self.run_pipeline()
        self.assertEqual(code, 0)
        for name in ("triage.jsonl", "screen_tasks.jsonl", "probe.jsonl"):
            with self.subTest(ledger=name):
                self.assertTrue(self.ledger(name), f"{name} is empty")

    def test_the_whole_pipeline_runs_at_a_smaller_window(self):
        # The window is the user's to choose, and the depth measurements follow it.
        # Pinned at 48,000 tokens, a smaller window made the probe prompt too large
        # to fit, so every model read as truncated and triage rejected the field.
        code, _ = self.run_pipeline("--ctx", "32768")
        self.assertEqual(code, 0)
        for rec in self.ledger("probe.jsonl"):
            with self.subTest(model=rec["model"]):
                self.assertEqual(rec["num_ctx"], 32768)
                self.assertEqual(rec["depth_target"], 24576)
                self.assertLess(rec["depth_target"], rec["num_ctx"],
                                "the probe prompt cannot fit the window it was sent to")
        page = (self.tmp / "report.html").read_text(encoding="utf-8")
        self.assertIn("32,768", page)
        self.assertNotIn("65,536", page)

    def test_the_report_renders(self):
        self.run_pipeline()
        page = (self.tmp / "report.html").read_text(encoding="utf-8")
        self.assertIn("<title>", page)
        self.assertIn("m:1", page)

    def test_no_task_is_measured_twice(self):
        self.run_pipeline()
        seen = [r["task"] for r in self.ledger("screen_tasks.jsonl") if r["run"] == 1]
        self.assertEqual(len(seen), len(set(seen)),
                         "triage and the screen both measured the same task")
        # Every task graded exactly once, whichever stage reached it first.
        from codesift.tasks import TASKS
        self.assertEqual(len(seen), len(TASKS))

    def test_the_model_is_released_by_every_stage_that_loads_it(self):
        self.run_pipeline()
        # Triage and the screen. The probe stage loads nothing here: triage's
        # context gate takes the deep measurement and records it, so the probe
        # finds the model already measured and skips it.
        self.assertGreaterEqual(self.client.unloaded.count("m:1"), 2)

    def test_a_second_run_measures_nothing_again(self):
        self.run_pipeline()
        before = len(self.client.asked)
        self.run_pipeline()
        self.assertEqual(len(self.client.asked), before,
                         "a resumed pipeline repeated finished work")

    def test_a_model_triage_rejects_never_reaches_the_screen(self):
        slow = {"message": {"content": "x"}, "_wall": 0.1, "eval_count": 5,
                "eval_duration": 5 * 10 ** 9, "load_duration": 10 ** 9}
        with mock.patch.object(self.client, "chat", return_value=slow):
            self.run_pipeline()
        self.assertFalse(self.ledger("screen.jsonl"),
                         "the screen ran on a model triage had rejected")
        # The finding names what stopped it; a separate gate field said the same.
        [rec] = self.ledger("triage.jsonl")
        self.assertFalse(rec["passed"])
        self.assertEqual([f["code"] for f in rec["findings"]], ["slow_generation"])


if __name__ == "__main__":
    unittest.main()


class TestTheReportKeepsUp(unittest.TestCase):
    """Triage covers the field, then each model is finished and written out.

    A sweep is hours long. Rendering once at the end meant a run that died at the
    seventh model left nothing to look at, and nothing to look at while it ran.
    """

    def setUp(self):
        progress.reset()
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.cwd = Path.cwd()
        os.chdir(self.tmp)
        self.addCleanup(os.chdir, self.cwd)
        client = FakeOllama()
        for module in (cli.screen, cli.probe, cli.triage):
            self.enterContext(mock.patch.object(module, "Ollama",
                                                lambda *a, **k: client))
            self.enterContext(mock.patch.object(module.gpulock, "acquire"))
        self.rendered = []
        real = cli.report.run
        def spy(cfg, path, models=None):
            self.rendered.append(list(models or []))
            return real(cfg, path, models)
        self.enterContext(mock.patch.object(cli.report, "run", spy))

    def test_the_report_is_rewritten_after_every_model(self):
        with mock.patch("sys.stdout", new=io.StringIO()):
            cli.main(["run", "--models", "a:1", "b:1", "c:1",
                      "--results-dir", "results", "-o", "report.html"])
        # Once per model, and each render covers the whole field: a model not yet
        # reached reads as not screened, and one triage rejected is named with what
        # stopped it rather than being absent.
        self.assertEqual(self.rendered, [["a:1", "b:1", "c:1"]] * 3)

    def test_triage_still_covers_the_field_before_any_of_it(self):
        # The cheapest stage answers "which of these cannot be used" first, which
        # is the question a reader has before any model is screened.
        with mock.patch("sys.stdout", new=io.StringIO()):
            cli.main(["run", "--models", "a:1", "b:1", "--results-dir", "results",
                      "-o", "report.html"])
        judged = [json.loads(l)["model"] for l in
                  (self.tmp / "results" / "triage.jsonl").read_text().splitlines()]
        screened = [json.loads(l)["model"] for l in
                    (self.tmp / "results" / "screen_tasks.jsonl").read_text().splitlines()]
        self.assertEqual(judged, ["a:1", "b:1"])
        self.assertEqual(sorted(set(screened)), ["a:1", "b:1"])
