"""The whole pipeline, run for real, with only the model faked.

Every other test of `run` mocks each stage, which proves the stages are called
and nothing about what happens inside them. Two of the worst faults this project
has had were invisible to that: a repository seeded at a relative path produced a
doubled path so the grader could not open its own check script, and an edit
placed in the wrong function left a name undefined on a line no unit test reached.
Both would have failed here on the first run.

So nothing internal is patched. The Ollama client is replaced, because there is
no server, and opencode is replaced, because there is no model -- and everything
between them is the real thing: seeding, grading, ledgers, resumption, the
verdicts and the rendered report. The working directory is a temporary one and
the results directory is given relatively, since that is the shape that broke.
"""
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from codesift import agent, cli
from codesift.tasks import TASKSETS
from tests import reference

REFERENCE = Path(__file__).parent / "tasklist_reference"

# The fake model only sees a prompt, so its answers are looked up by one.
TASK_BY_PROMPT = {t["prompt"]: t
                  for name in TASKSETS for t in TASKSETS[name]}


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

    def chat_messages(self, model, messages, **kwargs):
        return {"message": {"content": "1"}, "_wall": 0.1,
                "prompt_eval_count": 100, "prompt_eval_duration": 5 * 10 ** 8}

    def placement(self, model):
        return {"pct_gpu": 60.0, "total_gb": 10.0, "vram_gb": 6.0}

    def show(self, model):
        return {"model_info": {"general.architecture": "fake",
                               "fake.expert_count": 128, "fake.expert_used_count": 8}}

    def unload(self, model):
        self.unloaded.append(model)


class FakeOpencode:
    """Stands in for the harness: writes the answer, then exits like opencode."""

    def __init__(self, cmd, **kwargs):
        self.returncode = 0
        repo = Path(cmd[cmd.index("--dir") + 1])
        task = cmd[cmd.index("--title") + 1].split("-", 1)[1]
        self._write(repo, task)

    def _write(self, repo, task):
        if task == "ag_module":
            shutil.copytree(REFERENCE, repo, dirs_exist_ok=True)
            return
        for rel, body in (reference.AGENT.get(task) or {}).items():
            (repo / rel).parent.mkdir(parents=True, exist_ok=True)
            (repo / rel).write_text(body, encoding="utf-8")

    def communicate(self, timeout=None):
        return ('{"type": "step_start", "part": {}}\n'
                '{"type": "step_finish", "part": {"tokens": {"input": 900, '
                '"output": 120, "total": 1020}}}\n', "")

    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0


class _OnlyPopenFaked:
    """Replaces `subprocess` inside the agent module, and nothing else.

    Patching the attribute on the real module would have replaced Popen for the
    whole process, and `subprocess.run` is built on Popen -- so the graders, which
    run candidate code in a subprocess, would have been answered by the fake
    harness instead of executing anything.
    """

    Popen = FakeOpencode

    def __getattr__(self, name):
        return getattr(subprocess, name)


class TestTheWholePipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.cwd = Path.cwd()
        os.chdir(self.tmp)
        self.addCleanup(os.chdir, self.cwd)
        self.client = FakeOllama()

        for module in (cli.screen, cli.probe, cli.prefixcache, cli.triage, agent):
            self.enterContext(mock.patch.object(module, "Ollama",
                                                lambda *a, **k: self.client))
            if hasattr(module, "gpulock"):
                self.enterContext(mock.patch.object(module.gpulock, "acquire"))
        self.enterContext(mock.patch.object(agent, "preflight", lambda models: None))
        self.enterContext(mock.patch.object(agent, "subprocess", _OnlyPopenFaked()))

    def run_pipeline(self, *extra):
        out = io.StringIO()
        with mock.patch("sys.stdout", new=out):
            code = cli.main(["run", "--models", "m:1", "--results-dir", "results",
                             "--runs", "1", "-o", "report.html", *extra])
        return code, out.getvalue()

    def ledger(self, name):
        path = self.tmp / "results" / name
        if not path.exists():
            return []
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

    def test_every_stage_records_something(self):
        code, _ = self.run_pipeline()
        self.assertEqual(code, 0)
        for name in ("triage.jsonl", "screen.jsonl", "screen_tasks.jsonl",
                     "probe.jsonl", "prefix_cache.jsonl", "agentic.jsonl"):
            with self.subTest(ledger=name):
                self.assertTrue(self.ledger(name), f"{name} is empty")

    def test_the_report_renders(self):
        self.run_pipeline()
        page = (self.tmp / "report.html").read_text(encoding="utf-8")
        self.assertIn("<title>", page)
        self.assertIn("m:1", page)

    def test_a_relative_results_directory_does_not_double_the_path(self):
        # The exact shape that broke: everything downstream runs with the
        # repository as its working directory, so a relative path resolves
        # against the repository instead of against here.
        self.run_pipeline()
        for rec in self.ledger("agentic.jsonl"):
            with self.subTest(task=rec["task"]):
                self.assertTrue(Path(rec["repo"]).is_absolute())
                self.assertNotIn("can't open file", rec["detail"])

    def test_the_graded_task_is_graded_rather_than_crashing(self):
        self.run_pipeline()
        graded = [r for r in self.ledger("agentic.jsonl") if r.get("checks")]
        self.assertTrue(graded, "no task produced per-check results")
        for rec in graded:
            with self.subTest(task=rec["task"]):
                self.assertTrue(rec["passed"],
                                f"the reference answer was rejected: {rec['detail']}")

    def test_the_hard_set_is_measured_once(self):
        _, text = self.run_pipeline()
        hard = [r for r in self.ledger("screen_tasks.jsonl")
                if r["taskset"] == "hard" and r["run"] == 1]
        seen = [r["task"] for r in hard]
        self.assertEqual(len(seen), len(set(seen)),
                         "triage and the screen both measured it")
        self.assertIn("complete", text)

    def test_the_model_is_released_by_every_stage(self):
        self.run_pipeline()
        self.assertGreaterEqual(self.client.unloaded.count("m:1"), 4)

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
        self.assertEqual(self.ledger("triage.jsonl")[0]["gate"], "speed")


if __name__ == "__main__":
    unittest.main()
