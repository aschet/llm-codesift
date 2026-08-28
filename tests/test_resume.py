# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""Interrupted runs must continue, not restart and not silently skip work."""
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from codesift import probe, progress, screen
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

    def placement(self, model):
        return {"pct_gpu": 50.0, "total_gb": 10.0, "vram_gb": 5.0}

    def unload(self, model):
        self.unloaded.append(model)


class ResumeCase(unittest.TestCase):
    def setUp(self):
        progress.reset()
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

    def test_a_task_is_stored_once_however_often_it_is_measured(self):
        self.run_screen()
        self.run_screen(redo=True)
        rows = [json.loads(l) for l in
                (self.tmp / "screen_tasks.jsonl").read_text(encoding="utf-8").splitlines()]
        keys = [(r["model"], r["run"], r["task"]) for r in rows]
        self.assertEqual(len(keys), len(set(keys)), "duplicate task records")

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

    def test_redo_measures_again_rather_than_reading_back(self):
        """--redo exists to replace a record, so it must not be answered by one."""
        from codesift import probe
        probe.record(self.cfg, dict(model="m:1", num_ctx=self.cfg.ctx, prefill_s=4.0,
                                    depth_target=100, gen_tok_s=50.0, retrieved=True))
        with mock.patch.object(probe, "Ollama", self.client_factory), \
             mock.patch.object(probe.gpulock, "acquire"):
            probe.run(self.cfg, depth=100, redo=True)
        self.assertGreater(self.total_calls, 0, "the stored record answered instead")

    def test_failed_measurements_are_retried(self):
        """A record carrying an error must not count as done."""
        (self.tmp / "probe.jsonl").write_text(
            json.dumps({"model": "m:1", "num_ctx": 4096, "error": "boom"}) + "\n",
            encoding="utf-8")
        with mock.patch.object(probe, "Ollama", self.client_factory), \
             mock.patch.object(probe.gpulock, "acquire"):
            probe.run(self.cfg, depth=100)
        self.assertGreater(self.total_calls, 0, "an errored model should be retried")


class TestTheDepthIsMeasuredNotAssumed(unittest.TestCase):
    """The prompt is sized by the model's own tokenizer, and the server is asked.

    A fixed characters-per-token figure sizes the prompt wrong by up to a third,
    and the error runs the wrong way: a prompt built for three quarters of the
    window arrives at the whole of it, and the overflow is discarded in silence.
    """

    class Counter:
        """Reports a token count derived from a fixed characters-per-token rate."""

        def __init__(self, chars_per_token=2.5, overhead=7, cap=None):
            self.rate, self.overhead, self.cap = chars_per_token, overhead, cap
            self.lengths, self.prompts = [], []

        def chat(self, model, prompt, **kwargs):
            self.lengths.append(len(prompt))
            self.prompts.append(prompt)
            n = self.overhead + int(len(prompt) / self.rate)
            return {"message": {"content": "quartz-mongoose-8814"}, "_wall": 0.1,
                    "eval_count": 4, "eval_duration": 10 ** 8,
                    "prompt_eval_count": min(n, self.cap) if self.cap else n,
                    "prompt_eval_duration": 10 ** 9}

        def placement(self, model):
            return {"pct_gpu": 100.0, "total_gb": 4.0}

    def test_the_ratio_comes_from_the_server(self):
        client = self.Counter(chars_per_token=2.5)
        self.assertAlmostEqual(probe.calibrate(client, "m:1", 4096, 24576), 2.5,
                               places=1)

    def test_the_ratio_is_measured_on_the_text_that_will_be_sent(self):
        # The block index grows through the filler and handler_7 does not
        # tokenize like handler_431, so the sample has to come from where the
        # prompt actually is, not from its first few blocks.
        client = self.Counter()
        probe.calibrate(client, "m:1", 4096, 24576)
        sample = client.prompts[-1]
        whole = probe._filler(24576, probe.DEFAULT_CHARS_PER_TOKEN)
        self.assertIn(sample, whole, "the sample is a slice of the real prompt")
        self.assertLessEqual(len(sample), probe.CALIBRATION_CHARS)
        indices = [int(n) for n in re.findall(r"def handler_(\d+)", sample)]
        highest = whole.count("def handler_") - 1
        self.assertGreater(min(indices), highest // 4,
                           "a slice from the start prices numbers that are shorter")

    def test_the_prompt_is_sized_to_the_measured_ratio(self):
        # The one that matters: at 2.5 chars/token a prompt built at 3.5 would be
        # 40% over, which is what pushed a model past its window.
        client = self.Counter(chars_per_token=2.5)
        rec = probe.measure(client, "m:1", 32768, 24576)
        self.assertAlmostEqual(rec["depth_tokens"] / 24576, 1.0, delta=0.1)
        self.assertFalse(rec["likely_truncated"])

    def test_a_prompt_the_window_could_not_hold_is_seen(self):
        # The server reports what it read, so a cut prompt is a number, not a guess.
        client = self.Counter(chars_per_token=2.5, cap=16384)
        rec = probe.measure(client, "m:1", 32768, 24576)
        self.assertEqual(rec["depth_tokens"], 16384)
        self.assertTrue(rec["likely_truncated"])

    def test_an_unreadable_count_falls_back_to_the_shorter_prompt(self):
        class Mute(self.Counter):
            def chat(self, model, prompt, **kwargs):
                out = super().chat(model, prompt, **kwargs)
                out["prompt_eval_count"] = None
                return out

        self.assertEqual(probe.calibrate(Mute(), "m:1", 4096, 24576),
                         probe.DEFAULT_CHARS_PER_TOKEN)


class TestTheDeepPromptIsPaidOnce(ResumeCase):
    """Triage and the probe stage take the same measurement.

    The deep prompt costs a minute on a large model, so whichever of them runs
    first writes it to the probe ledger and the other reads it. Only a record
    taken at depth counts: the speed gate writes one too, and it holds a rate.
    """

    def setUp(self):
        progress.reset()
        super().setUp()
        self.cfg.models = ["m:1"]

    def test_a_stored_measurement_is_found(self):
        from codesift import probe
        probe.record(self.cfg, dict(model="m:1", num_ctx=self.cfg.ctx, prefill_s=4.0,
                                    depth_target=24576, gen_tok_s=50.0, retrieved=True))
        self.assertIsNotNone(probe.stored(self.cfg, "m:1", self.cfg.ctx))

    def test_a_shallow_record_is_not_mistaken_for_a_measurement(self):
        # The speed gate stops before the deep prompt and stores what it did
        # measure; treating that as complete would skip the deep measurement.
        from codesift import probe
        probe.record(self.cfg, dict(model="m:1", num_ctx=self.cfg.ctx, gen_tok_s=50.0,
                                    depth_target=0))
        self.assertIsNone(probe.stored(self.cfg, "m:1", self.cfg.ctx))
        self.assertFalse(probe.at_depth(dict(model="m:1", gen_tok_s=50.0)))

    def test_a_measurement_at_another_window_does_not_count(self):
        from codesift import probe
        probe.record(self.cfg, dict(model="m:1", num_ctx=8192, prefill_s=4.0,
                                    depth_target=6144))
        self.assertIsNone(probe.stored(self.cfg, "m:1", self.cfg.ctx))

    def test_a_second_write_replaces_the_first(self):
        # Triage writes what the speed gate measured, then the deep measurement
        # supersedes it. One model at one window is one record.
        from codesift import probe
        probe.record(self.cfg, dict(model="m:1", num_ctx=self.cfg.ctx, gen_tok_s=50.0))
        probe.record(self.cfg, dict(model="m:1", num_ctx=self.cfg.ctx, prefill_s=4.0,
                                    depth_target=24576, retrieved=True))
        rows = [json.loads(l) for l in
                (self.tmp / "probe.jsonl").read_text().splitlines() if l.strip()]
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["retrieved"])

    def test_another_model_is_not_disturbed(self):
        from codesift import probe
        probe.record(self.cfg, dict(model="keep:1", num_ctx=self.cfg.ctx, prefill_s=1.0))
        probe.record(self.cfg, dict(model="m:1", num_ctx=self.cfg.ctx, prefill_s=4.0))
        probe.record(self.cfg, dict(model="m:1", num_ctx=self.cfg.ctx, prefill_s=5.0))
        rows = [json.loads(l) for l in
                (self.tmp / "probe.jsonl").read_text().splitlines() if l.strip()]
        self.assertEqual({r["model"] for r in rows}, {"keep:1", "m:1"})
        self.assertEqual(len(rows), 2)

    def test_a_failed_measurement_does_not_count(self):
        from codesift import probe
        probe.record(self.cfg, dict(model="m:1", num_ctx=self.cfg.ctx, prefill_s=4.0,
                                    error="deep: timeout"))
        self.assertIsNone(probe.stored(self.cfg, "m:1", self.cfg.ctx))


if __name__ == "__main__":
    unittest.main()


class TestPartialSelection(ResumeCase):
    """`--only` re-measures one task; it must not rewrite the model's standing.

    Built from the tasks one invocation selected, the summary went from 29 tasks
    to 2, and the report ranked the model on those 2 -- which is the opposite of
    what the flag is for, since it exists to make re-measuring one task cheap.
    """

    def tasks(self):
        return [json.loads(l) for l in
                (self.tmp / "screen_tasks.jsonl").read_text().splitlines() if l.strip()]

    def screen_with(self, **kwargs):
        with mock.patch.object(screen, "Ollama", self.client_factory):
            screen.run(self.cfg, **kwargs)

    def test_re_measuring_one_task_keeps_the_rest_on_record(self):
        self.screen_with(only=["fmt_oneword", "tc_single"])
        before = {t["task"] for t in self.tasks()}
        self.screen_with(only=["fmt_oneword"], redo=True)
        self.assertEqual({t["task"] for t in self.tasks()}, before,
                         "re-measuring one task dropped the others")

    def test_a_re_measured_task_supersedes_the_stored_one(self):
        self.screen_with(only=["fmt_oneword", "tc_single"])
        self.screen_with(only=["fmt_oneword"], redo=True)
        self.assertEqual(len([t for t in self.tasks() if t["task"] == "fmt_oneword"]), 1)

    def test_an_unknown_task_id_is_refused(self):
        # Silently measuring less than was asked for is the worst answer: the
        # flag exists to re-measure one task, and a typo would report success
        # having never run it.
        with self.assertRaises(SystemExit):
            self.screen_with(only=["fmt_oneword", "no_such_task"])
