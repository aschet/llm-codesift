"""Discarding models is destructive, so the parts that bound the damage are tested.

What matters is that only the named models lose records, that the originals
survive alongside, that unparseable lines are not collateral, and that a discard
stops a sweep without stopping a deliberate run.
"""
import io
import json
import tempfile
import unittest
from pathlib import Path

from codesift import prune
from codesift.config import Config

OFFLINE = "http://127.0.0.1:1"


def screen_record(model, taskset="basic", rate=100.0):
    tasks = [dict(task="t1", kind="codegen", passed=rate >= 50, format_ok=True,
                  detail="ok", wall=1.0),
             dict(task="t2", kind="toolcall", passed=True, format_ok=True,
                  detail="ok", wall=0.5)]
    return dict(model=model, run=1, taskset=taskset, ctx=65536, n=14,
                passed=int(rate / 100 * 14), pass_rate=rate, format_ok_rate=100.0,
                hit_cap_n=0, median_wall=1.0, total_s=10.0, tasks=tasks)


def probe_record(model, gen=45.0):
    return dict(model=model, num_ctx=65536, gen_tok_s=gen, prefill_tok_s=800.0,
                prefill_s=60.0, prefill_toks=48000, likely_truncated=False,
                retrieved=True, placement={"pct_gpu": 40.0})


class PruneCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.cfg = Config(host=OFFLINE, results_dir=self.tmp)
        self.out = io.StringIO()

    def write(self, name, records):
        with (self.tmp / name).open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")

    def field(self):
        """One model worth keeping and one the hard set ruled out."""
        self.write("screen.jsonl", [
            screen_record("keeper", "basic", 100.0), screen_record("keeper", "hard", 100.0),
            screen_record("dud", "basic", 57.0), screen_record("dud", "hard", 47.0)])
        self.write("probe.jsonl", [probe_record("keeper"), probe_record("dud", gen=280.0)])
        self.write("agentic.jsonl", [dict(model="dud", task="ag_fixbug", passed=False)])

    def lines(self, name):
        return [json.loads(l) for l in (self.tmp / name).read_text().splitlines() if l]


class TestSelection(PruneCase):
    def test_only_the_ruled_out_model_is_named(self):
        self.field()
        prune.run(self.cfg, stream=self.out)
        text = self.out.getvalue()
        self.assertIn("dud", text)
        self.assertNotIn("keeper", text)

    def test_a_dry_run_changes_nothing(self):
        self.field()
        prune.run(self.cfg, stream=self.out)
        self.assertIn("Dry run", self.out.getvalue())
        self.assertEqual(len(self.lines("screen.jsonl")), 4)
        self.assertEqual(prune.read_discarded(self.tmp), [])

    def test_named_models_override_the_verdict(self):
        self.field()
        prune.run(self.cfg, models=["keeper"], apply=True, stream=self.out)
        self.assertEqual(prune.read_discarded(self.tmp), ["keeper"])
        self.assertEqual({r["model"] for r in self.lines("screen.jsonl")}, {"dud"})


class TestRemoval(PruneCase):
    def test_records_go_and_the_original_stays_alongside(self):
        self.field()
        prune.run(self.cfg, apply=True, stream=self.out)
        self.assertEqual({r["model"] for r in self.lines("screen.jsonl")}, {"keeper"})
        self.assertEqual({r["model"] for r in self.lines("probe.jsonl")}, {"keeper"})
        self.assertEqual(self.lines("agentic.jsonl"), [])
        self.assertEqual(len(self.lines("screen.jsonl.bak")), 4,
                         "the original must survive a deletion this size")

    def test_an_unparseable_line_is_not_collateral(self):
        # It belongs to no model that can be identified, so it is not this tool's
        # to delete; losing it would destroy data nobody asked to touch.
        self.field()
        with (self.tmp / "screen.jsonl").open("a") as fh:
            fh.write('{"model": "dud", "trunc')
        prune.run(self.cfg, apply=True, stream=self.out)
        raw = (self.tmp / "screen.jsonl").read_text()
        self.assertIn('{"model": "dud", "trunc', raw)

    def test_keep_records_discards_the_name_only(self):
        self.field()
        prune.run(self.cfg, apply=True, keep_records=True, stream=self.out)
        self.assertEqual(len(self.lines("screen.jsonl")), 4)
        self.assertEqual(prune.read_discarded(self.tmp), ["dud"])

    def test_a_ledger_without_the_model_is_left_alone(self):
        self.field()
        self.write("prefix_cache.jsonl", [dict(model="keeper", turns=[])])
        prune.run(self.cfg, apply=True, stream=self.out)
        self.assertFalse((self.tmp / "prefix_cache.jsonl.bak").exists(),
                         "an untouched ledger should not be backed up")

    def test_a_retained_application_is_removed_with_its_model(self):
        self.field()
        app = self.tmp / "agent_apps" / "dud__ag_module"
        app.mkdir(parents=True)
        (app / "app.py").write_text("x", encoding="utf-8")
        keep = self.tmp / "agent_apps" / "keeper__ag_module"
        keep.mkdir(parents=True)
        prune.run(self.cfg, apply=True, stream=self.out)
        self.assertFalse(app.exists())
        self.assertTrue(keep.exists())


class TestDiscardList(PruneCase):
    def test_a_sweep_skips_a_discarded_model_but_a_named_run_does_not(self):
        prune.write_discarded(self.tmp, ["dud"])
        cfg = Config(host=OFFLINE, results_dir=self.tmp)
        import codesift.config as config
        real = config.installed_models
        config.installed_models = lambda host: ["keeper", "dud"]
        try:
            self.assertEqual(cfg.resolve_models(), ["keeper"])
            named = Config(host=OFFLINE, results_dir=self.tmp, models=["dud"])
            self.assertEqual(named.resolve_models(), ["dud"],
                             "naming a model must run it whatever the screen concluded")
        finally:
            config.installed_models = real

    def test_discarding_twice_accumulates_rather_than_replaces(self):
        prune.write_discarded(self.tmp, ["a"])
        prune.write_discarded(self.tmp, ["b"])
        self.assertEqual(prune.read_discarded(self.tmp), ["a", "b"])

    def test_comments_and_blanks_are_ignored(self):
        (self.tmp / prune.DISCARDED).write_text("# a note\n\ndud\n", encoding="utf-8")
        self.assertEqual(prune.read_discarded(self.tmp), ["dud"])

    def test_forget_restores_every_discarded_model(self):
        prune.write_discarded(self.tmp, ["a", "b"])
        prune.run(self.cfg, forget=True, stream=self.out)
        self.assertIn("Dry run", self.out.getvalue())
        self.assertEqual(prune.read_discarded(self.tmp), ["a", "b"])
        prune.run(self.cfg, forget=True, apply=True, stream=self.out)
        self.assertEqual(prune.read_discarded(self.tmp), [])


if __name__ == "__main__":
    unittest.main()
