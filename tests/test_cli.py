"""Argument parsing must reach each stage with the values the user supplied."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from codesift import cli


class TestParsing(unittest.TestCase):
    def test_defaults(self):
        args = cli.build_parser().parse_args(["screen"])
        cfg = cli._config(args)
        self.assertEqual(cfg.ctx, 65536)
        self.assertEqual(cfg.results_dir, Path("results"))
        self.assertEqual(cfg.models, [])
        self.assertEqual(args.taskset, "basic")

    def test_models_and_file_are_merged(self):
        with tempfile.TemporaryDirectory() as td:
            listing = Path(td) / "m.txt"
            listing.write_text("from-file:1\n# skip\n", encoding="utf-8")
            args = cli.build_parser().parse_args(
                ["screen", "--models", "cli:1", "cli:2", "--models-file", str(listing)])
            self.assertEqual(cli._config(args).models, ["cli:1", "cli:2", "from-file:1"])

    def test_host_normalisation_flows_through(self):
        args = cli.build_parser().parse_args(["probe", "--host", "box:11434"])
        self.assertEqual(cli._config(args).host, "http://box:11434")

    def test_unknown_taskset_is_rejected(self):
        with self.assertRaises(SystemExit):
            cli.build_parser().parse_args(["screen", "--taskset", "nonsense"])

    def test_subcommand_is_required(self):
        with self.assertRaises(SystemExit):
            cli.build_parser().parse_args([])


class TestDispatch(unittest.TestCase):
    """Each subcommand must call its own stage, with the parsed options."""

    def test_screen(self):
        with mock.patch.object(cli.screen, "run") as run:
            cli.main(["screen", "--taskset", "hard", "--runs", "3",
                      "--models", "a:1", "--ctx", "8192"])
        cfg, kwargs = run.call_args[0][0], run.call_args[1]
        self.assertEqual(kwargs["taskset"], "hard")
        self.assertEqual(kwargs["runs"], 3)
        self.assertEqual(cfg.models, ["a:1"])
        self.assertEqual(cfg.ctx, 8192)

    def test_probe_passes_depth(self):
        with mock.patch.object(cli.probe, "run") as run:
            cli.main(["probe", "--depth", "12000"])
        self.assertEqual(run.call_args[1]["depth"], 12000)

    def test_cache(self):
        with mock.patch.object(cli.prefixcache, "run") as run:
            cli.main(["cache", "--redo"])
        self.assertTrue(run.call_args[1]["redo"])

    def test_agent_returns_its_exit_code(self):
        with mock.patch.object(cli.agent, "run", return_value=2) as run:
            self.assertEqual(cli.main(["agent"]), 2)
        self.assertEqual(run.call_args[1]["timeout"], 1200)

    def test_report_uses_requested_output_path(self):
        with mock.patch.object(cli.report, "run",
                               return_value=Path("out.html")) as run:
            cli.main(["report", "-o", "custom.html"])
        self.assertEqual(run.call_args[0][1], Path("custom.html"))

    def test_run_executes_every_stage_in_order(self):
        with mock.patch.object(cli.triage, "run"), \
             mock.patch.object(cli.screen, "run") as screen, \
             mock.patch.object(cli.probe, "run") as probe, \
             mock.patch.object(cli.prefixcache, "run") as cache, \
             mock.patch.object(cli.agent, "run", return_value=0) as agent, \
             mock.patch.object(cli.report, "run", return_value=Path("r.html")):
            cli.main(["run", "--models", "a:1"])
        self.assertEqual([c[1]["taskset"] for c in screen.call_args_list],
                         ["basic", "hard"])
        probe.assert_called_once()
        cache.assert_called_once()
        agent.assert_called_once()

    def test_run_can_omit_the_agent_stage(self):
        with mock.patch.object(cli.triage, "run"), \
             mock.patch.object(cli.screen, "run"), \
             mock.patch.object(cli.probe, "run"), \
             mock.patch.object(cli.prefixcache, "run"), \
             mock.patch.object(cli.agent, "run") as agent, \
             mock.patch.object(cli.report, "run", return_value=Path("r.html")):
            cli.main(["run", "--skip-agent"])
        agent.assert_not_called()

    def test_run_continues_when_the_agent_stage_is_unavailable(self):
        """A missing opencode must not cost the measurements already taken."""
        with mock.patch.object(cli.triage, "run"), \
             mock.patch.object(cli.screen, "run"), \
             mock.patch.object(cli.probe, "run"), \
             mock.patch.object(cli.prefixcache, "run"), \
             mock.patch.object(cli.agent, "run", return_value=2), \
             mock.patch.object(cli.report, "run", return_value=Path("r.html")) as report:
            self.assertEqual(cli.main(["run"]), 0)
        report.assert_called_once()


if __name__ == "__main__":
    unittest.main()


class TestTriageRunsFirst(unittest.TestCase):
    """The cascade is worthless if it runs after the stages it exists to skip."""

    def test_the_full_run_triages_before_it_screens(self):
        order = []
        with mock.patch.object(cli.triage, "run", side_effect=lambda *a, **k: order.append("triage")), \
             mock.patch.object(cli.screen, "run", side_effect=lambda *a, **k: order.append("screen")), \
             mock.patch.object(cli.probe, "run", side_effect=lambda *a, **k: order.append("probe")), \
             mock.patch.object(cli.prefixcache, "run"), \
             mock.patch.object(cli.agent, "run", return_value=0), \
             mock.patch.object(cli.report, "run", return_value=Path("r.html")):
            cli.main(["run", "--models", "a:1"])
        self.assertEqual(order[0], "triage")
        self.assertLess(order.index("triage"), order.index("screen"))

    def test_a_model_triage_rejected_is_not_screened_even_if_named(self):
        # Naming a model overrides the discard list by design, so the pipeline has
        # to read the triage result itself or triage would do nothing here.
        import tempfile, json as _json
        from pathlib import Path as _Path
        tmp = _Path(self.enterContext(tempfile.TemporaryDirectory())) \
            if hasattr(self, "enterContext") else None
        with tempfile.TemporaryDirectory() as d:
            (_Path(d) / "triage.jsonl").write_text(
                _json.dumps({"model": "a:1", "passed": False, "gate": "speed"}) + "\n",
                encoding="utf-8")
            with mock.patch.object(cli.triage, "run"), \
                 mock.patch.object(cli.screen, "run") as screen, \
                 mock.patch.object(cli.probe, "run"), \
                 mock.patch.object(cli.prefixcache, "run"), \
                 mock.patch.object(cli.agent, "run", return_value=0), \
                 mock.patch.object(cli.report, "run", return_value=Path("r.html")):
                cli.main(["run", "--models", "a:1", "--results-dir", d])
            screen.assert_not_called()

    def test_the_full_run_acts_on_what_triage_rejected(self):
        # Without --apply the rejections are only printed, and the stages below
        # would measure the models triage just ruled out.
        with mock.patch.object(cli.triage, "run") as triage_run, \
             mock.patch.object(cli.screen, "run"), \
             mock.patch.object(cli.probe, "run"), \
             mock.patch.object(cli.prefixcache, "run"), \
             mock.patch.object(cli.agent, "run", return_value=0), \
             mock.patch.object(cli.report, "run", return_value=Path("r.html")):
            cli.main(["run", "--models", "a:1"])
        self.assertTrue(triage_run.call_args[1]["apply"])
