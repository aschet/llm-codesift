# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""Argument parsing must reach each stage with the values the user supplied."""
import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from codesift import cli
from codesift.args import EXECUTION_WARNING, config_from as A_config
from codesift.config import DEFAULT_CTX, working_depth


class TestParsing(unittest.TestCase):
    """The shared options mean the same thing wherever they are parsed."""

    def stage_args(self, name, argv):
        from codesift import args as A
        return A.stage(name, "", measuring=(name != "report")).parse_args(argv)

    def test_defaults(self):
        parsed = self.stage_args("screen", [])
        cfg = A_config(parsed)
        self.assertEqual(cfg.ctx, 32768)
        self.assertEqual(cfg.results_dir, Path("report_data"))
        self.assertEqual(cfg.models, [])

    def test_the_records_follow_the_report(self):
        # A report and its measurements are one result, so naming the report
        # names the store, and two reports do not share one.
        cfg = A_config(cli.build_parser().parse_args(
            ["run", "-o", "/tmp/sweeps/june.html"]))
        self.assertEqual(cfg.results_dir, Path("/tmp/sweeps/june_data"))

    def test_an_explicit_store_wins_over_the_report_name(self):
        cfg = A_config(cli.build_parser().parse_args(
            ["run", "-o", "june.html", "--results-dir", "shared"]))
        self.assertEqual(cfg.results_dir, Path("shared"))

    def test_models_and_file_are_merged(self):
        with tempfile.TemporaryDirectory() as td:
            listing = Path(td) / "m.txt"
            listing.write_text("from-file:1\n# skip\n", encoding="utf-8")
            parsed = self.stage_args(
                "screen", ["--models", "cli:1", "cli:2", "--models-file", str(listing)])
            self.assertEqual(A_config(parsed).models, ["cli:1", "cli:2", "from-file:1"])

    def test_host_normalisation_flows_through(self):
        parsed = self.stage_args("probe", ["--host", "box:11434"])
        self.assertEqual(A_config(parsed).host, "http://box:11434")

    def test_an_unknown_option_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.stage_args("screen", ["--taskset", "hard"])

    def test_subcommand_is_required(self):
        with self.assertRaises(SystemExit):
            cli.build_parser().parse_args([])


class TestDispatch(unittest.TestCase):
    """Each subcommand must call its own stage, with the parsed options."""

    def test_screen(self):
        from codesift import screen
        with mock.patch.object(screen, "run") as run:
            screen.main(["--models", "a:1", "--ctx", "8192"])
        cfg, kwargs = run.call_args[0][0], run.call_args[1]
        self.assertEqual(cfg.ctx, 8192)
        self.assertNotIn("runs", kwargs, "repetition is no longer a setting")
        self.assertNotIn("taskset", kwargs, "one task set, so nothing to choose")
        self.assertEqual(cfg.models, ["a:1"])
        self.assertEqual(cfg.ctx, 8192)

    def test_repetition_cannot_be_asked_for(self):
        # Greedy decoding with a fixed seed answers the same way every time, so a
        # second run measured nothing and cost as much as the first.
        from codesift import screen
        with self.assertRaises(SystemExit):
            screen.main(["--runs", "3"])

    def test_probe_passes_the_depth_the_window_implies(self):
        from codesift import probe
        with mock.patch.object(probe, "run") as run:
            probe.main(["--ctx", "16384"])
        self.assertEqual(run.call_args[1]["depth"], 12288)


    def test_report_uses_requested_output_path(self):
        from codesift import report
        with mock.patch.object(report, "run", return_value=Path("out.html")) as run:
            report.main(["-o", "custom.html"])
        self.assertEqual(run.call_args[0][1], Path("custom.html"))

    def test_run_executes_every_stage_in_order(self):
        with mock.patch.object(cli.triage, "run"), \
             mock.patch.object(cli.screen, "run") as screen, \
             mock.patch.object(cli.probe, "run") as probe, \
             mock.patch.object(cli.report, "run", return_value=Path("r.html")):
            cli.main(["run", "--models", "a:1"])
        screen.assert_called_once()
        probe.assert_called_once()


if __name__ == "__main__":
    unittest.main()


STAGES = ("screen", "probe", "triage", "regrade", "report")


class TestTheHelpLeadsWithWhatAUserRuns(unittest.TestCase):
    """A user has `run` and `discover`. The help offers those and nothing else.

    The stages run one step of `run` or maintain what it stored, and exist so the
    harness can be iterated on quickly. Nobody using this tool reaches them, so
    naming them in the help only invited it. Each is still `python -m
    codesift.<name>`, with its own --help.
    """

    def top_help(self):
        return cli.build_parser().format_help()

    def test_only_the_two_are_offered(self):
        listed = self.top_help().split("positional arguments:")[1].split("options:")[0]
        self.assertIn("run", listed)
        self.assertIn("discover", listed)
        for gone in STAGES:
            with self.subTest(command=gone):
                self.assertNotIn(f"    {gone} ", listed)
                with self.assertRaises(SystemExit):
                    cli.build_parser().parse_args([gone])

    def test_every_stage_is_a_module_you_can_run(self):
        import importlib
        for name in STAGES:
            with self.subTest(stage=name):
                mod = importlib.import_module(f"codesift.{name}")
                self.assertTrue(callable(getattr(mod, "main", None)),
                                f"{name} has no main()")

    def test_the_help_does_not_point_at_the_stages(self):
        # Checked on the epilog rather than the whole page: "report" and
        # "screening" occur in the prose describing what run and discover do.
        self.assertIsNone(cli.build_parser().epilog)


class TestTheSandboxWarningReachesTheHelp(unittest.TestCase):
    """The readme states it, and nobody reads the readme before the first run.

    It belongs on every entry point that runs what a model wrote, which includes
    regrade -- grading a reply means executing it, whether the reply arrives now or
    was recorded last week.
    """

    def help_of(self, command):
        import importlib, io, contextlib
        if command == "codesift":
            return cli.build_parser().format_help()
        p = cli.build_parser()
        [sub] = [a for a in p._actions if isinstance(a, argparse._SubParsersAction)]
        if command in sub.choices:
            return sub.choices[command].format_help()
        mod = importlib.import_module(f"codesift.{command}")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), mock.patch.object(mod, "run"):
            try:
                mod.main(["--help"])
            except SystemExit:
                pass
        return buf.getvalue()

    def test_every_entry_point_that_executes_a_reply_warns(self):
        for command in ("codesift", "run", "screen", "triage", "regrade"):
            with self.subTest(command=command):
                self.assertIn("executed without a sandbox", self.help_of(command))

    def test_the_warning_is_wrapped_for_a_terminal(self):
        # The parsers that print it use the raw formatter, which leaves a
        # paragraph on a single line.
        for line in EXECUTION_WARNING.splitlines():
            self.assertLessEqual(len(line), 80, line)


class TestOptionsMatchWhatAStageReads(unittest.TestCase):
    """An option accepted and ignored is worse than one that is absent.

    `discover` inherited the measuring options -- a model list, a context window,
    a per-request timeout -- and read none of them: it reads a web page rather than
    driving a model. The help promised they did something.
    """

    def options(self, command):
        import importlib
        if command in ("run", "discover"):
            p = cli.build_parser()
            [sub] = [a for a in p._actions if isinstance(a, argparse._SubParsersAction)]
            parser = sub.choices[command]
        else:
            mod = importlib.import_module(f"codesift.{command}")
            with mock.patch.object(mod, "run"):
                try:
                    mod.main(["--help"])
                except SystemExit:
                    pass
            parser = self._parser_of(mod, command)
        return {s for a in parser._actions for s in a.option_strings if s.startswith("--")}

    def _parser_of(self, mod, command):
        import io, contextlib
        from codesift import args as A
        holder = {}
        real = A.stage
        with mock.patch.object(A, "stage",
                               lambda *a, **k: holder.setdefault("p", real(*a, **k))):
            with contextlib.redirect_stdout(io.StringIO()), \
                 mock.patch.object(mod, "run"):
                try:
                    mod.main(["--help"])
                except SystemExit:
                    pass
        return holder["p"]

    def test_discover_does_not_take_the_measuring_options(self):
        opts = self.options("discover")
        for dead in ("--models", "--models-file", "--ctx", "--timeout"):
            with self.subTest(option=dead):
                self.assertNotIn(dead, opts)

    def test_discover_keeps_only_the_host(self):
        # It reads the server to see what is already installed, and stores nothing,
        # so it has no results directory either.
        opts = self.options("discover")
        self.assertIn("--host", opts)
        self.assertNotIn("--results-dir", opts)
        self.assertNotIn("--refresh", opts)

    def test_each_stage_takes_only_what_it_reads(self):
        # regrade re-grades every stored record and talks to nothing; report also
        # names the host it measured against.
        self.assertEqual(self.options("regrade"), {"--help", "--results-dir", "--apply"})
        self.assertNotIn("--ctx", self.options("report"))
        self.assertIn("--host", self.options("report"))

    def test_a_measuring_stage_still_takes_them_all(self):
        opts = self.options("screen")
        self.assertLessEqual({"--host", "--results-dir", "--models", "--models-file",
                              "--ctx", "--timeout"}, opts)


class TestTheParameterLimitIsStatedNotGuessed(unittest.TestCase):
    """The default is a stated figure, never one read off the local machine.

    It was derived from whatever GPU this process could see: about three times its
    memory. That is the wrong machine when the server is elsewhere, absent on Apple
    Silicon and AMD, and follows a tunnel to the wrong card -- so the default moved
    with the vendor and with where Ollama happened to be running.
    """

    def test_the_default_admits_a_mixture_past_the_memory_it_fits_in(self):
        # A mixture of experts runs well past its card: on 12GB the field cleared
        # the floor up to 2.2 times the card, at 47 to 78 tok/s. Scaled to 24GB that
        # is about 70B. A limit set from what fits resident would have excluded the
        # two highest-ranked models in the measured field, both 35B mixtures.
        args = cli.build_parser().parse_args(["discover"])
        self.assertEqual(args.max_params, 70.0)
        self.assertGreater(args.max_params, 35.0)

    def test_a_stated_limit_is_used(self):
        args = cli.build_parser().parse_args(["discover", "--max-params", "8"])
        self.assertEqual(args.max_params, 8.0)

    def test_the_limit_never_depends_on_the_local_machine(self):
        from codesift import config
        self.assertFalse(hasattr(config, "params_ceiling"))


class TestTheWindowIsTheOnlySetting(unittest.TestCase):
    """One knob. The user says how much context; the software sizes the rest.

    A depth settable on its own could be larger than the window it is read into,
    which reports every model as truncated rather than refusing.
    """

    def stages(self):
        return (mock.patch.object(cli.probe, "run"),
                mock.patch.object(cli.triage, "run"))

    def depth_for(self, *argv):
        from codesift import probe
        with mock.patch.object(probe, "run") as run:
            probe.main(list(argv))
        return run.call_args[1]["depth"]

    def test_the_default_window_probes_at_three_quarters_of_itself(self):
        self.assertEqual(self.depth_for(), working_depth(DEFAULT_CTX))

    def test_a_32k_window_probes_at_24k(self):
        self.assertEqual(self.depth_for("--ctx", "32768"), 24576)

    def test_the_prompt_fits_whatever_window_is_asked_for(self):
        for ctx in (8192, 16384, 32768, 65536, 131072):
            with self.subTest(ctx=ctx):
                self.assertLess(self.depth_for("--ctx", str(ctx)), ctx)

    def test_depth_cannot_be_set_apart_from_the_window(self):
        from codesift import probe
        with self.assertRaises(SystemExit):
            probe.main(["--depth", "48000"])

    def test_triage_measures_at_the_same_depth_as_the_probe(self):
        # They apply the same rule to the same measurement; disagreeing on depth
        # would let the gate reject a model the report then calls intact.
        from codesift import triage
        with mock.patch.object(triage, "run") as run:
            triage.main(["--ctx", "32768"])
        self.assertEqual(run.call_args[1]["depth"],
                         self.depth_for("--ctx", "32768"))


class TestTriageRunsFirst(unittest.TestCase):
    """The cascade is worthless if it runs after the stages it exists to skip."""

    def test_the_full_run_triages_before_it_screens(self):
        order = []
        with mock.patch.object(cli.triage, "run", side_effect=lambda *a, **k: order.append("triage")), \
             mock.patch.object(cli.screen, "run", side_effect=lambda *a, **k: order.append("screen")), \
             mock.patch.object(cli.probe, "run", side_effect=lambda *a, **k: order.append("probe")), \
             mock.patch.object(cli.report, "run", return_value=Path("r.html")):
            cli.main(["run", "--models", "a:1"])
        self.assertEqual(order[0], "triage")
        self.assertLess(order.index("triage"), order.index("screen"))

    def test_a_model_triage_rejected_is_not_screened_even_if_named(self):
        # Naming a model overrides the discard list by design, so the pipeline has
        # to read the triage result itself or triage would do nothing here.
        import tempfile, json as _json
        from pathlib import Path as _Path
        _Path(self.enterContext(tempfile.TemporaryDirectory())) \
            if hasattr(self, "enterContext") else None
        with tempfile.TemporaryDirectory() as d:
            (_Path(d) / "triage.jsonl").write_text(
                _json.dumps({"model": "a:1", "passed": False, "gate": "speed"}) + "\n",
                encoding="utf-8")
            with mock.patch.object(cli.triage, "run"), \
                 mock.patch.object(cli.screen, "run") as screen, \
                 mock.patch.object(cli.probe, "run"), \
                     mock.patch.object(cli.report, "run", return_value=Path("r.html")):
                cli.main(["run", "--models", "a:1", "--results-dir", d])
            screen.assert_not_called()

    def test_triage_needs_no_apply_to_govern_the_run(self):
        # --apply existed only to write the discard list. The pipeline reads the
        # triage result directly, so a rejection governs the stages below whether
        # or not anything was written for a later sweep.
        with mock.patch.object(cli.triage, "run") as triage_run, \
             mock.patch.object(cli.screen, "run"), \
             mock.patch.object(cli.probe, "run"), \
             mock.patch.object(cli.report, "run", return_value=Path("r.html")):
            cli.main(["run", "--models", "a:1"])
        self.assertNotIn("apply", triage_run.call_args[1])
