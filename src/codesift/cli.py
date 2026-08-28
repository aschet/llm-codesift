# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""Command line interface."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from . import __version__, library, probe, report, screen, triage
from . import args, progress
from .config import DEFAULT_HOST, working_depth


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codesift",
        description="Screening harness for locally hosted coding models served by Ollama."
                    f"\n\n{args.EXECUTION_WARNING}",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"codesift {__version__}")
    # The two commands there are. The stage modules -- screen, probe, triage,
    # report and regrade -- run one step of `run` or maintain what it stored, and
    # exist so the harness can be iterated on quickly; the help does not offer them,
    # but `python -m codesift.<name>` reaches them with its own --help.
    sub = parser.add_subparsers(dest="command", required=True,
                                metavar="{run,discover}")

    p = sub.add_parser("run", help="every stage in order, then the report",
                       description="Every stage in order, then the report."
                                   f"\n\n{args.EXECUTION_WARNING}",
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    args.add_common(p)
    args.add_output(p)

    # Reads the library listing and the server's model list. It stores nothing, so
    # it has no results directory.
    p = sub.add_parser("discover", help="models in the Ollama library worth screening")
    p.add_argument("--host", default=DEFAULT_HOST,
                   help="Ollama server URL (default: %(default)s, or $OLLAMA_HOST)")
    p.add_argument("--since", default="18m", metavar="WHEN",
                   help="ignore models not updated since this date, year, or "
                        "number of months back (default: %(default)s)")
    # A rough ceiling, to keep out what certainly will not run well on a consumer
    # card. It is not a prediction: triage measures the generation rate for real in
    # about ten seconds, and this only spares the reader a listing of 405B models.
    #
    # 70 rather than something near what fits resident, because a mixture of experts
    # runs well past its card -- only the active parameters move per token. Nothing
    # here detects the card, and the server does not report one, so state your own
    # with --max-params.
    p.add_argument("--max-params", type=float, default=70.0, metavar="B",
                   help="largest parameter count to consider, in billions (default: "
                        "%(default)g, a rough ceiling for a consumer card)")
    p.add_argument("--min-params", type=float, default=4.0, metavar="B",
                   help="smallest parameter count to consider, which excludes models "
                        "too small to write code (default: %(default)s)")
    p.add_argument("--coding", action="store_true", dest="require_coding",
                   help="keep only models whose listing names programming work")
    p.add_argument("--include-installed", action="store_true",
                   help="also list models already on the server")
    p.add_argument("--sort", default="date", choices=list(library.SORTS),
                   help="ordering, by a fact the listing states (default: %(default)s)")
    p.add_argument("--match", metavar="REGEX",
                   help="keep only models whose name or description matches")
    p.add_argument("--exclude", metavar="REGEX",
                   help="drop models whose name or description matches")
    p.add_argument("--json", action="store_true", dest="as_json",
                   help="print the suggestions as JSON")
    p.add_argument("--write-models", metavar="PATH",
                   help="write the suggestions as ollama pull commands; the same file "
                        "is accepted by --models-file")

    return parser


def main(argv: list[str] | None = None) -> int:
    opts = build_parser().parse_args(argv)
    cfg = args.config_from(opts)

    if opts.command == "discover":
        return library.run(cfg, since=opts.since, max_params_b=opts.max_params,
                           min_params_b=opts.min_params,
                           match=opts.match, exclude=opts.exclude,
                           require_coding=opts.require_coding,
                           include_installed=opts.include_installed,
                           sort=opts.sort, as_json=opts.as_json,
                           write_models=Path(opts.write_models) if opts.write_models else None)

    # Cheapest decisive question first: a model rejected here never reaches the
    # stages below, which is where the hours are.
    with progress.document():
        with progress.group("triage"):
            triage.run(cfg, depth=working_depth(cfg.ctx))
        judged = triage.read_ledger(cfg)
        # Everything the run was asked to cover, kept for the report: silence cannot
        # distinguish a model that failed from one that was never run, and the first
        # is a finding.
        field = cfg.resolve_models()
        cfg.models = [m for m in field if judged.get(m, {}).get("passed", True)]
        if not cfg.models:
            # Not a bail: every model was measured and every one answered. There
            # is simply nothing left that the stages below could measure.
            progress.note("every model was rejected in triage; nothing further to run")
            return 0
        # Triage is a pass over the whole field because it is the cheapest thing
        # here and answers the question a reader has first: which of these cannot
        # be used at all. The stages that cost hours then run one model at a time,
        # so the report is rewritten after each and a sweep that dies at the
        # seventh model leaves six complete ones and a report of them.
        for model in cfg.models:
            with progress.group(model):
                one = replace(cfg, models=[model])
                screen.run(one)
                probe.run(one, depth=working_depth(cfg.ctx))
            page = report.run(cfg, Path(opts.output), field)
        progress.note(f"wrote {page}")
    return 0
