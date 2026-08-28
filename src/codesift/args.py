# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""Command line options shared by every entry point.

Each stage is a command in its own right -- `python -m codesift.probe` -- and the
pipeline calls the same stages in order. Both need the same options to mean the
same thing, so they are defined once here rather than in either.

They live outside `cli` because a stage cannot import the module that imports it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import (DEFAULT_CTX, DEFAULT_HOST, DEFAULT_OUTPUT, Config, data_dir,
                     normalise_host, read_model_file)

# Printed by every entry point that runs model-written code. A reader who never
# opens the readme still reaches --help, and this is the one fact about the tool
# that costs something to learn late.
# Wrapped here rather than by argparse: the parsers that print it use the raw
# formatter, which leaves a paragraph on one line.
EXECUTION_WARNING = (
    "WARNING: model-written code is executed without a sandbox. It runs with the\n"
    "privileges of the invoking user and with unrestricted access to the filesystem\n"
    "and the network, and can damage the host system. Running the harness inside a\n"
    "virtual machine is strongly recommended.")


# A subcommand takes what it reads and nothing else: an option that is accepted and
# ignored is worse than one that is absent, because the help promises it does
# something.
def add_results(parser: argparse.ArgumentParser) -> None:
    # No default of its own: it follows the report unless it is given, and only
    # an argument the user actually typed may override that.
    parser.add_argument("--results-dir", metavar="DIR",
                        help=f"where measurement records are stored (default: "
                             f"{data_dir(DEFAULT_OUTPUT)}, beside the report)")


def add_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, metavar="PATH",
                        help="where to write the report; the records are stored "
                             "beside it (default: %(default)s)")


def add_server(parser: argparse.ArgumentParser) -> None:
    add_results(parser)
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help="Ollama server URL (default: %(default)s, or $OLLAMA_HOST)")


def add_models(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--models", nargs="+", metavar="MODEL",
                        help="models to evaluate (default: every model on the server)")
    parser.add_argument("--models-file", metavar="PATH",
                        help="file listing one model per line; # comments allowed")


def add_common(parser: argparse.ArgumentParser) -> None:
    """Everything a stage that drives a model needs."""
    add_server(parser)
    add_models(parser)
    parser.add_argument("--ctx", type=int, default=DEFAULT_CTX,
                        help="context window for every request (default: %(default)s). The "
                             "depth the probe and the context gate measure at follows from it")
    parser.add_argument("--timeout", type=int, default=2400, metavar="SECONDS",
                        help="per-request timeout (default: %(default)s)")


def config_from(args: argparse.Namespace) -> Config:
    """The measuring options are absent on subcommands that do not measure."""
    models = list(getattr(args, "models", None) or [])
    if getattr(args, "models_file", None):
        models += read_model_file(args.models_file)
    cfg = Config(models=models)
    # The records belong to the report they were built for, so an output path
    # names the store as well, unless --results-dir says otherwise.
    if getattr(args, "results_dir", None):
        cfg.results_dir = Path(args.results_dir)
    elif getattr(args, "output", None):
        cfg.results_dir = data_dir(args.output)
    if getattr(args, "host", None):
        cfg.host = normalise_host(args.host)
    for name in ("ctx", "timeout"):
        if getattr(args, name, None) is not None:
            setattr(cfg, name, getattr(args, name))
    return cfg



def stage(name: str, description: str, *, models: bool = True,
          measuring: bool = True, executes: bool = False) -> argparse.ArgumentParser:
    """A parser for one stage, run on its own.

    `codesift` itself lists only the two commands a user runs. A stage is reached
    as `python -m codesift.<name>`, which puts its own flags beside the code that
    reads them while the shared ones stay defined once.

    `executes` marks the stages that run what a model wrote -- grading a reply
    means executing it, whether the reply arrives now or was recorded last week.
    The probe and the report measure and render without running anything.
    """
    if executes:
        description = f"{description}\n\n{EXECUTION_WARNING}"
    parser = argparse.ArgumentParser(prog=f"codesift.{name}", description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    if measuring:
        add_common(parser)
    else:
        add_results(parser)
        if models:
            add_models(parser)
    return parser
