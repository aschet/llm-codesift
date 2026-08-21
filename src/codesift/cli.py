"""Command line interface."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import (__version__, agent, library, opencode, prefixcache, probe,
               prune, regrade, report, screen, triage)
from .config import DEFAULT_CTX, DEFAULT_HOST, Config, read_model_file


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help="Ollama server URL (default: %(default)s, or $OLLAMA_HOST)")
    parser.add_argument("--models", nargs="+", metavar="MODEL",
                        help="models to evaluate (default: every model on the server)")
    parser.add_argument("--models-file", metavar="PATH",
                        help="file listing one model per line; # comments allowed")
    parser.add_argument("--results-dir", default="results", metavar="DIR",
                        help="where measurement records are stored (default: %(default)s)")
    parser.add_argument("--ctx", type=int, default=DEFAULT_CTX,
                        help="context window for every request (default: %(default)s)")
    parser.add_argument("--timeout", type=int, default=2400, metavar="SECONDS",
                        help="per-request timeout (default: %(default)s)")


def _config(args: argparse.Namespace) -> Config:
    models = list(args.models or [])
    if getattr(args, "models_file", None):
        models += read_model_file(args.models_file)
    return Config(host=args.host, results_dir=Path(args.results_dir),
                  ctx=args.ctx, models=models, timeout=args.timeout)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codesift",
        description="Screening harness for locally hosted coding models served by Ollama.")
    parser.add_argument("--version", action="version", version=f"codesift {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("screen", help="prompt-level tasks: code, edit, format, tools, tracing")
    _add_common(p)
    p.add_argument("--taskset", default="basic", choices=["basic", "hard"])
    p.add_argument("--num-predict", type=int, default=screen.NUM_PREDICT, metavar="TOKENS",
                   help="output budget per reply, covering reasoning as well as the "
                        "answer (default: %(default)s)")
    p.add_argument("--runs", type=int, default=1, help="repetitions, for run-to-run spread")
    p.add_argument("--only", nargs="+", metavar="TASK_ID", help="restrict to these tasks")
    p.add_argument("--redo", action="store_true", help="ignore stored results and re-measure")

    p = sub.add_parser("probe", help="latency and context behaviour at working depth")
    _add_common(p)
    p.add_argument("--depth", type=int, default=48000, help="prompt size in tokens")
    p.add_argument("--redo", action="store_true")

    p = sub.add_parser("cache", help="prefix cache reuse across turns")
    _add_common(p)
    p.add_argument("--redo", action="store_true")

    p = sub.add_parser("agent", help="task completion under the opencode harness")
    _add_common(p)
    p.add_argument("--only", nargs="+", metavar="TASK_ID")
    p.add_argument("--keep", action="store_true", help="retain the scratch repositories")
    p.add_argument("--redo", action="store_true")
    p.add_argument("--task-timeout", type=int, default=1200, metavar="SECONDS")
    p.add_argument("--select", default="suitable,limited", metavar="VERDICTS",
                   help="which screen verdicts to run, comma separated: suitable, "
                        "limited, unsuitable, or all (default: %(default)s). The "
                        "application task allows an hour per model, so models the "
                        "screen ruled out are skipped unless asked for")

    p = sub.add_parser("triage", help="reject unusable models as cheaply as possible")
    _add_common(p)
    p.add_argument("--depth", type=int, default=48000)
    p.add_argument("--redo", action="store_true", help="re-triage models already judged")
    p.add_argument("--apply", action="store_true",
                   help="add the rejected models to the discard list")

    p = sub.add_parser("regrade", help="re-apply the grader to replies already recorded")
    _add_common(p)
    p.add_argument("--apply", action="store_true", help="rewrite the records; without it, list and stop")

    p = sub.add_parser("prune", help="discard ruled-out models and their records")
    _add_common(p)
    p.add_argument("--select", default="unsuitable", metavar="VERDICTS",
                   help="which verdicts to discard, comma separated (default: %(default)s)")
    p.add_argument("--apply", action="store_true", help="act; without it, list and stop")
    p.add_argument("--keep-records", action="store_true",
                   help="only stop future sweeps measuring them; leave the records")
    p.add_argument("--forget", action="store_true",
                   help="restore every discarded model to future sweeps")

    p = sub.add_parser("report", help="render the HTML report")
    _add_common(p)
    p.add_argument("-o", "--output", default="report.html", metavar="PATH")

    p = sub.add_parser("run", help="every stage in order, then the report")
    _add_common(p)
    p.add_argument("--runs", type=int, default=3, help="screen repetitions per taskset")
    p.add_argument("--depth", type=int, default=48000)
    p.add_argument("--skip-agent", action="store_true",
                   help="omit the agent stage, which requires opencode")
    p.add_argument("-o", "--output", default="report.html", metavar="PATH")

    p = sub.add_parser("discover", help="models in the Ollama library worth screening")
    _add_common(p)
    p.add_argument("--since", default="18m", metavar="WHEN",
                   help="ignore models not updated since this date, year, or "
                        "number of months back (default: %(default)s)")
    p.add_argument("--max-size-gb", type=float, default=32.0, metavar="GB",
                   help="largest download to consider (default: %(default)s)")
    p.add_argument("--min-size-gb", type=float, default=4.0, metavar="GB",
                   help="smallest download to consider, which excludes models too "
                        "small to write code (default: %(default)s)")
    p.add_argument("--min-context", type=int, metavar="TOKENS",
                   help="smallest advertised context window (default: --ctx)")
    p.add_argument("--coding", action="store_true", dest="require_coding",
                   help="keep only models whose listing names programming work")
    p.add_argument("--include-installed", action="store_true",
                   help="also list models already on the server")
    p.add_argument("--limit", type=int, default=20, help="suggestions to print (default: %(default)s)")
    p.add_argument("--sort", default="date", choices=list(library.SORTS),
                   help="ordering, by a fact the listing states (default: %(default)s)")
    p.add_argument("--match", metavar="REGEX",
                   help="keep only models whose name or description matches")
    p.add_argument("--exclude", metavar="REGEX",
                   help="drop models whose name or description matches")
    p.add_argument("--json", action="store_true", dest="as_json",
                   help="print the suggestions as JSON")
    p.add_argument("--write-models", metavar="PATH",
                   help="write the suggestions as a model list for --models-file")
    p.add_argument("--refresh", action="store_true", help="ignore cached pages")

    p = sub.add_parser("sync-opencode", help="declare Ollama models in opencode's config")
    _add_common(p)
    p.add_argument("--config", metavar="PATH", help="opencode config file to update")
    p.add_argument("--write", action="store_true", help="apply changes, keeping a backup")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _config(args)

    if args.command == "screen":
        screen.run(cfg, taskset=args.taskset, runs=args.runs,
                   num_predict=args.num_predict, redo=args.redo, only=args.only)
    elif args.command == "probe":
        probe.run(cfg, depth=args.depth, redo=args.redo)
    elif args.command == "cache":
        prefixcache.run(cfg, redo=args.redo)
    elif args.command == "agent":
        picked = [] if args.select.strip().lower() == "all" else args.select.split(",")
        return agent.run(cfg, timeout=args.task_timeout, redo=args.redo,
                         only=args.only, keep=args.keep, select_verdicts=picked)
    elif args.command == "triage":
        return triage.run(cfg, depth=args.depth, redo=args.redo, apply=args.apply)
    elif args.command == "regrade":
        return regrade.run(cfg, apply=args.apply)
    elif args.command == "prune":
        return prune.run(cfg, verdicts=args.select.split(","), models=cfg.models or None,
                         apply=args.apply, forget=args.forget,
                         keep_records=args.keep_records)
    elif args.command == "report":
        out = report.run(cfg, Path(args.output), cfg.models or None)
        print(f"wrote {out}")
    elif args.command == "discover":
        return library.run(cfg, since=args.since, max_size_gb=args.max_size_gb,
                           min_size_gb=args.min_size_gb, min_context=args.min_context,
                           match=args.match, exclude=args.exclude,
                           require_coding=args.require_coding,
                           include_installed=args.include_installed,
                           limit=args.limit, sort=args.sort, as_json=args.as_json,
                           write_models=Path(args.write_models) if args.write_models else None,
                           refresh=args.refresh)
    elif args.command == "sync-opencode":
        return opencode.run(cfg, Path(args.config) if args.config else None, args.write)
    elif args.command == "run":
        # Cheapest decisive question first: a model rejected here never reaches the
        # stages below, which is where the hours are. The discard list alone is not
        # enough to enforce that here, because naming a model explicitly overrides
        # it by design -- so the pipeline reads the triage result directly.
        triage.run(cfg, depth=args.depth, apply=True)
        judged = triage.read_ledger(cfg)
        cfg.models = [m for m in cfg.resolve_models()
                      if judged.get(m, {}).get("passed", True)]
        if not cfg.models:
            print("every model was rejected in triage; nothing further to run",
                  file=sys.stderr)
            return 0
        for taskset in ("basic", "hard"):
            screen.run(cfg, taskset=taskset, runs=args.runs)
        probe.run(cfg, depth=args.depth)
        prefixcache.run(cfg)
        if not args.skip_agent:
            if agent.run(cfg, select_verdicts=["suitable", "limited"]) != 0:
                print("agent stage skipped; continuing to the report", file=sys.stderr)
        out = report.run(cfg, Path(args.output), cfg.models or None)
        print(f"wrote {out}")
    return 0
