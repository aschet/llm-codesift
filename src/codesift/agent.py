"""Task completion under a real agent harness (opencode).

Each task seeds an isolated repository and lets the harness drive the model
through it. Grading inspects the resulting files and runs the repository's own
checks; the model's account of what it did is disregarded.

Small tasks are graded pass or fail. A task wide enough that a partial result is
still informative sets `graded`, and its check script reports one line per
requirement, which is scored as a fraction instead.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import gpulock
from .config import Config
from .ollama import Ollama
from .tasks import AGENT_TASKS

CHECKS_DIR = Path(__file__).parent / "tasks" / "checks"

# How much of the model's own output is kept with each result. A session that ends
# having written nothing is only diagnosable from what the model said instead.
SAID_KEPT = 8000

# The harness's raw stream, written beside a retained repository.
RAW_SESSION = 2_000_000


def preflight(models: list[str]) -> str | None:
    """Return an error message if the harness cannot resolve these models."""
    if not shutil.which("opencode"):
        return ("opencode is not on PATH. The agent stage needs it to drive the model; "
                "the other stages have no such dependency.")
    try:
        listing = subprocess.run(["opencode", "models"], capture_output=True,
                                 text=True, timeout=120).stdout
    except Exception:
        return None      # cannot verify; proceed and let the run report failures
    if "ollama/" not in listing:
        return ("opencode has no 'ollama' provider configured, so 'ollama/<model>' "
                "cannot be resolved and every task would fail for that reason alone.\n"
                "Run: codesift sync-opencode --write")
    missing = [m for m in models if f"ollama/{m}" not in listing]
    if missing:
        return (f"opencode does not list: {', '.join(missing)}\n"
                "Run: codesift sync-opencode --write")
    return denied_tools()


def denied_tools(config_path: Path | None = None) -> str | None:
    """Report tools the configuration refuses outright.

    The stage runs opencode with --auto, which approves anything not explicitly
    denied, so a task fails only if the user's own configuration forbids the tool
    it needs. That failure looks exactly like an incapable model, so it is caught
    here instead.
    """
    from .opencode import DEFAULT_CONFIG, strip_comments
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    try:
        data = json.loads(strip_comments(path.read_text(encoding="utf-8")))
    except Exception:
        return None
    perms = data.get("permission")
    if not isinstance(perms, dict):
        return None
    denied = sorted(k for k, v in perms.items()
                    if (v if isinstance(v, str) else (v or {}).get("*")) == "deny")
    if not denied:
        return None
    return (f"{path} denies: {', '.join(denied)}. The agent stage cannot edit or run "
            "anything it is refused, and every task would fail for that reason "
            "rather than for anything about the model. Remove those entries or set "
            "them to 'allow'.")


def _digest(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def seed(task: dict, workdir: Path, at: Path | None = None) -> Path:
    """Create the task's starting repository and return its absolute path.

    Absolute because everything downstream runs with the repository as its working
    directory: a relative path handed to one of those resolves against the
    repository rather than against here, which produced a doubled path and a
    grader that failed to find its own check script.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    if at is not None:
        at = Path(at).resolve()
        shutil.rmtree(at, ignore_errors=True)
        at.mkdir(parents=True)
        repo = at
    else:
        repo = Path(tempfile.mkdtemp(prefix=f"{task['id']}_", dir=workdir))
    repo = repo.resolve()
    for rel, content in task["files"].items():
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return repo


def parse_checks(output: str) -> list[dict]:
    """Read `CHECK <name> PASS|FAIL <detail>` lines from a graded check script."""
    checks = []
    for line in output.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) >= 3 and parts[0] == "CHECK" and parts[2] in ("PASS", "FAIL"):
            checks.append(dict(name=parts[1], passed=parts[2] == "PASS",
                               detail=parts[3] if len(parts) > 3 else ""))
    return checks


def verify(task: dict, repo: Path) -> tuple[bool, str, list[dict]]:
    """Run the task's check without a shell.

    Shell strings are not portable: heredocs are unavailable on Windows and the
    interpreter is named differently, so checks are argv lists using a {py}
    placeholder, or a script written into the repository.
    """
    source = task.get("verify_src")
    if not source and task.get("verify_src_path"):
        # A tool outside this package supplies an absolute path to its own checker.
        source = Path(task["verify_src_path"]).read_text(encoding="utf-8")
    if not source and task.get("verify_src_file"):
        source = (CHECKS_DIR / task["verify_src_file"]).read_text(encoding="utf-8")
    if source:
        script = (repo / "_verify_check.py").resolve()
        script.write_text(source, encoding="utf-8")
        cmd = [sys.executable, str(script)]
    else:
        cmd = [sys.executable if arg == "{py}" else arg for arg in task["verify"]]
    try:
        proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True,
                              timeout=task.get("verify_timeout", 60))
    except subprocess.TimeoutExpired as exc:
        raw = (exc.stdout or b"") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        partial = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
        return False, "verify timed out", parse_checks(partial)
    output = (proc.stdout or "") + (proc.stderr or "")

    if task.get("graded"):
        # A graded task has no single expected line: the score is how many of its
        # requirements were met, and "passed" keeps its plain meaning of all of them.
        checks = parse_checks(output)
        if not checks:
            lines = output.strip().splitlines()
            return False, (lines[-1][:160] if lines else "the check script reported nothing"), []
        won = sum(1 for c in checks if c["passed"])
        missed = [c["name"] for c in checks if not c["passed"]]
        detail = f"{won}/{len(checks)} checks"
        if missed:
            detail += ": missing " + ", ".join(missed[:5])
            if len(missed) > 5:
                detail += f" and {len(missed) - 5} more"
        return won == len(checks), detail, checks

    if task["expect_stdout"] not in output:
        lines = output.strip().splitlines()
        return False, (lines[-1][:160] if lines else "no output"), []
    return True, "ok", []


def parse_events(stdout: str) -> tuple[list[str], int, list[str], dict, int, str]:
    """One JSON event per line: {type, sessionID, part:{type, ...}}.

    The model's own words are returned alongside the counts. A session that ends
    after two turns having written nothing looks identical whether the model gave
    up, answered in prose instead of calling a tool, or hit something in the
    harness -- and without what it said, the three cannot be told apart. The same
    omission in the screen stage made a grading bug undiagnosable for a day.
    """
    tools, steps, errors, said = [], 0, [], []
    tokens = {"input": 0, "output": 0, "total": 0, "reasoning": 0}
    peak_input = 0
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except Exception:
            continue
        kind = event.get("type") or ""
        part = event.get("part") or {}
        if kind == "error" or part.get("type") == "error":
            err = event.get("error") or {}
            errors.append(err.get("name") or str(err)[:80] or "error")
        if kind == "step_start":
            steps += 1
        if part.get("type") in ("tool", "tool-invocation") or kind == "tool":
            name = part.get("tool") or part.get("name") or event.get("tool")
            if name:
                tools.append(name)
        # Part types observed in the stream: text, reasoning, tool, step-start,
        # step-finish. Both text and reasoning are kept, because the failure worth
        # explaining is a turn that produced neither a tool call nor an answer: one
        # model spent 6457 tokens and two minutes reasoning about what it was about
        # to do, then stopped, and only the reasoning says so.
        if part.get("type") in ("text", "reasoning"):
            body = part.get("text") or part.get("reasoning") or ""
            if body:
                said.append(body)
        if kind == "step_finish":
            counts = part.get("tokens") or {}
            for key in tokens:
                tokens[key] += counts.get(key, 0) or 0
            peak_input = max(peak_input, counts.get("input", 0) or 0)
    return tools, steps, errors, tokens, peak_input, "\n\n".join(said)


def _kill_tree(proc) -> None:
    """Take down the harness and everything it started."""
    if proc is None or proc.poll() is not None:
        return
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=15)
    except Exception:
        try:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            pass


def _slug(model: str) -> str:
    return "".join(c if c.isalnum() or c in "-." else "_" for c in model)


def run_task(model: str, task: dict, workdir: Path, timeout: int,
             retain_dir: Path | None = None) -> dict:
    # A task that produces a whole application is worth keeping and reading, so it
    # is built at a path named after the model rather than in a temporary one.
    at = None
    if task.get("retain") and retain_dir is not None:
        at = retain_dir / f"{_slug(model)}__{task['id']}"
    repo = seed(task, workdir, at=at)
    timeout = max(timeout, task.get("min_timeout", 0))
    protected = {f: _digest(repo / f) for f in task.get("immutable", [])}
    prompt = task["prompt"].replace("{py}", Path(sys.executable).stem)
    cmd = ["opencode", "run", "--format", "json", "--auto",
           "-m", f"ollama/{model}", "--dir", str(repo),
           "--title", f"codesift-{task['id']}", prompt]

    # A model that cannot install its package into the machine cannot contaminate
    # the next model graded on it. PEP 668 already refuses this on most Linux
    # distributions, but that refusal names --break-system-packages in its own error
    # text, and a model driven with --auto simply takes the suggestion; one did.
    # Requiring a virtualenv is not overridable by that flag, and a virtualenv the
    # model makes for itself lives inside the repository, where it belongs.
    env = dict(os.environ, PIP_REQUIRE_VIRTUALENV="1")

    # The model is given a shell, and it uses it: it starts servers, runs test
    # suites, leaves things listening. Killing opencode alone would leave those
    # behind to hold ports and memory for the rest of the run, so the harness takes
    # its own process group and takes the group down with it.
    group = {"start_new_session": True} if hasattr(os, "setsid") else {}
    started, timed_out = time.time(), False
    proc = None
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, cwd=repo, env=env, **group)
        stdout, stderr = proc.communicate(timeout=timeout)
        code = proc.returncode
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        stdout, stderr = proc.communicate()
        stderr, code, timed_out = (stderr or "") + "\ntimed out", -1, True
    except FileNotFoundError:
        stdout, stderr, code = "", "opencode not found on PATH", 127
    finally:
        _kill_tree(proc)

    tools, turns, errors, tokens, peak_input, said = parse_events(stdout)
    passed, detail, checks = verify(task, repo)
    tampered = [f for f, h in protected.items() if _digest(repo / f) != h]
    if tampered:
        passed, detail = False, f"modified protected file(s): {tampered}"
        checks = []

    if at is not None:
        # What is kept is meant to be read. The check script, its scratch database
        # and the caches left by running the tests are not part of what the model
        # built, and a 20KB grader dropped beside a small application is noise.
        for litter in ("_verify_check.py", "_grading_store", "_grading_store.json",
                       ".pytest_cache"):
            path = repo / litter
            shutil.rmtree(path, ignore_errors=True) if path.is_dir() else (
                path.unlink() if path.exists() else None)
        for cache in repo.rglob("__pycache__"):
            shutil.rmtree(cache, ignore_errors=True)

    # The harness's own output, kept verbatim beside the work. Parsing it into
    # counts throws away the one thing that explains a session which ended having
    # written nothing, and a parser written against a guessed event shape captures
    # nothing at all without saying so -- which is exactly what happened.
    if at is not None and stdout:
        (repo / "opencode.jsonl").write_text(stdout[-RAW_SESSION:], encoding="utf-8")

    shot = repo / "ui.png"
    return dict(
        model=model, task=task["id"], passed=passed, detail=detail,
        checks=checks, score=(round(100 * sum(c["passed"] for c in checks) / len(checks), 1)
                              if checks else None),
        screenshot=str(shot) if shot.exists() else None,
        retained=bool(at),
        wall_s=round(time.time() - started, 1), timed_out=timed_out, returncode=code,
        tool_calls=len(tools), tools=tools[:40], turns=turns, errors=errors[:10],
        tokens=tokens, peak_input_tokens=peak_input,
        repo=str(repo), stderr=(stderr or "")[-600:], ts=time.time(),
        said=said[-SAID_KEPT:],
    )


def select(cfg: Config, models: list[str], verdicts: list[str]) -> list[str]:
    """The models worth an agent run, taken from the screen's own verdicts.

    The application task allows an hour per model, so running it on a model the
    screen has already ruled out spends a working day learning nothing. An empty
    verdict list means the caller chose the models itself.
    """
    if not verdicts:
        return list(models)
    from .report import analyse
    return analyse(cfg, models).by_verdict(verdicts)


def run(cfg: Config, timeout: int = 1200, redo: bool = False,
        only: list[str] | None = None, keep: bool = False,
        select_verdicts: list[str] | None = None) -> int:
    models = cfg.resolve_models()
    if select_verdicts:
        wanted = select(cfg, models, select_verdicts)
        dropped = [m for m in models if m not in wanted]
        if dropped:
            print(f"skipping {len(dropped)} model(s) the screen ruled out: "
                  + ", ".join(dropped), flush=True)
        if not wanted:
            print("no model matches " + ",".join(select_verdicts)
                  + ". Screen the models first, or pass --select all to run anyway.",
                  file=sys.stderr)
            return 2
        models = wanted
    problem = preflight(models)
    if problem:
        print(problem, file=sys.stderr)
        return 2

    gpulock.acquire("agent", endpoint=cfg.host)
    client = Ollama(cfg.host, cfg.timeout)
    path = cfg.path("agentic.jsonl")
    workdir = cfg.results_dir / "agent_work"
    retain_dir = cfg.results_dir / "agent_apps"

    # (model, task) -> completed at least once. The outcome is kept, not just the
    # fact of measurement, because the second tier is gated on it.
    done = {}
    if path.exists() and not redo:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            key = (rec["model"], rec["task"])
            done[key] = done.get(key, False) or bool(rec.get("passed"))

    chosen = [t for t in AGENT_TASKS if not only or t["id"] in set(only)]
    pending = [(m, t) for m in models for t in chosen if (m, t["id"]) not in done]
    if pending:
        budget = sum(max(timeout, t.get("min_timeout", 0)) for _, t in pending)
        print(f"{len(pending)} task(s) to run; at worst {budget / 3600:.1f}h if every "
              f"one runs to its limit", flush=True)

    def attempt(model, task):
        """Run one task and record it. Returns whether the model completed it."""
        if (model, task["id"]) in done:
            print(f"{model} / {task['id']}: already measured, skipping", flush=True)
            return done[(model, task["id"])]
        print(f"{model} / {task['id']}: running", flush=True)
        rec = run_task(model, task, workdir, timeout, retain_dir=retain_dir)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"  {'PASS' if rec['passed'] else 'FAIL'} {rec['wall_s']}s  "
              f"tools={rec['tool_calls']} turns={rec['turns']}"
              f"{' TIMEOUT' if rec['timed_out'] else ''}  {rec['detail'][:70]}",
              flush=True)
        for c in rec.get("checks") or []:
            print(f"    {'ok  ' if c['passed'] else 'FAIL'} {c['name']:16} "
                  f"{c['detail'][:60]}", flush=True)
        if rec.get("retained"):
            print(f"    kept at {rec['repo']}", flush=True)
            if rec.get("screenshot"):
                print(f"    screenshot {rec['screenshot']}", flush=True)
        elif not keep:
            shutil.rmtree(rec["repo"], ignore_errors=True)
        return bool(rec["passed"])

    for model in models:
        for task in chosen:
            attempt(model, task)
        client.unload(model)
    return 0
