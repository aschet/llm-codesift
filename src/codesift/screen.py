# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""Prompt-level screening: generation, editing, output format, tool calls, tracing.

Results are appended per task, so an interrupted run resumes without regenerating
anything already measured.
"""
from __future__ import annotations

import ast
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import gpulock, ledger, progress
from . import args
from .config import Config
from .ollama import Ollama
from .tasks import TASKS

EXEC_TIMEOUT = 15

# A candidate's own stdout is otherwise encoded with whatever the platform's
# locale defaults to -- cp1252 on Windows, and conceivably ASCII in a bare POSIX
# locale -- and a model demonstrating its answer with so much as a checkmark then
# crashes the interpreter before any check runs. Read back with the same fixed
# encoding, since a parent decoding what the child wrote as UTF-8 through its own
# locale can itself fail: capture_output runs that decode on a background thread,
# where the error surfaces as noise on the real stderr rather than a result the
# caller can catch, and the task is left looking like it produced no output.
_CHILD_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}

# The budget covers a model's reasoning as well as its answer, and exists to stop a
# runaway rather than to constrain a legitimate reply. It is roughly 2.5 times the
# longest passing reply measured: too low silently converts a verbose model into an
# incompetent one, while too high only costs time, under two minutes a task.
#
# The figure to watch is not this one but how often it is reached. Past a few percent
# the budget has started deciding scores, and the report names every model it bound.
NUM_PREDICT = 6144

# How much of each reply is kept, so that a changed grader can be applied to it
# without asking the model again.
RAW_KEPT = 12000


def _declarations_only(code: str) -> str:
    """Drop statements that exist to demonstrate the code rather than to be it.

    A bare expression at the top level -- `print(divide(10, 0))`, `flaky_call()` --
    is a demonstration. It is not part of the answer, and running it alongside the
    tests grades the model on its illustration; one that fails at random would also
    make the result differ between runs of identical input. A top-level `for` or
    `while` loop is the same habit spelled out longer -- a model showing its work
    by calling the function over some sample inputs and printing the result -- and
    is dropped for the same reason: it can fail or print output unrelated to
    whether the answer itself is correct.

    Everything that declares something is kept, in order, so imports introduced
    beside a first attempt remain available to the version that supersedes it, and
    a redefinition later in the reply wins exactly as Python would have it. Code
    that will not parse is returned untouched, since the grader is about to run it
    and report the syntax error.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    keep = []
    for node in tree.body:
        if isinstance(node, ast.Expr):
            continue                      # a bare call, or a stray docstring
        if isinstance(node, (ast.For, ast.While)):
            continue                      # a self-test loop demonstrating the answer
        if isinstance(node, ast.If) and _is_main_guard(node):
            continue                      # a demonstration behind a guard
        keep.append(node)
    if not keep:
        return code
    return ast.unparse(ast.Module(body=keep, type_ignores=[]))


def _is_main_guard(node: ast.If) -> bool:
    test = node.test
    return (isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name) and test.left.id == "__name__")


def extract_code(text: str, entry: str | None = None) -> str:
    """Pull the answer out of a reply, tolerating fences and surrounding prose.

    A reply usually holds several fenced blocks, and which one is the answer is not
    a question of size or of position: a model reasoning through a bug shows the
    broken version before the fixed one, and one that documents a correct answer
    shows example calls after it.

    So the whole reply is the answer: every fenced block, joined in order, with
    demonstrations removed by `_declarations_only`. Later definitions supersede
    earlier ones the way the interpreter would, imports stay in scope, and nothing
    depends on guessing which block the model considered final.

    `entry` names what the task asked for. It is used only when the joined blocks
    will not parse, where a reply that is part prose and part code can still be
    salvaged by taking the block defining what was asked for.
    """
    if not text:
        return ""
    fences = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if not fences:
        return _declarations_only(text.strip())
    joined = "\n\n".join(f.strip() for f in fences)
    try:
        ast.parse(joined)
    except SyntaxError:
        if entry:
            named = [f for f in fences if _defines(f, entry)]
            if named:
                return _declarations_only(named[-1].strip())
        return max(fences, key=len).strip()
    return _declarations_only(joined)


def _defines(block: str, name: str) -> bool:
    return re.search(r"^\s*(?:async\s+def|def|class)\s+" + re.escape(name) + r"\b",
                     block, re.M) is not None


# Runs the candidate, then the task's checks one statement at a time in a shared
# namespace, so a check that fails costs itself and not the ones after it.
_PER_CHECK = """
import ast, sys
ns = {}
try:
    exec(compile(CODE, "<candidate>", "exec"), ns)
except Exception as exc:
    print("SETUP", type(exc).__name__ + ": " + str(exc)[:120]); raise SystemExit
passed = failed = 0
first = ""
tree = ast.parse(TESTS)
for node in tree.body:
    src = ast.get_source_segment(TESTS, node) or ""
    is_check = "assert" in src
    try:
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<checks>", "exec"), ns)
    except Exception as exc:
        detail = type(exc).__name__ + ((": " + str(exc)[:90]) if str(exc) else "")
        if not is_check:
            print("SETUP", detail); raise SystemExit
        failed += 1
        if not first:
            first = src.strip().splitlines()[0][:70] + " -> " + detail
    else:
        if is_check:
            passed += 1
print("RESULT", passed, failed, first)
"""


@contextlib.contextmanager
def _scratch():
    """A directory to run a candidate in, removed however the attempt ends.

    Not TemporaryDirectory: a process killed at the timeout can still hold its
    working directory for a moment on Windows, and the removal would then raise
    out of the grader instead of the task simply scoring nothing.
    """
    path = tempfile.mkdtemp(prefix="codesift-")
    try:
        yield Path(path)
    finally:
        shutil.rmtree(path, ignore_errors=True)


def run_checks(code: str, tests: str) -> tuple[int, int, str]:
    """Return (checks met, checks total, what the first shortfall was).

    A task's assertions are graded one at a time rather than as a single script.
    Most failing answers are near misses, and scoring those the same as code that
    does not run reports a difference that is not there.

    The statements run in order in one namespace, because a task's checks are not
    independent: some build the object the later ones examine. A setup statement
    that raises ends the attempt, since nothing after it means anything.
    """
    if not code.strip():
        return 0, 1, "empty"
    script = ("CODE = " + repr(code) + "\nTESTS = " + repr(tests) + "\n" + _PER_CHECK)
    with _scratch() as td:
        path = td / "checks.py"
        path.write_text(script, encoding="utf-8")
        try:
            proc = subprocess.run([sys.executable, str(path)], capture_output=True,
                                  encoding="utf-8", errors="replace",
                                  timeout=EXEC_TIMEOUT, cwd=str(td), env=_CHILD_ENV)
        except subprocess.TimeoutExpired:
            return 0, 1, "timeout"
    for line in (proc.stdout or "").splitlines():
        if line.startswith("RESULT "):
            _, met, missed, *rest = line.split(" ", 3)
            return int(met), int(met) + int(missed), (rest[0] if rest else "")
        if line.startswith("SETUP "):
            return 0, 1, line[6:]
    err = (proc.stderr or proc.stdout or "").strip().splitlines()
    return 0, 1, (err[-1][:160] if err else "no output")


def run_tests(code: str, tests: str) -> tuple[bool, str]:
    """Execute candidate code and its assertions in a subprocess.

    The timeout and scratch directory guard against runaway loops; this is not a
    security sandbox, and the code being run was produced by a language model.
    """
    if not code.strip():
        return False, "empty"
    script = f"{code}\n\n{tests}\nprint('__PASS__')\n"
    with _scratch() as td:
        path = td / "candidate.py"
        path.write_text(script, encoding="utf-8")
        try:
            proc = subprocess.run([sys.executable, str(path)], capture_output=True,
                                  encoding="utf-8", errors="replace",
                                  timeout=EXEC_TIMEOUT, cwd=str(td), env=_CHILD_ENV)
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if "__PASS__" in proc.stdout:
        return True, "ok"
    err = (proc.stderr or proc.stdout or "").strip().splitlines()
    return False, (err[-1][:160] if err else "no output")


def _def_has_annotations(code: str) -> bool:
    m = re.search(r"def\s+\w+\s*\((.*?)\)\s*(->[^:]*)?:", code, re.S)
    return bool(m) and (bool(m.group(2)) or ":" in (m.group(1) or ""))


def grade(task: dict, response: dict) -> tuple[bool, bool, str, float]:
    """Return (passed, parseable, detail, score).

    `parseable` is tracked separately because a reply the harness cannot parse
    costs a turn regardless of whether the model knew the answer.

    `score` is the fraction of the task met. For code it is the share of the
    task's checks satisfied, so missing one edge case is a penalty rather than a
    zero; everything else is answered or not, and scores one or nothing.
    """
    msg = response.get("message") or {}
    text = (msg.get("content") or "").strip()
    kind = task["kind"]

    if kind in ("codegen", "edit"):
        code = extract_code(text, task.get("entry"))
        if not code:
            return False, False, "no code emitted", 0.0
        met, total, detail = run_checks(code, task["tests"])
        if met == total:
            return True, True, "ok", 1.0
        return False, True, f"{met}/{total} checks: {detail}", met / total

    if kind == "trace":
        got = text.strip().strip("`").splitlines()[-1].strip() if text else ""
        norm = lambda s: re.sub(r"\s+", "", s)
        hit = norm(task["expect"]) == norm(got)
        return hit, bool(text), f"got={got[:60]!r}", float(hit)

    if kind == "toolcall":
        calls = msg.get("tool_calls") or []
        want = task.get("want")
        if want is None:                        # answering directly is correct
            if calls:
                name = (calls[0].get("function") or {}).get("name")
                return False, True, f"called {name} instead of answering", 0.0
            return bool(text), bool(text), "answered directly", float(bool(text))
        if not calls:
            return False, False, "no tool_call emitted", 0.0
        fn = calls[0].get("function") or {}
        if fn.get("name") != want:
            return False, True, f"called={fn.get('name')}, wanted={want}", 0.0
        for key, typ in (task.get("want_args") or {}).items():
            given = fn.get("arguments")
            if isinstance(given, str):
                try:
                    given = json.loads(given)
                except Exception:
                    return False, False, "arguments not valid JSON", 0.0
            given = given or {}
            if key not in given:
                return False, True, f"missing arg {key!r}", 0.0
            if not isinstance(given[key], typ):
                return False, True, f"arg {key!r} is {type(given[key]).__name__}", 0.0
        return True, True, f"called={fn.get('name')}", 1.0

    if kind == "format":
        check = task["check"]
        if check == "json_exact":
            raw = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
            try:
                hit = json.loads(raw) == task["expect"]
                return hit, True, "parsed", float(hit)
            except Exception:
                return False, False, f"unparseable: {text[:60]!r}", 0.0
        if check == "bare_code":
            if "```" in text:
                return False, False, "used fences despite instruction", 0.0
            ok, detail = run_tests(text, "assert add(2,3)==5\n")
            return ok, True, detail, float(ok)
        if check == "no_comments":
            if "```" in text:
                return False, False, "used fences despite instruction", 0.0
            if "#" in text:
                return False, True, "contains a comment", 0.0
            if '"""' in text or "'''" in text:
                return False, True, "contains a docstring", 0.0
            if _def_has_annotations(text):
                return False, True, "contains type hints", 0.0
            ok, detail = run_tests(
                text,
                "assert is_palindrome('A man, a plan, a canal: Panama')\n"
                "assert is_palindrome('')\n"
                "assert not is_palindrome('hello')\n"
                "assert is_palindrome('No lemon, no melon')\n")
            return ok, True, detail, float(ok)
        if check == "one_word":
            word = text.strip().strip(".`").lower()
            hit = word == task["expect"]
            return hit, len(word.split()) == 1, f"got={word[:40]!r}", float(hit)

    return False, False, f"unknown task kind {kind!r}", 0.0


def run_turns(client, model: str, task: dict, *, ctx: int,
              num_predict: int) -> tuple[dict, int]:
    """Put a task to the model, answering the calls it makes on the way.

    One turn unless the task offers more. Where it does, a call to a tool other
    than the one the task asks for is answered from `results` and the model is
    asked again: a model that lists the test files before running them has not
    called the wrong tool, and a harness would have served that call.

    Returns the last reply and how many turns it took, with the wall time of all
    of them, since the whole exchange is what a session would have paid.
    """
    prompt = task["prompt"]
    if task["kind"] == "edit":
        prompt = f"{prompt}\n\n```python\n{task['code']}\n```"
    messages = [{"role": "user", "content": prompt}]
    budget, wall = max(1, task.get("turns", 1)), 0.0
    for turn in range(1, budget + 1):
        resp = client.chat(model, messages if turn > 1 else prompt, ctx=ctx,
                           num_predict=num_predict, tools=task.get("tools"))
        wall += resp.get("_wall") or 0.0
        resp["_wall"] = round(wall, 2)
        message = resp.get("message") or {}
        calls = message.get("tool_calls") or []
        name = (calls[0].get("function") or {}).get("name") if calls else None
        served = (task.get("results") or {}).get(name)
        if turn == budget or name is None or name == task.get("want") or served is None:
            return resp, turn
        messages = messages + [message, {"role": "tool", "tool_name": name,
                                         "content": json.dumps(served)}]


def key(rec: dict) -> tuple:
    """What makes a result the same result: one model, one run, one task."""
    return rec["model"], rec["run"], rec["task"]


def load_ledger(path: Path) -> dict:
    """Every task result on file, by model, run and task."""
    return ledger.keyed(path, key)


def measure_tasks(client, cfg: Config, model: str, run_idx: int,
                  tasks: list[dict], done: dict, ledger_path: Path,
                  num_predict: int = NUM_PREDICT) -> list[dict]:
    """Run these tasks for one model and append each result to the ledger.

    Shared with the triage cascade, which grades the same tasks with the same
    grader. Writing to the same ledger is what stops the two measuring a model
    twice: whichever gets there first records the result, and the other finds it
    already done. A record here is a measurement wherever it came from.

    Tasks already measured in `done` are returned from it untouched. A task whose
    request failed is not one of them: it is asked again, since what is on record
    is the failure and not an answer.
    """
    records = []
    for task in tasks:
        ident = (model, run_idx, task["id"])
        if ident in done and measured(done[ident]):
            records.append(done[ident])
            continue
        try:
            resp, turns = run_turns(client, model, task, ctx=cfg.ctx,
                                    num_predict=num_predict)
        except Exception as exc:
            # The request failed, so the model said nothing to grade. Recorded as
            # the failure it is rather than as an empty answer: an unanswered task
            # scored zero and marked unparseable reads, everywhere downstream, as a
            # model that replied badly.
            rec = dict(model=model, run=run_idx, ctx=cfg.ctx, task=task["id"],
                       kind=task["kind"], error=f"{type(exc).__name__}: {exc}",
                       wall=None, ts=time.time())
        else:
            passed, parseable, detail, score = grade(task, resp)
            if turns > 1:
                # It reached the right tool, so it is not the wrong tool -- but it
                # took another round trip to get there, and the score is what one
                # direct call would have earned spread over the turns it took.
                score /= turns
                detail = f"{detail}, in {turns} turns"
            rec = dict(
                model=model, run=run_idx, ts=time.time(), ctx=cfg.ctx, turns=turns,
                task=task["id"], kind=task["kind"], passed=passed, score=score,
                format_ok=parseable, detail=detail, wall=resp["_wall"],
                gen_tok_s=round(resp["eval_count"] / (resp["eval_duration"] / 1e9), 1)
                if resp.get("eval_count") and resp.get("eval_duration") else None,
                hit_cap=bool(resp.get("eval_count")
                             and resp["eval_count"] >= num_predict),
                raw=((resp.get("message") or {}).get("content") or "")[:RAW_KEPT],
                tool_calls=(resp.get("message") or {}).get("tool_calls"),
            )
        records.append(rec)
        done[ident] = rec
        # Replace, do not append: the report reads this ledger directly, so a
        # superseded record would be counted twice. Written against what is on
        # disk rather than against `done`, which --redo empties.
        ledger.replace(ledger_path, rec, key)
        # A test point is all-or-nothing and a code task is not, so the share of
        # assertions met rides along in the diagnostics.
        progress.unit("screen", task["id"],
                      progress.OK if measured(rec) and rec["passed"] else progress.FAIL,
                      rec["wall"] or 0.0,
                      (rec.get("detail") or rec.get("error") or "")[:60],
                      score=rec.get("score"))
    return records


# Every task is measured once. Greedy decoding at a fixed seed reproduces exactly for
# all but one of the models measured, so a second run re-measures a function that has
# already answered; it is also what the code benchmarks do, ranking on greedy pass@1
# from a single sample.
#
# The uncertainty that remains is in the task sample, not in the decoding: at 29 tasks
# and a rate near 85% the 95% interval is about 13 points, and repeating the same tasks
# does not narrow it. Only more tasks do.
RUN = 1


def run(cfg: Config, num_predict: int = NUM_PREDICT, redo: bool = False,
        only: list[str] | None = None) -> None:
    gpulock.acquire("screen", endpoint=cfg.host)
    client = Ollama(cfg.host, cfg.timeout)
    tasks = TASKS
    ledger_path = cfg.path("screen_tasks.jsonl")
    stored = load_ledger(ledger_path)
    done = {} if redo else dict(stored)
    unknown = sorted(set(only or []) - {t["id"] for t in tasks})
    if unknown:
        # A typo would otherwise report success having measured nothing.
        raise SystemExit(f"unknown task id(s): {', '.join(unknown)}")
    selected = [t for t in tasks if not only or t["id"] in set(only)]

    models = cfg.resolve_models()
    for i, model in enumerate(models, 1):
        pending = [t for t in selected
                   if (model, RUN, t["id"]) not in done
                   or not measured(done[(model, RUN, t["id"])])]
        if pending:
            progress.subject(i, len(models), model, f"{len(pending)} task(s) to run")
        else:
            progress.subject(i, len(models), model)
        measure_tasks(client, cfg, model, RUN, selected, done, ledger_path,
                      num_predict)
        # Everything on record for this model, not whatever this invocation
        # selected, and a fresh measurement supersedes a stored one of the same task.
        held = {**stored, **done}
        _report(model, [held[(model, RUN, t["id"])] for t in tasks
                        if (model, RUN, t["id"]) in held])
        if pending:
            client.unload(model)


def measured(rec: dict) -> bool:
    """Whether this record holds an answer, as opposed to a request that failed.

    A failed request is on file so the run can say what happened, and it is not a
    measurement of the model: it counts towards no rate and settles no gate.
    """
    return not rec.get("error")


def _score(rec: dict) -> float:
    got = rec.get("score")
    return float(got) if got is not None else float(bool(rec.get("passed")))


def _report(model: str, records: list[dict]) -> None:
    """Say how the model did, from the records themselves."""
    failed = [r for r in records if not measured(r)]
    records = [r for r in records if measured(r)]
    n = len(records) or 1
    passed = sum(1 for r in records if r["passed"])
    rate = round(100 * sum(_score(r) for r in records) / n, 1)
    parseable = round(100 * sum(1 for r in records if r["format_ok"]) / n, 1)
    capped = sum(1 for r in records if r.get("hit_cap"))
    # The count and the rate measure different things -- tasks fully met, against
    # the mean score with partial credit per assertion -- so the line says which is
    # which rather than printing them as one figure and its percentage.
    progress.result(f"scored {rate}%, {passed} of {len(records)} tasks fully passed, "
                    f"parseable {parseable}%"
                    + (f", {len(failed)} task(s) unmeasured" if failed else "")
                    + (f", WARNING: {capped} replies hit the token cap" if capped else ""))


def main(argv: list[str] | None = None) -> int:
    parser = args.stage("screen", "Prompt-level tasks: code, edit, format, tools, tracing.",
                        executes=True)
    parser.add_argument("--num-predict", type=int, default=NUM_PREDICT, metavar="TOKENS",
                        help="output budget per reply (default: %(default)s)")
    parser.add_argument("--only", nargs="+", metavar="TASK_ID")
    parser.add_argument("--redo", action="store_true")
    a = parser.parse_args(argv)
    with progress.document():
        run(args.config_from(a), num_predict=a.num_predict, redo=a.redo, only=a.only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
