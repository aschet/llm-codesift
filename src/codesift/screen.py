"""Prompt-level screening: generation, editing, output format, tool calls, tracing.

Results are appended per task, so an interrupted run resumes without regenerating
anything already measured.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import gpulock
from .config import Config
from .ollama import Ollama
from .tasks import TASKSETS

EXEC_TIMEOUT = 15

# The budget has to cover a model's reasoning as well as its answer, and it exists to
# stop a runaway rather than to constrain a legitimate reply. At 2048 it was binding on
# the longest tasks: a reasoning model spent it thinking and returned no code at all,
# which the grader recorded as incompetence. Set high enough that reaching it is itself
# a finding, and reported as such rather than silently scored as a wrong answer.
NUM_PREDICT = 6144

# How much of each reply is kept. A grading bug in how code was pulled out of a reply
# could only be confirmed on the replies short enough to have survived this cap, and
# most of the ones in question had not; the record is worth the disk.
RAW_KEPT = 12000


def _declarations_only(code: str) -> str:
    """Drop statements that exist to demonstrate the code rather than to be it.

    A bare expression at the top level -- `print(divide(10, 0))`, `flaky_call()` --
    is a demonstration. It is not part of the answer, and running it alongside the
    tests grades the model on its illustration: one reply divided by zero on purpose
    to show the retry working, and another called a function that fails at random,
    which would have made the result differ between runs of identical input.

    Everything that declares something is kept, in order, so imports introduced
    beside a first attempt remain available to the version that supersedes it, and
    a redefinition later in the reply wins exactly as Python would have it. Code
    that will not parse is returned untouched, since the grader is about to run it
    and report the syntax error, which is the honest result.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    keep = []
    for node in tree.body:
        if isinstance(node, ast.Expr):
            continue                      # a bare call, or a stray docstring
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
    a question of size. Taking the longest graded whichever happened to be longer,
    so a model that documented a correct answer with a few example calls was scored
    on the examples and failed for a NameError -- a penalty for explaining an
    answer, recorded as incorrectness. Taking the first block that defines the entry
    point graded a draft the model had already withdrawn, since a model reasoning
    through a bug shows the broken version before the fixed one.

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


def run_checks(code: str, tests: str) -> tuple[int, int, str]:
    """Return (checks met, checks total, what the first shortfall was).

    A task's assertions are graded one at a time rather than as a single script.
    Most failing answers are near misses -- across one full sweep the median
    satisfied three quarters of the assertions it was given -- and scoring those
    the same as code that does not run reports a difference that is not there.

    The statements run in order in one namespace, because a task's checks are not
    independent: some build the object the later ones examine. A setup statement
    that raises ends the attempt, since nothing after it means anything.
    """
    if not code.strip():
        return 0, 1, "empty"
    script = ("CODE = " + repr(code) + "\nTESTS = " + repr(tests) + "\n" + _PER_CHECK)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "checks.py"
        path.write_text(script, encoding="utf-8")
        try:
            proc = subprocess.run([sys.executable, str(path)], capture_output=True,
                                  text=True, timeout=EXEC_TIMEOUT, cwd=td)
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
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "candidate.py"
        path.write_text(script, encoding="utf-8")
        try:
            proc = subprocess.run([sys.executable, str(path)], capture_output=True,
                                  text=True, timeout=EXEC_TIMEOUT, cwd=td)
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
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    return False, False, "arguments not valid JSON", 0.0
            args = args or {}
            if key not in args:
                return False, True, f"missing arg {key!r}", 0.0
            if not isinstance(args[key], typ):
                return False, True, f"arg {key!r} is {type(args[key]).__name__}", 0.0
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


def load_ledger(path: Path) -> dict:
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                done[(rec["model"], rec["run"], rec["taskset"], rec["task"])] = rec
            except Exception:
                pass
    return done


def measure_tasks(client, cfg: Config, model: str, taskset: str, run_idx: int,
                  tasks: list[dict], done: dict, ledger_path: Path,
                  num_predict: int = NUM_PREDICT) -> list[dict]:
    """Run these tasks for one model and append each result to the ledger.

    Shared with the triage cascade, which grades the same tasks with the same
    grader. Writing to the same ledger is what stops the two measuring a model
    twice: whichever gets there first records the result, and the other finds it
    already done. A record here is a measurement wherever it came from.

    Tasks already present in `done` are returned from it untouched.
    """
    records = []
    for task in tasks:
        key = (model, run_idx, taskset, task["id"])
        if key in done:
            records.append(done[key])
            continue
        prompt = task["prompt"]
        if task["kind"] == "edit":
            prompt = f"{prompt}\n\n```python\n{task['code']}\n```"
        try:
            resp = client.chat(model, prompt, ctx=cfg.ctx, num_predict=num_predict,
                               tools=task.get("tools"))
        except Exception as exc:
            rec = dict(model=model, run=run_idx, taskset=taskset,
                       task=task["id"], kind=task["kind"], passed=False, score=0.0,
                       format_ok=False, detail=f"{type(exc).__name__}: {exc}",
                       wall=None, ts=time.time())
        else:
            passed, parseable, detail, score = grade(task, resp)
            rec = dict(
                model=model, run=run_idx, taskset=taskset, ts=time.time(),
                task=task["id"], kind=task["kind"], passed=passed, score=score,
                format_ok=parseable, detail=detail, wall=resp["_wall"],
                gen_toks=resp.get("eval_count"),
                gen_tok_s=round(resp["eval_count"] / (resp["eval_duration"] / 1e9), 1)
                if resp.get("eval_count") and resp.get("eval_duration") else None,
                hit_cap=bool(resp.get("eval_count")
                             and resp["eval_count"] >= num_predict),
                raw=((resp.get("message") or {}).get("content") or "")[:RAW_KEPT],
                tool_calls=(resp.get("message") or {}).get("tool_calls"),
            )
        records.append(rec)
        done[key] = rec
        with ledger_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"  {task['id']:16} {'PASS' if rec['passed'] else 'FAIL':4} "
              f"{rec['wall'] or 0:6.1f}s  {rec['detail'][:60]}", flush=True)
    return records


def run(cfg: Config, taskset: str = "basic", runs: int = 1, num_predict: int = NUM_PREDICT, redo: bool = False,
        only: list[str] | None = None) -> None:
    gpulock.acquire(f"screen {taskset}", endpoint=cfg.host)
    client = Ollama(cfg.host, cfg.timeout)
    tasks = TASKSETS[taskset]
    ledger_path = cfg.path("screen_tasks.jsonl")
    summary_path = cfg.path("screen.jsonl")
    done = {} if redo else load_ledger(ledger_path)
    selected = [t for t in tasks if not only or t["id"] in set(only)]

    for run_idx in range(1, runs + 1):
        for model in cfg.resolve_models():
            pending = [t for t in selected
                       if (model, run_idx, taskset, t["id"]) not in done]
            if pending:
                print(f"\n{model} [{taskset} run {run_idx}]: {len(pending)} task(s) to run",
                      flush=True)
            else:
                # Nothing left to measure, but the summary may never have been
                # written: triage records task results without one, so a set it
                # completed would otherwise be missing from the report entirely.
                print(f"{model} [{taskset} run {run_idx}]: complete", flush=True)
            records = measure_tasks(client, cfg, model, taskset, run_idx, selected,
                                    done, ledger_path, num_predict)
            _write_summary(summary_path, model, run_idx, taskset, cfg.ctx, records)
            if pending:
                client.unload(model)


def _score(rec: dict) -> float:
    got = rec.get("score")
    return float(got) if got is not None else float(bool(rec.get("passed")))


def _write_summary(path: Path, model: str, run_idx: int, taskset: str, ctx: int,
                   records: list[dict]) -> None:
    """Rewrite this model's row for this run and task set.

    `total_s` is the sum of the tasks' own times rather than the wall clock of the
    process, so a set finished across two sessions reports the same figure as one
    finished in a single sitting.
    """
    n = len(records) or 1
    walls = sorted(r["wall"] for r in records if r.get("wall"))
    summary = dict(
        model=model, run=run_idx, taskset=taskset, ctx=ctx, ts=time.time(),
        n=len(records),
        passed=sum(1 for r in records if r["passed"]),
        # The mean of what each task scored, not the count of tasks fully met. A
        # record written before scores existed has none, so its outcome stands in.
        pass_rate=round(100 * sum(_score(r) for r in records) / n, 1),
        fully_passed_rate=round(100 * sum(1 for r in records if r["passed"]) / n, 1),
        format_ok_rate=round(100 * sum(1 for r in records if r["format_ok"]) / n, 1),
        hit_cap_n=sum(1 for r in records if r.get("hit_cap")),
        median_wall=walls[len(walls) // 2] if walls else None,
        total_s=round(sum(r.get("wall") or 0 for r in records), 1),
        tasks=records,
    )
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                if not (rec.get("model") == model and rec.get("run") == run_idx
                        and rec.get("taskset") == taskset):
                    rows.append(rec)
            except Exception:
                pass
    rows.append(summary)
    with path.open("w", encoding="utf-8") as fh:
        for rec in rows:
            fh.write(json.dumps(rec) + "\n")
    print(f"  -> {summary['passed']}/{summary['n']} passed ({summary['pass_rate']}%), "
          f"parseable {summary['format_ok_rate']}%, {summary['total_s']}s"
          + (f"  [WARNING: {summary['hit_cap_n']} replies hit the token cap]"
             if summary["hit_cap_n"] else ""), flush=True)
