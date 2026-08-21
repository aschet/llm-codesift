"""Grade the task-list module.

Written to be the opposite of the application task in one respect. There, the
application had to start before thirteen of the eighteen checks could even be
attempted, so a model that mis-named one file scored the same as a model that
wrote nothing. Here only the import can cascade: once the module loads, every
function is exercised on its own, and a model that gets eight of them right
scores eight.

Standard library only, and no server, port or browser is involved, so the whole
thing takes about a second.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

REPO = Path.cwd()
SRC = REPO / "src"
HEX = re.compile(r"^#[0-9a-fA-F]{6}$")
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), str(detail)[:200]))
    return bool(ok)


def failed(name, exc):
    line = traceback.format_exception_only(type(exc), exc)[-1].strip()
    return check(name, False, line)


def check_layout():
    module = SRC / "tasklist.py"
    tests = list((REPO / "tests").glob("test_*.py")) if (REPO / "tests").is_dir() else []
    missing = []
    if not module.exists():
        missing.append("src/tasklist.py")
    if not tests:
        missing.append("tests/test_*.py")
    check("layout", not missing, "missing: " + ", ".join(missing) if missing else "ok")


def load_module():
    sys.path.insert(0, str(SRC))
    try:
        import tasklist
    except Exception as exc:
        failed("importable", exc)
        return None
    origin = os.path.abspath(getattr(tasklist, "__file__", "") or "")
    if not origin.startswith(str(REPO) + os.sep):
        check("importable", False, f"imported from outside the repository: {origin}")
        return None
    # A function that is absent costs the checks that need it and nothing else.
    # Failing the import here instead would cascade all thirteen, so a model that
    # wrote eight of the nine functions would score what a model that wrote none
    # scores -- which is the fault this task exists to avoid, and which it had.
    missing = [n for n in ("new_store", "add", "get", "update", "delete", "tasks",
                           "reorder", "save", "load") if not hasattr(tasklist, n)]
    check("importable", True,
          "no " + ", ".join(missing) if missing else "every function is present")
    return tasklist


def run_check(name, fn):
    """Each check stands alone, so one exception costs one point and no more."""
    try:
        ok, detail = fn()
    except Exception as exc:
        return failed(name, exc)
    return check(name, ok, detail)


def build(m):
    """A store with three tasks, used by the checks that need one."""
    store = m.new_store()
    a = m.add(store, "Buy milk", "Semi-skimmed", "#3366cc")
    b = m.add(store, "Write report", "Quarterly summary", "#cc7722")
    c = m.add(store, "Walk the dog", "", "#22aa88")
    return store, a, b, c


def checks_for(m):
    def add():
        store, a, b, c = build(m)
        faults = []
        for key, typ in (("id", int), ("title", str), ("description", str),
                         ("color", str), ("done", bool), ("position", int)):
            if key not in a:
                faults.append(f"no {key!r}")
            elif not isinstance(a[key], typ) or (typ is int and isinstance(a[key], bool)):
                faults.append(f"{key} is {type(a[key]).__name__}")
        if a.get("done") is not False:
            faults.append("a new task is not open")
        if isinstance(a.get("color"), str) and not HEX.match(a["color"]):
            faults.append(f"color {a['color']!r} is not #rrggbb")
        if len({a["id"], b["id"], c["id"]}) != 3:
            faults.append("ids are not unique")
        if [t["id"] for t in m.tasks(store)] != [a["id"], b["id"], c["id"]]:
            faults.append("not placed last, or not listed by position")
        return not faults, "; ".join(faults[:4]) or "ok"

    def get():
        store, a, _, _ = build(m)
        faults = []
        if (m.get(store, a["id"]) or {}).get("title") != "Buy milk":
            faults.append("did not return the task")
        if m.get(store, 987654) is not None:
            faults.append("an unknown id did not return None")
        return not faults, "; ".join(faults) or "ok"

    def update():
        store, a, _, _ = build(m)
        after = m.update(store, a["id"], title="Buy oat milk", color="#aa2200")
        faults = []
        if (after or {}).get("title") != "Buy oat milk":
            faults.append("the title did not change")
        if ((after or {}).get("color") or "").lower() != "#aa2200":
            faults.append("the colour did not change")
        if (after or {}).get("description") != "Semi-skimmed":
            faults.append("a field that was not named was overwritten")
        if m.update(store, 987654, title="x") is not None:
            faults.append("an unknown id did not return None")
        return not faults, "; ".join(faults) or "ok"

    def update_rejects_unknown():
        store, a, _, _ = build(m)
        try:
            m.update(store, a["id"], position=99)
        except ValueError:
            return True, "ok"
        return False, "an unknown field was accepted"

    def toggle_done():
        store, a, _, _ = build(m)
        after = m.update(store, a["id"], done=True)
        if (after or {}).get("done") is not True:
            return False, f"done is {(after or {}).get('done')!r}"
        if (after or {}).get("title") != "Buy milk":
            return False, "checking it off discarded the title"
        return True, "ok"

    def delete():
        store, a, _, _ = build(m)
        faults = []
        if m.delete(store, a["id"]) is not True:
            faults.append("did not report the removal")
        if m.get(store, a["id"]) is not None:
            faults.append("still present afterwards")
        if m.delete(store, 987654) is not False:
            faults.append("an unknown id did not return False")
        return not faults, "; ".join(faults) or "ok"

    def filter_done():
        store, a, b, c = build(m)
        m.update(store, b["id"], done=True)
        faults = []
        if [t["id"] for t in m.tasks(store, done=True)] != [b["id"]]:
            faults.append("done=True did not return only the checked task")
        if [t["id"] for t in m.tasks(store, done=False)] != [a["id"], c["id"]]:
            faults.append("done=False did not return only the open tasks")
        if len(m.tasks(store)) != 3:
            faults.append("an unfiltered listing lost tasks")
        return not faults, "; ".join(faults) or "ok"

    def filter_text():
        store, a, b, c = build(m)
        faults = []
        if [t["id"] for t in m.tasks(store, q="MILK")] != [a["id"]]:
            faults.append("no case-insensitive match on the title")
        if [t["id"] for t in m.tasks(store, q="quarterly")] != [b["id"]]:
            faults.append("no match on the description")
        if m.tasks(store, q="zzz"):
            faults.append("matched something it should not have")
        return not faults, "; ".join(faults) or "ok"

    def reorder():
        store, a, b, c = build(m)
        wanted = [c["id"], a["id"], b["id"]]
        m.reorder(store, wanted)
        listed = m.tasks(store)
        faults = []
        if [t["id"] for t in listed] != wanted:
            faults.append("the listing did not follow the new order")
        elif [t["position"] for t in listed] != sorted(t["position"] for t in listed):
            faults.append("position no longer ascends with the listing")
        return not faults, "; ".join(faults) or "ok"

    def reorder_rejects_unknown():
        store, a, b, c = build(m)
        try:
            m.reorder(store, [a["id"], b["id"], 987654])
        except ValueError:
            return True, "ok"
        return False, "an unknown id was accepted"

    def persist():
        store, a, b, c = build(m)
        m.update(store, a["id"], done=True, title="Buy oat milk")
        m.reorder(store, [c["id"], a["id"], b["id"]])
        path = REPO / "_grading_store.json"
        if path.exists():
            path.unlink()
        m.save(store, str(path))
        if not path.exists():
            return False, "save wrote nothing at the path it was given"
        again = m.load(str(path))
        listed = m.tasks(again)
        faults = []
        if [t["id"] for t in listed] != [c["id"], a["id"], b["id"]]:
            faults.append("the order did not survive")
        if not any(t["done"] for t in listed):
            faults.append("a checked task came back open")
        if not any(t["title"] == "Buy oat milk" for t in listed):
            faults.append("an edited title did not survive")
        return not faults, "; ".join(faults) or "ok"

    def load_missing():
        gone = REPO / "_grading_absent.json"
        if gone.exists():
            gone.unlink()
        store = m.load(str(gone))
        empty = not m.tasks(store)
        return empty, "ok" if empty else "a missing file did not give an empty store"

    def empty_title():
        store = m.new_store()
        try:
            m.add(store, "")
        except ValueError:
            return True, "ok"
        return False, "an empty title was accepted"

    return [("add", add), ("get", get), ("update", update),
            ("update_rejects_unknown", update_rejects_unknown),
            ("toggle_done", toggle_done), ("delete", delete),
            ("filter_done", filter_done), ("filter_text", filter_text),
            ("reorder", reorder), ("reorder_rejects_unknown", reorder_rejects_unknown),
            ("persist", persist), ("load_missing", load_missing),
            ("empty_title", empty_title)]


def check_own_tests():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SRC)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"],
            cwd=str(REPO), env=env, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return check("own_tests", False, "the test run did not finish")
    except FileNotFoundError:
        return check("own_tests", False, "pytest unavailable")
    text = (proc.stdout or "") + (proc.stderr or "")
    passed = int((re.search(r"(\d+) passed", text) or [0, 0])[1])
    broken = sum(int(n) for n in re.findall(r"(\d+) (?:failed|error)", text))
    if proc.returncode == 5 or (passed == 0 and not broken):
        return check("own_tests", False, "no tests were collected")
    if broken:
        return check("own_tests", False, f"{passed} passed, {broken} failed")
    check("own_tests", passed >= 4,
          f"{passed} passed" + ("" if passed >= 4 else ", too few to cover the module"))


def main():
    check_layout()
    module = load_module()
    if module is None:
        for name, _ in checks_for(_Missing()):
            check(name, False, "the module could not be imported")
    else:
        for name, fn in checks_for(module):
            run_check(name, fn)

    check_own_tests()

    for name, ok, detail in RESULTS:
        print(f"CHECK {name} {'PASS' if ok else 'FAIL'} {detail}")
    print(f"CHECKS {sum(1 for _, ok, _ in RESULTS if ok)}/{len(RESULTS)}")


class _Missing:
    """Stands in for the module so the check names are still reported."""

    def __getattr__(self, name):
        raise ImportError("module unavailable")


if __name__ == "__main__":
    main()
