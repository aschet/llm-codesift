"""
Agentic tasks: each seeds a throwaway repo, then opencode must drive tools
(read / edit / bash) to reach a state that `verify` accepts.

Graded on the resulting files only -- never on what the model claimed it did.
"""

MODULE_SPEC = '''# Task List Module

One Python module that keeps a list of tasks and can save and load them. No web
server, no packaging, no dependencies beyond the standard library. This file is
the contract; it is graded against, so build to it exactly. Do not edit it.

## Layout

    src/tasklist.py           the module
    tests/test_tasklist.py    your tests

## The task

A dict, exactly these keys:

    {"id": 3, "title": "Buy milk", "description": "Semi-skimmed",
     "color": "#3366cc", "done": False, "position": 2}

`color` is a CSS hex colour of the form `#rrggbb`. `position` orders the list.

## The functions

Every one takes the store as its first argument.

    new_store()                 an empty store
    add(store, title, description="", color="#888888")
                                the created task: open, placed last, with an id
                                unique within the store. An empty title raises
                                ValueError.
    get(store, task_id)         the task, or None if there is no such id
    update(store, task_id, **fields)
                                the updated task, or None if there is no such
                                id. Only title, description, color and done may
                                be set; any other keyword raises ValueError.
                                Fields not named are left alone.
    delete(store, task_id)      True if it was removed, False if there was no
                                such id
    tasks(store, done=None, q=None)
                                the tasks, ordered by position ascending.
                                `done` keeps only the checked or only the open
                                ones when it is True or False. `q` keeps those
                                whose title or description contains it, ignoring
                                case. Both may be given at once.
    reorder(store, order)       the tasks in the order given, which later
                                listings keep. `order` is a list of ids; an id
                                that is not in the store raises ValueError.
    save(store, path)           write the store to that path
    load(path)                  a store holding the same tasks in the same
                                order. A path that does not exist gives an empty
                                store rather than an error.

## Before you finish

Import it and run your own tests. Both of these must work from the repository
root:

    PYTHONPATH=src python -c "import tasklist; tasklist.new_store()"
    PYTHONPATH=src python -m pytest tests
'''


TASKS = [
    dict(
        id="ag_fixbug",
        prompt="The test suite fails. Run `{py} tests/test_stats.py` to see the failure, "
               "find the bug in src/stats.py, and fix it. Do not modify the tests.",
        files={
            "src/stats.py": '''def median(xs):
    if not xs:
        raise ValueError("empty")
    s = sorted(xs)
    return s[len(s) // 2]


def mean(xs):
    if not xs:
        raise ValueError("empty")
    return sum(xs) / len(xs)
''',
            "tests/test_stats.py": '''import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from stats import median, mean

assert median([1, 3, 2]) == 2
assert median([1, 2, 3, 4]) == 2.5, "even-length median must average the middle pair"
assert mean([1, 2, 3]) == 2
print("ALL PASS")
''',
        },
        # tests must be untouched: the fix has to land in the source
        immutable=["tests/test_stats.py"],
        verify=["{py}", "tests/test_stats.py"],
        expect_stdout="ALL PASS",
    ),
    dict(
        id="ag_feature",
        prompt="Add a function `parse_kv(text)` to src/config.py that parses lines of the form "
               "key=value into a dict. Ignore blank lines and lines starting with #. Strip "
               "whitespace around keys and values. A line without '=' raises ValueError. "
               "Then add tests for it in tests/test_config.py following the existing style, "
               "and make sure `{py} tests/test_config.py` passes.",
        files={
            "src/config.py": '''def load(path):
    with open(path) as f:
        return f.read()
''',
            "tests/test_config.py": '''import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from config import load

print("ALL PASS")
''',
        },
        verify_src="import sys; sys.path.insert(0, 'src')\n"
                 "from config import parse_kv\n"
                 "assert parse_kv('a=1') == {'a':'1'}\n"
                 "assert parse_kv('a = 1 \\n b=2') == {'a':'1','b':'2'}\n"
                 "assert parse_kv('# c=3\\na=1') == {'a':'1'}\n"
                 "assert parse_kv('') == {}\n"
                 "try:\n"
                 "    parse_kv('nope'); raise SystemExit('should have raised')\n"
                 "except ValueError: pass\n"
                 "print('ALL PASS')\n",
        expect_stdout="ALL PASS",
    ),
    dict(
        id="ag_refactor",
        prompt="src/reader.py and src/writer.py both contain an identical `normalize_path` "
               "function. Extract it into a new module src/paths.py and make both files import "
               "it from there instead of defining it. Do not change its behaviour. Verify with "
               "`{py} tests/test_paths.py`.",
        files={
            "src/reader.py": '''import os


def normalize_path(p):
    return os.path.normpath(os.path.expanduser(p))


def read(p):
    with open(normalize_path(p)) as f:
        return f.read()
''',
            "src/writer.py": '''import os


def normalize_path(p):
    return os.path.normpath(os.path.expanduser(p))


def write(p, data):
    with open(normalize_path(p), "w") as f:
        f.write(data)
''',
            "tests/test_paths.py": '''import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from reader import normalize_path as a
from writer import normalize_path as b
from paths import normalize_path as c

assert a("./x/../y") == c("./x/../y") == b("./x/../y")
src = open(os.path.join(os.path.dirname(__file__), "..", "src", "reader.py")).read()
assert "def normalize_path" not in src, "reader.py still defines it"
src = open(os.path.join(os.path.dirname(__file__), "..", "src", "writer.py")).read()
assert "def normalize_path" not in src, "writer.py still defines it"
print("ALL PASS")
''',
        },
        immutable=["tests/test_paths.py"],
        verify=["{py}", "tests/test_paths.py"],
        expect_stdout="ALL PASS",
    ),
    dict(
        id="ag_module",
        prompt="Read SPEC.md and write what it describes: one module and your own "
               "tests for it. Import it and run your tests with `{py}` to confirm it "
               "behaves as the specification says, and fix what does not. "
               "Do not edit SPEC.md.",
        files={"SPEC.md": MODULE_SPEC},
        immutable=["SPEC.md"],
        # Graded per function rather than pass or fail. Unlike the application task,
        # nothing here cascades except the import itself: once the module loads,
        # every function is exercised on its own, so a model that gets eight of them
        # right scores eight rather than scoring the same as one that wrote nothing.
        graded=True,
        pivot="importable",
        verify_src_file="module_check.py",
        verify_timeout=300,
        # Minutes rather than an hour. This asks whether a model can implement
        # ordinary things from a written contract at all, which the three small
        # tasks do not: they each edit code that already exists.
        min_timeout=900,
        retain=True,
    ),
]
