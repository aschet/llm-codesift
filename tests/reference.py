"""Known-good solutions for the task suites.

Every task must be solvable, and every seeded defect must genuinely fail. These
solutions prove both, so a task that silently accepts anything is caught here
rather than in a measurement run.
"""

BASIC = {
"cg_version": """
def parse_version(s):
    parts = [int(p) for p in s.split('.')][:3]
    return tuple(parts + [0] * (3 - len(parts)))
""",
"cg_intervals": """
def merge_intervals(intervals):
    if not intervals:
        return []
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out
""",
"cg_flatten": """
def flatten(d, sep='.'):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in flatten(v, sep).items():
                out[f"{k}{sep}{k2}"] = v2
        else:
            out[k] = v
    return out
""",
"cg_retry": """
import functools
def retry(times):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            last = None
            for _ in range(times):
                try:
                    return fn(*a, **kw)
                except Exception as e:
                    last = e
            raise last
        return wrapper
    return deco
""",
"ed_offbyone": """
def chunk(items, n):
    return [items[i:i + n] for i in range(0, len(items), n)]
""",
"ed_addparam": """
def normalize(text, strip_punct=False):
    r = ' '.join(text.lower().split())
    if strip_punct:
        for c in '.,!?':
            r = r.replace(c, '')
        r = ' '.join(r.split())
    return r
""",
"ed_method": """
class Counter2:
    def __init__(self):
        self.counts = {}
    def add(self, item):
        self.counts[item] = self.counts.get(item, 0) + 1
    def most_common(self, k):
        return sorted(self.counts.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
""",
}

HARD = {
"h_cg_roman": """
def to_roman(n):
    if not isinstance(n, int) or n < 1 or n > 3999:
        raise ValueError(n)
    vals = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),
            (50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]
    out = []
    for v, s in vals:
        while n >= v:
            out.append(s)
            n -= v
    return ''.join(out)
""",
"h_cg_semver": """
def _pre(p):
    out = []
    for part in p.split('.'):
        out.append((0, int(part), '') if part.isdigit() else (1, 0, part))
    return out
def cmp_semver(a, b):
    def split(v):
        core, _, pre = v.partition('-')
        return [int(x) for x in core.split('.')], pre
    ca, pa = split(a)
    cb, pb = split(b)
    if ca != cb:
        return -1 if ca < cb else 1
    if pa == pb:
        return 0
    if not pa:
        return 1
    if not pb:
        return -1
    la, lb = _pre(pa), _pre(pb)
    if la != lb:
        return -1 if la < lb else 1
    return 0
""",
"h_cg_tokenize": r"""
def tokenize(s):
    toks, cur, in_q, has = [], [], False, False
    i = 0
    while i < len(s):
        c = s[i]
        if in_q:
            if c == '\\' and i + 1 < len(s):
                cur.append(s[i + 1]); i += 2; continue
            if c == '"':
                in_q = False; i += 1; continue
            cur.append(c); i += 1; continue
        if c == '"':
            in_q = True; has = True; i += 1; continue
        if c.isspace():
            if cur or has:
                toks.append(''.join(cur)); cur, has = [], False
            i += 1; continue
        cur.append(c); i += 1
    if in_q:
        raise ValueError('unterminated quote')
    if cur or has:
        toks.append(''.join(cur))
    return toks
""",
"h_ed_cache": """
from collections import OrderedDict
class Cache:
    def __init__(self, cap):
        self.cap = cap
        self.data = OrderedDict()
    def get(self, k):
        if k not in self.data:
            return None
        self.data.move_to_end(k)
        return self.data[k]
    def put(self, k, v):
        if k in self.data:
            self.data.move_to_end(k)
        elif len(self.data) >= self.cap:
            self.data.popitem(last=False)
        self.data[k] = v
""",
"h_ed_paginate": """
def paginate(items, page, per_page):
    if per_page < 1:
        raise ValueError('per_page must be >= 1')
    if page < 1:
        return []
    start = (page - 1) * per_page
    return items[start:start + per_page]
""",
}

# Files to overwrite in a seeded repository to solve each agent task.
def _tasklist():
    """The reference module, flattened to the {path: text} shape the tasks use."""
    from pathlib import Path
    root = Path(__file__).parent / "tasklist_reference"
    return {str(path.relative_to(root)): path.read_text(encoding="utf-8")
            for path in sorted(root.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts}


AGENT = {
    "ag_module": _tasklist(),
"ag_fixbug": {"src/stats.py": """
def median(xs):
    if not xs:
        raise ValueError("empty")
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def mean(xs):
    if not xs:
        raise ValueError("empty")
    return sum(xs) / len(xs)
"""},
"ag_feature": {"src/config.py": """
def load(path):
    with open(path) as f:
        return f.read()


def parse_kv(text):
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(line)
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    return out
"""},
"ag_refactor": {
    "src/paths.py": """
import os


def normalize_path(p):
    return os.path.normpath(os.path.expanduser(p))
""",
    "src/reader.py": """
from paths import normalize_path


def read(p):
    with open(normalize_path(p)) as f:
        return f.read()
""",
    "src/writer.py": """
from paths import normalize_path


def write(p, data):
    with open(normalize_path(p), "w") as f:
        f.write(data)
""",
},
}


# The two format tasks that forbid fences also forbid comments, docstrings and
# type hints, so their answers cannot come from the solutions above, which are
# written for readability.
BARE = {
    "fmt_barecode": "def add(a, b):\n    return a + b",
    "h_fmt_negative": ("def is_palindrome(s):\n"
                       "    t = [c.lower() for c in s if c.isalnum()]\n"
                       "    return t == t[::-1]"),
}

# A sample value per type, for filling in the arguments a tool task wants.
SAMPLE = {str: "src", int: 3, bool: True, list: ["a"], dict: {"a": 1}, float: 1.5}


def perfect_answer(task):
    """The reply a model that knew everything would give, as a chat response.

    Derived from the task's own definition -- the reference solution, the trace
    output it states, the tool it names -- so a fake model is graded by the real
    grader rather than around it. If this stops scoring full marks, a task and its
    grader have drifted apart, which is worth failing a test over.
    """
    import json as _json
    kind = task["kind"]
    if kind in ("codegen", "edit"):
        body = (BASIC.get(task["id"]) or HARD.get(task["id"]) or "")
        return {"content": f"```python\n{body}\n```"}
    if kind == "trace":
        return {"content": task["expect"]}
    if kind == "format":
        check = task["check"]
        if check == "json_exact":
            return {"content": _json.dumps(task["expect"])}
        if check == "one_word":
            return {"content": task["expect"]}
        return {"content": BARE[task["id"]]}
    if kind == "toolcall":
        want = task.get("want")
        if want is None:
            return {"content": "No tool is needed; the answer is 4."}
        args = {k: SAMPLE[typ] for k, typ in (task.get("want_args") or {}).items()}
        return {"content": "",
                "tool_calls": [{"function": {"name": want, "arguments": args}}]}
    raise AssertionError(f"no answer for task kind {kind!r}")
