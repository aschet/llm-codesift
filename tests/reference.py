# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""Known-good solutions for the tasks that run code.

Every task must be solvable. These solutions prove it, so a task that silently
accepts anything is caught here rather than in a measurement run.
"""

SOLUTIONS = {
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

"cg_roman": """
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
"cg_semver": """
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
"cg_tokenize": r"""
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
"ed_cache": """
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
"ed_paginate": """
def paginate(items, page, per_page):
    if per_page < 1:
        raise ValueError('per_page must be >= 1')
    if page < 1:
        return []
    start = (page - 1) * per_page
    return items[start:start + per_page]
""",
"cg_topo": """
def topo_sort(graph):
    nodes = set(graph)
    for after in graph.values():
        nodes.update(after)
    state, order = {}, []
    def visit(n):
        if state.get(n) == 'done':
            return True
        if state.get(n) == 'open':
            return False
        state[n] = 'open'
        for m in graph.get(n, []):
            if not visit(m):
                return False
        state[n] = 'done'
        order.append(n)
        return True
    for n in sorted(nodes, key=str):
        if not visit(n):
            return None
    return order[::-1]
""",
"cg_pathnorm": """
def normalise_path(path):
    absolute = path.startswith('/')
    out = []
    for part in path.split('/'):
        if part in ('', '.'):
            continue
        if part == '..':
            if out and out[-1] != '..':
                out.pop()
            elif not absolute:
                out.append('..')
            continue
        out.append(part)
    joined = '/'.join(out)
    if absolute:
        return '/' + joined
    return joined or '.'
""",
"cg_csvline": """
def split_csv(line):
    fields, field, i, quoted = [], [], 0, False
    while i < len(line):
        c = line[i]
        if quoted:
            if c == '"':
                if i + 1 < len(line) and line[i + 1] == '"':
                    field.append('"')
                    i += 1
                else:
                    quoted = False
            else:
                field.append(c)
        elif c == '"':
            quoted = True
        elif c == ',':
            fields.append(''.join(field))
            field = []
        else:
            field.append(c)
        i += 1
    fields.append(''.join(field))
    return fields
""",
"cg_rle": """
def rle_encode(text):
    out, i = [], 0
    while i < len(text):
        j = i
        while j < len(text) and text[j] == text[i]:
            j += 1
        run = j - i
        if run > 1:
            out.append(text[i] + '#' + str(run) + '#')
        else:
            out.append(text[i] if text[i] != '#' else '##')
        i = j
    return ''.join(out)

def rle_decode(code):
    out, i = [], 0
    while i < len(code):
        c = code[i]
        if i + 1 < len(code) and code[i + 1] == '#':
            end = code.index('#', i + 2)
            out.append(c * int(code[i + 2:end]))
            i = end + 1
        elif c == '#':
            out.append('#')
            i += 2
        else:
            out.append(c)
            i += 1
    return ''.join(out)
""",
"cg_bsearch": """
def insert_point(sorted_items, value):
    lo, hi = 0, len(sorted_items)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_items[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo
""",
"cg_expr": """
def evaluate(expression):
    text = expression.replace(' ', '')
    pos = 0
    def number():
        nonlocal pos
        start = pos
        while pos < len(text) and text[pos].isdigit():
            pos += 1
        return int(text[start:pos])
    def atom():
        nonlocal pos
        if text[pos] == '(':
            pos += 1
            value = expr()
            pos += 1
            return value
        if text[pos] == '-':
            pos += 1
            return -atom()
        return number()
    def term():
        nonlocal pos
        value = atom()
        while pos < len(text) and text[pos] in '*/':
            op = text[pos]
            pos += 1
            rhs = atom()
            value = value * rhs if op == '*' else value / rhs
        return value
    def expr():
        nonlocal pos
        value = term()
        while pos < len(text) and text[pos] in '+-':
            op = text[pos]
            pos += 1
            rhs = term()
            value = value + rhs if op == '+' else value - rhs
        return value
    return expr()
""",
"cg_wrap": """
def wrap_text(text, width):
    words = text.split()
    lines, current = [], ''
    for word in words:
        candidate = word if not current else current + ' ' + word
        if current and len(candidate) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines
""",
"cg_duration": """
import re
def parse_duration(text):
    if not text or not re.fullmatch(r'(\\d+[dhms])+', text):
        raise ValueError(f'cannot read {text!r} as a duration')
    seconds = {'d': 86400, 'h': 3600, 'm': 60, 's': 1}
    return sum(int(n) * seconds[u] for n, u in re.findall(r'(\\d+)([dhms])', text))
""",
"ed_window": """
def rolling_max(items, size):
    return [max(items[i:i + size]) for i in range(len(items) - size + 1)]
""",
"ed_swallow": """
def load_all(rows, parse):
    out, errors = [], []
    for row in rows:
        try:
            out.append(parse(row))
        except ValueError as exc:
            errors.append(str(exc))
    return out, errors
""",
"ed_earlyreturn": """
def find_all(items, predicate):
    return [i for i, item in enumerate(items) if predicate(item)]
""",
"ed_keycollide": """
def memoise(fn):
    cache = {}

    def wrapper(*args):
        key = args
        if key not in cache:
            cache[key] = fn(*args)
        return cache[key]

    return wrapper
""",
}

# Files to overwrite in a seeded repository to solve each agent task.
BARE = {
    "fmt_barecode": "def add(a, b):\n    return a + b",
    "fmt_negative": ("def is_palindrome(s):\n"
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
        body = SOLUTIONS.get(task["id"]) or ""
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
