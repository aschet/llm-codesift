"""
Tier-2 tasks: built to discriminate where the tier-1 screen saturates.

Design rule -- hard through *care*, not obscurity. Every answer is deterministic
CPython semantics, no version-specific or implementation-defined behaviour.
Target: strong 30B models land 50-80%, not 100%.
"""

TASKS = [
    # ---------- trace: the tier-1 discriminator, pushed harder ----------
    dict(id="h_tr_finally", kind="trace", expect="2",
         prompt="What does this print? Reply with only the printed value.\n\n"
                "def f():\n    try:\n        return 1\n    finally:\n        return 2\n"
                "print(f())"),
    dict(id="h_tr_zipself", kind="trace", expect="[(1, 2), (3, 4)]",
         prompt="What does this print? Reply with only the printed value.\n\n"
                "it = iter([1,2,3,4])\nprint(list(zip(it, it)))"),
    dict(id="h_tr_genexhaust", kind="trace", expect="[0, 1, 2] []",
         prompt="What does this print? Reply with only the printed value.\n\n"
                "def gen():\n    for i in range(3):\n        yield i\n"
                "g = gen()\nprint(list(g), list(g))"),
    dict(id="h_tr_classattr", kind="trace", expect="[1] [1]",
         prompt="What does this print? Reply with only the printed value.\n\n"
                "class A:\n    items = []\n"
                "a, b = A(), A()\na.items.append(1)\nprint(a.items, b.items)"),
    dict(id="h_tr_chained", kind="trace", expect="False",
         prompt="What does this print? Reply with only the printed value.\n\n"
                "print(1 in [1,2] == True)"),
    dict(id="h_tr_dupkey", kind="trace", expect="{'a': 2}",
         prompt="What does this print? Reply with only the printed value.\n\n"
                "print({k: v for k, v in [('a',1),('a',2)]})"),

    # ---------- codegen: correctness hinges on edge cases ----------
    dict(id="h_cg_roman", kind="codegen", entry="to_roman",
         prompt="Write `to_roman(n)` converting an integer 1..3999 to a Roman numeral string, "
                "using subtractive notation (4=IV, 9=IX, 40=XL, 90=XC, 400=CD, 900=CM). "
                "Raise ValueError for n outside 1..3999.",
         tests="""
assert to_roman(1) == 'I'
assert to_roman(4) == 'IV'
assert to_roman(9) == 'IX'
assert to_roman(14) == 'XIV'
assert to_roman(40) == 'XL'
assert to_roman(90) == 'XC'
assert to_roman(400) == 'CD'
assert to_roman(900) == 'CM'
assert to_roman(1994) == 'MCMXCIV'
assert to_roman(3999) == 'MMMCMXCIX'
for bad in (0, -1, 4000):
    try:
        to_roman(bad); assert False, f'should raise for {bad}'
    except ValueError: pass
"""),
    dict(id="h_cg_semver", kind="codegen", entry="cmp_semver",
         prompt="Write `cmp_semver(a, b)` comparing two semantic versions, returning -1, 0 or 1. "
                "Numeric parts compare numerically. A version with a pre-release suffix "
                "(after '-') sorts BEFORE the same version without one: 1.0.0-alpha < 1.0.0. "
                "Pre-release identifiers are dot-separated; numeric ones compare numerically and "
                "rank below non-numeric ones; otherwise compare as strings. A longer pre-release "
                "with an equal prefix sorts higher: 1.0.0-a < 1.0.0-a.1",
         tests="""
assert cmp_semver('1.0.0','1.0.1') == -1
assert cmp_semver('1.0.1','1.0.0') == 1
assert cmp_semver('1.0.0','1.0.0') == 0
assert cmp_semver('1.0.0-alpha','1.0.0') == -1
assert cmp_semver('1.0.0','1.0.0-alpha') == 1
assert cmp_semver('1.0.0-alpha','1.0.0-beta') == -1
assert cmp_semver('1.0.0-a','1.0.0-a.1') == -1
assert cmp_semver('1.0.0-1','1.0.0-alpha') == -1
assert cmp_semver('2.0.0','10.0.0') == -1
"""),
    dict(id="h_cg_tokenize", kind="codegen", entry="tokenize",
         prompt="Write `tokenize(s)` splitting a string on whitespace into a list, except that "
                "double-quoted spans stay as one token with the quotes removed. A backslash "
                "inside quotes escapes the next character literally. Unterminated quotes raise "
                "ValueError. Empty quoted spans produce an empty-string token.",
         tests="""
assert tokenize('a b c') == ['a','b','c']
assert tokenize('a "b c" d') == ['a','b c','d']
assert tokenize('"hello world"') == ['hello world']
assert tokenize('a ""') == ['a','']
assert tokenize(r'"a\\"b"') == ['a"b']
assert tokenize('  spaced   out  ') == ['spaced','out']
try:
    tokenize('a "unterminated'); assert False, 'should raise'
except ValueError: pass
"""),

    # ---------- edit: the naive fix breaks something else ----------
    dict(id="h_ed_cache", kind="edit", entry="Cache",
         prompt="This LRU cache evicts the wrong entry: it drops the oldest *inserted* key rather "
                "than the least recently *used*. A `get` must count as a use. Fix it while keeping "
                "the same public API. Return the complete class.",
         code="""class Cache:
    def __init__(self, cap):
        self.cap = cap
        self.data = {}
    def get(self, k):
        return self.data.get(k)
    def put(self, k, v):
        if k not in self.data and len(self.data) >= self.cap:
            del self.data[next(iter(self.data))]
        self.data[k] = v""",
         tests="""
c = Cache(2)
c.put('a',1); c.put('b',2)
assert c.get('a') == 1          # 'a' is now most-recently-used
c.put('c',3)                    # must evict 'b', not 'a'
assert c.get('b') is None, 'evicted the wrong key'
assert c.get('a') == 1
assert c.get('c') == 3
d = Cache(2)
d.put('x',1); d.put('x',2)      # update must not count as growth
d.put('y',3)
assert d.get('x') == 2 and d.get('y') == 3
"""),
    dict(id="h_ed_paginate", kind="edit", entry="paginate",
         prompt="This paginator is wrong for the final page and for empty input, and it accepts "
                "invalid page sizes silently. Fix it so out-of-range pages return an empty list, "
                "page numbers are 1-based, and per_page < 1 raises ValueError. Keep the signature. "
                "Return the complete function.",
         code="""def paginate(items, page, per_page):
    start = page * per_page
    return items[start:start + per_page]""",
         tests="""
xs = [1,2,3,4,5]
assert paginate(xs,1,2) == [1,2]
assert paginate(xs,2,2) == [3,4]
assert paginate(xs,3,2) == [5]
assert paginate(xs,4,2) == []
assert paginate([],1,3) == []
assert paginate(xs,1,10) == xs
try:
    paginate(xs,1,0); assert False, 'should raise'
except ValueError: pass
"""),

    # ---------- format: multiple simultaneous constraints ----------
    dict(id="h_fmt_constraints", kind="format", check="json_exact",
         expect={"ok": True, "count": 3, "items": ["a", "b", "c"]},
         prompt='Reply with ONLY a JSON object and nothing else. No markdown fences, no prose. '
                'It must have exactly three keys: "ok" (boolean true), "count" (the number 3, '
                'not a string), and "items" (a list of the three lowercase strings a, b and c '
                'in that order).'),
    dict(id="h_fmt_negative", kind="format", check="no_comments",
         prompt="Write a Python function `is_palindrome(s)` returning True if s reads the same "
                "forwards and backwards, ignoring case and any non-alphanumeric characters. "
                "Output ONLY the code. Do not include any comments, docstrings, type hints, "
                "or markdown fences."),

    # ---------- toolcall: knowing when NOT to call ----------
    dict(id="h_tc_restraint", kind="toolcall", want=None,
         prompt="What is 17 + 25? Answer directly.",
         tools=[{"type": "function", "function": {
             "name": "search_web",
             "description": "Search the web for current information about a topic",
             "parameters": {"type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"]}}}]),
    dict(id="h_tc_nested", kind="toolcall", want="create_issue",
         want_args={"title": str, "labels": list},
         prompt="Open a bug report titled 'Crash on empty input' and tag it with both "
                "'bug' and 'urgent'. Use the tools.",
         tools=[{"type": "function", "function": {
             "name": "create_issue",
             "description": "Create a new issue in the tracker",
             "parameters": {"type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "body": {"type": "string"},
                                "labels": {"type": "array", "items": {"type": "string"}}},
                            "required": ["title", "labels"]}}}]),
]
