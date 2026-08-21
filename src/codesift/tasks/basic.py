"""
Screening tasks. Small on purpose -- this filters obvious losers, it does not rank.

Kinds:
  codegen   prompt -> code -> run asserts
  edit      existing code + instruction -> full rewritten code -> run asserts
  format    strict output constraint -> regex check (no execution)
  toolcall  tools offered -> must emit a valid call with the right name/args
  trace     predict what code prints -> string match (comprehension, no generation)
"""

TASKS = [
    # ---------- codegen ----------
    dict(
        id="cg_version", kind="codegen", entry="parse_version",
        prompt="Write a Python function `parse_version(s)` that turns a version string into a "
               "tuple of ints. '1.2.3' -> (1,2,3). Missing parts default to 0, so '1.2' -> (1,2,0) "
               "and '2' -> (2,0,0). Extra parts beyond three are ignored.",
        tests="""
assert parse_version('1.2.3') == (1,2,3)
assert parse_version('1.2') == (1,2,0)
assert parse_version('2') == (2,0,0)
assert parse_version('1.2.3.4') == (1,2,3)
""",
    ),
    dict(
        id="cg_intervals", kind="codegen", entry="merge_intervals",
        prompt="Write a Python function `merge_intervals(intervals)` taking a list of [start, end] "
               "pairs and returning them merged and sorted by start. Touching intervals like "
               "[1,2] and [2,3] merge into [1,3].",
        tests="""
assert merge_intervals([[1,3],[2,6],[8,10]]) == [[1,6],[8,10]]
assert merge_intervals([[1,2],[2,3]]) == [[1,3]]
assert merge_intervals([]) == []
assert merge_intervals([[5,6],[1,2]]) == [[1,2],[5,6]]
""",
    ),
    dict(
        id="cg_flatten", kind="codegen", entry="flatten",
        prompt="Write a Python function `flatten(d, sep='.')` that flattens a nested dict into a "
               "single level, joining keys with sep. {'a':{'b':1}} -> {'a.b':1}. Non-dict values "
               "are kept as-is. Empty dicts produce no keys.",
        tests="""
assert flatten({'a':{'b':1}}) == {'a.b':1}
assert flatten({'a':1,'b':{'c':{'d':2}}}) == {'a':1,'b.c.d':2}
assert flatten({}) == {}
assert flatten({'a':{}}) == {}
assert flatten({'a':{'b':1}}, sep='/') == {'a/b':1}
""",
    ),
    dict(
        id="cg_retry", kind="codegen", entry="retry",
        prompt="Write a Python decorator `retry(times)` that retries the wrapped function when it "
               "raises, up to `times` total attempts, re-raising the last exception if all fail. "
               "It must return the function's value on success and preserve __name__.",
        tests="""
calls = []
@retry(3)
def flaky():
    calls.append(1)
    if len(calls) < 3: raise ValueError('nope')
    return 'ok'
assert flaky() == 'ok'
assert len(calls) == 3
assert flaky.__name__ == 'flaky'

@retry(2)
def always():
    raise KeyError('bad')
try:
    always(); assert False, 'should have raised'
except KeyError: pass
""",
    ),

    # ---------- edit ----------
    dict(
        id="ed_offbyone", kind="edit", entry="chunk",
        prompt="The function below is supposed to split a list into chunks of size n, but the last "
               "partial chunk is being dropped. Fix it. Return the complete corrected function.",
        code="""def chunk(items, n):
    out = []
    for i in range(0, len(items) - n + 1, n):
        out.append(items[i:i+n])
    return out""",
        tests="""
assert chunk([1,2,3,4,5], 2) == [[1,2],[3,4],[5]]
assert chunk([1,2,3,4], 2) == [[1,2],[3,4]]
assert chunk([], 3) == []
assert chunk([1], 5) == [[1]]
""",
    ),
    dict(
        id="ed_addparam", kind="edit", entry="normalize",
        prompt="Add an optional `strip_punct=False` parameter to the function below. When True, it "
               "should also remove the characters . , ! and ? from the result. Existing behaviour "
               "must be unchanged when the parameter is not passed. Return the complete function.",
        code="""def normalize(text):
    return ' '.join(text.lower().split())""",
        tests="""
assert normalize('  Hello   World ') == 'hello world'
assert normalize('Hi, there!') == 'hi, there!'
assert normalize('Hi, there!', strip_punct=True) == 'hi there'
assert normalize('A. B? C!', strip_punct=True) == 'a b c'
""",
    ),
    dict(
        id="ed_method", kind="edit", entry="Counter2",
        prompt="Add a `most_common(k)` method to the class below, returning the k highest-count "
               "items as (item, count) tuples sorted by count descending, ties broken by item "
               "ascending. Return the complete class.",
        code="""class Counter2:
    def __init__(self):
        self.counts = {}
    def add(self, item):
        self.counts[item] = self.counts.get(item, 0) + 1""",
        tests="""
c = Counter2()
for x in ['a','b','a','c','b','a']: c.add(x)
assert c.most_common(2) == [('a',3),('b',2)]
assert c.most_common(1) == [('a',3)]
d = Counter2()
d.add('z'); d.add('y')
assert d.most_common(2) == [('y',1),('z',1)]
""",
    ),

    # ---------- format compliance ----------
    dict(
        id="fmt_jsononly", kind="format",
        prompt='Reply with ONLY a JSON object, no prose, no markdown fences. It must have exactly '
               'the keys "language" and "year", with values "python" and 1991.',
        check="json_exact", expect={"language": "python", "year": 1991},
    ),
    dict(
        id="fmt_barecode", kind="format",
        prompt="Output ONLY the Python source of a function `add(a, b)` returning a+b. "
               "No markdown fences, no explanation, no example usage. Just the two lines of code.",
        check="bare_code",
    ),
    dict(
        id="fmt_oneword", kind="format",
        prompt="What keyword declares a function in Python? Answer with exactly one word, "
               "lowercase, nothing else.",
        check="one_word", expect="def",
    ),

    # ---------- tool calling ----------
    dict(
        id="tc_single", kind="toolcall", want="search_files",
        prompt="Find every file in the repository whose name contains 'config'. Use the tools.",
        tools=[{
            "type": "function",
            "function": {
                "name": "search_files",
                "description": "Search for files by a substring of their name",
                "parameters": {
                    "type": "object",
                    "properties": {"pattern": {"type": "string", "description": "substring to match"}},
                    "required": ["pattern"],
                },
            },
        }],
    ),
    dict(
        id="tc_choose", kind="toolcall", want="run_tests",
        prompt="The test suite is failing and I need to see the output. Use the tools.",
        tools=[
            {"type": "function", "function": {
                "name": "search_files",
                "description": "Search for files by a substring of their name",
                "parameters": {"type": "object",
                               "properties": {"pattern": {"type": "string"}},
                               "required": ["pattern"]}}},
            {"type": "function", "function": {
                "name": "run_tests",
                "description": "Run the project's test suite and return the output",
                "parameters": {"type": "object",
                               "properties": {"path": {"type": "string"}},
                               "required": []}}},
        ],
    ),

    # ---------- comprehension ----------
    dict(
        id="tr_mutate", kind="trace", expect="[1, 2, 2, 4]",
        prompt="What does this print? Reply with only the printed value.\n\n"
               "def f(xs, extra=[]):\n"
               "    extra.append(len(xs))\n"
               "    return xs + extra\n"
               "print(f([1,2], ) + [4])" ,
    ),
    dict(
        id="tr_closure", kind="trace", expect="[2, 2, 2]",
        prompt="What does this print? Reply with only the printed value.\n\n"
               "fs = [lambda: i for i in range(3)]\n"
               "print([f() for f in fs])",
    ),
]
