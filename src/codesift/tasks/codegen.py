# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""Tasks of kind `codegen`: a specification -> code -> run assertions against it"""

TASKS = [
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
    dict(id="cg_roman", kind="codegen", entry="to_roman",
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
    dict(id="cg_semver", kind="codegen", entry="cmp_semver",
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
    dict(id="cg_tokenize", kind="codegen", entry="tokenize",
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
    dict(id="cg_topo", kind="codegen", entry="topo_sort",
         prompt="Write `topo_sort(graph)` where graph maps a node to a list of nodes it "
                "must come before. Return a list of all nodes in a valid order. Return None "
                "if the graph has a cycle. Nodes with no edges may appear anywhere.",
         tests="""
order = topo_sort({'a': ['b'], 'b': ['c'], 'c': []})
assert order.index('a') < order.index('b') < order.index('c')
assert sorted(topo_sort({'a': [], 'b': []})) == ['a', 'b']
assert topo_sort({'a': ['b'], 'b': ['a']}) is None
assert topo_sort({}) == []
deep = topo_sort({'x': ['y', 'z'], 'y': ['z'], 'z': []})
assert deep.index('x') < deep.index('y') < deep.index('z')
"""),
    dict(id="cg_pathnorm", kind="codegen", entry="normalise_path",
         prompt="Write `normalise_path(path)` for POSIX-style paths. Collapse repeated "
                "slashes, resolve '.' and '..', and drop any trailing slash. An absolute "
                "path keeps its leading slash and cannot rise above the root. A relative "
                "path may keep leading '..' segments. The empty result of a relative path "
                "is '.'.",
         tests="""
assert normalise_path('/a//b/./c') == '/a/b/c'
assert normalise_path('/a/b/../c') == '/a/c'
assert normalise_path('/../..') == '/'
assert normalise_path('a/../../b') == '../b'
assert normalise_path('a/b/') == 'a/b'
assert normalise_path('./a/.') == 'a'
assert normalise_path('a/..') == '.'
assert normalise_path('/') == '/'
"""),
    dict(id="cg_csvline", kind="codegen", entry="split_csv",
         prompt="Write `split_csv(line)` returning the fields of one CSV line. Fields are "
                "comma separated. A field may be wrapped in double quotes, in which case it "
                "may contain commas, and a doubled quote inside it means one literal quote. "
                "Quotes are removed from the result. Do not use the csv module.",
         tests="""
assert split_csv('a,b,c') == ['a', 'b', 'c']
assert split_csv('a,"b,c",d') == ['a', 'b,c', 'd']
assert split_csv('"he said ""hi"" twice",x') == ['he said "hi" twice', 'x']
assert split_csv('a,,b') == ['a', '', 'b']
assert split_csv('""') == ['']
assert split_csv('a,"b"') == ['a', 'b']
"""),
    dict(id="cg_rle", kind="codegen", entry="rle_encode",
         prompt="Write `rle_encode(text)` and `rle_decode(code)` so that decoding an "
                "encoded string returns the original exactly. Encoding must shorten long "
                "runs of a repeated character. The text may contain digits and punctuation, "
                "so an encoding that appends a bare count is ambiguous; the scheme is yours "
                "to choose as long as it round-trips.",
         tests="""
for sample in ('aaabbc', 'abc', '', 'a1112', 'zzzzzzzzzzzz', '111'):
    assert rle_decode(rle_encode(sample)) == sample, sample
assert len(rle_encode('aaaaaaaa')) < len('aaaaaaaa')
"""),
    dict(id="cg_bsearch", kind="codegen", entry="insert_point",
         prompt="Write `insert_point(sorted_items, value)` returning the index where value "
                "would be inserted to keep the list sorted, before any equal items. Do not "
                "use the bisect module. It must not scan the list item by item: the list may "
                "hold a million entries.",
         tests="""
assert insert_point([1, 3, 5], 0) == 0
assert insert_point([1, 3, 5], 6) == 3
assert insert_point([1, 3, 3, 3, 5], 3) == 1
assert insert_point([], 1) == 0
assert insert_point([2, 2, 2], 2) == 0

class Counted:
    def __init__(self, n):
        self.n, self.reads = n, 0

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        self.reads += 1
        return i

big = Counted(1000000)
assert insert_point(big, 999999) == 999999
assert big.reads < 100, f'read {big.reads} entries; the list may hold a million'
"""),
    dict(id="cg_expr", kind="codegen", entry="evaluate",
         prompt="Write `evaluate(expression)` for arithmetic over integers with + - * / and "
                "parentheses, honouring the usual precedence. Division is float division. "
                "Whitespace is insignificant. Do not use eval or exec.",
         tests="""
assert evaluate('1+2*3') == 7
assert evaluate('(1+2)*3') == 9
assert evaluate('10/4') == 2.5
assert evaluate('2*(3+4)-5') == 9
assert evaluate(' 8 / 2 / 2 ') == 2.0
assert evaluate('1+2*3-4/2') == 5.0
"""),
    dict(id="cg_wrap", kind="codegen", entry="wrap_text",
         prompt="Write `wrap_text(text, width)` returning a list of lines. Break on spaces "
                "only, never mid-word, and put as many words on a line as fit within width. "
                "A word longer than width goes on a line of its own, unbroken. Runs of "
                "whitespace collapse to one space and the result has no trailing spaces.",
         tests="""
assert wrap_text('the quick brown fox', 10) == ['the quick', 'brown fox']
assert wrap_text('a  b   c', 5) == ['a b c']
assert wrap_text('extraordinarily long', 5) == ['extraordinarily', 'long']
assert wrap_text('', 10) == []
assert wrap_text('one two', 7) == ['one two']
assert all(not l.endswith(' ') for l in wrap_text('aa bb cc dd', 5))
"""),
    dict(id="cg_duration", kind="codegen", entry="parse_duration",
         prompt="Write `parse_duration(text)` turning a duration into whole seconds. The "
                "text is a sequence of a number and a unit, such as '1h30m' or '2d' or "
                "'45s'. Units are d, h, m and s. Raise ValueError on anything else, "
                "including an empty string and a bare number.",
         tests="""
assert parse_duration('45s') == 45
assert parse_duration('1h30m') == 5400
assert parse_duration('2d') == 172800
assert parse_duration('1h1m1s') == 3661
rejected = []
for bad in ('', '10', 'h', '1x', '1h 30m', 'abc'):
    try:
        parse_duration(bad)
    except ValueError:
        rejected.append(bad)
assert len(rejected) == 6, f'accepted {6 - len(rejected)} malformed durations'
"""),
]
