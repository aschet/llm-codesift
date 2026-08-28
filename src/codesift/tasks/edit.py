# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""Tasks of kind `edit`: existing code and an instruction -> the full rewrite -> assertions"""

TASKS = [
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
    dict(id="ed_cache", kind="edit", entry="Cache",
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
    dict(id="ed_paginate", kind="edit", entry="paginate",
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
    dict(id="ed_window", kind="edit", entry="rolling_max",
         prompt="This should return the maximum of every window of `size` consecutive "
                "items, left to right. It misses the last window. Fix it and return the "
                "full function.",
         code="""def rolling_max(items, size):
    out = []
    for i in range(len(items) - size):
        out.append(max(items[i:i + size]))
    return out""",
         tests="""
assert rolling_max([1, 3, 2, 5], 2) == [3, 3, 5]
assert rolling_max([4, 4, 4], 3) == [4]
assert rolling_max([1, 2], 2) == [2]
assert rolling_max([7], 1) == [7]
assert rolling_max([], 1) == []
"""),
    dict(id="ed_keycollide", kind="edit", entry="memoise",
         prompt="This caches results per argument list, but two different calls can collide "
                "and the second gets the first's answer. Fix the collision, keep the caching, "
                "and return the full function.",
         code="""def memoise(fn):
    cache = {}

    def wrapper(*args):
        key = ''.join(str(a) for a in args)
        if key not in cache:
            cache[key] = fn(*args)
        return cache[key]

    return wrapper""",
         tests="""
calls = []

@memoise
def join(*parts):
    calls.append(parts)
    return '-'.join(str(p) for p in parts)

assert join('a', 'bc') == 'a-bc'
assert join('ab', 'c') == 'ab-c'
assert join(1, 23) == '1-23'
assert join(12, 3) == '12-3'
assert join('a', 'bc') == 'a-bc'
assert len(calls) == 4, f'cached {4 - len(calls)} calls wrongly'
"""),
    dict(id="ed_swallow", kind="edit", entry="load_all",
         prompt="This hides every failure, including the ones that mean the caller passed "
                "something unusable. Change it so a ValueError from the parser is skipped "
                "and recorded in the returned list of errors, while every other exception "
                "propagates to the caller. Return the full function.",
         code="""def load_all(rows, parse):
    out, errors = [], []
    for row in rows:
        try:
            out.append(parse(row))
        except Exception as exc:
            errors.append(str(exc))
    return out, errors""",
         tests="""
def parse(row):
    if row == 'bad':
        raise ValueError('bad row')
    if row == 'fatal':
        raise TypeError('unusable input')
    return int(row)

out, errors = load_all(['1', 'bad', '2'], parse)
assert out == [1, 2]
assert len(errors) == 1 and 'bad row' in errors[0]
propagated = False
try:
    load_all(['1', 'fatal'], parse)
except TypeError:
    propagated = True
assert propagated, 'anything but a ValueError must reach the caller'
"""),
    dict(id="ed_earlyreturn", kind="edit", entry="find_all",
         prompt="This should return every index at which the predicate holds, but it stops "
                "at the first. Fix it and return the full function.",
         code="""def find_all(items, predicate):
    for i, item in enumerate(items):
        if predicate(item):
            return [i]
    return []""",
         tests="""
assert find_all([1, 2, 3, 4], lambda x: x % 2 == 0) == [1, 3]
assert find_all([1, 3], lambda x: x % 2 == 0) == []
assert find_all([2], lambda x: True) == [0]
assert find_all([], lambda x: True) == []
assert find_all([5, 5, 5], lambda x: x == 5) == [0, 1, 2]
"""),
]
