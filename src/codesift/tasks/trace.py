# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""Tasks of kind `trace`: predict what code prints -> string match, nothing generated"""

TASKS = [
    dict(
        id="tr_mutate", kind="trace", expect="[1, 2, 2, 4]",
        prompt="What does this print? Reply with only the printed value.\n\n"
               "def f(xs, extra=[]):\n"
               "    extra.append(len(xs))\n"
               "    return xs + extra\n"
               "print(f([1,2], ) + [4])",
    ),
    dict(
        id="tr_closure", kind="trace", expect="[2, 2, 2]",
        prompt="What does this print? Reply with only the printed value.\n\n"
               "fs = [lambda: i for i in range(3)]\n"
               "print([f() for f in fs])",
    ),
    dict(id="tr_finally", kind="trace", expect="2",
         prompt="What does this print? Reply with only the printed value.\n\n"
                "def f():\n    try:\n        return 1\n    finally:\n        return 2\n"
                "print(f())"),
    dict(id="tr_zipself", kind="trace", expect="[(1, 2), (3, 4)]",
         prompt="What does this print? Reply with only the printed value.\n\n"
                "it = iter([1,2,3,4])\nprint(list(zip(it, it)))"),
    dict(id="tr_genexhaust", kind="trace", expect="[0, 1, 2] []",
         prompt="What does this print? Reply with only the printed value.\n\n"
                "def gen():\n    for i in range(3):\n        yield i\n"
                "g = gen()\nprint(list(g), list(g))"),
    dict(id="tr_classattr", kind="trace", expect="[1] [1]",
         prompt="What does this print? Reply with only the printed value.\n\n"
                "class A:\n    items = []\n"
                "a, b = A(), A()\na.items.append(1)\nprint(a.items, b.items)"),
    dict(id="tr_chained", kind="trace", expect="False",
         prompt="What does this print? Reply with only the printed value.\n\n"
                "print(1 in [1,2] == True)"),
    dict(id="tr_dupkey", kind="trace", expect="{'a': 2}",
         prompt="What does this print? Reply with only the printed value.\n\n"
                "print({k: v for k, v in [('a',1),('a',2)]})"),
    dict(id="tr_tupleaug", kind="trace", expect="(1, [2, 3])",
         prompt="What does this print? Reply with only the printed value.\n\n"
                "t = (1, [2])\n"
                "try:\n    t[1] += [3]\nexcept TypeError:\n    pass\n"
                "print(t)"),
    dict(id="tr_boolkey", kind="trace", expect="{True: 'c'}",
         prompt="What does this print? Reply with only the printed value.\n\n"
                "print({True: 'a', 1: 'b', 1.0: 'c'})"),
    dict(id="tr_listmul", kind="trace", expect="[[9], [9], [9]]",
         prompt="What does this print? Reply with only the printed value.\n\n"
                "grid = [[0]] * 3\ngrid[0][0] = 9\nprint(grid)"),
    dict(id="tr_roundeven", kind="trace", expect="0,2,2",
         prompt="What does this print? Reply with only the printed value.\n\n"
                "print(f'{round(0.5)},{round(1.5)},{round(2.5)}')"),
    dict(id="tr_exceptdel", kind="trace", expect="gone",
         prompt="What does this print? Reply with only the printed value.\n\n"
                "e = 'kept'\n"
                "try:\n    raise ValueError('boom')\nexcept ValueError as e:\n    pass\n"
                "print(locals().get('e', 'gone'))"),
    dict(id="tr_sortstable", kind="trace", expect="[('c', 0), ('b', 1), ('a', 1)]",
         prompt="What does this print? Reply with only the printed value.\n\n"
                "pairs = [('b', 1), ('a', 1), ('c', 0)]\n"
                "print(sorted(pairs, key=lambda p: p[1]))"),
    dict(id="tr_defaulteval", kind="trace", expect="1",
         prompt="What does this print? Reply with only the printed value.\n\n"
                "i = 1\ndef f(x=i):\n    return x\ni = 99\nprint(f())"),
    dict(id="tr_slicestep", kind="trace", expect="[9, 1, 9, 3, 9, 5]",
         prompt="What does this print? Reply with only the printed value.\n\n"
                "xs = [0, 1, 2, 3, 4, 5]\nxs[::2] = [9, 9, 9]\nprint(xs)"),
]
