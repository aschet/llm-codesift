# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""Tasks of kind `format`: a strict output constraint -> checked, nothing executed"""

TASKS = [
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
    dict(id="fmt_constraints", kind="format", check="json_exact",
         expect={"ok": True, "count": 3, "items": ["a", "b", "c"]},
         prompt='Reply with ONLY a JSON object and nothing else. No markdown fences, no prose. '
                'It must have exactly three keys: "ok" (boolean true), "count" (the number 3, '
                'not a string), and "items" (a list of the three lowercase strings a, b and c '
                'in that order).'),
    dict(id="fmt_negative", kind="format", check="no_comments",
         prompt="Write a Python function `is_palindrome(s)` returning True if s reads the same "
                "forwards and backwards, ignoring case and any non-alphanumeric characters. "
                "Output ONLY the code. Do not include any comments, docstrings, type hints, "
                "or markdown fences."),
]
