"""Grading must be strict in both directions.

A grader that accepts a wrong answer inflates every score; one that rejects a
correct answer makes a model look broken. Each branch is exercised with both.
"""
import unittest

from codesift import screen
from codesift.screen import extract_code, grade
from codesift.tasks import AGENT_TASKS, TASKSETS

BASIC = {t["id"]: t for t in TASKSETS["basic"]}
HARD = {t["id"]: t for t in TASKSETS["hard"]}


def reply(content="", tool_calls=None):
    message = {"content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"message": message}


def call(name, arguments):
    return [{"function": {"name": name, "arguments": arguments}}]


class TestExtractCode(unittest.TestCase):
    def test_plain_code(self):
        self.assertEqual(extract_code("def f():\n    pass"), "def f():\n    pass")

    def test_fenced_code(self):
        self.assertEqual(extract_code("prose\n```python\nx = 1\n```\nmore"), "x = 1")

    def test_longest_fence_wins(self):
        text = "```\nshort\n```\ntext\n```python\nlonger block here\n```"
        self.assertEqual(extract_code(text), "longer block here")

    def test_empty(self):
        self.assertEqual(extract_code(""), "")


class TestCodegenGrading(unittest.TestCase):
    def test_correct_solution_passes(self):
        passed, parseable, _, _ = grade(BASIC["cg_version"], reply(
            "```python\ndef parse_version(s):\n"
            "    p = [int(x) for x in s.split('.')][:3]\n"
            "    return tuple(p + [0] * (3 - len(p)))\n```"))
        self.assertTrue(passed)
        self.assertTrue(parseable)

    def test_plausible_but_wrong_solution_fails(self):
        # Omits zero padding, which the assertions require.
        passed, parseable, _, _ = grade(BASIC["cg_version"], reply(
            "```python\ndef parse_version(s):\n"
            "    return tuple(int(x) for x in s.split('.')[:3])\n```"))
        self.assertFalse(passed)
        self.assertTrue(parseable, "code was present, so it was parseable")

    def test_prose_without_code_is_unparseable(self):
        passed, parseable, detail, _ = grade(BASIC["cg_version"], reply(""))
        self.assertFalse(passed)
        self.assertFalse(parseable)
        self.assertIn("no code", detail)


class TestTraceGrading(unittest.TestCase):
    def test_exact_answer(self):
        self.assertTrue(grade(BASIC["tr_closure"], reply("[2, 2, 2]"))[0])

    def test_whitespace_is_ignored(self):
        self.assertTrue(grade(BASIC["tr_closure"], reply("[2,2,2]"))[0])

    def test_wrong_answer(self):
        self.assertFalse(grade(BASIC["tr_closure"], reply("[0, 1, 2]"))[0])


class TestToolCallGrading(unittest.TestCase):
    def test_correct_call(self):
        self.assertTrue(grade(BASIC["tc_single"],
                              reply("", call("search_files", {"pattern": "config"})))[0])

    def test_wrong_tool(self):
        passed, parseable, _, _ = grade(BASIC["tc_choose"],
                                     reply("", call("search_files", {"pattern": "x"})))
        self.assertFalse(passed)
        self.assertTrue(parseable, "a valid call to the wrong tool is still parseable")

    def test_missing_call_is_unparseable(self):
        passed, parseable, _, _ = grade(BASIC["tc_single"], reply("I will look for that"))
        self.assertFalse(passed)
        self.assertFalse(parseable)

    def test_restraint_answering_directly(self):
        self.assertTrue(grade(HARD["h_tc_restraint"], reply("42"))[0])

    def test_restraint_calling_a_tool_fails(self):
        passed, _, detail, _ = grade(HARD["h_tc_restraint"],
                                  reply("", call("search_web", {"query": "17+25"})))
        self.assertFalse(passed)
        self.assertIn("search_web", detail)

    def test_argument_types_are_checked(self):
        good = call("create_issue", {"title": "x", "labels": ["bug", "urgent"]})
        self.assertTrue(grade(HARD["h_tc_nested"], reply("", good))[0])
        wrong_type = call("create_issue", {"title": "x", "labels": "bug"})
        self.assertFalse(grade(HARD["h_tc_nested"], reply("", wrong_type))[0])
        missing = call("create_issue", {"title": "x"})
        self.assertFalse(grade(HARD["h_tc_nested"], reply("", missing))[0])

    def test_arguments_supplied_as_json_string(self):
        encoded = call("create_issue", '{"title": "x", "labels": ["bug"]}')
        self.assertTrue(grade(HARD["h_tc_nested"], reply("", encoded))[0])

    def test_arguments_that_are_not_json(self):
        passed, parseable, _, _ = grade(HARD["h_tc_nested"], reply("", call("create_issue", "{oops")))
        self.assertFalse(passed)
        self.assertFalse(parseable)


class TestFormatGrading(unittest.TestCase):
    def test_json_exact(self):
        self.assertTrue(grade(BASIC["fmt_jsononly"],
                              reply('{"language":"python","year":1991}'))[0])

    def test_json_with_wrong_value(self):
        self.assertFalse(grade(BASIC["fmt_jsononly"],
                               reply('{"language":"python","year":"1991"}'))[0])

    def test_unparseable_json_is_flagged(self):
        passed, parseable, _, _ = grade(BASIC["fmt_jsononly"], reply("Sure! Here you go."))
        self.assertFalse(passed)
        self.assertFalse(parseable)

    def test_one_word(self):
        self.assertTrue(grade(BASIC["fmt_oneword"], reply("def"))[0])
        self.assertFalse(grade(BASIC["fmt_oneword"], reply("the def keyword"))[0])

    def test_bare_code_rejects_fences(self):
        passed, parseable, detail, _ = grade(
            BASIC["fmt_barecode"], reply("```python\ndef add(a, b):\n    return a + b\n```"))
        self.assertFalse(passed)
        self.assertFalse(parseable)
        self.assertIn("fences", detail)

    def test_bare_code_accepts_clean_code(self):
        self.assertTrue(grade(BASIC["fmt_barecode"],
                              reply("def add(a, b):\n    return a + b"))[0])

    def test_no_comments_rejects_each_violation(self):
        body = ("def is_palindrome(s):\n"
                "    t = [c.lower() for c in s if c.isalnum()]\n"
                "    return t == t[::-1]")
        self.assertTrue(grade(HARD["h_fmt_negative"], reply(body))[0])
        for bad, expected in (
                (body.replace("def", "# note\ndef"), "comment"),
                ('def is_palindrome(s):\n    """doc"""\n    return True', "docstring"),
                (body.replace("def is_palindrome(s):",
                              "def is_palindrome(s: str) -> bool:"), "type hints"),
                ("```python\n" + body + "\n```", "fences")):
            with self.subTest(violation=expected):
                passed, _, detail, _ = grade(HARD["h_fmt_negative"], reply(bad))
                self.assertFalse(passed)
                self.assertIn(expected, detail)

    def test_no_comments_still_requires_correctness(self):
        # Clean, but ignores punctuation and case as the prompt requires.
        self.assertFalse(grade(HARD["h_fmt_negative"],
                               reply("def is_palindrome(s):\n    return s == s[::-1]"))[0])


class TestUnknownKind(unittest.TestCase):
    def test_unrecognised_kind_never_passes(self):
        passed, parseable, detail, _ = grade({"kind": "nonsense"}, reply("anything"))
        self.assertFalse(passed)
        self.assertFalse(parseable)
        self.assertIn("unknown", detail)


if __name__ == "__main__":
    unittest.main()


class TestCodeExtraction(unittest.TestCase):
    """Which fenced block is graded.

    Taking the longest block graded whichever happened to be longer, so a model
    that answered correctly and then illustrated it with a few example calls was
    graded on the examples and failed for a NameError. That is a penalty for
    documenting an answer, recorded as incorrectness.
    """

    def test_examples_after_the_answer_are_not_graded_instead_of_it(self):
        reply = (
            "```python\n"
            "def parse_version(s):\n"
            "    parts = s.split('.')\n"
            "    return tuple([int(p) for p in parts[:3]] + [0] * (3 - len(parts)))\n"
            "```\n\n**Examples:**\n```python\n"
            "parse_version('1.2.3')  # (1, 2, 3)\n"
            "parse_version('1.2')    # (1, 2, 0)\n"
            "parse_version('2')      # (2, 0, 0)\n"
            "parse_version('1.2.3.4')  # extra parts ignored\n"
            "```")
        code = screen.extract_code(reply)
        self.assertIn("def parse_version", code)
        passed, _ = screen.run_tests(code, "assert parse_version('1.2') == (1, 2, 0)\n")
        self.assertTrue(passed, "a correct answer must not fail on its own examples")

    def test_an_answer_split_across_blocks_is_kept_whole(self):
        reply = ("```python\ndef helper(x):\n    return x + 1\n```\n"
                 "and then\n"
                 "```python\ndef main(x):\n    return helper(x) * 2\n```")
        passed, detail = screen.run_tests(screen.extract_code(reply, "main"),
                                          "assert main(1) == 4\n")
        self.assertTrue(passed, detail)

    def test_a_demonstration_that_raises_is_not_run(self):
        # Observed: a correct retry decorator, illustrated by dividing by zero on
        # purpose to show the retry working. Executing the illustration alongside
        # the tests graded the model on its own example.
        reply = ("```python\ndef double(x):\n    return x * 2\n```\n"
                 "For example:\n"
                 "```python\n@staticmethod\ndef unused():\n    pass\n"
                 "print(1 / 0)\n```")
        passed, detail = screen.run_tests(screen.extract_code(reply, "double"),
                                          "assert double(3) == 6\n")
        self.assertTrue(passed, detail)

    def test_a_withdrawn_draft_does_not_outrank_the_correction(self):
        # Observed: four successive attempts at the same function, the first two
        # wrong. Grading the first scored a version the model had already replaced.
        reply = ("First attempt:\n```python\ndef chunk(xs, n):\n"
                 "    return [xs[i:i+n] for i in range(0, len(xs) - n + 1, n)]\n```\n"
                 "That drops the tail. Corrected:\n```python\ndef chunk(xs, n):\n"
                 "    return [xs[i:i+n] for i in range(0, len(xs), n)]\n```")
        passed, detail = screen.run_tests(screen.extract_code(reply, "chunk"),
                                          "assert chunk([1,2,3,4,5], 2) == [[1,2],[3,4],[5]]\n")
        self.assertTrue(passed, detail)

    def test_an_import_beside_a_draft_survives_into_the_correction(self):
        # The revision often omits the import, because the model considers it
        # already stated. Dropping the superseded block dropped the import with it.
        reply = ("```python\nfrom functools import wraps\n\ndef deco(f):\n"
                 "    return f\n```\n"
                 "Better, preserving the name:\n"
                 "```python\ndef deco(f):\n    @wraps(f)\n"
                 "    def inner(*a):\n        return f(*a)\n    return inner\n```")
        code = screen.extract_code(reply, "deco")
        passed, detail = screen.run_tests(
            code, "def g():\n    return 1\nassert deco(g)().__class__ is int\n")
        self.assertTrue(passed, detail)

    def test_a_main_guard_is_not_executed(self):
        reply = ("```python\ndef f():\n    return 5\n\n"
                 "if __name__ == '__main__':\n    raise SystemExit('demo')\n```")
        passed, detail = screen.run_tests(screen.extract_code(reply, "f"),
                                          "assert f() == 5\n")
        self.assertTrue(passed, detail)

    def test_code_that_will_not_parse_is_handed_to_the_grader_as_written(self):
        # The syntax error is the finding; hiding it behind a fallback would
        # report something else instead.
        reply = "```python\ndef f(:\n    return 1\n```"
        self.assertIn("def f(:", screen.extract_code(reply, "f"))

    def test_a_single_block_is_unchanged(self):
        reply = "Here you go:\n```python\ndef f():\n    return 7\n```\nHope that helps."
        self.assertEqual(screen.extract_code(reply), "def f():\n    return 7")

    def test_a_reply_of_declarations_keeps_them_and_drops_the_output(self):
        reply = "```python\nTABLE = {'a': 1}\n```\nwhich gives\n```python\nprint(TABLE)\n```"
        code = screen.extract_code(reply)
        self.assertIn("TABLE", code)
        self.assertNotIn("print", code)

    def test_an_unfenced_reply_is_taken_as_written(self):
        self.assertEqual(screen.extract_code("def f():\n    return 1"),
                         "def f():\n    return 1")
        self.assertEqual(screen.extract_code(""), "")

    def test_classes_and_async_definitions_count_as_the_answer(self):
        for head in ("class Thing:\n    pass", "async def go():\n    return 1"):
            with self.subTest(head=head.split()[0]):
                reply = f"```python\n{head}\n```\n```python\nprint('a much longer usage line')\n```"
                self.assertIn(head.split()[0], screen.extract_code(reply))


class TestPartialCredit(unittest.TestCase):
    """A near miss is a near miss, not a zero.

    Across one full sweep the median failing answer satisfied three quarters of
    the assertions it was given, and scored what code that does not run scored.
    One task's whole discriminating power sat on a single edge case because of it.
    """

    FLATTEN = BASIC["cg_flatten"]
    CORRECT = ("def flatten(d, sep='.'):\n"
               "    out = {}\n"
               "    for k, v in d.items():\n"
               "        if isinstance(v, dict):\n"
               "            for k2, v2 in flatten(v, sep).items():\n"
               "                out[k + sep + k2] = v2\n"
               "        else:\n"
               "            out[k] = v\n"
               "    return out\n")

    def score(self, code):
        return grade(self.FLATTEN, reply(f"```python\n{code}\n```"))

    def test_a_correct_answer_scores_everything(self):
        passed, _, detail, score = self.score(self.CORRECT)
        self.assertTrue(passed)
        self.assertEqual(score, 1.0)
        self.assertEqual(detail, "ok")

    def test_missing_one_edge_case_is_a_penalty_not_a_zero(self):
        # Keeps an empty nested dict as a key, which the prompt forbids. Every
        # other assertion still holds.
        keeps_empty = self.CORRECT.replace(
            "        if isinstance(v, dict):",
            "        if isinstance(v, dict) and not v:\n"
            "            out[k] = v\n"
            "        elif isinstance(v, dict):")
        passed, _, detail, score = self.score(keeps_empty)
        self.assertFalse(passed, "it is still a failure")
        self.assertEqual(score, 0.8, "four of the five checks were met")
        self.assertIn("4/5", detail)

    def test_code_that_does_not_run_scores_nothing(self):
        passed, _, _, score = self.score("def flatten(d, sep='.'):\n    return 1 / 0\n")
        self.assertFalse(passed)
        self.assertEqual(score, 0.0)

    def test_a_near_miss_outscores_a_broken_answer(self):
        near = self.score(self.CORRECT.replace("out[k] = v", "out[k] = v  # noqa"))[3]
        broken = self.score("def flatten(d, sep='.'):\n    return 1 / 0\n")[3]
        self.assertGreater(near, broken)

    def test_an_empty_reply_still_scores_nothing(self):
        passed, parseable, detail, score = grade(self.FLATTEN, reply(""))
        self.assertEqual((passed, parseable, score), (False, False, 0.0))
        self.assertEqual(detail, "no code emitted")

    def test_setup_that_raises_ends_the_attempt(self):
        # h_ed_cache builds a cache before examining it; if the constructor throws
        # there is nothing left to check, and the checks after it are not credited.
        passed, _, _, score = grade(HARD["h_ed_cache"],
                                    reply("```python\nclass Cache:\n"
                                          "    def __init__(self, n):\n"
                                          "        raise RuntimeError('no')\n```"))
        self.assertFalse(passed)
        self.assertEqual(score, 0.0)

    def test_every_other_kind_stays_all_or_nothing(self):
        for task, good, bad in (
                (BASIC["tr_closure"], "[2, 2, 2]", "[0, 1, 2]"),
                (BASIC["fmt_oneword"], "def", "a function"),
        ):
            with self.subTest(task=task["id"]):
                self.assertEqual(grade(task, reply(good))[3], 1.0)
                self.assertEqual(grade(task, reply(bad))[3], 0.0)
