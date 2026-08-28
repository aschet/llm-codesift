# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""A tool task may answer a lookup and ask again.

Grading the first call of a single turn cannot tell a model that went to the
wrong tool from one that looked before it acted. A task carrying `turns` serves
the lookup and asks again, so the two are different results.
"""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from codesift import screen
from codesift.config import Config


def call(name, **arguments):
    return {"message": {"content": "", "tool_calls":
                        [{"function": {"name": name, "arguments": arguments}}]},
            "_wall": 1.0, "eval_count": 10, "eval_duration": 1_000_000_000}


class Client:
    """Replies in the order given, keeping what it was asked each time."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.asked = []

    def chat(self, model, prompt, **kwargs):
        self.asked.append(prompt)
        return dict(self.replies.pop(0))

    def unload(self, model):
        pass


TASK = dict(id="t_choose", kind="toolcall", want="run_tests", turns=2,
            results={"search_files": ["tests/test_api.py"]},
            prompt="The suite is failing.", tools=[])
DIRECT = dict(id="t_one", kind="toolcall", want="run_tests",
              prompt="The suite is failing.", tools=[])


class TestRunTurns(unittest.TestCase):
    def turns(self, task, *replies):
        client = Client(*replies)
        resp, used = screen.run_turns(client, "m", task, ctx=4096, num_predict=64)
        return client, resp, used

    def test_a_direct_call_ends_the_exchange(self):
        client, _, used = self.turns(TASK, call("run_tests"))
        self.assertEqual(used, 1)
        self.assertEqual(len(client.asked), 1)

    def test_a_lookup_is_answered_and_the_model_asked_again(self):
        client, resp, used = self.turns(TASK, call("search_files", pattern="test"),
                                        call("run_tests", path="tests"))
        self.assertEqual(used, 2)
        second = client.asked[1]
        self.assertIsInstance(second, list, "the second turn carries the exchange")
        self.assertEqual(second[-1]["role"], "tool")
        self.assertEqual(json.loads(second[-1]["content"]), ["tests/test_api.py"])
        name = resp["message"]["tool_calls"][0]["function"]["name"]
        self.assertEqual(name, "run_tests")

    def test_a_task_without_turns_is_asked_once(self):
        client, _, used = self.turns(DIRECT, call("search_files", pattern="test"))
        self.assertEqual((used, len(client.asked)), (1, 1))

    def test_a_call_to_a_tool_with_no_stored_result_ends_it(self):
        # Nothing can be served back, so inventing a reply would measure the
        # invention rather than the model.
        client, _, used = self.turns(TASK, call("deploy"))
        self.assertEqual((used, len(client.asked)), (1, 1))

    def test_the_wall_time_covers_every_turn(self):
        _, resp, _ = self.turns(TASK, call("search_files", pattern="t"),
                                call("run_tests"))
        self.assertEqual(resp["_wall"], 2.0)


class TestWhatTheRecordSays(unittest.TestCase):
    def measure(self, task, *replies):
        with tempfile.TemporaryDirectory() as td:
            ledger = Path(td) / "screen_tasks.jsonl"
            with contextlib.redirect_stdout(io.StringIO()):
                recs = screen.measure_tasks(Client(*replies), Config(results_dir=Path(td)),
                                            "m", 1, [task], {}, ledger)
        return recs[0]

    def test_reaching_the_tool_after_a_lookup_is_not_a_failure(self):
        rec = self.measure(TASK, call("search_files", pattern="test"),
                           call("run_tests", path="tests"))
        self.assertTrue(rec["passed"], "it called the tool the task asked for")
        self.assertTrue(rec["format_ok"])

    def test_the_extra_round_trip_costs_score(self):
        direct = self.measure(TASK, call("run_tests"))
        looked = self.measure(TASK, call("search_files", pattern="test"),
                              call("run_tests"))
        self.assertEqual(direct["score"], 1.0)
        self.assertEqual(looked["score"], 0.5)
        self.assertEqual(looked["turns"], 2)
        self.assertIn("2 turns", looked["detail"])

    def test_the_wrong_tool_is_still_the_wrong_tool(self):
        # Both turns spent on a tool the task did not ask for.
        rec = self.measure(TASK, call("search_files", pattern="test"),
                           call("search_files", pattern="spec"))
        self.assertFalse(rec["passed"])
        self.assertIn("wanted=run_tests", rec["detail"])


if __name__ == "__main__":
    unittest.main()
