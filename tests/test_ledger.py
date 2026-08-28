# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""The store every stage keeps its measurements in.

What matters is that a damaged line costs only itself, and that writing a record
replaces the one it supersedes instead of joining it.
"""
import json
import tempfile
import unittest
from pathlib import Path

from codesift import ledger

BY_NAME = lambda rec: rec["name"]


class LedgerCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.path = self.tmp / "store.jsonl"

    def put(self, *records):
        ledger.write(self.path, records)

    def lines(self):
        return self.path.read_text(encoding="utf-8").splitlines()


class TestReading(LedgerCase):
    def test_a_missing_file_is_no_records_rather_than_an_error(self):
        self.assertEqual(ledger.read(self.tmp / "never-written.jsonl"), [])

    def test_records_come_back_in_the_order_they_were_written(self):
        self.put({"name": "a"}, {"name": "b"})
        self.assertEqual([r["name"] for r in ledger.read(self.path)], ["a", "b"])

    def test_a_line_cut_off_mid_write_costs_only_itself(self):
        # A run killed while writing leaves a partial line; everything measured
        # before it is still good and must survive.
        self.put({"name": "a"})
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write('{"name": "b"')
        self.assertEqual([r["name"] for r in ledger.read(self.path)], ["a"])


class TestKeying(LedgerCase):
    def test_the_last_record_of_a_key_wins(self):
        self.put({"name": "a", "v": 1}, {"name": "a", "v": 2})
        self.assertEqual(ledger.keyed(self.path, BY_NAME)["a"]["v"], 2)

    def test_a_record_the_key_cannot_be_taken_from_is_skipped(self):
        self.put({"name": "a"}, {"nothing": True})
        self.assertEqual(list(ledger.keyed(self.path, BY_NAME)), ["a"])


class TestReplacing(LedgerCase):
    def test_a_rewritten_record_supersedes_rather_than_joins(self):
        self.put({"name": "a", "v": 1}, {"name": "b", "v": 1})
        ledger.replace(self.path, {"name": "a", "v": 2}, BY_NAME)
        stored = ledger.read(self.path)
        self.assertEqual(len(stored), 2)
        self.assertEqual(ledger.keyed(self.path, BY_NAME)["a"]["v"], 2)

    def test_other_keys_are_left_alone(self):
        self.put({"name": "keep", "v": 1})
        ledger.replace(self.path, {"name": "new", "v": 1}, BY_NAME)
        self.assertEqual({r["name"] for r in ledger.read(self.path)}, {"keep", "new"})

    def test_it_writes_a_store_that_does_not_exist_yet(self):
        deep = self.tmp / "made" / "up" / "store.jsonl"
        ledger.replace(deep, {"name": "a"}, BY_NAME)
        self.assertEqual([r["name"] for r in ledger.read(deep)], ["a"])

    def test_a_damaged_line_is_not_carried_forward(self):
        # Rewriting is the one chance to drop what cannot be read; keeping the
        # text would mean writing back something no reader can use.
        self.path.write_text('{"name": "a"}\n{"name":\n', encoding="utf-8")
        ledger.replace(self.path, {"name": "b"}, BY_NAME)
        self.assertEqual([json.loads(l)["name"] for l in self.lines()], ["a", "b"])

    def test_every_line_is_one_record(self):
        self.put({"name": "a"})
        ledger.replace(self.path, {"name": "b", "text": "two\nlines"}, BY_NAME)
        self.assertEqual(len(self.lines()), 2)


if __name__ == "__main__":
    unittest.main()
