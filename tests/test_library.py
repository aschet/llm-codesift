# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""The library listing is third-party markup that can change without notice.

These tests pin the shape the parser expects and, more importantly, the decisions
made on top of it: a suggestion that cannot be pulled, cannot hold the context, or
cannot call a tool wastes a screening slot, which is the expensive thing here.
"""
import contextlib
import datetime as dt
import io
import json
import re
import tempfile
import unittest
from pathlib import Path

from codesift import library
from codesift.config import Config

OFFLINE = "http://127.0.0.1:1"
TODAY = dt.date(2026, 8, 21)


def card(name, desc="", caps=(), sizes=(), pulls="1.2K", tags=3, updated="Aug 19, 2026"):
    cap_spans = "".join(
        f'<span class="inline-flex items-center rounded-md bg-indigo-50 px-2">{c}</span>'
        for c in caps)
    size_spans = "".join(
        f'<span class="inline-flex items-center rounded-md bg-[#ddf4ff] px-2">{s}</span>'
        for s in sizes)
    return f'''<li class="flex items-baseline border-b py-6">
      <a href="/library/{name}" class="group w-full space-y-5">
        <div title="{name}" class="flex flex-col">
          <h2><span class="group-hover:underline truncate">{name}</span></h2>
          <p class="max-w-lg break-words text-neutral-800 text-md">{desc}</p>
        </div>
        <div class="flex flex-col space-y-2">
          <div class="flex flex-wrap space-x-2">{cap_spans}{size_spans}</div>
          <p class="my-4 flex space-x-5 text-[13px]">
            <span class="flex items-center"><svg></svg>
              <span >{pulls}</span><span class="hidden sm:flex">&nbsp;Pulls</span></span>
            <span class="flex items-center"><svg></svg>
              <span >{tags}</span><span class="hidden sm:flex">&nbsp;Tags</span></span>
            <span class="flex items-center" title="{updated} 6:06 PM UTC"><svg></svg>
              <span class="hidden sm:flex">Updated&nbsp;</span><span >yesterday</span></span>
          </p>
        </div>
      </a>
    </li>'''


def index(*cards):
    return '<div id="repo"><ul role="list">' + "".join(cards) + "</ul></div>"


class FakeFetcher:
    """Serves saved pages, and records what was asked for."""

    def __init__(self, pages):
        self.pages = pages
        self.asked = []

    def get(self, url):
        self.asked.append(url)
        try:
            return self.pages[url]
        except KeyError:
            raise library.LibraryError(f"{url}: not found") from None


class TestParseIndex(unittest.TestCase):
    def test_reads_every_field_of_a_card(self):
        page = index(card("qwen3-coder", "Qwen3-Coder is agentic code generation.",
                          caps=("tools",), sizes=("30b", "480b"),
                          pulls="8.5M", tags=9, updated="Sep 23, 2025"))
        [model] = library.parse_index(page)
        self.assertEqual(model.name, "qwen3-coder")
        self.assertEqual(model.description, "Qwen3-Coder is agentic code generation.")
        self.assertEqual(model.capabilities, ("tools",))
        self.assertEqual(model.sizes, ("30b", "480b"))
        self.assertEqual(model.pulls, 8_500_000)
        self.assertEqual(model.tag_count, 9)
        self.assertEqual(model.updated, dt.date(2025, 9, 23))

    def test_capabilities_and_sizes_are_not_confused(self):
        page = index(card("m", caps=("vision", "tools", "thinking"), sizes=("9b", "35b")))
        [model] = library.parse_index(page)
        self.assertTrue(model.has_tools)
        self.assertNotIn("9b", model.capabilities)

    def test_a_card_without_recognisable_markup_is_skipped(self):
        page = index("<li>nothing here</li>", card("real"))
        self.assertEqual([m.name for m in library.parse_index(page)], ["real"])

    def test_unrelated_markup_yields_nothing_rather_than_raising(self):
        self.assertEqual(library.parse_index("<html><body>maintenance</body></html>"), [])


class TestCodingClaims(unittest.TestCase):
    """A claim in a description is evidence the claim was made, and nothing else."""

    def claims(self, name, desc):
        [model] = library.parse_index(index(card(name, desc, caps=("tools",))))
        return library.coding_claims(model)

    def test_reports_the_words_the_listing_uses(self):
        self.assertEqual(
            self.claims("devstral", "Explores codebases and powers software engineering agents."),
            ("codebases", "software engineering"))

    def test_a_word_inside_another_word_is_not_a_claim(self):
        # "encoder-decoder" is not a statement about writing code.
        self.assertEqual(self.claims("glm-ocr", "A GLM-V encoder-decoder architecture."), ())

    def test_a_model_named_for_coding_counts(self):
        self.assertIn("coder", self.claims("qwen3-coder", "Long context models."))

    def test_requiring_a_coding_claim_filters_rather_than_ranks(self):
        models = library.parse_index(index(
            card("plain", "A capable assistant.", caps=("tools",), sizes=("24b",)),
            card("stated", "Built for coding agents.", caps=("tools",), sizes=("24b",))))
        self.assertEqual(len(library.suggest(models)), 2)
        only = library.suggest(models, require_coding=True)
        self.assertEqual([p.model.name for p in only], ["stated"])


class TestSince(unittest.TestCase):
    def test_accepts_a_year_a_date_and_a_month_count(self):
        self.assertEqual(library.parse_since("2025"), dt.date(2025, 1, 1))
        self.assertEqual(library.parse_since("2025-06-01"), dt.date(2025, 6, 1))
        self.assertEqual(library.parse_since("6m", TODAY), dt.date(2026, 2, 19))

    def test_rejects_anything_else(self):
        with self.assertRaises(ValueError):
            library.parse_since("last tuesday")


class SuggestCase(unittest.TestCase):
    def suggest(self, models, _fetcher=None, **kw):
        return library.suggest(models, **kw)


class TestNothingIsDefinedAndUnused(unittest.TestCase):
    """A refactor that removes a consumer must remove what fed it.

    A property reading a field the dataclass no longer carries raises on any call,
    and nothing calls it, so nothing reports it.
    """

    def test_every_dataclass_field_and_property_resolves(self):
        v = library.Variant(ref="x:30b", label="30b", params_b=30.0)
        for name in dir(v):
            if name.startswith("_"):
                continue
            with self.subTest(member=name):
                getattr(v, name)          # a stale property raises here

    def test_the_module_names_nothing_it_does_not_define(self):
        import ast
        src = Path(library.__file__).read_text()
        defined = set(dir(library))
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id.isupper() and node.id not in defined:
                    self.fail(f"{node.id} is referenced but no longer defined")


class TestSuggest(SuggestCase):
    """Every constraint is a fact the listing prints, and the listing is one request."""

    def test_excludes_what_the_harness_cannot_use(self):
        models = library.parse_index(index(
            card("embedder", "Text embeddings.", caps=("embedding", "tools"), sizes=("8b",)),
            card("chatty", "A general chat model with coding skill.", caps=(), sizes=("8b",)),
            card("stale", "An older coding model.", caps=("tools",), sizes=("8b",),
                 updated="Nov 12, 2024"),
            card("keeper", "Agentic coding and refactoring.", caps=("tools",), sizes=("24b",))))
        picks = self.suggest(models, since=dt.date(2025, 1, 1))
        self.assertEqual([p.model.name for p in picks], ["keeper"])

    def test_tool_calling_cannot_be_relaxed(self):
        models = library.parse_index(index(
            card("mute", "Writes excellent code.", caps=("vision",), sizes=("24b",))))
        self.assertEqual(self.suggest(models), [])

    def test_a_cloud_model_with_no_size_is_dropped(self):
        # Nothing to pull, so it can never be screened locally.
        models = library.parse_index(index(
            card("cloudy", "Frontier model.", caps=("tools", "cloud"), sizes=())))
        self.assertEqual(self.suggest(models), [])

    def test_a_cloud_model_that_also_ships_sizes_is_kept(self):
        # The badge means "also available in the cloud", not "only".
        models = library.parse_index(index(
            card("both", "Frontier model.", caps=("tools", "cloud"), sizes=("9b", "27b"))))
        [pick] = self.suggest(models)
        self.assertEqual(pick.variant.ref, "both:27b")

    def test_the_largest_tag_within_the_limit_is_chosen(self):
        models = library.parse_index(index(
            card("fam", "Coding.", caps=("tools",), sizes=("4b", "12b", "27b", "70b"))))
        [pick] = self.suggest(models, max_params_b=30.0, min_params_b=4.0)
        self.assertEqual(pick.variant.ref, "fam:27b")
        self.assertEqual(pick.variant.params_b, 27.0)

    def test_a_model_with_nothing_inside_the_limits_is_skipped(self):
        models = library.parse_index(index(
            card("huge", "Coding.", caps=("tools",), sizes=("405b",))))
        skipped = []
        self.assertEqual(self.suggest(models, max_params_b=40.0,
                                      on_skip=lambda n, w: skipped.append(n)), [])
        self.assertEqual(skipped, ["huge"])

    def test_a_badge_that_states_no_count_is_not_turned_into_one(self):
        # e4b is an effective count and 8x7b names experts; both read as numbers
        # and are not. They are offered without a parameter count rather than a
        # guessed one, and no size limit applies to them.
        for sizes, ref in ((("e2b", "e4b"), "x:e4b"), (("8x7b",), "x:8x7b")):
            with self.subTest(sizes=sizes):
                models = library.parse_index(index(
                    card("x", "Coding.", caps=("tools",), sizes=sizes)))
                [pick] = self.suggest(models, max_params_b=1.0)
                self.assertEqual(pick.variant.ref, ref)
                self.assertIsNone(pick.variant.params_b)

    def test_a_readable_badge_is_preferred_over_one_that_is_not(self):
        models = library.parse_index(index(
            card("gem", "Coding.", caps=("tools",), sizes=("e2b", "e4b", "12b", "26b"))))
        [pick] = self.suggest(models, max_params_b=40.0)
        self.assertEqual(pick.variant.ref, "gem:26b")

    def test_sorts_newest_first_by_default(self):
        models = library.parse_index(index(
            card("older", "Coding.", caps=("tools",), sizes=("24b",), updated="Jan 4, 2026"),
            card("newer", "Coding.", caps=("tools",), sizes=("24b",), updated="Jul 4, 2026")))
        self.assertEqual([p.model.name for p in self.suggest(models)], ["newer", "older"])

    def test_every_ordering_is_a_fact_from_the_listing(self):
        models = library.parse_index(index(
            card("zebra", "Coding.", caps=("tools",), sizes=("24b",), updated="Jul 4, 2026"),
            card("alpha", "Coding.", caps=("tools",), sizes=("24b",), updated="Jan 4, 2026")))
        names = lambda how: [p.model.name for p in self.suggest(models, sort=how)]
        self.assertEqual(names("date"), ["zebra", "alpha"])
        self.assertEqual(names("name"), ["alpha", "zebra"])

    def test_installed_tags_are_left_out_unless_asked_for(self):
        models = library.parse_index(index(
            card("here", "Coding.", caps=("tools",), sizes=("24b",))))
        self.assertEqual(self.suggest(models, installed={"here:24b"}), [])
        [pick] = self.suggest(models, installed={"here:24b"}, include_installed=True)
        self.assertTrue(pick.installed)

    def test_holding_one_tag_does_not_hide_another(self):
        # Having fam:26b says nothing about fam:31b, which was never measured.
        models = library.parse_index(index(
            card("fam", "Coding.", caps=("tools",), sizes=("26b", "31b"))))
        [pick] = self.suggest(models, installed={"fam:26b"}, max_params_b=40.0)
        self.assertEqual(pick.variant.ref, "fam:31b")
        self.assertFalse(pick.installed)
        self.assertEqual(pick.have, ("fam:26b",))

    def test_the_reader_decides_what_is_off_topic(self):
        models = library.parse_index(index(
            card("reader", "An OCR model.", caps=("tools",), sizes=("8b",)),
            card("worker", "A coding model.", caps=("tools",), sizes=("8b",))))
        self.assertEqual([p.model.name for p in self.suggest(models, exclude="OCR")],
                         ["worker"])
        self.assertEqual([p.model.name for p in self.suggest(models, match="OCR")],
                         ["reader"])


class TestRun(SuggestCase):
    def setUp(self):
        # run() reports problems and file writes on stderr; the tests assert on the
        # stream they pass in, so keep the noise out of the test output.
        self.enterContext(contextlib.redirect_stderr(io.StringIO()))
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.cfg = Config(host=OFFLINE, results_dir=self.tmp)
        self.fetcher = FakeFetcher({
            library.LIBRARY_URL: index(
                card("keeper", "Agentic coding and refactoring.", caps=("tools",),
                     sizes=("8b", "24b"), pulls="1.2M", updated="Aug 01, 2026")),
        })

    def test_the_whole_listing_costs_one_request(self):
        library.run(self.cfg, fetcher=self.fetcher, stream=io.StringIO())
        self.assertEqual(self.fetcher.asked, [library.LIBRARY_URL])

    def test_the_coding_column_answers_rather_than_quotes(self):
        out = io.StringIO()
        library.run(self.cfg, fetcher=self.fetcher, stream=out)
        row = [l for l in out.getvalue().splitlines() if l.startswith("keeper:")][0]
        self.assertRegex(row, r"\byes\b")
        self.assertNotIn("codebases", row)

    def test_matches_are_not_cut_off_at_a_count(self):
        many = index(*[card(f"m{i}", "Coding.", caps=("tools",), sizes=("24b",),
                            updated="Aug 01, 2026") for i in range(30)])
        out = io.StringIO()
        library.run(self.cfg, fetcher=FakeFetcher({library.LIBRARY_URL: many}), stream=out)
        self.assertEqual(sum(1 for l in out.getvalue().splitlines()
                             if re.match(r"m\d+:", l)), 30)

    def test_the_table_carries_only_what_bears_on_screening(self):
        out = io.StringIO()
        library.run(self.cfg, fetcher=self.fetcher, stream=out)
        header = [l for l in out.getvalue().splitlines() if l.startswith("model")][0]
        self.assertEqual(header.split(), ["model", "params", "updated", "coding"])

    def test_the_publishers_description_is_not_in_the_table(self):
        out = io.StringIO()
        library.run(self.cfg, fetcher=self.fetcher, stream=out)
        self.assertNotIn("Agentic coding and refactoring", out.getvalue())

    def test_json_still_carries_the_description_whole(self):
        out = io.StringIO()
        library.run(self.cfg, fetcher=self.fetcher, stream=out, as_json=True)
        [row] = json.loads(out.getvalue())
        self.assertEqual(row["description"], "Agentic coding and refactoring.")

    def test_prints_a_table(self):
        out = io.StringIO()
        code = library.run(self.cfg, fetcher=self.fetcher, stream=out)
        self.assertEqual(code, 0)
        self.assertIn("keeper:24b", out.getvalue())
        self.assertIn("24B", out.getvalue())

    def test_json_output_is_machine_readable(self):
        out = io.StringIO()
        library.run(self.cfg, fetcher=self.fetcher, stream=out, as_json=True)
        [row] = json.loads(out.getvalue())
        self.assertEqual(row["ref"], "keeper:24b")
        self.assertEqual(row["params_b"], 24.0)

    def test_writes_a_model_list_the_harness_can_read(self):
        from codesift.config import read_model_file
        path = self.tmp / "suggested.txt"
        library.run(self.cfg, fetcher=self.fetcher, stream=io.StringIO(),
                    write_models=path)
        self.assertEqual(read_model_file(path), ["keeper:24b"])

    def test_the_list_installs_what_it_names(self):
        # Nothing can be screened before it is installed, so the file that names
        # the candidates is also the file that pulls them.
        path = self.tmp / "suggested.txt"
        library.run(self.cfg, fetcher=self.fetcher, stream=io.StringIO(),
                    write_models=path)
        self.assertIn("ollama pull keeper:24b", path.read_text(encoding="utf-8"))

    def test_an_unreadable_listing_fails_without_a_traceback(self):
        out = io.StringIO()
        code = library.run(self.cfg, fetcher=FakeFetcher({}), stream=out)
        self.assertEqual(code, 1)

    def test_markup_that_parses_to_nothing_is_reported(self):
        out = io.StringIO()
        code = library.run(self.cfg, stream=out,
                           fetcher=FakeFetcher({library.LIBRARY_URL: "<html></html>"}))
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
