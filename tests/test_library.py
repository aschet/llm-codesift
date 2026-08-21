"""The library listing is third-party markup that can change without notice.

These tests pin the shape the parser expects and, more importantly, the decisions
made on top of it: a suggestion that cannot be pulled, cannot hold the context, or
cannot call a tool wastes a screening slot, which is the expensive thing here.
"""
import contextlib
import datetime as dt
import io
import json
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


def tag_row(ref, size="19GB", context="256K", age="3 months ago",
            digest="06c1097efce0"):
    """Cloud tags print a usage tier where a local tag prints its download size."""
    return (f'<a href="/library/{ref}" class="group"><div>'
            f'<span class="font-mono">{digest}</span> &bull; {size} &bull; '
            f'{context} context window &bull; Text input &bull; {age}</div></a>')


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


class TestParseTags(unittest.TestCase):
    def test_local_and_cloud_tags_are_distinguished(self):
        page = (tag_row("gpt-oss:20b", "14GB", "128K")
                + tag_row("gpt-oss:120b", "65GB", "128K")
                + tag_row("gpt-oss:20b-cloud", "Low Usage", "128K"))
        variants = {v.ref: v for v in library.parse_tags("gpt-oss", page)}
        self.assertEqual(variants["gpt-oss:20b"].size_gb, 14.0)
        self.assertEqual(variants["gpt-oss:20b"].context, 131072)
        self.assertFalse(variants["gpt-oss:20b"].cloud)
        self.assertTrue(variants["gpt-oss:20b-cloud"].cloud)
        self.assertIsNone(variants["gpt-oss:20b-cloud"].size_gb)

    def test_the_manifest_digest_is_read(self):
        [v] = library.parse_tags("m", tag_row("m:8b", digest="d8b269ad5c7c"))
        self.assertEqual(v.digest, "d8b269ad5c7c")

    def test_a_tag_is_read_once_despite_repeated_markup(self):
        page = tag_row("m:8b") + tag_row("m:8b")
        self.assertEqual(len(library.parse_tags("m", page)), 1)

    def test_another_models_tags_are_not_picked_up(self):
        page = tag_row("qwen3:8b") + tag_row("qwen3-coder:30b")
        self.assertEqual([v.ref for v in library.parse_tags("qwen3", page)], ["qwen3:8b"])


def model_page(arch, params, quant="Q4_K_M"):
    return (f'<div class="flex"><span class="font-mono">24277f07f62d</span>'
            f'<span>15GB</span><span>model arch</span><span>{arch}</span>'
            f'<span>&middot;</span><span>parameters</span><span>{params}</span>'
            f'<span>&middot;</span><span>quantization</span><span>{quant}</span></div>')


class TestModelPage(unittest.TestCase):
    def test_reads_the_architecture_and_parameter_count(self):
        self.assertEqual(library.parse_model_page(model_page("mistral3", "24B")),
                         ("mistral3", "24B"))

    def test_a_page_without_the_field_yields_nothing_rather_than_raising(self):
        self.assertEqual(library.parse_model_page("<html>maintenance</html>"), ("", ""))

    def test_mixture_of_experts_is_claimed_only_when_the_name_says_so(self):
        v = lambda arch: library.Variant("m:1", 1.0, 1024, False, arch=arch)
        self.assertTrue(v("qwen3moe").moe)
        self.assertTrue(v("gptoss").moe)          # a mixture of experts without "moe"
        self.assertFalse(v("mistral3").moe)
        # Publishers give a mixture of experts a plain family name often enough that
        # calling this one dense would be a guess, so it stays unanswered.
        self.assertIsNone(v("gemma4").moe)
        self.assertIsNone(v("").moe)


class TestVariantChoice(unittest.TestCase):
    def variants(self):
        return [
            library.Variant("m:latest", 19.0, 262144, False),
            library.Variant("m:30b", 19.0, 262144, False),
            library.Variant("m:30b-a3b-q8_0", 32.0, 262144, False),
            library.Variant("m:480b", 290.0, 262144, False),
            library.Variant("m:30b-cloud", None, 262144, True),
        ]

    def test_prefers_the_default_build_over_a_larger_quantisation(self):
        # 30b-a3b-q8_0 is bigger and fits, but the intended download is 30b.
        best = library._best_variant(self.variants(), max_size_gb=32.0, min_context=None)
        self.assertEqual(best.ref, "m:30b")

    def test_oversized_and_cloud_tags_are_never_chosen(self):
        best = library._best_variant(self.variants(), max_size_gb=20.0, min_context=None)
        self.assertEqual(best.ref, "m:30b")
        self.assertIsNone(library._best_variant(
            [library.Variant("m:480b", 290.0, 262144, False),
             library.Variant("m:x-cloud", None, 262144, True)],
            max_size_gb=32.0, min_context=None))

    def test_a_short_context_disqualifies_the_tag(self):
        small = [library.Variant("m:8b", 5.0, 32768, False)]
        self.assertIsNone(library._best_variant(small, None, 65536))
        self.assertIsNotNone(library._best_variant(small, None, 32768))

    def test_falls_back_when_only_qualified_tags_exist(self):
        only = [library.Variant("m:q4_K_M", 18.0, 131072, False),
                library.Variant("m:fp16", 60.0, 131072, False)]
        best = library._best_variant(only, max_size_gb=32.0, min_context=None)
        self.assertEqual(best.ref, "m:q4_K_M")


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
            card("plain", "A capable assistant.", caps=("tools",)),
            card("stated", "Built for coding agents.", caps=("tools",))))
        fetcher = FakeFetcher({
            f"{library.LIBRARY_URL}/plain/tags": tag_row("plain:24b", "15GB", "128K"),
            f"{library.LIBRARY_URL}/stated/tags": tag_row("stated:24b", "15GB", "128K")})
        both = library.suggest(models, fetcher)
        self.assertEqual(len(both), 2)
        only = library.suggest(models, fetcher, require_coding=True)
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
    def fetcher(self, **tags):
        pages = {f"{library.LIBRARY_URL}/{name}/tags": page for name, page in tags.items()}
        return FakeFetcher(pages)

    def suggest(self, models, fetcher, **kw):
        return library.suggest(models, fetcher, **kw)


class TestSuggest(SuggestCase):
    def test_excludes_what_the_harness_cannot_use(self):
        models = library.parse_index(index(
            card("embedder", "Text embeddings.", caps=("embedding", "tools")),
            card("chatty", "A general chat model with coding skill.", caps=()),
            card("stale", "An older coding model.", caps=("tools",),
                 updated="Nov 12, 2024"),
            card("keeper", "Agentic coding and refactoring.", caps=("tools",)),
        ))
        fetcher = self.fetcher(keeper=tag_row("keeper:24b", "15GB", "128K"))
        picks = self.suggest(models, fetcher, since=dt.date(2025, 1, 1))
        self.assertEqual([p.variant.ref for p in picks], ["keeper:24b"])
        # Pages are fetched only for the survivor: its tags, then the chosen tag for
        # the architecture. The three rejections cost nothing beyond the listing.
        self.assertEqual(fetcher.asked, [f"{library.LIBRARY_URL}/keeper/tags",
                                         f"{library.LIBRARY_URL}/keeper:24b"])

    def test_tool_calling_cannot_be_relaxed(self):
        # A harness reaches the filesystem through tool calls, so a model without them
        # is not a weaker candidate, it is not a candidate. There is no flag for it.
        import inspect
        models = library.parse_index(index(
            card("mute", "Excellent at coding, no tools.", caps=("thinking",))))
        fetcher = self.fetcher(mute=tag_row("mute:24b", "15GB", "128K"))
        self.assertEqual(self.suggest(models, fetcher), [])
        for func in (library.suggest, library.run):
            self.assertNotIn("require_tools", inspect.signature(func).parameters)

    def test_a_cloud_only_model_is_dropped_at_the_tag_stage(self):
        models = library.parse_index(index(
            card("cloudy", "Agentic coding.", caps=("tools", "cloud"))))
        fetcher = self.fetcher(cloudy=tag_row("cloudy:480b", "Medium Usage", "256K"))
        skipped = []
        picks = self.suggest(models, fetcher, on_skip=lambda n, w: skipped.append(n))
        self.assertEqual(picks, [])
        self.assertEqual(skipped, ["cloudy"])

    def test_sorts_newest_first_by_default(self):
        models = library.parse_index(index(
            card("older", "Coding model.", caps=("tools",), updated="Mar 01, 2025"),
            card("newer", "Coding model.", caps=("tools",), updated="Aug 01, 2026"),
        ))
        fetcher = self.fetcher(older=tag_row("older:24b", "15GB", "128K"),
                               newer=tag_row("newer:24b", "15GB", "128K"))
        picks = self.suggest(models, fetcher)
        self.assertEqual([p.model.name for p in picks], ["newer", "older"])

    def test_every_ordering_is_a_fact_from_the_listing(self):
        # Pull counts are deliberately not an ordering: they accumulate with age, so
        # a new model can never rank well on them however good it is.
        models = library.parse_index(index(
            card("zebra", "A model.", caps=("tools",),
                 pulls="2M", updated="Mar 01, 2025"),
            card("alpha", "A model.", caps=("tools",),
                 pulls="10K", updated="Aug 01, 2026"),
        ))
        fetcher = self.fetcher(zebra=tag_row("zebra:24b", "15GB", "128K"),
                               alpha=tag_row("alpha:24b", "15GB", "128K"))
        names = lambda how: [p.model.name for p in self.suggest(models, fetcher, sort=how)]
        self.assertEqual(names("date"), ["alpha", "zebra"])
        self.assertEqual(names("name"), ["alpha", "zebra"])
        self.assertNotIn("pulls", library.SORTS)

    def test_installed_tags_are_left_out_unless_asked_for(self):
        models = library.parse_index(index(card("here", "Coding.", caps=("tools",))))
        fetcher = self.fetcher(here=tag_row("here:24b", "15GB", "128K"))
        self.assertEqual(self.suggest(models, fetcher, installed={"here", "here:24b"}), [])
        picks = self.suggest(models, fetcher, installed={"here:24b"},
                             include_installed=True)
        self.assertTrue(picks[0].installed)

    def test_the_architecture_is_carried_through_from_the_tag_page(self):
        models = library.parse_index(index(card("dev", "Coding.", caps=("tools",))))
        fetcher = FakeFetcher({
            f"{library.LIBRARY_URL}/dev/tags": tag_row("dev:24b", "15GB", "128K"),
            f"{library.LIBRARY_URL}/dev:24b": model_page("mistral3", "24B")})
        [pick] = self.suggest(models, fetcher)
        self.assertEqual(pick.variant.arch, "mistral3")
        self.assertEqual(pick.variant.params, "24B")
        self.assertFalse(pick.variant.moe)

    def test_a_missing_architecture_page_does_not_lose_the_suggestion(self):
        models = library.parse_index(index(card("dev", "Coding.", caps=("tools",))))
        fetcher = self.fetcher(dev=tag_row("dev:24b", "15GB", "128K"))
        [pick] = self.suggest(models, fetcher)
        self.assertEqual(pick.variant.ref, "dev:24b")
        self.assertEqual(pick.variant.arch, "")

    def test_holding_one_tag_does_not_hide_another(self):
        # gemma4:26b on the server says nothing about gemma4:31b, which is a different
        # weight set that has never been measured. It is a candidate, marked with what
        # is already held so the relationship is visible.
        page = (tag_row("fam:26b", "18GB", "256K") + tag_row("fam:31b", "20GB", "256K"))
        models = library.parse_index(index(card("fam", "Coding.", caps=("tools",))))
        [pick] = self.suggest(models, self.fetcher(fam=page),
                              installed={"fam", "fam:26b"})
        self.assertEqual(pick.variant.ref, "fam:31b")
        self.assertFalse(pick.installed)
        self.assertEqual(pick.have, ("fam:26b",))

    def test_an_alias_of_installed_weights_is_recognised_by_digest(self):
        # ":latest" and ":q4_K_M" are routinely one download under two names; only the
        # manifest digest says so, and without it the same weights get screened twice.
        page = (tag_row("fam:latest", "18GB", "256K", digest="d8b269ad5c7c")
                + tag_row("fam:q4_K_M", "18GB", "256K", digest="d8b269ad5c7c"))
        models = library.parse_index(index(card("fam", "Coding.", caps=("tools",))))
        self.assertEqual(self.suggest(models, self.fetcher(fam=page),
                                      installed={"fam:q4_K_M", "d8b269ad5c7c"}), [])
        # Without the digest the alias would read as a model that is not held.
        self.assertEqual(len(self.suggest(models, self.fetcher(fam=page),
                                          installed={"fam:q4_K_M"})), 1)

    def test_the_reader_decides_what_is_off_topic(self):
        # No weighting of a publisher's description can establish that a model is or
        # is not worth screening, so the text is exposed as a filter, not a verdict.
        models = library.parse_index(index(
            card("reader", "An OCR model for document understanding.", caps=("tools",)),
            card("worker", "A general model with strong tool use.", caps=("tools",))))
        fetcher = self.fetcher(reader=tag_row("reader:8b", "6GB", "128K"),
                               worker=tag_row("worker:8b", "6GB", "128K"))
        self.assertEqual(len(self.suggest(models, fetcher)), 2)
        kept = self.suggest(models, fetcher, exclude="ocr")
        self.assertEqual([p.model.name for p in kept], ["worker"])
        kept = self.suggest(models, fetcher, match="tool use")
        self.assertEqual([p.model.name for p in kept], ["worker"])

    def test_a_model_whose_tag_page_fails_is_reported_not_fatal(self):
        models = library.parse_index(index(
            card("gone", "Coding.", caps=("tools",)),
            card("fine", "Coding.", caps=("tools",))))
        fetcher = self.fetcher(fine=tag_row("fine:24b", "15GB", "128K"))
        skipped = []
        picks = self.suggest(models, fetcher, on_skip=lambda n, w: skipped.append(n))
        self.assertEqual([p.model.name for p in picks], ["fine"])
        self.assertEqual(skipped, ["gone"])


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
                     pulls="1.2M", updated="Aug 01, 2026")),
            f"{library.LIBRARY_URL}/keeper/tags": tag_row("keeper:24b", "15GB", "128K"),
        })

    def test_prints_a_table(self):
        out = io.StringIO()
        code = library.run(self.cfg, fetcher=self.fetcher, stream=out)
        self.assertEqual(code, 0)
        self.assertIn("keeper:24b", out.getvalue())
        self.assertIn("15GB", out.getvalue())

    def test_json_output_is_machine_readable(self):
        out = io.StringIO()
        library.run(self.cfg, fetcher=self.fetcher, stream=out, as_json=True)
        [row] = json.loads(out.getvalue())
        self.assertEqual(row["ref"], "keeper:24b")
        self.assertEqual(row["size_gb"], 15.0)

    def test_writes_a_model_list_the_harness_can_read(self):
        from codesift.config import read_model_file
        path = self.tmp / "suggested.txt"
        library.run(self.cfg, fetcher=self.fetcher, stream=io.StringIO(),
                    write_models=path)
        self.assertEqual(read_model_file(path), ["keeper:24b"])

    def test_an_unreadable_listing_fails_without_a_traceback(self):
        out = io.StringIO()
        code = library.run(self.cfg, fetcher=FakeFetcher({}), stream=out)
        self.assertEqual(code, 1)

    def test_markup_that_parses_to_nothing_is_reported(self):
        out = io.StringIO()
        code = library.run(self.cfg, stream=out,
                           fetcher=FakeFetcher({library.LIBRARY_URL: "<html></html>"}))
        self.assertEqual(code, 1)


class TestFetcherCache(unittest.TestCase):
    def test_a_fresh_cached_page_is_reused(self):
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        fetcher = library.Fetcher(cache_dir=tmp, timeout=0.5)
        path = fetcher._cache_path(library.LIBRARY_URL)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("cached", encoding="utf-8")
        self.assertEqual(fetcher.get(library.LIBRARY_URL), "cached")

    def test_an_expired_page_is_refetched(self):
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        # No network in tests, so expiry must surface as a fetch attempt that fails.
        fetcher = library.Fetcher(cache_dir=tmp, ttl_hours=0, timeout=0.5)
        path = fetcher._cache_path("http://127.0.0.1:1/library")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale", encoding="utf-8")
        with self.assertRaises(library.LibraryError):
            fetcher.get("http://127.0.0.1:1/library")


if __name__ == "__main__":
    unittest.main()
