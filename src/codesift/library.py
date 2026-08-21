"""Finds models in the Ollama library that are worth screening.

Screening costs GPU hours, so the shortlist it runs on should be chosen rather
than pulled at random. This module reads the public library listing and the tag
page of each surviving model, then applies the constraints the harness itself
imposes: the model must run locally, emit tool calls, hold a working context,
fit the machine, and be recent enough that a screen result still means something.

Every column it reports is copied from the page. Nothing here ranks models by how
well they might write code: the listing carries only the publisher's own description,
and no weighting of that text is evidence of anything. Deciding which of the
candidates is any good is what the screen is for.
"""
from __future__ import annotations

import datetime as dt
import html
import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path

LIBRARY_URL = "https://ollama.com/library"
USER_AGENT = "codesift"

# The listing shows capabilities as badges; these are the ones that matter here.
CAP_TOOLS = "tools"
CAP_CLOUD = "cloud"
CAP_EMBEDDING = "embedding"

# Architectures whose name does not contain "moe" but which are mixtures of experts,
# and ones known to be dense. Anything absent from both is reported as printed rather
# than guessed at, since only the weights themselves settle it.
MOE_ARCHITECTURES = frozenset({"gptoss", "deepseek2", "deepseek3"})
DENSE_ARCHITECTURES = frozenset({
    "llama", "mistral", "mistral3", "gemma", "gemma2", "gemma3", "qwen2", "qwen3",
    "phi3", "phi4", "starcoder2", "cohere", "cohere2", "olmo2", "nemotron",
})

# Words that mean a publisher claims the model does programming work. They are
# reported, and can be required, but they are never weighted into a score: a claim
# in a description is evidence that the claim was made, and nothing more.
CODING_WORDS = (
    "code", "coder", "coding", "codebase", "codebases", "programming",
    "software engineering", "swe-bench", "developer", "refactor", "refactoring",
    "debugging", "code completion", "fill-in-the-middle",
)

class LibraryError(RuntimeError):
    """The listing could not be read or understood."""


@dataclass(frozen=True)
class Variant:
    """One pullable tag of a model."""

    ref: str                     # "qwen3-coder:30b"
    size_gb: float | None        # download size; None for cloud tags
    context: int | None          # advertised context window in tokens
    cloud: bool                  # served by Ollama's cloud, not pullable
    age: str = ""                # as printed on the page, e.g. "11 months ago"
    digest: str = ""             # short manifest digest, which identifies the weights
    arch: str = ""               # architecture as the tag's page names it
    params: str = ""             # parameter count as the tag's page states it

    @property
    def moe(self) -> bool | None:
        """Whether the architecture is a mixture of experts.

        Positive only when the name says so. Several publishers give a mixture of
        experts a plain family name, so a False here would be a guess; those return
        None and the architecture is reported as printed for the reader to judge.
        """
        if not self.arch:
            return None
        name = self.arch.lower()
        if "moe" in name or name in MOE_ARCHITECTURES:
            return True
        if name in DENSE_ARCHITECTURES:
            return False
        return None

    @property
    def tag(self) -> str:
        return self.ref.partition(":")[2]


@dataclass(frozen=True)
class LibraryModel:
    """One entry in the library listing."""

    name: str
    description: str = ""
    capabilities: tuple[str, ...] = ()
    sizes: tuple[str, ...] = ()
    pulls: int = 0
    tag_count: int = 0
    updated: dt.date | None = None

    @property
    def has_tools(self) -> bool:
        return CAP_TOOLS in self.capabilities

    @property
    def is_embedding(self) -> bool:
        return CAP_EMBEDDING in self.capabilities


@dataclass
class Suggestion:
    """A model and the specific tag that satisfies the constraints."""

    model: LibraryModel
    variant: Variant
    installed: bool = False        # this exact tag is on the server
    have: tuple[str, ...] = ()     # other tags of the same model that are

    @property
    def coding_words(self) -> tuple[str, ...]:
        return coding_claims(self.model)

    def as_dict(self) -> dict:
        return dict(model=self.model.name, ref=self.variant.ref,
                    size_gb=self.variant.size_gb, context=self.variant.context,
                    arch=self.variant.arch, params=self.variant.params,
                    moe=self.variant.moe,
                    updated=self.model.updated.isoformat() if self.model.updated else None,
                    pulls=self.model.pulls,
                    capabilities=list(self.model.capabilities),
                    installed=self.installed, have=list(self.have),
                    coding_words=list(self.coding_words),
                    description=self.model.description)


# ---------------------------------------------------------------- fetching

def _get(url: str, timeout: float) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", "replace")
    except urllib.error.URLError as exc:
        raise LibraryError(f"{url}: {exc}") from exc


class Fetcher:
    """Reads library pages, keeping a copy on disk.

    The listing changes daily at most, so repeated runs during a screening session
    should not re-request it. A cached page older than the lifetime is refetched.
    """

    def __init__(self, cache_dir: Path | None = None, ttl_hours: float = 24.0,
                 timeout: float = 30.0, refresh: bool = False) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.ttl = ttl_hours * 3600
        self.timeout = timeout
        self.refresh = refresh

    def _cache_path(self, url: str) -> Path | None:
        if not self.cache_dir:
            return None
        slug = re.sub(r"[^A-Za-z0-9]+", "_", url.split("://", 1)[-1]).strip("_")
        return self.cache_dir / f"{slug}.html"

    def get(self, url: str) -> str:
        path = self._cache_path(url)
        if path and path.exists() and not self.refresh:
            age = dt.datetime.now().timestamp() - path.stat().st_mtime
            if age < self.ttl:
                return path.read_text(encoding="utf-8")
        body = _get(url, self.timeout)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding="utf-8")
        return body


# ---------------------------------------------------------------- parsing

_TAGS = re.compile(r"<[^>]+>")


def _text(fragment: str) -> str:
    return html.unescape(_TAGS.sub(" ", fragment)).replace("\xa0", " ").strip()


def _squash(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _pulls(value: str) -> int:
    """"13.3K" and "1.2M" as printed on the listing."""
    m = re.match(r"([\d.]+)\s*([KMB]?)", value.strip(), re.I)
    if not m:
        return 0
    scale = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}[m.group(2).upper()]
    try:
        return int(float(m.group(1)) * scale)
    except ValueError:
        return 0


def _context(value: str) -> int | None:
    """"256K" as printed beside a tag."""
    m = re.match(r"([\d.]+)\s*([KM]?)", value.strip(), re.I)
    if not m:
        return None
    scale = {"": 1, "K": 1024, "M": 1024 * 1024}[m.group(2).upper()]
    try:
        return int(float(m.group(1)) * scale)
    except ValueError:
        return None


def parse_index(page: str) -> list[LibraryModel]:
    """Read the library listing into models. Unrecognised cards are skipped."""
    out: list[LibraryModel] = []
    seen: set[str] = set()
    for card in re.findall(r"<li\b[^>]*>(.*?)</li>", page, re.S):
        m = re.search(r'href="/library/([^"/?#]+)"', card)
        if not m or m.group(1) in seen:
            continue
        name = html.unescape(m.group(1))
        seen.add(name)
        desc = re.search(r'<p class="max-w-lg[^"]*"[^>]*>(.*?)</p>', card, re.S)
        # Capability badges and size badges differ only by their background colour.
        caps = re.findall(r'<span[^>]*class="[^"]*bg-(?:indigo-50|cyan-50)[^"]*"[^>]*>([^<]*)</span>', card)
        sizes = re.findall(r'<span[^>]*class="[^"]*bg-\[#ddf4ff\][^"]*"[^>]*>([^<]*)</span>', card)
        pulls = re.search(r"<span[^>]*>\s*([\d.]+[KMB]?)\s*</span>\s*<span[^>]*>\s*&nbsp;Pulls", card)
        tags = re.search(r"<span[^>]*>\s*([\d,]+)\s*</span>\s*<span[^>]*>\s*&nbsp;Tags", card)
        out.append(LibraryModel(
            name=name,
            description=_squash(_text(desc.group(1))) if desc else "",
            capabilities=tuple(c.strip().lower() for c in caps if c.strip()),
            sizes=tuple(s.strip().lower() for s in sizes if s.strip()),
            pulls=_pulls(pulls.group(1)) if pulls else 0,
            tag_count=int(tags.group(1).replace(",", "")) if tags else 0,
            updated=_updated(card),
        ))
    return out


def _updated(card: str) -> dt.date | None:
    """The listing carries the exact timestamp in a tooltip beside the relative text."""
    m = re.search(r'title="([A-Z][a-z]{2} \d{1,2}, \d{4})[^"]*"', card)
    if not m:
        return None
    try:
        return dt.datetime.strptime(m.group(1), "%b %d, %Y").date()
    except ValueError:
        return None


_ARCH = re.compile(r"model arch (\S+)\s*[\u00b7&#183;]*\s*parameters (\S+)")


def parse_model_page(page: str) -> tuple[str, str]:
    """Architecture and parameter count, as the page for one tag prints them."""
    text = _squash(_text(page))
    m = re.search(r"model arch (\S+) . parameters (\S+)", text)
    return (m.group(1), m.group(2)) if m else ("", "")


def parse_tags(name: str, page: str) -> list[Variant]:
    """Read a model's tag page.

    Local tags print a download size; cloud tags print a usage tier in the same
    position, which is how the two are told apart.
    """
    out: list[Variant] = []
    seen: set[str] = set()
    pattern = re.compile(r'href="/library/(' + re.escape(name) + r':[^"]+)"')
    parts = pattern.split(page)
    # split() yields [prefix, ref, body, ref, body, ...]
    for ref, body in zip(parts[1::2], parts[2::2]):
        ref = html.unescape(ref)
        if ref in seen:
            continue
        seen.add(ref)
        text = _squash(_text(body[:4000]))
        size = re.search(r"([\d.]+)\s*GB\b", text)
        ctx = re.search(r"([\d.]+[KM]?)\s*context window", text)
        age = re.search(r"(\d+ \w+ ago|yesterday|today)", text)
        digest = re.search(r"\b([0-9a-f]{12})\b", text)
        cloud = size is None and re.search(r"\b\w+ Usage\b", text) is not None
        out.append(Variant(ref=ref,
                           size_gb=float(size.group(1)) if size else None,
                           context=_context(ctx.group(1)) if ctx else None,
                           cloud=cloud,
                           age=age.group(1) if age else "",
                           digest=digest.group(1) if digest else ""))
    return out


# ---------------------------------------------------------------- selection

# A tag naming only a parameter count is the publisher's default build, which for
# almost every model in the library means Q4_K_M. Fully qualified tags exist beside
# it at other quantisations, and picking the largest of those would suggest an fp16
# or q8 weight set where the intended download is a quarter of the size.
_PLAIN_TAG = re.compile(r"^(latest|[\d.]+[bm]|[\d.]+x[\d.]+b)$", re.I)


def _best_variant(variants: list[Variant], max_size_gb: float | None,
                  min_context: int | None, min_size_gb: float = 0.0) -> Variant | None:
    """The largest default build that still fits, since larger is generally stronger.

    Tags whose page states no size are treated as unknown rather than as fitting:
    a suggestion the machine cannot run wastes more time than one it never made.
    """
    usable = [v for v in variants if not v.cloud and v.size_gb is not None]
    if max_size_gb is not None:
        usable = [v for v in usable if v.size_gb <= max_size_gb]
    if min_size_gb:
        usable = [v for v in usable if v.size_gb >= min_size_gb]
    if min_context is not None:
        usable = [v for v in usable if v.context is None or v.context >= min_context]
    if not usable:
        return None
    plain = [v for v in usable if _PLAIN_TAG.match(v.tag)]
    if not plain:
        # Some publishers ship only qualified tags; take the customary quantisation.
        plain = [v for v in usable if "q4" in v.tag.lower()] or usable
    # Among tags of the same size, prefer the more specific name: ":latest" duplicates
    # a numbered tag and does not say which build was measured.
    plain.sort(key=lambda v: (v.size_gb or 0, v.tag != "latest", -len(v.tag)))
    return plain[-1]


SORTS = ("date", "name")


def _mentions(term: str, haystack: str) -> bool:
    """Whole-word match.

    Substring matching reads "encoder-decoder" as a mention of both code and coder,
    which is how a text search turns into a false claim about what a model is for.
    """
    pattern = re.escape(term).replace(r"\ ", r"\s+")
    return re.search(r"(?<!\w)" + pattern + r"(?!\w)", haystack) is not None


def coding_claims(model: LibraryModel) -> tuple[str, ...]:
    """The coding words the listing actually uses about a model, in the page's terms."""
    text = f"{model.name} {model.description}".lower()
    return tuple(w for w in CODING_WORDS if _mentions(w, text))


def suggest(models: list[LibraryModel], fetcher: Fetcher, *,
            max_size_gb: float | None = None,
            min_size_gb: float = 0.0,
            min_context: int | None = None,
            since: dt.date | None = None,
            require_coding: bool = False,
            match: str | None = None,
            exclude: str | None = None,
            installed: set[str] | None = None,
            include_installed: bool = False,
            limit: int = 20,
            sort: str = "date",
            on_skip=None) -> list[Suggestion]:
    """Keep the models that meet every stated constraint, in the requested order.

    Each constraint is a fact printed on the page: a capability badge, a date, a
    download size, a context window. Ordering is by one of those facts too. What the
    publisher claims about a model's ability is carried through as its description
    and left for the reader, since the listing offers no way to test it.

    The listing alone settles capability, age and text, and is one request. Only
    models that pass those are looked up individually, so tag pages are fetched for
    a shortlist rather than for the whole library.
    """
    installed = installed or set()
    kept: list[LibraryModel] = []
    hunt = re.compile(match, re.I) if match else None
    drop = re.compile(exclude, re.I) if exclude else None

    for model in models:
        if model.is_embedding:
            continue           # nothing to prompt: it returns vectors
        if not model.has_tools:
            continue           # a harness drives a model through tools or not at all
        if since and (not model.updated or model.updated < since):
            continue           # too old, or carries no date to show it is not
        if require_coding and not coding_claims(model):
            continue           # the listing does not claim it does programming work
        text = f"{model.name} {model.description}"
        if hunt and not hunt.search(text):
            continue
        if drop and drop.search(text):
            continue
        kept.append(model)

    # Ordering by download size would mean fetching every candidate's tag page before
    # any could be ranked, so size is a filter here rather than an ordering. Pull
    # counts are not offered as one either: they accumulate with age, so ranking by
    # them buries exactly the recent models this is meant to surface.
    order = {
        "date": lambda m: (m.updated or dt.date.min, m.name),
        "name": lambda m: m.name,
    }
    kept.sort(key=order.get(sort, order["date"]), reverse=(sort != "name"))

    out: list[Suggestion] = []
    for model in kept:
        if len(out) >= limit:
            break
        try:
            variants = parse_tags(model.name, fetcher.get(f"{LIBRARY_URL}/{model.name}/tags"))
        except LibraryError as exc:
            if on_skip:
                on_skip(model.name, str(exc))
            continue
        variant = _best_variant(variants, max_size_gb, min_context, min_size_gb)
        if variant is None:
            if on_skip:
                on_skip(model.name, "no local tag within the size and context limits")
            continue
        # The architecture lives on the page for the individual tag, and decides more
        # about how a model behaves on a given card than its parameter count does.
        try:
            arch, params = parse_model_page(fetcher.get(f"{LIBRARY_URL}/{variant.ref}"))
            variant = replace(variant, arch=arch, params=params)
        except LibraryError:
            pass               # the size and context already justify the suggestion
        # Whether a model is already held is a question about the weights, not the
        # name: having gemma4:26b says nothing about gemma4:31b, which has never been
        # measured, while :latest and :q4_K_M are routinely the same download under
        # two names. The manifest digest settles both cases; sibling tags are reported.
        have = tuple(sorted(n for n in installed
                            if ":" in n and n.partition(":")[0] == model.name
                            and n != variant.ref))
        already = variant.ref in installed or bool(
            variant.digest and variant.digest in installed)
        if already and not include_installed:
            continue
        out.append(Suggestion(model=model, variant=variant, installed=already, have=have))
    return out


def installed_names(host: str, timeout: float = 10.0) -> set[str]:
    """What the server already holds: every tag, its bare name, and its digest.

    The digest is what makes the comparison exact. A tag name can be an alias for
    weights already downloaded under another name, and only the manifest says so.
    """
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as resp:
            data = json.loads(resp.read())
    except Exception:
        return set()
    out = set()
    for entry in data.get("models") or []:
        name = entry.get("name") or ""
        if name:
            out.add(name)
            out.add(name.partition(":")[0])
        digest = (entry.get("digest") or "").removeprefix("sha256:")
        if len(digest) >= 12:
            out.add(digest[:12])
    return out


# ---------------------------------------------------------------- entry point

def parse_since(value: str, today: dt.date | None = None) -> dt.date:
    """Accepts a date, a bare year, or a count of months back from today."""
    today = today or dt.date.today()
    value = value.strip()
    if re.fullmatch(r"\d{4}", value):
        return dt.date(int(value), 1, 1)
    if re.fullmatch(r"\d+\s*m(onths?)?", value, re.I):
        months = int(re.match(r"\d+", value).group())
        return today - dt.timedelta(days=round(months * 30.44))
    try:
        return dt.datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"cannot read {value!r} as a date, year, or month count") from exc


def _table(rows: list[Suggestion], width: int = 64) -> str:
    """Every column is copied from the listing; the description is the publisher's."""
    head = ("#", "model", "arch", "params", "size", "context", "updated", "you have",
            "coding words", "described as")
    body = []
    for i, s in enumerate(rows, 1):
        desc = s.model.description
        body.append((
            str(i),
            s.variant.ref + (" *" if s.installed else ""),
            (s.variant.arch or "-") + (" moe" if s.variant.moe else ""),
            s.variant.params or "-",
            f"{s.variant.size_gb:.0f}GB" if s.variant.size_gb is not None else "-",
            f"{s.variant.context // 1024}K" if s.variant.context else "-",
            f"{s.model.updated:%Y-%m}" if s.model.updated else "-",
            ", ".join(t.partition(":")[2] for t in s.have[:2]) or "-",
            ", ".join(s.coding_words[:3]) or "-",
            desc if len(desc) <= width else desc[:width - 1].rstrip() + "\u2026",
        ))
    widths = [max(len(r[i]) for r in (head, *body)) for i in range(len(head) - 1)]
    lines = []
    for row in (head, *body):
        cells = [c.ljust(w) for c, w in zip(row, widths)] + [row[-1]]
        lines.append("  ".join(cells).rstrip())
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def run(cfg, *, since: str = "18m", max_size_gb: float | None = 32.0,
        min_size_gb: float = 4.0, min_context: int | None = None,
        require_coding: bool = False,
        match: str | None = None,
        exclude: str | None = None,
        include_installed: bool = False, limit: int = 20, sort: str = "date",
        as_json: bool = False, write_models: Path | None = None,
        cache_dir: Path | None = None, refresh: bool = False,
        fetcher: Fetcher | None = None, stream=None) -> int:
    """Print the models worth screening, and optionally write them as a model list."""
    stream = stream or sys.stdout
    cutoff = parse_since(since)
    min_context = cfg.ctx if min_context is None else min_context
    fetcher = fetcher or Fetcher(cache_dir or cfg.results_dir / "library-cache",
                                 refresh=refresh)

    try:
        models = parse_index(fetcher.get(LIBRARY_URL))
    except LibraryError as exc:
        print(f"could not read the library listing: {exc}", file=sys.stderr)
        return 1
    if not models:
        print("the library listing returned no entries; its markup may have changed",
              file=sys.stderr)
        return 1

    skipped: list[str] = []
    picks = suggest(models, fetcher,
                    max_size_gb=max_size_gb, min_size_gb=min_size_gb,
                    min_context=min_context, since=cutoff,
                    match=match, exclude=exclude,
                    require_coding=require_coding,
                    installed=installed_names(cfg.host),
                    include_installed=include_installed, limit=limit, sort=sort,
                    on_skip=lambda name, why: skipped.append(f"{name}: {why}"))

    if as_json:
        json.dump([s.as_dict() for s in picks], stream, indent=2)
        stream.write("\n")
    else:
        print(f"{len(models)} models listed, {len(picks)} worth screening "
              f"(updated since {cutoff}, {min_size_gb:g} to {max_size_gb:g}GB, "
              f"at least {min_context // 1024}K context, tool calling"
              f"{', naming coding work' if require_coding else ''}), "
              f"by {sort}", file=stream)
        print(file=stream)
        print(_table(picks) if picks else "nothing matched", file=stream)
        if any(s.installed for s in picks):
            print("\n* already on the server", file=stream)

    if write_models:
        write_models = Path(write_models)
        write_models.parent.mkdir(parents=True, exist_ok=True)
        write_models.write_text(
            "\n".join([f"# suggested by codesift discover on {dt.date.today()}"]
                      + [s.variant.ref for s in picks]) + "\n", encoding="utf-8")
        print(f"wrote {write_models}", file=sys.stderr)
    return 0
