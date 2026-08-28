# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
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
from dataclasses import dataclass
from pathlib import Path

LIBRARY_URL = "https://ollama.com/library"
USER_AGENT = "codesift"

# The listing shows capabilities as badges; these are the ones that matter here.
CAP_TOOLS = "tools"
CAP_CLOUD = "cloud"
CAP_EMBEDDING = "embedding"

CODING_WORDS = (
    "code", "coder", "coding", "codebase", "codebases", "programming",
    "software engineering", "swe-bench", "developer", "refactor", "refactoring",
    "debugging", "code completion", "fill-in-the-middle",
)

class LibraryError(RuntimeError):
    """The listing could not be read or understood."""


@dataclass(frozen=True)
class Variant:
    """One pullable tag, as the library listing names it.

    Everything here comes off the listing page, which is one request for the whole
    library. The size badge gives the parameter count; the context window and the
    architecture are not read from the tag pages, since triage tests the context a
    model actually holds and the probe times what it actually generates.
    """

    ref: str                     # "qwen3-coder:30b"
    label: str                   # the size badge as printed: "30b", "e4b", "8x7b"
    params_b: float | None       # parameter count in billions, or None if unstated

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

    @property
    def cloud_only(self) -> bool:
        """Served by Ollama's cloud with nothing to download.

        A cloud badge alone does not mean that: gemma4 and qwen3.5 are offered both
        ways and list their local sizes. A cloud badge with no size badge does --
        there is no tag to pull, so screening it locally is impossible.
        """
        return CAP_CLOUD in self.capabilities and not self.sizes


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
                    params_b=self.variant.params_b,
                    pulls=self.model.pulls,
                    updated=self.model.updated.isoformat() if self.model.updated else None,
                    capabilities=list(self.model.capabilities),
                    coding_words=list(self.coding_words),
                    installed=self.installed, have=list(self.have),
                    description=self.model.description)


def _get(url: str, timeout: float) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", "replace")
    except urllib.error.URLError as exc:
        raise LibraryError(f"{url}: {exc}") from exc


def fetch(url: str, timeout: float = 30.0) -> str:
    """One request, no cache: it costs about half a second, and a cached listing
    could hand back a stale view of a library that exists to say what is new."""
    return _get(url, timeout)


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


_SIZE = re.compile(r"^([\d.]+)([bm])$", re.I)          # 30b, 9b, 270m


def parse_size(label: str) -> float | None:
    """A size badge as a parameter count in billions, or None if it does not state one.

    Only plain counts are read. The other two forms look like numbers and are not:
    `e4b` is an effective count, and the model behind gemma4:e4b reports 8.0B
    parameters and ships 9.61GB, so reading it as four billion understates it by
    half. `8x7b` names experts, and Mixtral 8x7B totals about 46.7B rather than the
    56B the multiplication gives, because the experts share layers.

    A badge that cannot be read honestly is left unstated rather than guessed. The
    label itself is still printed, so the reader sees `e4b` and judges it.
    """
    m = _SIZE.match(label)
    if not m:
        return None
    n = float(m.group(1))
    return n if m.group(2).lower() == "b" else n / 1000


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


def pick_variant(model: LibraryModel, max_b: float | None,
                 min_b: float) -> Variant | None:
    """The largest tag within the parameter limits, or None if nothing fits.

    Largest, because a bigger model is the better screening candidate when both
    fit -- the point of a limit is to name what the card can hold, not to prefer
    the smallest thing under it.

    A badge that states no readable count -- `e4b`, `8x7b` -- cannot be ranked or
    limited, so it is used only when a model offers nothing else, and then the last
    one listed is taken, the listing being in ascending order. A model with no badge
    at all is offered as `latest`, which is how such a model is pulled. Both carry
    no parameter count rather than a guessed one, and neither is size-limited, since
    there is no size to test.
    """
    sized = [(b, s) for b, s in ((parse_size(s), s) for s in model.sizes)
             if b is not None]
    if not sized:
        if model.sizes:
            return Variant(f"{model.name}:{model.sizes[-1]}", model.sizes[-1], None)
        return Variant(f"{model.name}:latest", "latest", None)
    fits = [(b, s) for b, s in sized if (max_b is None or b <= max_b) and b >= min_b]
    if not fits:
        return None
    b, label = max(fits)
    return Variant(f"{model.name}:{label}", label, b)


def suggest(models: list[LibraryModel], *,
            max_params_b: float | None = None,
            min_params_b: float = 0.0,
            since: dt.date | None = None,
            require_coding: bool = False,
            match: str | None = None,
            exclude: str | None = None,
            installed: set[str] | None = None,
            include_installed: bool = False,
            sort: str = "date",
            on_skip=None) -> list[Suggestion]:
    """Keep the models that meet every stated constraint, in the requested order.

    Every constraint is a fact the listing prints: a capability badge, a date, a
    size badge, the publisher's own description. One request answers all of them,
    so this reads the whole library in about a second.

    What the publisher claims about a model's ability is carried through as its
    description and left for the reader, since the listing offers no way to test it.
    Nothing here is a measurement -- the stages after it are.
    """
    installed = installed or set()
    hunt = re.compile(match, re.I) if match else None
    drop = re.compile(exclude, re.I) if exclude else None

    kept: list[LibraryModel] = []
    for model in models:
        if model.is_embedding:
            continue           # nothing to prompt: it returns vectors
        if not model.has_tools:
            continue           # a harness drives a model through tools or not at all
        if model.cloud_only:
            continue           # served by Ollama's cloud and never pullable
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

    # Not by pulls: they accumulate with age, so ranking on them buries exactly
    # the recent models this exists to surface.
    order = {
        "date": lambda m: (m.updated or dt.date.min, m.name),
        "name": lambda m: m.name,
    }
    kept.sort(key=order.get(sort, order["date"]), reverse=(sort != "name"))

    # Not capped: the filters are what narrow the list, and cutting a sorted list
    # at some length drops matches for no reason the reader can see.
    out: list[Suggestion] = []
    for model in kept:
        variant = pick_variant(model, max_params_b, min_params_b)
        if variant is None:
            if on_skip:
                on_skip(model.name, "no tag within the parameter limits")
            continue
        # Sibling tags are reported rather than counted as having it: holding
        # gemma4:26b says nothing about gemma4:31b, which has never been measured.
        have = tuple(sorted(n for n in installed
                            if ":" in n and n.partition(":")[0] == model.name
                            and n != variant.ref))
        already = variant.ref in installed
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


def _table(rows: list[Suggestion]) -> str:
    """Every column is copied from the listing, and only what bears on whether to
    screen a model. The pull count is left out, since it measures how long a model
    has been popular rather than whether it is worth an hour of GPU time, and so is
    the description, which does not survive a cut to terminal width. `--json`
    carries everything.
    """
    head = ("model", "params", "updated", "coding")
    body = []
    for s in rows:
        body.append((
            s.variant.ref + (" *" if s.installed else ""),
            f"{s.variant.params_b:g}B" if s.variant.params_b else "-",
            f"{s.model.updated:%Y-%m}" if s.model.updated else "-",
            # Whether the listing claims programming work, not which words it used.
            "yes" if s.coding_words else "no",
        ))
    widths = [max(len(r[i]) for r in (head, *body)) for i in range(len(head))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip()
             for row in (head, *body)]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def run(cfg, *, since: str = "18m", max_params_b: float | None = 70.0,
        min_params_b: float = 4.0,
        require_coding: bool = False,
        match: str | None = None,
        exclude: str | None = None,
        include_installed: bool = False, sort: str = "date",
        as_json: bool = False, write_models: Path | None = None,
        fetcher=None, stream=None) -> int:
    """Print the models worth screening, and optionally write them as a model list."""
    stream = stream or sys.stdout
    cutoff = parse_since(since)
    read = fetcher.get if fetcher is not None else fetch
    try:
        models = parse_index(read(LIBRARY_URL))
    except LibraryError as exc:
        print(f"could not read the library listing: {exc}", file=sys.stderr)
        return 1
    if not models:
        print("the library listing returned no entries; its markup may have changed",
              file=sys.stderr)
        return 1

    skipped: list[str] = []
    picks = suggest(models,
                    max_params_b=max_params_b, min_params_b=min_params_b,
                    since=cutoff, match=match, exclude=exclude,
                    require_coding=require_coding,
                    installed=installed_names(cfg.host),
                    include_installed=include_installed, sort=sort,
                    on_skip=lambda name, why: skipped.append(f"{name}: {why}"))

    if as_json:
        json.dump([s.as_dict() for s in picks], stream, indent=2)
        stream.write("\n")
    else:
        span = (f"{min_params_b:g}B to {max_params_b:g}B" if max_params_b
                else f"{min_params_b:g}B and up")
        print(f"{len(models)} models listed, {len(picks)} worth screening "
              f"(updated since {cutoff}, {span}, tool calling"
              f"{', naming coding work' if require_coding else ''}), "
              f"by {sort}", file=stream)
        print(file=stream)
        print(_table(picks) if picks else "nothing matched", file=stream)
        if any(s.installed for s in picks):
            print("\n* already on the server", file=stream)

    if write_models:
        write_models = Path(write_models)
        write_models.parent.mkdir(parents=True, exist_ok=True)
        # Written as pull commands rather than as bare names: nothing can be
        # screened before it is installed, and `--models-file` strips the prefix
        # back off, so one file serves both steps.
        write_models.write_text(
            "\n".join([f"# suggested by codesift discover on {dt.date.today()}"]
                      + [f"ollama pull {s.variant.ref}" for s in picks]) + "\n",
            encoding="utf-8")
        print(f"wrote {write_models}", file=sys.stderr)
    return 0
