"""Adaptive, offset-preserving chunking.

Invariant for every strategy: text[chunk.start:chunk.end] == chunk.text, and every
non-whitespace character is in some chunk. A search hit is therefore already a
citation (doc_hash + char span), and quotes found in chunks verify against the
full document.

Why adaptive: onboarding documents mix shapes, and one fixed window treats them all
badly - it cuts "Start Date:" from its value in a form, splits table rows, and
glues unrelated policy paragraphs together. So:

1. Segment lines into blocks: heading, kv ("Label: value"), list item, table row,
   prose. Blank lines end a block.
2. Group blocks into sections under their heading.
3. Profile each section by its dominant block kind and pick a strategy + size:
       kv     target 450 / max 700   never split a field from its value
       list   target 650 / max 900   cut only between items
       table  target 800 / max 1100  cut only between rows
       prose  target 700 / max 1000  semantic: cut where adjacent sentences are
                                     least similar (topic shift)
   A section that fits within its target stays whole.
4. Cut scoring: block boundaries (blank line) are natural cuts (-1); inside prose the
   score is the cosine similarity of the sentence windows on either side - the
   lowest-scoring boundary in the [min, max] window wins.
5. Overlap only when a cut was forced through coherent text (score above the
   section's own mean): the next chunk repeats the last unit. Natural cuts get no
   overlap, so we don't pay for duplicated text where it buys nothing.
6. Tiny sections (a heading and one line) merge with their neighbours.
7. Each chunk carries its kind and section heading; the heading is prepended for
   embedding (contextual retrieval) but never changes the chunk's offsets.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass(frozen=True)
class Chunk:
    idx: int
    start: int
    end: int
    text: str
    kind: str = "prose"
    heading: Optional[str] = None

    def embed_text(self) -> str:
        return f"{self.heading}\n{self.text}" if self.heading else self.text


@dataclass
class _Unit:
    start: int
    end: int
    kind: str
    block: int  # units in the same block share this id; a change = natural boundary


@dataclass
class _Section:
    heading: Optional[str]
    units: list[_Unit] = field(default_factory=list)


PARAMS = {  # (target, max)
    "kv": (450, 700),
    "list": (650, 900),
    "table": (800, 1100),
    "prose": (700, 1000),
}
MIN_CHARS = 200
SINGLE_CHUNK_MAX = 700

_KV = re.compile(r"^\s*[A-Za-z][\w /()&'.-]{0,40}:\s*\S")
_LIST = re.compile(r"^\s*(?:[-*•]|\d{1,3}[.)]|[a-z][.)])\s+\S")
_MD_HEADING = re.compile(r"^\s*#{1,6}\s+\S")
_SECTION_WORD = re.compile(r"^\s*(?:section|article|part|appendix)\s+[\w.]+", re.I)
_SENT_END = re.compile(r"(?<=[.!?])\s+(?=\S)")


def _classify(line: str) -> str:
    s = line.strip()
    if not s:
        return "blank"
    if _MD_HEADING.match(s):
        return "heading"
    if s.count("|") >= 2 or s.count("\t") >= 2:
        return "table"
    if _LIST.match(s):
        return "list"
    if _KV.match(s):
        return "kv"
    words = s.split()
    if len(s) <= 80 and len(words) <= 10 and s[-1] not in ".,;:!?":
        caps = sum(w[:1].isupper() or not w[:1].isalpha() for w in words) / len(words)
        if s.isupper() or caps >= 0.7 or _SECTION_WORD.match(s):
            return "heading"
    return "prose"


def _segment(text: str) -> list[_Section]:
    sections = [_Section(heading=None)]
    block_id = 0
    prev_kind = "blank"
    pos = 0
    para_start: Optional[int] = None
    para_end = 0

    def flush_para():
        nonlocal para_start, block_id
        if para_start is None:
            return
        block_id += 1
        span = text[para_start:para_end]
        cursor = 0
        for m in list(_SENT_END.finditer(span)) + [None]:
            end = m.start() if m else len(span)
            if span[cursor:end].strip():
                a, b = para_start + cursor, para_start + end
                while text[a].isspace():
                    a += 1
                sections[-1].units.append(_Unit(a, b, "prose", block_id))
            cursor = m.end() if m else len(span)
        para_start = None

    for line in text.split("\n"):
        start, end = pos, pos + len(line)
        pos = end + 1
        kind = _classify(line)
        if kind != "prose" or prev_kind == "blank":
            flush_para()
        if kind == "blank":
            prev_kind = "blank"
            continue
        a = start + (len(line) - len(line.lstrip()))
        b = start + len(line.rstrip())
        if kind == "heading":
            sections.append(_Section(heading=line.strip().lstrip("#").strip()))
            block_id += 1
            sections[-1].units.append(_Unit(a, b, "heading", block_id))
        elif kind == "prose":
            if para_start is None:
                para_start = a
            para_end = b
        else:
            if kind != prev_kind:
                block_id += 1
            sections[-1].units.append(_Unit(a, b, kind, block_id))
        prev_kind = kind
    flush_para()
    return [s for s in sections if s.units]


def _profile(sec: _Section) -> str:
    weight: dict[str, int] = {}
    for u in sec.units:
        if u.kind != "heading":
            weight[u.kind] = weight.get(u.kind, 0) + (u.end - u.start)
    return max(weight, key=weight.get) if weight else "prose"


def _boundary_scores(text: str, units: list[_Unit], embed: Optional[Callable]) -> list[float]:
    """score[i] = cohesion across the boundary between units[i] and units[i+1]:
    -1.0 = natural break (different block) - always preferred; prose: cosine of the
    2-sentence windows, clipped to [0, 1] (signed hashing can go negative, which
    must never outrank a real paragraph break)."""
    n = len(units)
    scores = [-1.0] * max(0, n - 1)
    prose_pairs = [i for i in range(n - 1) if units[i].block == units[i + 1].block and units[i].kind == "prose"]
    if prose_pairs and embed is not None:
        vecs = embed([text[u.start:u.end] for u in units])
        for i in prose_pairs:
            left = vecs[max(0, i - 1):i + 1].sum(axis=0)
            right = vecs[i + 1:i + 3].sum(axis=0)
            denom = (np.linalg.norm(left) * np.linalg.norm(right)) or 1.0
            scores[i] = min(1.0, max(0.0, float(left @ right / denom)))
    for i in range(n - 1):
        if units[i].block == units[i + 1].block and units[i].kind != "prose":
            scores[i] = 0.5  # inside a form / list / table: allowed, but not preferred
    return scores


def _hard_split(text: str, u: _Unit, max_chars: int) -> list[tuple[int, int]]:
    """A single unit longer than max (a run-on sentence): cut at whitespace."""
    spans, a = [], u.start
    while a < u.end:
        b = min(a + max_chars, u.end)
        if b < u.end:
            ws = text.rfind(" ", a + max_chars // 2, b)
            b = ws if ws > a else b
        spans.append((a, b))
        a = b
        while a < u.end and text[a].isspace():
            a += 1
    return spans


def _pack(text: str, sec: _Section, kind: str, embed) -> list[tuple[int, int, str]]:
    target, max_chars = PARAMS[kind]
    units = sec.units
    size = lambda i, j: units[j].end - units[i].start  # noqa: E731
    if size(0, len(units) - 1) <= target:
        return [(units[0].start, units[-1].end, kind)]
    scores = _boundary_scores(text, units, embed if kind == "prose" else None)
    prose_scores = [s for s, u in zip(scores, units) if u.kind == "prose" and s >= 0]
    forced_threshold = (sum(prose_scores) / len(prose_scores)) if prose_scores else 1.0
    out: list[tuple[int, int, str]] = []
    i = 0
    while i < len(units):
        if units[i].end - units[i].start > max_chars:
            out += [(a, b, kind) for a, b in _hard_split(text, units[i], max_chars)]
            i += 1
            continue
        j = i
        while j + 1 < len(units) and size(i, j + 1) <= max_chars:
            j += 1
        last = len(units) - 1
        if j == last and size(i, j) <= target:
            out.append((units[i].start, units[j].end, kind))
            break
        # Candidate cuts after unit k: chunk >= MIN, and if the rest would fit in this
        # chunk anyway, the remainder must also be >= MIN (no confetti chunks).
        cands = [k for k in range(i, j + 1 if j < last else last)
                 if size(i, k) >= MIN_CHARS and (j < last or size(k + 1, last) >= MIN_CHARS)]
        if not cands:
            if j == last:
                out.append((units[i].start, units[j].end, kind))
                break
            cands = [j]
        cut = min(cands, key=lambda k: (scores[k], -k))
        out.append((units[i].start, units[cut].end, kind))
        forced = scores[cut] > forced_threshold and units[cut].kind == "prose"
        i = cut if forced and cut > i else cut + 1  # overlap one unit only on forced cuts
    return out


def chunk_text(text: str, embed: Optional[Callable] = None, **_ignored) -> list[Chunk]:
    """Adaptive chunking (see module doc). `embed` (list[str] -> ndarray) enables
    semantic cuts in prose; without it, prose cuts fall back to block boundaries."""
    if not text.strip():
        return []
    sections = _segment(text)
    first, last = sections[0].units[0].start, sections[-1].units[-1].end
    if last - first <= SINGLE_CHUNK_MAX:
        kinds = {_profile(s) for s in sections}
        return [Chunk(0, first, last, text[first:last], kinds.pop() if len(kinds) == 1 else "mixed",
                      sections[0].heading or (sections[1].heading if len(sections) > 1 else None))]

    raw: list[tuple[int, int, str, Optional[str], bool]] = []  # start, end, kind, heading, whole_section
    for sec in sections:
        kind = _profile(sec)
        packed = _pack(text, sec, kind, embed)
        raw += [(a, b, k, sec.heading, len(packed) == 1) for a, b, k in packed]

    merged: list[list] = []
    for a, b, kind, heading, whole in raw:
        if merged:
            pa, pb, pkind, pheading, pwhole = merged[-1]
            small = (b - a) < MIN_CHARS and whole
            prev_small = (pb - pa) < MIN_CHARS and pwhole
            if (small or prev_small) and (b - pa) <= PARAMS["kv"][0] and pwhole and whole:
                merged[-1] = [pa, b, pkind if pkind == kind else "mixed", pheading or heading, True]
                continue
        merged.append([a, b, kind, heading, whole])
    return [Chunk(i, a, b, text[a:b], kind, heading) for i, (a, b, kind, heading, _w) in enumerate(merged)]
