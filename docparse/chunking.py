"""Element-aware chunking: ExtractionResult -> retrieval-ready Chunks.

Design notes (why this differs from the naive "10 chunk types" recipe):

* Chunks are built from parsed Elements, never from raw character streams —
  the whole point of docparse is that structure survives to this stage.
* Text is packed sentence-by-sentence inside a (section_path, role) group up
  to a token budget, with sentence-level (not token-slice) overlap. Headings
  are never chunks themselves; they become the breadcrumb prefix of
  ``embed_text`` so every vector carries its ancestry.
* Claim chunks embed the claim sentence + a *compact* local context window,
  not the whole parent paragraph — otherwise claim vectors are near-duplicates
  of their section chunks. Each claim links to its parent chunk so retrieval
  can deduplicate.
* Tables get a summary chunk + row-group chunks (key-value serialization,
  header semantics repeated in every group). Per-cell "triple" chunks are OFF
  by default: a 25-row x 5-col table would explode into 100+ near-identical
  micro-vectors that crowd top-k. The triples are instead stored as
  ``metadata["facts"]`` on each row-group chunk for exact-lookup post-filters.
* Figure chunks bundle caption + VLM description + the paragraphs that mention
  the figure. Panel chunks are emitted only when a VLM actually produced
  panels — the pipeline must degrade gracefully when enrichment wasn't run.
* Reference chunks are emitted but flagged ``retrieval_default: exclude`` so a
  retriever doesn't return bibliography entries as evidence unless asked.
* Every chunk carries char offsets into ``result.document_text()`` so a
  long-context embedder can do *late chunking* (embed the whole section, pool
  the chunk span) without re-parsing.
* ``ChunkingResult.stats`` reports per-type counts and a token histogram —
  chunking is an indexing decision that must be benchmarked, not trusted.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from docparse.models import Element, ExtractionResult
from docparse.serialize import rows_to_kv as _rows_to_kv

# --------------------------------------------------------------------------- schema

ChunkType = str  # section_text | claim | table_summary | table_row_group | table_fact
#                  figure | figure_panel | formula | reference | summary


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    chunk_type: ChunkType
    text: str                                   # what a human reads / the LLM sees
    embed_text: str = ""                        # breadcrumb-prefixed text to embed
    section_path: list[str] = field(default_factory=list)
    role: str | None = None                     # abstract/methods/results/...
    pages: list[int] = field(default_factory=list)
    element_ids: list[str] = field(default_factory=list)
    char_start: int | None = None               # span into document_text() -> late chunking
    char_end: int | None = None
    citations: list[dict] = field(default_factory=list)
    quantities: list[dict] = field(default_factory=list)
    linked_figures: list[str] = field(default_factory=list)   # "Figure 3"
    linked_tables: list[str] = field(default_factory=list)    # "Table 1"
    linked_formulas: list[int] = field(default_factory=list)  # equation numbers
    parent_chunk_id: str | None = None          # claim -> its section chunk, panel -> figure
    visual_ref: dict | None = None
    token_estimate: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v not in (None, [], {}, "")}


@dataclass
class ChunkingConfig:
    # token budgets (estimated; see _est_tokens)
    target_tokens: int = 400          # pack sentences up to this
    max_tokens: int = 560             # hard ceiling before forced split
    min_tokens: int = 40              # merge smaller chunks into a neighbour
    overlap_sentences: int = 1        # sentence-level overlap between adjacent chunks

    # claims
    claim_context_sentences: int = 1  # sentences of local context each side of a claim

    # tables
    table_rows_per_group: int = 6
    emit_table_facts: bool = False    # per-cell triple chunks (OFF: index bloat)

    # figures
    emit_figure_panels: bool = True   # only fires when VLM panels exist
    figure_mention_max: int = 3       # nearby mentioning paragraphs to bundle

    # references
    emit_references: bool = True      # flagged retrieval_default: exclude

    # summaries
    summary_roles: tuple[str, ...] = ("abstract", "conclusion")
    summary_max_tokens: int = 600     # hard cap; truncate at sentence boundary

    # defense-in-depth: sentences/lines matching these never enter chunk text,
    # even if upstream cleaning missed them (extend per corpus)
    boilerplate_patterns: tuple[str, ...] = (
        r"open access article under the terms",
        r"creative ?commons licen[cs]e", r"creativecommons\.org",
        r"©\s*(?:19|20)\d{2}", r"all rights reserved",
        r"^\s*\d+\s+of\s+\d+\s*$",                    # "1 of 17"
        r"https?://doi\.org/\S+",
        r"downloaded (?:from|by)", r"terms and conditions",
        r"wiley online library", r"rules of use",
        r"^\s*(?:international\s+)?journal of [a-z][a-z ]{0,40}$",
    )
    min_section_tokens: int = 120     # merge adjacent section chunks below this


@dataclass
class ChunkingResult:
    doc_id: str
    source: str
    chunks: list[Chunk] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"doc_id": self.doc_id, "source": self.source, "stats": self.stats,
                "warnings": self.warnings, "chunks": [c.to_dict() for c in self.chunks]}


# --------------------------------------------------------------------------- helpers

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(“\"'0-9])")
_WORD = re.compile(r"\S+")
_FIG_MENTION = re.compile(r"\b(?:fig(?:ure)?s?\.?)\s*(\d+[A-Za-z]?)", re.I)
_TAB_MENTION = re.compile(r"\btables?\s*(\d+[A-Za-z]?)", re.I)
_EQ_MENTION = re.compile(r"\b(?:eq(?:uation)?s?\.?)\s*\(?(\d{1,2})\)?", re.I)

_TEXT_KINDS = ("narrative", "list_item")
_NUMISH = re.compile(r"\d|°C|%")


def _boiler_re(patterns: tuple[str, ...]) -> re.Pattern:
    return re.compile("|".join(patterns), re.IGNORECASE)


def _scrub(text: str, rx: re.Pattern) -> str:
    """Drop boilerplate lines, then boilerplate sentences, from a text block."""
    lines = [ln for ln in text.split("\n") if not rx.search(ln) or len(ln) > 300]
    out_sents = []
    for ln in lines:
        for s_ in _sentences(ln):
            if not rx.search(s_):
                out_sents.append(s_)
    return " ".join(out_sents)


def _fake_section(entry: str) -> bool:
    """Mirror of scientific._fake_heading (same exemptions): table headers,
    glyph garbage, and front-matter banners must not survive as section-path
    entries even if extraction let them through."""
    from docparse.scientific import _FRONT_MATTER_HEADING, _fake_heading
    return bool(_FRONT_MATTER_HEADING.match(entry)) or _fake_heading(entry)


def _clean_path(section_path: Iterable[str]) -> list[str]:
    return [re.sub(r"\s+", " ", p).strip()
            for p in section_path if p and not _fake_section(p)]


def _est_tokens(text: str) -> int:
    """Cheap, deterministic token estimate (~1.33 tokens/word for English prose)."""
    return max(1, round(len(_WORD.findall(text)) * 4 / 3))


def _sentences(text: str, max_tokens: int | None = None) -> list[str]:
    parts = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    parts = parts or ([text.strip()] if text.strip() else [])
    if max_tokens is None:
        return parts
    out: list[str] = []
    for p in parts:
        if _est_tokens(p) <= max_tokens:
            out.append(p)
            continue
        words = _WORD.findall(p)                 # punctuation-free run: word windows
        step = max(1, int(max_tokens * 3 / 4))   # tokens -> words
        for i in range(0, len(words), step):
            out.append(" ".join(words[i:i + step]))
    return out


def _breadcrumb(title: str | None, section_path: Iterable[str], extra: str = "") -> str:
    bits = [b for b in ([title] if title else []) + list(section_path) if b]
    dedup: list[str] = []
    for b in bits:
        if not dedup or dedup[-1] != b:
            dedup.append(b)
    crumb = " > ".join(dedup)
    if extra:
        crumb = f"{crumb} > {extra}" if crumb else extra
    return crumb


def _links_from_text(chunk: Chunk, text: str) -> None:
    chunk.linked_figures = sorted({f"Figure {m}" for m in _FIG_MENTION.findall(text)})
    chunk.linked_tables = sorted({f"Table {m}" for m in _TAB_MENTION.findall(text)})
    chunk.linked_formulas = sorted({int(m) for m in _EQ_MENTION.findall(text)})


def _collect_meta(chunk: Chunk, els: list[Element]) -> None:
    cites, qtys = [], []
    for el in els:
        cites += el.metadata.get("citations", []) or []
        qtys += el.metadata.get("quantities_labeled", []) or []
    seen: set[str] = set()
    chunk.citations = [c for c in cites if not (c.get("key") in seen or seen.add(c.get("key")))]
    seen = set()
    chunk.quantities = [q for q in qtys if not (q.get("value") in seen or seen.add(q.get("value")))]
    pages = sorted({el.page for el in els if el.page})
    chunk.pages = pages
    chunk.element_ids = [el.id for el in els]
    starts = [el.char_start for el in els if el.char_start is not None]
    ends = [el.char_end for el in els if el.char_end is not None]
    if starts and ends:
        chunk.char_start, chunk.char_end = min(starts), max(ends)


class _Ids:
    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.counters: dict[str, int] = {}

    def next(self, kind: str) -> str:
        n = self.counters.get(kind, 0) + 1
        self.counters[kind] = n
        return f"{self.doc_id}::{kind}::{n:04d}"


# --------------------------------------------------------------------------- builders

def _build_section_text_chunks(result: ExtractionResult, cfg: ChunkingConfig,
                               ids: _Ids, title: str | None) -> list[Chunk]:
    """Group consecutive narrative/list_item elements by (section_path, role),
    then pack sentences to the token budget with sentence-level overlap."""
    chunks: list[Chunk] = []
    group: list[Element] = []
    key: tuple | None = None

    def flush() -> None:
        nonlocal group
        if not group:
            return
        chunks.extend(_pack_group(group, cfg, ids, title))
        group = []

    for el in result.elements:
        if el.kind not in _TEXT_KINDS or not el.text:
            if el.kind in ("heading", "title") and not el.metadata.get("pseudo_heading"):
                flush()
            continue
        if el.metadata.get("role") == "references":
            continue                                     # handled as reference chunks
        k = (tuple(el.section_path), el.metadata.get("role"))
        if k != key:
            flush()
            key = k
        group.append(el)
    flush()

    # merge undersized trailing chunks into their predecessor within a section
    merged: list[Chunk] = []
    for c in chunks:
        prev = merged[-1] if merged else None
        tiny = c.token_estimate < cfg.min_tokens
        if (prev is not None and tiny
                and prev.token_estimate + c.token_estimate <= cfg.max_tokens):
            if prev.section_path != c.section_path:
                prev.metadata.setdefault("absorbed_sections", []).append(
                    " > ".join(c.section_path))
            prev.text = f"{prev.text}\n\n{c.text}"
            prev.embed_text = f"{prev.embed_text}\n\n{c.text}"
            prev.element_ids = list(dict.fromkeys(prev.element_ids + c.element_ids))
            prev.pages = sorted(set(prev.pages) | set(c.pages))
            prev.citations += [x for x in c.citations if x not in prev.citations]
            prev.quantities += [x for x in c.quantities if x not in prev.quantities]
            prev.char_end = c.char_end or prev.char_end
            prev.token_estimate = _est_tokens(prev.text)
            _links_from_text(prev, prev.text)
        else:
            merged.append(c)

    # pass 2: adjacent section chunks below min_section_tokens merge upward when
    # they share role + top-level section (two-column papers fragment badly)
    packed: list[Chunk] = []
    for c in merged:
        prev = packed[-1] if packed else None
        same_scope = (prev is not None and prev.role == c.role
                      and prev.section_path[:1] == c.section_path[:1])
        if (same_scope
                and min(prev.token_estimate, c.token_estimate) < cfg.min_section_tokens
                and prev.token_estimate + c.token_estimate <= cfg.target_tokens):
            joiner = "\n\n" if prev.section_path != c.section_path else " "
            prev.text = f"{prev.text}{joiner}{c.text}"
            prev.embed_text = f"{prev.embed_text}{joiner}{c.text}"
            prev.element_ids = list(dict.fromkeys(prev.element_ids + c.element_ids))
            prev.pages = sorted(set(prev.pages) | set(c.pages))
            prev.citations += [x for x in c.citations if x not in prev.citations]
            prev.quantities += [x for x in c.quantities if x not in prev.quantities]
            prev.char_end = c.char_end or prev.char_end
            prev.token_estimate = _est_tokens(prev.text)
            _links_from_text(prev, prev.text)
        else:
            packed.append(c)
    return packed


def _pack_group(group: list[Element], cfg: ChunkingConfig, ids: _Ids,
                title: str | None) -> list[Chunk]:
    section_path = _clean_path(group[0].section_path)
    role = group[0].metadata.get("role")
    crumb = _breadcrumb(title, section_path)

    # sentence stream, remembering which element each sentence came from;
    # boilerplate sentences are dropped here so they never reach an embedding
    rx = _boiler_re(cfg.boilerplate_patterns)
    stream: list[tuple[str, Element]] = []
    for el in group:
        for s in _sentences(el.text, max_tokens=cfg.max_tokens):
            if not rx.search(s):
                stream.append((s, el))

    out: list[Chunk] = []
    buf: list[tuple[str, Element]] = []
    buf_tokens = 0

    def emit() -> None:
        nonlocal buf, buf_tokens
        if not buf:
            return
        text = " ".join(s for s, _ in buf)
        c = Chunk(chunk_id=ids.next("sec"), doc_id=ids.doc_id, chunk_type="section_text",
                  text=text, section_path=section_path, role=role,
                  embed_text=f"[{crumb}]\n{text}" if crumb else text,
                  token_estimate=_est_tokens(text))
        uniq: dict[str, Element] = {}
        for _, el in buf:
            uniq.setdefault(el.id, el)
        _collect_meta(c, list(uniq.values()))
        _links_from_text(c, text)
        out.append(c)
        # sentence-level overlap into the next buffer — but never carry over a
        # huge piece (word-window splits would then duplicate whole chunks)
        tail = buf[-cfg.overlap_sentences:] if cfg.overlap_sentences else []
        if sum(_est_tokens(s) for s, _ in tail) > max(1, cfg.target_tokens // 4):
            tail = []
        buf = list(tail)
        buf_tokens = sum(_est_tokens(s) for s, _ in buf)

    def pieces(sent: str) -> list[str]:
        if _est_tokens(sent) <= cfg.max_tokens:
            return [sent]
        words = sent.split()
        step = max(1, cfg.target_tokens * 3 // 4)          # words per piece
        return [" ".join(words[i:i + step]) for i in range(0, len(words), step)]

    for raw_sent, el in stream:
        for sent in pieces(raw_sent):
            t = _est_tokens(sent)
            if buf and (buf_tokens + t > cfg.max_tokens
                        or (buf_tokens >= cfg.target_tokens)):
                emit()
            buf.append((sent, el))
            buf_tokens += t
    if buf:
        # avoid emitting an overlap-only remainder that duplicates the previous chunk
        if not (out and len(buf) <= cfg.overlap_sentences):
            emit()
    return out


def _build_claim_chunks(result: ExtractionResult, cfg: ChunkingConfig, ids: _Ids,
                        title: str | None, section_chunks: list[Chunk]) -> list[Chunk]:
    claims = result.stats.get("claims") or []
    by_id = {e.id: e for e in result.elements}
    # map element -> containing section chunk for dedup links
    el_to_chunk = {eid: c.chunk_id for c in section_chunks for eid in c.element_ids}
    out: list[Chunk] = []
    seen_texts: set[str] = set()
    for cl in claims:
        text = (cl.get("text") or "").strip()
        if not text or text in seen_texts:
            continue
        seen_texts.add(text)
        el = by_id.get(cl.get("element_id"))
        context = ""
        if el is not None and cfg.claim_context_sentences:
            sents = _sentences(el.text)
            try:
                i = next(j for j, s in enumerate(sents) if text[:60] in s or s in text)
            except StopIteration:
                i = None
            if i is not None:
                lo = max(0, i - cfg.claim_context_sentences)
                hi = min(len(sents), i + cfg.claim_context_sentences + 1)
                context = " ".join(sents[lo:i] + sents[i + 1:hi])
        crumb = _breadcrumb(title, el.section_path if el is not None else [],
                            extra=f"claim/{cl.get('type', 'claim')}")
        body = text if not context else f"{text}\nContext: {context}"
        c = Chunk(chunk_id=ids.next("claim"), doc_id=ids.doc_id, chunk_type="claim",
                  text=body, embed_text=f"[{crumb}]\n{body}" if crumb else body,
                  role=cl.get("role"), token_estimate=_est_tokens(body),
                  parent_chunk_id=el_to_chunk.get(cl.get("element_id")),
                  metadata={"claim_type": cl.get("type")})
        if el is not None:
            c.section_path = _clean_path(el.section_path)
            c.element_ids = [el.id]
            c.pages = [el.page] if el.page else []
            c.char_start, c.char_end = el.char_start, el.char_end
        c.quantities = cl.get("quantities", [])
        _links_from_text(c, el.text if el is not None else text)
        out.append(c)
    return out


def _table_label(el: Element, captions: dict[str, Element]) -> tuple[str | None, str]:
    label = el.metadata.get("label") or (el.structured or {}).get("table_number")
    cap = captions.get(el.id)
    return label, (cap.text if cap is not None else "")


def _build_table_chunks(result: ExtractionResult, cfg: ChunkingConfig, ids: _Ids,
                        title: str | None) -> list[Chunk]:
    captions = {e.parent_id: e for e in result.elements if e.kind == "caption" and e.parent_id}
    out: list[Chunk] = []
    for el in result.elements:
        if el.kind != "table" or not isinstance(el.structured, dict):
            continue
        header = el.structured.get("header") or []
        rows = el.structured.get("rows") or []
        label, caption = _table_label(el, captions)
        name = label or "Table"
        crumb = _breadcrumb(title, el.section_path, extra=name)

        # 1) summary chunk — discovery ("which table has X?")
        cols = ", ".join(h for h in header if h) or "(no header)"
        summary = (f"{name}: {caption}" if caption else f"{name}.")
        summary += f"\nColumns: {cols}. Rows: {len(rows)}."
        sc = Chunk(chunk_id=ids.next("tab"), doc_id=ids.doc_id, chunk_type="table_summary",
                   text=summary, embed_text=f"[{crumb}]\n{summary}",
                   section_path=_clean_path(el.section_path), role=el.metadata.get("role"),
                   token_estimate=_est_tokens(summary), visual_ref=el.visual_ref,
                   metadata={"label": label, "n_rows": len(rows), "columns": header})
        _collect_meta_single(sc, el, captions.get(el.id))
        out.append(sc)

        # 2) row-group chunks — lookup ("what is the 1.5°C year for CanESM5?")
        for gi in range(0, len(rows), cfg.table_rows_per_group):
            grp = rows[gi:gi + cfg.table_rows_per_group]
            kv, _ = _rows_to_kv([header] + grp, max_rows=len(grp) + 1)
            body = (f"{name} ({caption[:160]})\n{kv}" if caption else f"{name}\n{kv}")
            rc = Chunk(chunk_id=ids.next("tabrows"), doc_id=ids.doc_id,
                       chunk_type="table_row_group", text=body,
                       embed_text=f"[{crumb} rows {gi + 1}-{gi + len(grp)}]\n{body}",
                       section_path=_clean_path(el.section_path), role=el.metadata.get("role"),
                       parent_chunk_id=sc.chunk_id, token_estimate=_est_tokens(body),
                       visual_ref=el.visual_ref,
                       metadata={"label": label, "row_start": gi + 1,
                                 "row_end": gi + len(grp),
                                 # per-cell facts live HERE, not as separate vectors
                                 "facts": [{"row_key": r[0] if r else "",
                                            "column": h, "value": v}
                                           for r in grp
                                           for h, v in zip(header[1:], r[1:]) if v]})
            _collect_meta_single(rc, el, None)
            out.append(rc)

            if cfg.emit_table_facts:                      # opt-in micro-fact vectors
                for f in rc.metadata["facts"]:
                    ft = f"{name}: {f['row_key']} — {f['column']}: {f['value']}."
                    out.append(Chunk(chunk_id=ids.next("fact"), doc_id=ids.doc_id,
                                     chunk_type="table_fact", text=ft,
                                     embed_text=f"[{crumb}]\n{ft}",
                                     parent_chunk_id=rc.chunk_id,
                                     section_path=_clean_path(el.section_path),
                                     token_estimate=_est_tokens(ft),
                                     metadata={"label": label, **f}))
    return out


def _collect_meta_single(chunk: Chunk, el: Element, cap: Element | None) -> None:
    els = [el] + ([cap] if cap is not None else [])
    _collect_meta(chunk, els)


def _build_figure_chunks(result: ExtractionResult, cfg: ChunkingConfig, ids: _Ids,
                         title: str | None) -> list[Chunk]:
    captions = {e.parent_id: e for e in result.elements if e.kind == "caption" and e.parent_id}
    # index paragraphs by the figure labels they mention
    mentions: dict[str, list[Element]] = {}
    for el in result.elements:
        if el.kind in _TEXT_KINDS and el.text:
            for m in _FIG_MENTION.findall(el.text):
                mentions.setdefault(f"Figure {m}", []).append(el)

    out: list[Chunk] = []
    for el in result.elements:
        if el.kind != "figure":
            continue
        cap = captions.get(el.id)
        label = el.metadata.get("label") or (cap.metadata.get("label") if cap is not None else None)
        if not label and cap is not None and cap.text:
            if (m := re.match(r"\s*(FIGURE|FIG\.?)\s+(\d+)", cap.text, re.I)):
                label = f"Figure {m.group(2)}"
        name = label or "Figure"
        crumb = _breadcrumb(title, el.section_path, extra=name)
        parts = []
        if cap is not None and cap.text:
            parts.append(f"Caption: {cap.text}")
        if el.text:                                        # VLM description, when enriched
            parts.append(f"Description: {el.text}")
        near = [p.text for p in (mentions.get(label, []) if label else [])
                [:cfg.figure_mention_max]]
        if near:
            parts.append("Mentioned in text: " + " ".join(near))
        if not parts:
            continue                                       # nothing retrievable
        body = f"{name}.\n" + "\n".join(parts)
        fc = Chunk(chunk_id=ids.next("fig"), doc_id=ids.doc_id, chunk_type="figure",
                   text=body, embed_text=f"[{crumb}]\n{body}",
                   section_path=_clean_path(el.section_path), role=el.metadata.get("role"),
                   token_estimate=_est_tokens(body), visual_ref=el.visual_ref,
                   metadata={"label": label,
                             "figure_type": (el.structured or {}).get("figure_type")
                             or el.metadata.get("vlm_type")})
        _collect_meta_single(fc, el, cap)
        out.append(fc)

        panels = ((el.structured or {}).get("panels") or []) if cfg.emit_figure_panels else []
        if cfg.emit_figure_panels and not panels and el.text:
            panels = _panels_from_text(el.text)         # VLM prose, no JSON panels
        for p in panels:
            if not isinstance(p, dict):
                continue
            bits = [f"{name}{p.get('id', '')}: {p.get('label') or ''}".strip()]
            for k in ("variable", "unit", "takeaway"):
                if p.get(k):
                    bits.append(f"{k}: {p[k]}")
            for nv in p.get("notable_values") or []:
                if isinstance(nv, dict):
                    bits.append(f"{nv.get('item', '')}: {nv.get('value', '')}")
            ptext = ". ".join(b for b in bits if b)
            if _est_tokens(ptext) < 8:
                continue                                   # not worth a vector
            out.append(Chunk(chunk_id=ids.next("figpanel"), doc_id=ids.doc_id,
                             chunk_type="figure_panel", text=ptext,
                             embed_text=f"[{crumb} panel {p.get('id', '?')}]\n{ptext}",
                             parent_chunk_id=fc.chunk_id,
                             section_path=_clean_path(el.section_path),
                             pages=[el.page] if el.page else [],
                             visual_ref=el.visual_ref,
                             token_estimate=_est_tokens(ptext),
                             metadata={"label": label, "panel": p.get("id")}))
    return out


_PANEL_MARK = re.compile(
    r"(?:\bpanels?\s+\(?([A-Ha-h])\)?\b[:.]?|(?:^|[.;]\s+)\(([a-hA-H])\)\s)", re.I)


def _panels_from_text(desc: str) -> list[dict]:
    """Synthesize panel dicts from a prose figure description ("Panel A shows
    ..., (b) the change at 1.5°C, ...") when the VLM returned no JSON panels.
    Requires at least two distinct markers to avoid false positives."""
    hits = [(m.start(), (m.group(1) or m.group(2)).upper())
            for m in _PANEL_MARK.finditer(desc)]
    dedup: list[tuple[int, str]] = []
    for pos, pid in hits:
        if not any(p == pid for _, p in dedup):
            dedup.append((pos, pid))
    if len(dedup) < 2:
        return []
    panels = []
    for i, (pos, pid) in enumerate(dedup):
        end = dedup[i + 1][0] if i + 1 < len(dedup) else len(desc)
        seg = desc[pos:end].strip().rstrip(";,")
        panels.append({"id": pid, "takeaway": seg[:400]})
    return panels


def _build_formula_chunks(result: ExtractionResult, ids: _Ids, title: str | None) -> list[Chunk]:
    by_id = {e.id: e for e in result.elements}
    out: list[Chunk] = []
    for el in result.elements:
        if el.kind != "formula":
            continue
        s = el.structured or {}
        num = s.get("number")
        name = f"Equation ({num})" if num else "Equation"
        before = by_id.get(el.metadata.get("context_before"))
        after = by_id.get(el.metadata.get("context_after"))
        parts = [f"{name}: {s.get('expression') or el.text}"]
        if before is not None:
            parts.append(f"Context before: {_sentences(before.text)[-1]}")
        if after is not None:
            parts.append(f"Context after: {_sentences(after.text)[0]}")
        body = "\n".join(parts)
        crumb = _breadcrumb(title, el.section_path, extra=name)
        c = Chunk(chunk_id=ids.next("eq"), doc_id=ids.doc_id, chunk_type="formula",
                  text=body, embed_text=f"[{crumb}]\n{body}",
                  section_path=_clean_path(el.section_path), role=el.metadata.get("role"),
                  token_estimate=_est_tokens(body),
                  metadata={"equation_number": num, "expression": s.get("expression")})
        _collect_meta(c, [el])
        out.append(c)
    return out


def _build_reference_chunks(result: ExtractionResult, ids: _Ids) -> list[Chunk]:
    graph = result.stats.get("citation_graph") or {}
    ref_to_key = {v.get("ref_id"): k for k, v in graph.items() if v.get("ref_id")}
    ref_mentions = {v.get("ref_id"): v.get("mentions", []) for v in graph.values()}
    out = []
    for el in result.elements:
        if el.kind != "reference" or not el.text:
            continue
        c = Chunk(chunk_id=ids.next("ref"), doc_id=ids.doc_id, chunk_type="reference",
                  text=el.text, embed_text=el.text, role="references",
                  token_estimate=_est_tokens(el.text),
                  metadata={"citation_key": ref_to_key.get(el.id),
                            "mentioned_by_elements": ref_mentions.get(el.id, []),
                            "retrieval_default": "exclude"})
        _collect_meta(c, [el])
        out.append(c)
    return out


def _build_summary_chunks(result: ExtractionResult, cfg: ChunkingConfig, ids: _Ids,
                          title: str | None) -> list[Chunk]:
    out = []
    for role in cfg.summary_roles:
        # first CONTIGUOUS run of this role only — if role propagation ever
        # leaks (bad heading detection), the summary must not absorb the paper
        els, in_run, seen = [], False, False
        for e in result.elements:
            if e.kind in _TEXT_KINDS and e.metadata.get("role") == role and e.text:
                if seen and not in_run:
                    break
                in_run = seen = True
                els.append(e)
            else:
                in_run = False
        if not els:
            continue
        rx = _boiler_re(cfg.boilerplate_patterns)
        text = "\n".join(_scrub(e.text, rx) for e in els)
        text = "\n".join(ln for ln in text.split("\n") if ln.strip())
        if _est_tokens(text) < cfg.min_tokens:
            continue                                    # role misfire / fragment
        if _est_tokens(text) > cfg.summary_max_tokens:  # cap at sentence boundary
            kept: list[str] = []
            for s_ in _sentences(text):
                if _est_tokens(" ".join(kept + [s_])) > cfg.summary_max_tokens:
                    break
                kept.append(s_)
            text = " ".join(kept)
        crumb = _breadcrumb(title, [], extra=role)
        c = Chunk(chunk_id=ids.next("sum"), doc_id=ids.doc_id, chunk_type="summary",
                  text=text, embed_text=f"[{crumb}]\n{text}", role=role,
                  token_estimate=_est_tokens(text))
        _collect_meta(c, els)
        _links_from_text(c, text)
        out.append(c)
    return out


# --------------------------------------------------------------------------- entry

def chunk_result(result: ExtractionResult, cfg: ChunkingConfig | None = None,
                 doc_id: str | None = None) -> ChunkingResult:
    cfg = cfg or ChunkingConfig()
    paper = result.stats.get("paper") or {}
    title = paper.get("title") or next(
        (e.text for e in result.elements if e.kind == "title" and e.text), None)
    doc_id = doc_id or re.sub(r"[^A-Za-z0-9_-]+", "_",
                              (result.source or "doc").rsplit("/", 1)[-1])[:60]
    ids = _Ids(doc_id)
    cr = ChunkingResult(doc_id=doc_id, source=result.source)

    section_chunks = _build_section_text_chunks(result, cfg, ids, title)
    cr.chunks += section_chunks
    cr.chunks += _build_claim_chunks(result, cfg, ids, title, section_chunks)
    cr.chunks += _build_table_chunks(result, cfg, ids, title)
    cr.chunks += _build_figure_chunks(result, cfg, ids, title)
    cr.chunks += _build_formula_chunks(result, ids, title)
    cr.chunks += _build_summary_chunks(result, cfg, ids, title)
    if cfg.emit_references:
        cr.chunks += _build_reference_chunks(result, ids)

    # ---- validation & stats (the eval loop needs these) -------------------
    counts: dict[str, int] = {}
    toks = []
    for c in cr.chunks:
        counts[c.chunk_type] = counts.get(c.chunk_type, 0) + 1
        toks.append(c.token_estimate)
        if not c.embed_text:
            c.embed_text = c.text
    over = sum(1 for c in cr.chunks
               if c.chunk_type == "section_text" and c.token_estimate > cfg.max_tokens)
    if over:
        cr.warnings.append(f"{over} section chunk(s) exceed max_tokens "
                           f"(long unsplittable sentences)")
    chunked_ids = {eid for c in cr.chunks for eid in c.element_ids}
    orphans = [e.id for e in result.elements
               if e.kind in _TEXT_KINDS and e.text and e.id not in chunked_ids
               and e.metadata.get("role") != "references"]
    if orphans:
        cr.warnings.append(f"{len(orphans)} text element(s) not covered by any chunk")
    toks_sorted = sorted(toks) or [0]
    by_type_tokens: dict[str, list[int]] = {}
    for c in cr.chunks:
        by_type_tokens.setdefault(c.chunk_type, []).append(c.token_estimate)
    tokens_by_type = {}
    for t, xs in sorted(by_type_tokens.items()):
        xs.sort()
        tokens_by_type[t] = {"n": len(xs), "min": xs[0],
                             "median": xs[len(xs) // 2], "max": xs[-1]}
    cr.stats = {
        "n_chunks": len(cr.chunks),
        "by_type": counts,
        "tokens": {"min": toks_sorted[0], "median": toks_sorted[len(toks_sorted) // 2],
                   "max": toks_sorted[-1], "total": sum(toks)},
        "tokens_by_type": tokens_by_type,
        "doc_title": title,
        "config": {k: v for k, v in asdict(cfg).items()},
    }
    return cr
