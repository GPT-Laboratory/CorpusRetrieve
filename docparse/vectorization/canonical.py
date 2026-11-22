"""Type-aware canonicalization for portable indexing.

Two text views are retained per chunk:
  raw_text        — untouched, for quoting/grounding/audit
  canonical_text  — what actually gets embedded: role-tagged, breadcrumbed,
                    unicode/whitespace-normalized

Plus one structured view: quantities parsed to (value_norm, unit) so the KG
and payload filters can reason over numbers exactly.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

_WS = re.compile(r"\s+")
PAYLOAD_SCHEMA_VERSION = "2.0.0"

_TYPE_TAG = {
    "section_text": "SECTION", "claim": "CLAIM", "table_summary": "TABLE",
    "table_row_group": "TABLE_ROWS", "figure": "FIGURE", "figure_panel": "PANEL",
    "formula": "FORMULA", "reference": "REF", "summary": "SUMMARY",
}


def canonical_text(chunk: dict) -> str:
    """Build the normalized embedding view from a retrieval chunk."""
    base = chunk.get("embed_text") or chunk.get("text") or ""
    tag = _TYPE_TAG.get(chunk.get("chunk_type", ""), "CHUNK")
    role = chunk.get("role")
    head = f"[{tag}{f' role={role}' if role else ''}]"
    text = unicodedata.normalize("NFKC", base)
    # joining every "word- word" here a second time, which destroyed Finnish
    # coordinated compounds ("maa- ja metsätalous" -> "maaja"). Line-break
    # repair happens once, upstream, in docparse.scientific.
    text = _WS.sub(" ", text).strip()
    return f"{head} {text}"


# ------------------------------------------------------------------ quantities

_QTY_NUM = re.compile(r"(-?\d+(?:[.,]\d+)?)")
_RANGE = re.compile(r"(-?\d+(?:[.,]\d+)?)\s*(?:°c|%|k)?\s*(?:–|—|‒|-|to|till)\s*"
                    r"(-?\d+(?:[.,]\d+)?)", re.I)


def _num(value: str) -> float:
    return float(value.replace(",", "."))

_UNIT_RULES = [
    (re.compile(r"°c\s*[- ]?days?", re.I), "degC_days"),
    (re.compile(r"°c|\bdeg(?:rees?)?\s*c\b", re.I), "degC"),
    (re.compile(r"%"), "percent"),
    (re.compile(r"‰"), "permille"),
    (re.compile(r"kg/ha", re.I), "kg_per_ha"),
    (re.compile(r"t/ha", re.I), "t_per_ha"),
    (re.compile(r"mg/kg", re.I), "mg_per_kg"),
    (re.compile(r"kwh", re.I), "kWh"),
    (re.compile(r"mwh", re.I), "MWh"),
    (re.compile(r"€|\beur\b", re.I), "EUR"),
    (re.compile(r"\bha\b", re.I), "ha"),
    (re.compile(r"\bkg\b", re.I), "kg"),
    (re.compile(r"\bt\b"), "t"),
    (re.compile(r"\byears?\b", re.I), "years"),
    (re.compile(r"\bdays?\b", re.I), "days"),
    (re.compile(r"\bhpa\b", re.I), "hPa"),
    (re.compile(r"\bmm\b", re.I), "mm"),
    (re.compile(r"\bk\b"), "K"),
    (re.compile(r"[- ](?:fold|times)\b", re.I), "ratio"),
]


def normalize_quantity(q: dict) -> dict:
    """{'value': '60%–80%', 'type': 'percentage_change'} ->
       adds value_norm / value_norm_min+max and unit."""
    out = dict(q)
    raw = str(q.get("value", "")).replace("\u2212", "-")     # Unicode minus
    unit = next((u for rx, u in _UNIT_RULES if rx.search(raw)), None)
    if unit:
        out["unit"] = unit
    if (m := _RANGE.search(raw)):
        out["value_norm_min"] = _num(m.group(1))
        out["value_norm_max"] = _num(m.group(2))
    elif (m := _QTY_NUM.search(raw)):
        out["value_norm"] = _num(m.group(1))
    return out


BIBLIOGRAPHIC_FIELDS = ("title", "authors", "year", "doi", "handle_url", "online_url",
                        "source", "language", "source_doc_id", "finna_id", "urn",
                        "format", "organization", "license", "rights", "series",
                        "isbn", "issn", "publisher")


def bibliographic_record(doc: dict, pages: list[int] | None) -> dict | None:
    """Resolve the publication a unit belongs to.

    ``document_metadata.records`` lists every publication stored in the file,
    each with an optional ``page_start``/``page_end``; a multi-publication file
    (a journal issue) therefore resolves each unit to its own article. A file
    with a single record (or flat metadata) resolves every unit to it.
    """
    meta = doc.get("metadata") or {}
    records = meta.get("records") or []
    if not records and any(meta.get(f) for f in ("title", "handle_url", "doi")):
        records = [{k: meta[k] for k in BIBLIOGRAPHIC_FIELDS if meta.get(k) is not None}]
    records = [r for r in records if isinstance(r, dict)]
    if not records:
        return None
    # A mere empty record or organizational tag is not a publication identity.
    # Both manifest records and PDF-derived DOI fallbacks must contain at least
    # one externally intelligible locator or title.
    def valid(rec: dict) -> bool:
        return any(rec.get(k) for k in ("title", "doi", "handle_url", "online_url", "urn", "isbn"))
    if len(records) == 1:
        return dict(records[0]) if valid(records[0]) else None
    page_set = set(pages or [])
    titles = {str(r.get("title", "")).casefold().strip() for r in records}
    has_ranges = any(r.get("page_start") is not None for r in records)
    if len(titles) == 1 and "" not in titles and not has_ranges:
        # one publication catalogued several times (print and e-version, or
        # two organisations): the same work, so not ambiguous
        out = dict(records[0])
        out["alternate_records"] = [dict(r) for r in records[1:]]
        return out if valid(out) else None
    if page_set:
        matches = [rec for rec in records
                   if rec.get("page_start") is not None and rec.get("page_end") is not None
                   and all(rec["page_start"] <= page <= rec["page_end"] for page in page_set)]
        if len(matches) == 1 and valid(matches[0]):
            return dict(matches[0])
    return {"ambiguous": True, "candidates": [r.get("title") for r in records]}


def normalize_chunk(chunk: dict) -> dict:
    """Return a shallow-copied chunk with canonical_text and normalized
    quantities attached. Raw fields are never mutated."""
    c = dict(chunk)
    c["canonical_text"] = canonical_text(chunk)
    if chunk.get("quantities"):
        c["quantities_norm"] = [normalize_quantity(q) for q in chunk["quantities"]]
    return c


# --------------------------------------------------------------- payload build

def payload_for(chunk: dict, doc: dict, embed_model: str, embed_version: str,
                tenant: str) -> dict[str, Any]:
    """Build a filterable, provenance-bearing vector payload."""
    md = chunk.get("metadata", {}) or {}
    return {
        "tenant_id": tenant,
        "bibliographic": bibliographic_record(doc, chunk.get("pages")),
        "spans": chunk.get("spans", []),
        "locators": chunk.get("locators", []),
        "language": chunk.get("language"),
        "derived_text": bool(md.get("derived_text")),
        "corpus_id": doc.get("corpus_id", "default"),
        "external_id": doc.get("external_id"),
        "document_metadata": doc.get("metadata", {}),
        "doc_id": chunk.get("doc_id") or doc.get("doc_id"),
        # the file the unit came from: with content-hash document IDs this is
        # the only human-readable origin when no bibliographic record exists
        "source_file": doc.get("source"),
        "chunk_id": chunk["chunk_id"],
        "chunk_type": chunk["chunk_type"],
        "role": chunk.get("role"),
        "section_path": chunk.get("section_path", []),
        "pages": chunk.get("pages", []),
        "figure_label": md.get("label") if chunk["chunk_type"].startswith("figure") else None,
        "table_label": md.get("label") if chunk["chunk_type"].startswith("table") else None,
        "citation_ids": [c.get("key") for c in chunk.get("citations", []) if c.get("key")],
        "quantities": chunk.get("quantities_norm") or chunk.get("quantities") or [],
        "linked_figures": chunk.get("linked_figures", []),
        "linked_tables": chunk.get("linked_tables", []),
        "parent_chunk_id": chunk.get("parent_chunk_id"),
        "retrieval_default": md.get("retrieval_default", "include"),
        "raw_text": chunk.get("text", ""),
        "canonical_text": chunk.get("canonical_text", ""),
        "char_start": chunk.get("char_start"),
        "char_end": chunk.get("char_end"),
        "visual_ref": chunk.get("visual_ref"),
        "embedding_model": embed_model,
        "embedding_version": embed_version,
        "schema_version": PAYLOAD_SCHEMA_VERSION,
    }
