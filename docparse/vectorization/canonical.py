"""Type-aware canonicalization for Stage 5.

Two text views per chunk (the report's core normalization rule):
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
_HYPHEN_BREAK = re.compile(r"(\w)-\s+(\w)")

_TYPE_TAG = {
    "section_text": "SECTION", "claim": "CLAIM", "table_summary": "TABLE",
    "table_row_group": "TABLE_ROWS", "figure": "FIGURE", "figure_panel": "PANEL",
    "formula": "FORMULA", "reference": "REF", "summary": "SUMMARY",
}


def canonical_text(chunk: dict) -> str:
    """Build the embed-view. chunk['embed_text'] already carries the breadcrumb
    from Stage 4; here we add the role tag and normalize the surface form."""
    base = chunk.get("embed_text") or chunk.get("text") or ""
    tag = _TYPE_TAG.get(chunk.get("chunk_type", ""), "CHUNK")
    role = chunk.get("role")
    head = f"[{tag}{f' role={role}' if role else ''}]"
    text = unicodedata.normalize("NFKC", base)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)          # de-hyphenate line breaks
    text = _WS.sub(" ", text).strip()
    return f"{head} {text}"


# ------------------------------------------------------------------ quantities

_QTY_NUM = re.compile(r"(-?\d+(?:\.\d+)?)")
_RANGE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:°c|%|k)?\s*(?:–|—|‒|-|to)\s*"
                    r"(-?\d+(?:\.\d+)?)", re.I)

_UNIT_RULES = [
    (re.compile(r"°c\s*[- ]?days?", re.I), "degC_days"),
    (re.compile(r"°c|\bdeg(?:rees?)?\s*c\b", re.I), "degC"),
    (re.compile(r"%"), "percent"),
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
    raw = str(q.get("value", ""))
    unit = next((u for rx, u in _UNIT_RULES if rx.search(raw)), None)
    if unit:
        out["unit"] = unit
    if (m := _RANGE.search(raw)):
        out["value_norm_min"] = float(m.group(1))
        out["value_norm_max"] = float(m.group(2))
    elif (m := _QTY_NUM.search(raw)):
        out["value_norm"] = float(m.group(1))
    return out


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
    """Vector-DB payload following the report's metadata index set. Small,
    filterable, provenance-bearing; big raw text stays out of indexes but in
    the payload for grounding."""
    md = chunk.get("metadata", {}) or {}
    return {
        "tenant_id": tenant,
        "doc_id": chunk.get("doc_id") or doc.get("doc_id"),
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
        "schema_version": "2026-07",
    }
