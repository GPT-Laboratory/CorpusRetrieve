"""Table serializations: one structured view, several text renderings.

Evidence (structure-aware tabular chunking, 2026) says the *representation* of a
table often matters more than the vector DB behind it. So docparse keeps the
structured rows and can emit competing serializations side by side:

  * markdown — GitHub-flavoured table (default Element.text)
  * json     — list of {header: value} records
  * kv       — one "header: value; header: value" sentence per row
               (retrieves well for row-lookup questions)
"""
from __future__ import annotations

import json
from typing import Any, Iterable


def normalize_rows(rows: Iterable[Iterable[Any]]) -> list[list[str]]:
    rows = [list(r) for r in rows]
    if not rows:
        return []
    width = max(len(r) for r in rows)
    return [[("" if c is None else str(c)).replace("\n", " ").strip() for c in r]
            + [""] * (width - len(r)) for r in rows]


def rows_to_structured(rows: Iterable[Iterable[Any]]) -> dict[str, Any]:
    """The machine-usable view: header + body rows, always kept on the Element."""
    norm = normalize_rows(rows)
    if not norm:
        return {"header": [], "rows": []}
    return {"header": norm[0], "rows": norm[1:]}


def rows_to_markdown(rows: Iterable[Iterable[Any]], max_rows: int = 200) -> tuple[str, bool]:
    norm = [[c.replace("|", "\\|") for c in r] for r in normalize_rows(rows)]
    if not norm:
        return "", False
    width = len(norm[0])
    header, body = norm[0], norm[1:]
    truncated = len(body) > max_rows
    body = body[:max_rows]
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(out), truncated


def rows_to_json(rows: Iterable[Iterable[Any]], max_rows: int = 200) -> tuple[str, bool]:
    s = rows_to_structured(rows)
    header, body = s["header"], s["rows"]
    truncated = len(body) > max_rows
    recs = [dict(zip(header, r)) for r in body[:max_rows]]
    return json.dumps(recs, ensure_ascii=False), truncated


def rows_to_kv(rows: Iterable[Iterable[Any]], max_rows: int = 200) -> tuple[str, bool]:
    s = rows_to_structured(rows)
    header, body = s["header"], s["rows"]
    truncated = len(body) > max_rows
    lines = []
    for r in body[:max_rows]:
        pairs = [f"{h}: {v}" for h, v in zip(header, r) if v]
        if pairs:
            lines.append("; ".join(pairs) + ".")
    return "\n".join(lines), truncated


_SERIALIZERS = {"markdown": rows_to_markdown, "json": rows_to_json, "kv": rows_to_kv}


def serialize_table(rows: Iterable[Iterable[Any]], serializations: tuple[str, ...],
                    max_rows: int = 200) -> tuple[str, dict[str, str], dict[str, Any], bool]:
    """Return (primary_text, extra_serializations, structured_view, truncated).

    The first requested serialization becomes Element.text (markdown if listed,
    else whatever is first); the rest go to metadata["serializations"].
    """
    rows = [list(r) for r in rows]
    structured = rows_to_structured(rows)
    wanted = serializations or ("markdown",)
    primary_name = "markdown" if "markdown" in wanted else wanted[0]
    primary, truncated = _SERIALIZERS[primary_name](rows, max_rows)
    extras: dict[str, str] = {}
    for name in wanted:
        if name == primary_name or name not in _SERIALIZERS:
            continue
        extras[name], _ = _SERIALIZERS[name](rows, max_rows)
    return primary, extras, structured, truncated
