"""Tiny helpers for rendering rows as GitHub-flavoured markdown tables."""
from __future__ import annotations

from typing import Any, Iterable


def rows_to_markdown(rows: Iterable[Iterable[Any]], max_rows: int = 200) -> tuple[str, bool]:
    rows = [list(r) for r in rows]
    if not rows:
        return "", False
    width = max(len(r) for r in rows)
    norm = [[("" if c is None else str(c)).replace("\n", " ").replace("|", "\\|") for c in r]
            + [""] * (width - len(r)) for r in rows]
    header, body = norm[0], norm[1:]
    truncated = len(body) > max_rows
    body = body[:max_rows]
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(out), truncated
