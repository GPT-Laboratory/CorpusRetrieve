"""CSV/TSV, JSON/JSONL, plain text / markdown, and HTML extractors (encoding-aware)."""
from __future__ import annotations

import json
import re
from pathlib import Path

from docparse.router import register
from docparse.encoding import detect_encoding, read_text
from docparse.models import Element, ExtractionConfig, ExtractionResult
from docparse.serialize import serialize_table

_MD_HEADING = re.compile(r"^(#{1,6})\s+")


class CsvExtractor:
    doc_types = ("csv",)

    def extract(self, path: str, cfg: ExtractionConfig) -> ExtractionResult:
        import pandas as pd
        sep = "\t" if Path(path).suffix.lower() == ".tsv" else ","
        raw = Path(path).read_bytes()
        enc = detect_encoding(raw)
        result = ExtractionResult(source=path, doc_type="csv", mime="text/csv")

        # true row count (cheap; header excluded; RFC-4180 quoted newlines over-count)
        total_rows = max(raw.count(b"\n") + (0 if raw.endswith(b"\n") else 1) - 1, 0)

        df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False,
                         nrows=cfg.table_max_rows, encoding=enc)
        rows = [list(df.columns)] + df.values.tolist()
        text, extras, structured, _ = serialize_table(rows, cfg.table_serializations,
                                                      cfg.table_max_rows)
        truncated = total_rows > len(df)
        result.stats.update({"encoding": enc, "total_rows": total_rows})
        if truncated:
            result.warnings.append(
                f"serialized first {len(df)} of {total_rows} rows (table_max_rows)")
        meta = {"columns": list(df.columns), "sample_rows": len(df),
                "total_rows": total_rows, "truncated": truncated, "encoding": enc}
        if extras:
            meta["serializations"] = extras
        result.elements.append(Element(kind="table", text=text, order=1, extractor="pandas",
                                       structured=structured, metadata=meta))
        return result


class JsonExtractor:
    """JSON is evidence with structure — keep it, don't flatten it to prose."""

    doc_types = ("json",)

    def extract(self, path: str, cfg: ExtractionConfig) -> ExtractionResult:
        result = ExtractionResult(source=path, doc_type="json", mime="application/json")
        raw, enc = read_text(path)
        result.stats["encoding"] = enc
        order = 0

        def emit(obj, breadcrumb: str):
            nonlocal order
            # list of flat dicts -> a table (header = union of keys)
            if (isinstance(obj, list) and obj
                    and all(isinstance(r, dict) for r in obj)
                    and all(not isinstance(v, (dict, list)) for r in obj for v in r.values())):
                header: list[str] = []
                for r in obj:
                    header += [k for k in r if k not in header]
                rows = [header] + [[r.get(k, "") for k in header] for r in obj]
                text, extras, structured, trunc = serialize_table(
                    rows, cfg.table_serializations, cfg.table_max_rows)
                order += 1
                meta = {"path": breadcrumb or "$", "rows": len(obj), "truncated": trunc}
                if extras:
                    meta["serializations"] = extras
                result.elements.append(Element(kind="table", text=text, order=order,
                                               extractor="json", structured=structured,
                                               metadata=meta))
            elif isinstance(obj, dict):
                scalars = {k: v for k, v in obj.items() if not isinstance(v, (dict, list))}
                if scalars:
                    order += 1
                    result.elements.append(Element(
                        kind="other",
                        text="; ".join(f"{k}: {v}" for k, v in scalars.items()),
                        order=order, extractor="json",
                        structured={"object": scalars},
                        metadata={"path": breadcrumb or "$"}))
                for k, v in obj.items():
                    if isinstance(v, (dict, list)):
                        emit(v, f"{breadcrumb}.{k}" if breadcrumb else k)
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    if isinstance(v, (dict, list)):
                        emit(v, f"{breadcrumb}[{i}]")
            else:
                order += 1
                result.elements.append(Element(kind="other", text=str(obj), order=order,
                                               extractor="json",
                                               metadata={"path": breadcrumb or "$"}))

        try:
            if path.endswith(".jsonl") or "\n{" in raw.strip()[:2000] and not raw.lstrip().startswith(("[", "{")):
                records = [json.loads(line) for line in raw.splitlines() if line.strip()]
                emit(records, "$")
            else:
                emit(json.loads(raw), "")
        except Exception as exc:
            result.warnings.append(f"JSON parse failed, falling back to text: {exc}")
            order += 1
            result.elements.append(Element(kind="narrative", text=raw[:20000], order=order,
                                           extractor="json"))
        return result


class TextExtractor:
    doc_types = ("text",)

    def extract(self, path: str, cfg: ExtractionConfig) -> ExtractionResult:
        result = ExtractionResult(source=path, doc_type="text", mime="text/plain")
        raw, enc = read_text(path)
        result.stats["encoding"] = enc
        order = 0
        for block in [b.strip() for b in raw.split("\n\n") if b.strip()]:
            order += 1
            m = _MD_HEADING.match(block)
            if m:
                result.elements.append(Element(
                    kind="heading", text=block[m.end():].strip(), order=order,
                    level=len(m.group(1)), extractor="text", metadata={"encoding": enc}))
            else:
                result.elements.append(Element(kind="narrative", text=block, order=order,
                                               extractor="text", metadata={"encoding": enc}))
        return result


_BOILER_TAGS = ("nav", "footer", "header", "aside", "form", "noscript")


class HtmlExtractor:
    doc_types = ("html",)

    def extract(self, path: str, cfg: ExtractionConfig) -> ExtractionResult:
        from bs4 import BeautifulSoup
        result = ExtractionResult(source=path, doc_type="html", mime="text/html")
        raw, enc = read_text(path)
        result.stats["encoding"] = enc
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        if cfg.remove_boilerplate:
            removed = 0
            for tag in soup(list(_BOILER_TAGS)):
                tag.decompose()
                removed += 1
            result.stats["boilerplate_removed"] = removed
        order = 0
        if soup.title and soup.title.text.strip():
            order += 1
            result.elements.append(Element(kind="title", text=soup.title.text.strip(),
                                           order=order, level=0, extractor="bs4"))
        for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "table"]):
            if el.name == "table":
                rows = [[c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
                        for tr in el.find_all("tr")]
                rows = [r for r in rows if r]
                if not rows:
                    continue
                text, extras, structured, trunc = serialize_table(
                    rows, cfg.table_serializations, cfg.table_max_rows)
                order += 1
                meta = {"rows": len(rows), "truncated": trunc}
                if extras:
                    meta["serializations"] = extras
                result.elements.append(Element(kind="table", text=text, order=order,
                                               extractor="bs4", structured=structured,
                                               metadata=meta))
                continue
            text = el.get_text(" ", strip=True)
            if not text:
                continue
            if el.name.startswith("h"):
                order += 1
                result.elements.append(Element(kind="heading", text=text, order=order,
                                               level=int(el.name[1]), extractor="bs4"))
            else:
                order += 1
                kind = "list_item" if el.name == "li" else "narrative"
                result.elements.append(Element(kind=kind, text=text, order=order,
                                               extractor="bs4"))
        return result


register(CsvExtractor())
register(JsonExtractor())
register(TextExtractor())
register(HtmlExtractor())
