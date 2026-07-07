"""CSV/TSV, plain text / markdown, and HTML extractors."""
from __future__ import annotations

from pathlib import Path

from docparse._md import rows_to_markdown
from docparse.base import register
from docparse.lang import detect_language
from docparse.models import Element, ExtractionConfig, ExtractionResult


class CsvExtractor:
    doc_types = ("csv",)

    def extract(self, path: str, cfg: ExtractionConfig) -> ExtractionResult:
        import pandas as pd
        sep = "\t" if Path(path).suffix.lower() == ".tsv" else ","
        result = ExtractionResult(source=path, doc_type="csv", mime="text/csv")
        df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, nrows=cfg.table_max_rows + 1)
        rows = [list(df.columns)] + df.head(cfg.table_max_rows).values.tolist()
        md, trunc = rows_to_markdown(rows, cfg.table_max_rows)
        result.elements.append(Element(kind="table", text=md, order=1, extractor="pandas",
                                       metadata={"columns": list(df.columns), "sample_rows": min(len(df), cfg.table_max_rows),
                                                 "truncated": trunc}))
        return result


class TextExtractor:
    doc_types = ("text",)

    def extract(self, path: str, cfg: ExtractionConfig) -> ExtractionResult:
        result = ExtractionResult(source=path, doc_type="text", mime="text/plain")
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
        order = 0
        for block in [b.strip() for b in raw.split("\n\n") if b.strip()]:
            order += 1
            kind = "heading" if block.lstrip().startswith("#") else "narrative"
            result.elements.append(Element(kind=kind, text=block.lstrip("# ").strip() if kind == "heading" else block,
                                           order=order, extractor="text"))
        if cfg.detect_language:
            result.language = detect_language(raw[:4000])
            for el in result.elements:
                el.language = el.language or result.language
        return result


class HtmlExtractor:
    doc_types = ("html",)

    def extract(self, path: str, cfg: ExtractionConfig) -> ExtractionResult:
        from bs4 import BeautifulSoup
        result = ExtractionResult(source=path, doc_type="html", mime="text/html")
        soup = BeautifulSoup(Path(path).read_text(encoding="utf-8", errors="replace"), "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        order = 0
        if soup.title and soup.title.text.strip():
            order += 1
            result.elements.append(Element(kind="title", text=soup.title.text.strip(), order=order, extractor="bs4"))
        for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "table"]):
            text = el.get_text(" ", strip=True)
            if not text:
                continue
            kind = {"h1": "heading", "h2": "heading", "h3": "heading", "h4": "heading",
                    "li": "list_item", "table": "table", "p": "narrative"}[el.name]
            order += 1
            result.elements.append(Element(kind=kind, text=text, order=order, extractor="bs4"))
        if cfg.detect_language:
            joined = " ".join(e.text for e in result.elements if e.kind == "narrative")
            result.language = detect_language(joined[:4000])
        return result


register(CsvExtractor())
register(TextExtractor())
register(HtmlExtractor())
