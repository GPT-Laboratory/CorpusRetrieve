"""DOCX and XLSX extractors."""
from __future__ import annotations

from docparse._md import rows_to_markdown
from docparse.base import register
from docparse.lang import detect_language
from docparse.models import Element, ExtractionConfig, ExtractionResult


class DocxExtractor:
    doc_types = ("docx",)

    def extract(self, path: str, cfg: ExtractionConfig) -> ExtractionResult:
        import docx
        from docx.oxml.ns import qn
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        result = ExtractionResult(source=path, doc_type="docx",
                                  mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        document = docx.Document(path)
        order = 0
        sample: list[str] = []
        for child in document.element.body.iterchildren():
            if child.tag == qn("w:p"):
                para = Paragraph(child, document)
                text = para.text.strip()
                if not text:
                    continue
                style = (para.style.name or "").lower() if para.style else ""
                kind = "heading" if style.startswith(("heading", "title")) else "narrative"
                order += 1
                result.elements.append(Element(kind=kind, text=text, order=order, extractor="python-docx",
                                               metadata={"style": para.style.name if para.style else None}))
                if kind == "narrative" and len(sample) < 40:
                    sample.append(text)
            elif child.tag == qn("w:tbl"):
                table = Table(child, document)
                rows = [[c.text for c in row.cells] for row in table.rows]
                md, trunc = rows_to_markdown(rows, cfg.table_max_rows)
                if md:
                    order += 1
                    result.elements.append(Element(kind="table", text=md, order=order, extractor="python-docx",
                                                   metadata={"rows": len(rows), "truncated": trunc}))
        if cfg.detect_language:
            result.language = detect_language(" ".join(sample))
            for el in result.elements:
                if el.text and el.language is None:
                    el.language = result.language
        return result


class XlsxExtractor:
    doc_types = ("xlsx",)

    def extract(self, path: str, cfg: ExtractionConfig) -> ExtractionResult:
        from openpyxl import load_workbook

        result = ExtractionResult(source=path, doc_type="xlsx",
                                  mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        wb = load_workbook(path, read_only=True, data_only=True)
        order = 0
        for name in wb.sheetnames:
            ws = wb[name]
            rows = [list(r) for r in ws.iter_rows(values_only=True)]
            rows = [r for r in rows if any(c is not None and str(c).strip() for c in r)]
            if not rows:
                continue
            md, trunc = rows_to_markdown(rows, cfg.table_max_rows)
            order += 1
            result.elements.append(Element(kind="table", text=md, order=order, extractor="openpyxl",
                                           metadata={"sheet": name, "rows": len(rows),
                                                     "cols": max(len(r) for r in rows), "truncated": trunc}))
        wb.close()
        result.stats["sheets"] = wb.sheetnames if hasattr(wb, "sheetnames") else None
        return result


register(DocxExtractor())
register(XlsxExtractor())
