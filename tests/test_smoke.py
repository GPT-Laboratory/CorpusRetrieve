"""Smoke + structure tests. Golden corpus is built synthetically (no fixtures in git)."""
import json

import pytest

import docparse
from docparse import ExtractionConfig, extract

from .make_golden import build


def test_registry_has_core_types():
    for t in ("pdf", "docx", "xlsx", "csv", "html", "text", "image", "json"):
        assert t in docparse.registered_types()


def test_text_extraction_levels_sections_offsets(tmp_path):
    p = tmp_path / "note.md"
    p.write_text("# Title\n\n## Sub\n\nHello world. This paragraph is long enough to matter.\n")
    r = extract(str(p), ExtractionConfig())
    h1, h2, para = r.elements
    assert (h1.kind, h1.level) == ("heading", 1)
    assert (h2.kind, h2.level) == ("heading", 2)
    assert h2.section_path == ["Title"]
    assert para.section_path == ["Title", "Sub"]
    # char offsets index into document_text()
    doc = r.document_text()
    assert doc[para.char_start:para.char_end] == para.text


def test_csv_structured_and_row_count(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("a,b,c\n" + "\n".join(f"{i},{i+1},{i+2}" for i in range(300)) + "\n")
    r = extract(str(p), ExtractionConfig(table_max_rows=100,
                                         table_serializations=("markdown", "kv")))
    el = r.elements[0]
    assert el.structured["header"] == ["a", "b", "c"]
    assert el.metadata["total_rows"] == 300 and el.metadata["truncated"] is True
    assert "a: 0; b: 1; c: 2." in el.metadata["serializations"]["kv"]
    assert r.warnings  # truncation is reported, not silent


def test_json_becomes_table(tmp_path):
    p = tmp_path / "recs.json"
    p.write_text(json.dumps([{"name": "x", "v": 1}, {"name": "y", "v": 2}]))
    r = extract(str(p), ExtractionConfig())
    assert r.doc_type == "json"
    assert r.counts().get("table") == 1
    assert r.elements[0].structured["rows"] == [["x", "1"], ["y", "2"]]


def test_docx_levels_table_caption_link(tmp_path):
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_heading("Top", level=1)
    d.add_heading("Nested", level=2)
    d.add_paragraph("Body text under the nested heading, long enough for detection.")
    t = d.add_table(rows=2, cols=2)
    t.rows[0].cells[0].text, t.rows[0].cells[1].text = "k", "v"
    t.rows[1].cells[0].text, t.rows[1].cells[1].text = "a", "1"
    path = tmp_path / "x.docx"
    d.save(str(path))
    r = extract(str(path), ExtractionConfig())
    kinds = {e.kind: e for e in r.elements}
    assert kinds["table"].structured == {"header": ["k", "v"], "rows": [["a", "1"]]}
    body = next(e for e in r.elements if e.kind == "narrative")
    assert body.section_path == ["Top", "Nested"]


def test_pdf_structure(tmp_path):
    pytest.importorskip("fitz")
    golden = build(tmp_path / "golden")
    r = extract(str(golden / "report.pdf"), ExtractionConfig())
    counts = r.counts()
    # boilerplate stripped
    texts = [e.text for e in r.elements]
    assert not any("ACME Corp" in t for t in texts)
    assert not any(t.startswith("Page ") for t in texts)
    assert r.stats.get("boilerplate_removed", 0) >= 4
    # headings with levels; big title is level 1
    heads = {e.text: e.level for e in r.elements if e.kind == "heading"}
    assert heads.get("Quarterly Results") == 1
    # column order: left column before right column
    order = {e.text[:20]: e.order for e in r.elements if e.text}
    left = next(v for k, v in order.items() if k.startswith("Left column"))
    right = next(v for k, v in order.items() if k.startswith("Right column"))
    assert left < right
    # ruled table -> structured view
    tables = [e for e in r.elements if e.kind == "table"]
    assert tables and tables[0].structured["rows"]
    flat = [c for row in ([tables[0].structured["header"]] + tables[0].structured["rows"]) for c in row]
    assert "North" in " ".join(flat)
    # caption linked to the table
    cap = next(e for e in r.elements if e.kind == "caption")
    assert cap.parent_id == tables[0].id
    # narrative under its heading
    disc = next(e for e in r.elements if e.text.startswith("Discussion"))
    assert "Financial Table" in disc.section_path
    assert counts.get("narrative", 0) >= 4


def test_pdfplumber_fallback_keeps_contract(tmp_path):
    pytest.importorskip("pdfplumber")
    pytest.importorskip("fitz")
    golden = build(tmp_path / "golden")
    r = extract(str(golden / "report.pdf"), ExtractionConfig(pdf_backend="pdfplumber"))
    kinds = r.counts()
    assert kinds.get("heading", 0) >= 2          # no more page-blob-only fallback
    assert all(e.bbox for e in r.elements if e.kind in ("heading", "narrative"))
    tables = [e for e in r.elements if e.kind == "table"]
    assert tables and tables[0].structured["rows"]


def test_bench_harness(tmp_path):
    pytest.importorskip("fitz")
    from docparse.bench import run
    golden = build(tmp_path / "golden")
    report = run(str(golden), ExtractionConfig())
    agg = report["aggregate"]
    assert report["n_documents"] == 2
    assert agg["heading_recall"] >= 0.8
    assert agg["reading_order_tau"] is not None and agg["reading_order_tau"] >= 0.9
    assert agg["table_cell_f1"] is not None and agg["table_cell_f1"] >= 0.8


def test_crlf_windows_line_endings(tmp_path):
    """Files written on Windows (CRLF) must parse identically to LF files."""
    p = tmp_path / "crlf.md"
    p.write_bytes(b"# Title\r\n\r\n## Sub\r\n\r\nHello world. This paragraph is long enough to matter.\r\n")
    r = extract(str(p), ExtractionConfig())
    assert [(e.kind, e.level) for e in r.elements] == \
        [("heading", 1), ("heading", 2), ("narrative", None)]
    assert "\r" not in r.elements[2].text
