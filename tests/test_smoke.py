"""Smoke tests that need no network and no heavy binaries."""
import docparse
from docparse import ExtractionConfig, extract


def test_registry_has_core_types():
    for t in ("pdf", "docx", "xlsx", "csv", "html", "text", "image"):
        assert t in docparse.registered_types()


def test_text_extraction(tmp_path):
    p = tmp_path / "note.md"
    p.write_text("# Title\n\nHello world. This is a paragraph long enough to detect language.\n")
    r = extract(str(p), ExtractionConfig())
    kinds = r.counts()
    assert kinds.get("heading", 0) >= 1
    assert kinds.get("narrative", 0) >= 1


def test_csv_extraction(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("a,b,c\n1,2,3\n4,5,6\n")
    r = extract(str(p), ExtractionConfig())
    assert r.counts().get("table", 0) == 1
    assert "| a | b | c |" in r.elements[0].text
