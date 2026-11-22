"""Unit tests: deterministic IDs, PDF layout fixes, OCR accounting."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from docparse import ExtractionConfig, extract
from docparse.extractors import pdf as pdf_mod
from docparse.extractors.pdf import _caption_distance
from docparse.ocr import ocr_status

fitz = pytest.importorskip("fitz")
sys.path.insert(0, str(Path(__file__).parent))
from fixtures import bilingual_report, blank_pdf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = dict(ocr_langs="eng")


@pytest.fixture
def report(tmp_path):
    return bilingual_report(tmp_path / "report.pdf")


# --------------------------------------------------------- deterministic IDs

_ID_SCRIPT = """
import json, sys
from docparse import extract, chunk_result, ExtractionConfig
from docparse.vectorization.embedding import BM25SparseEncoder
from docparse.vectorization.canonical import normalize_chunk
r = extract(sys.argv[1], ExtractionConfig(ocr_langs="eng"))
c = chunk_result(r, doc_id="d")
enc = BM25SparseEncoder().fit(normalize_chunk(x.to_dict())["canonical_text"] for x in c.chunks)
print(json.dumps({"elements": [e.id for e in r.elements],
                  "chunks": [x.chunk_id for x in c.chunks],
                  "vocab": list(enc.vocab.items())}))
"""


def _ids_under_seed(path: Path, seed: str) -> dict:
    env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(ROOT)}
    out = subprocess.run([sys.executable, "-c", _ID_SCRIPT, str(path)], env=env,
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_element_chunk_ids_and_vocabulary_are_identical_across_processes(report):
    a, b = _ids_under_seed(report, "1"), _ids_under_seed(report, "987")
    assert a["elements"] == b["elements"]
    assert a["chunks"] == b["chunks"]
    assert a["vocab"] == b["vocab"]


def test_element_ids_change_with_content(report, tmp_path):
    other = tmp_path / "other.pdf"
    doc = fitz.open(str(report))
    doc[0].insert_text((72, 700), "extra line", fontsize=8)
    doc.save(str(other))
    ids_a = [e.id for e in extract(str(report), ExtractionConfig(**CFG)).elements]
    ids_b = [e.id for e in extract(str(other), ExtractionConfig(**CFG)).elements]
    assert set(ids_a).isdisjoint(ids_b)            # content-addressed


def test_references_survive_id_reassignment(report):
    r = extract(str(report), ExtractionConfig(**CFG))
    ids = {e.id for e in r.elements}
    linked = [e for e in r.elements if e.kind == "caption" and e.parent_id]
    assert linked and all(e.parent_id in ids for e in linked)


# --------------------------------------------- document-level heading levels

def test_heading_levels_are_consistent_across_pages(tmp_path):
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((72, 80), "Chapter One", fontsize=20)
    p1.insert_text((72, 120), "Section A", fontsize=14)
    p1.insert_text((72, 150), "Body text " * 8, fontsize=10)
    p2 = doc.new_page()                       # page with only a 14 pt heading
    p2.insert_text((72, 80), "Section B", fontsize=14)
    p2.insert_text((72, 120), "Body text " * 8, fontsize=10)
    path = tmp_path / "levels.pdf"
    doc.save(str(path))
    r = extract(str(path), ExtractionConfig(**CFG))
    levels = {e.text: e.level for e in r.elements if e.kind == "heading"}

    assert levels["Section A"] == levels["Section B"] == 2
    assert levels["Chapter One"] == 1


# ---------------------------------------------------- unique asset names

def test_every_crop_has_its_own_file(report, tmp_path):
    assets = tmp_path / "assets"
    r = extract(str(report), ExtractionConfig(assets_dir=str(assets), **CFG))
    paths = [e.visual_ref["image_path"] for e in r.elements
             if e.visual_ref and e.visual_ref.get("image_path")]
    assert len(paths) == len(set(paths))


# ----------------------------------------------- caption column awareness

def test_caption_does_not_link_to_a_figure_in_the_other_column():
    caption_left = [72, 300, 280, 312]
    figure_right = [320, 150, 540, 295]       # closer vertically, other column
    figure_left = [72, 100, 280, 240]
    assert _caption_distance(caption_left, figure_right) == 1e9
    assert _caption_distance(caption_left, figure_left) == pytest.approx(60, abs=1)


# ------------------------------------------- repeated images are not figures

def test_logo_repeated_on_every_page_is_marked(report):
    r = extract(str(report), ExtractionConfig(**CFG))
    figures = [e for e in r.elements if e.kind == "figure"]
    repeated = [e for e in figures if e.metadata.get("repeated_image")]
    assert len(repeated) == 4 and len(figures) == 5    # 4 logos + 1 real chart
    assert r.stats["repeated_images"] == 1


# ------------------------------------------------------- OCR is accounted

def test_missing_ocr_language_is_counted_per_page(tmp_path):
    path = blank_pdf(tmp_path / "scan.pdf", pages=2)
    status = ocr_status("eng+zzz")
    assert status["missing"] == ["zzz"]
    r = extract(str(path), ExtractionConfig(ocr_langs="eng+zzz"))
    assert r.stats["ocr_degraded_pages"] == 2
    assert any("zzz" in w for w in r.warnings)


def test_no_usable_ocr_language_fails_every_page(tmp_path):
    path = blank_pdf(tmp_path / "scan.pdf", pages=3)
    r = extract(str(path), ExtractionConfig(ocr_langs="zzz"))
    assert r.stats["ocr_failed_pages"] == 3


def test_missing_tesseract_binary_fails_every_page(tmp_path, monkeypatch):
    path = blank_pdf(tmp_path / "scan.pdf", pages=2)
    monkeypatch.setattr(pdf_mod, "ocr_status",
                        lambda langs: {"binary": None, "missing": ["eng"],
                                       "installed": [], "requested": ["eng"]})
    monkeypatch.setattr(pdf_mod, "usable_langs", lambda langs: None)
    r = extract(str(path), ExtractionConfig(**CFG))
    assert r.stats["ocr_failed_pages"] == 2
    assert any("Tesseract not found" in w for w in r.warnings)


# ------------------------------------------ garbled text layers go to OCR

def test_garbled_page_is_counted_and_routed_to_ocr(report, monkeypatch):
    monkeypatch.setattr(pdf_mod, "garbled_ratio", lambda text: 0.9)
    calls = []
    real = pdf_mod._ocr_lines
    monkeypatch.setattr(pdf_mod, "_ocr_lines",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    r = extract(str(report), ExtractionConfig(**CFG))
    assert r.stats["garbled_pages"] == 4 and len(calls) == 4


def test_page_rotation_is_recorded(tmp_path):
    doc = fitz.open()
    doc.new_page().set_rotation(90)
    path = tmp_path / "rot.pdf"
    doc.save(str(path))
    r = extract(str(path), ExtractionConfig(**CFG))
    assert r.stats["page_dims"]["1"]["rotation"] == 90
