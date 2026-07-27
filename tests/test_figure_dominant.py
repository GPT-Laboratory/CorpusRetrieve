"""Figure-dominant page handling (schematics / maps / engineering drawings).

A vector CAD drawing is mostly line-art with sparse label text; PyMuPDF's
find_tables() misreads its grid/wires as page-spanning tables, which then
suppress the one real (whole-page) figure and starve the VLM. These tests lock
in the fix: such pages yield a single full-page figure (routable to a VLM) and
are NOT chopped into phantom tables, while ordinary documents are unaffected.
"""
import glob
import os

import pytest

from docparse import ExtractionConfig, extract

SCHEMATIC = next(iter(glob.glob("/mnt/user-data/uploads/*wire*XC2484*.pdf")
                      + glob.glob("/mnt/user-data/uploads/*.pdf")), None)
PAPER = next(iter(glob.glob("/mnt/user-data/uploads/Embedding*RAG.pdf")), None)


@pytest.mark.skipif(not SCHEMATIC or "XC2484" not in (SCHEMATIC or ""),
                    reason="schematic fixture not available")
def test_schematic_becomes_one_full_page_figure(tmp_path):
    r = extract(SCHEMATIC, ExtractionConfig(assets_dir=str(tmp_path)))
    figs = [e for e in r.elements if e.kind == "figure"]
    full = [f for f in figs if f.metadata.get("full_page")]
    assert len(full) == 1                       # the whole schematic, kept as a figure
    # the figure crop is rendered to disk so a VLM can read it
    path = (full[0].visual_ref or {}).get("image_path")
    assert path and os.path.exists(path) and os.path.getsize(path) > 5000
    # phantom page-spanning "tables" are NOT produced on this page
    tables = [e for e in r.elements if e.kind == "table"]
    assert len(tables) == 0
    # page dimensions are recorded (needed for any provenance overlay)
    assert r.stats.get("page_dims", {}).get("1", {}).get("width", 0) > 0


@pytest.mark.skipif(not PAPER, reason="paper fixture not available")
def test_paper_not_misclassified_as_figure_dominant(tmp_path):
    r = extract(PAPER, ExtractionConfig(assets_dir=str(tmp_path)))
    full = [e for e in r.elements
            if e.kind == "figure" and e.metadata.get("full_page")]
    tables = [e for e in r.elements if e.kind == "table"]
    assert len(full) == 0        # a real paper is never a single page-figure
    assert len(tables) > 0       # its real tables are still detected


def test_figure_dominant_toggle_off_restores_old_behavior(tmp_path):
    if not SCHEMATIC or "XC2484" not in SCHEMATIC:
        pytest.skip("schematic fixture not available")
    r = extract(SCHEMATIC, ExtractionConfig(assets_dir=str(tmp_path),
                                            figure_dominant_detection=False))
    full = [e for e in r.elements
            if e.kind == "figure" and e.metadata.get("full_page")]
    assert len(full) == 0        # opt-out flag disables the new path
