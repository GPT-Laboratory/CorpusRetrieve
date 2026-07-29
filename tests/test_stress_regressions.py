"""New regression tests from the CorpusRetrieve v1.0 stress/regression pass.

Drop this file into tests/ as tests/test_stress_regressions.py. Each test is
written to FAIL against the current codebase and PASS once the corresponding
bug (see CorpusRetrieve_v1_Stress_Regression_Report.md) is fixed, so this file
can be used directly as a fix-verification / no-regression gate.

Covers:
  - BUG-1: table extraction must not emit zero-row/zero-col garbage Elements
  - BUG-2: numbered bibliography entries must not be promoted to kind="formula"
  - BUG-4: directory-mode batch extraction must isolate per-file failures,
           exactly like extract_zip already does
  - BUG-5: /extract and /chunk must return a clean, informative error (not a
           bare 500) when given an empty/corrupt/mislabeled upload
  - BUG-6: `python -m docparse --json <file>` must print valid JSON on
           stdout, with no third-party library output ahead of it
  - BUG-7: the sklearn-based projection fallback dependency must actually be
           installable from requirements.txt (documented-dependency check)
  - BUG-8: ingesting a chunks.json containing multiple documents must not
           silently drop every document but the first
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from docparse import ExtractionConfig, extract
from docparse.pipeline import extract_path


# --------------------------------------------------------------------------
# BUG-1: table extraction should never emit elements with 0 rows and 0 cols.
# A "table" with no extractable rows is not a table; it is a false-positive
# candidate from PyMuPDF's find_tables() and should be dropped, not surfaced
# as a first-class Element (which then becomes a table_summary chunk in
# Stage 4 and a real point in the `tables` Qdrant collection in Stage 5).
# --------------------------------------------------------------------------
def test_table_elements_never_have_zero_rows(tmp_path, monkeypatch):
    # PyMuPDF's find_tables() on real-world dense/wrapped multi-column pages
    # (verified on the report used for this test pass: 358 of 393 detected
    # "tables" ended up with zero structured body rows) reliably returns
    # single-row candidates -- a fragment of wrapped text, not a real table.
    # Because rows_to_structured() always treats the first extracted row as
    # the header and everything after it as the body, ANY single-row
    # candidate ends up with zero body rows by construction (raw
    # t.extract() output is never literally an empty list, so a naive
    # `if not rows` check does not catch this -- the real signature only
    # shows up after serialize_table()/rows_to_structured() run). Rather
    # than depend on PyMuPDF's version-specific, rendering-dependent
    # detection heuristics to reproduce that here, this test stubs both
    # observed shapes of the false positive directly.
    import fitz  # PyMuPDF

    class _FakeTable:
        bbox = (0, 0, 10, 10)
        row_count = 1
        col_count = 3

        def extract(self):
            # a single row of scattered text fragments -- the dominant
            # real-world garbage shape (e.g. wrapped citation-number columns)
            return [["(25,30,32,40-45,47,49,52,54-58)", "(5,38,49,57,60-66,86)", "(67-69)"]]

    class _FakeTableTrulyEmpty:
        bbox = (20, 20, 30, 30)
        row_count = 0
        col_count = 0

        def extract(self):
            return []

    class _FakeFindTablesResult:
        tables = [_FakeTable(), _FakeTableTrulyEmpty()]

    pdf_path = tmp_path / "one_page.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((36, 72), "Ordinary narrative content, no real table here.", fontsize=11)
    doc.save(str(pdf_path))
    doc.close()

    monkeypatch.setattr(
        fitz.Page, "find_tables",
        lambda self, *a, **kw: _FakeFindTablesResult(),
    )

    result = extract(str(pdf_path), ExtractionConfig())
    tables = [e for e in result.elements if e.kind == "table"]
    empty = [t for t in tables if not t.structured.get("rows")]

    assert not empty, (
        f"{len(empty)} of {len(tables)} table elements have zero structured "
        "body rows -- both single-row and truly-empty find_tables() "
        "candidates are false positives and should be filtered before "
        "becoming Elements (see BUG-1)."
    )


# --------------------------------------------------------------------------
# BUG-2: a numbered bibliography entry must not be misclassified as a
# formula just because it is short and starts with "(N)". This reproduces
# the exact false positives found in the real report ("(42)" and
# "(1) MMM. Kansallinen ilmastonmuutokseen sopeutumissuunnitelma 2022. 2014;5.").
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ref_text", [
    "(42)",
    "(1) MMM. Kansallinen ilmastonmuutokseen sopeutumissuunnitelma 2022. 2014;5.",
    "(2) Sorvali J. Ilmastonmuutoksen haitalliset vaikutukset ja toimialojen haavoittuvuus. 2013.",
])
def test_short_numbered_reference_lines_are_not_formulas(ref_text):
    from docparse.models import Element, ExtractionResult
    from docparse.scientific import clean

    r = ExtractionResult(source="x", doc_type="pdf")
    r.elements = [Element(kind="narrative", text=ref_text)]
    clean(r, ExtractionConfig())

    assert r.elements[0].kind != "formula", (
        f"Bibliography-style entry {ref_text!r} was promoted to kind='formula'. "
        "The equation heuristic must require an actual math symbol, or must "
        "not fire inside/adjacent to a references section (see BUG-2)."
    )


# --------------------------------------------------------------------------
# BUG-4: extract_path() must isolate per-file failures in directory mode the
# same way extract_zip() already does for archive members. One bad file must
# not prevent every other file in the directory from being processed.
# --------------------------------------------------------------------------
def test_directory_batch_isolates_per_file_failures(tmp_path):
    good_dir = tmp_path / "batch"
    good_dir.mkdir()

    (good_dir / "good1.txt").write_text("Some perfectly fine text content.\n")
    (good_dir / "good2.csv").write_text("col1,col2\n1,2\n3,4\n")
    (good_dir / "bad_empty.pdf").write_bytes(b"")  # 0-byte "PDF" -> PyMuPDF raises

    # This must NOT raise, and must return results for the two good files
    # even though the third file is unreadable.
    results = extract_path(str(good_dir), ExtractionConfig())

    sources = {os.path.basename(r.source) for r in results}
    assert "good1.txt" in sources
    assert "good2.csv" in sources
    # The bad file should show up as a warning-bearing result, not a crash.
    assert "bad_empty.pdf" in sources
    bad = next(r for r in results if os.path.basename(r.source) == "bad_empty.pdf")
    assert bad.warnings, "the failed file should carry a warning, not silently vanish"


# --------------------------------------------------------------------------
# BUG-7: the /api/jobs/{job_id}/projection PCA-fallback path imports sklearn.
# If sklearn isn't declared as a (real or optional) dependency, a clean
# `pip install -r requirements.txt` leaves this endpoint permanently broken.
# This test doesn't require the service to run -- it just checks that any
# module the codebase actually imports somewhere is declared in
# requirements.txt (or is stdlib), catching this whole class of gap.
# --------------------------------------------------------------------------
def test_sklearn_fallback_dependency_is_declared():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    req_path = os.path.join(repo_root, "requirements.txt")
    web_py_path = os.path.join(repo_root, "service", "web.py")

    with open(web_py_path, encoding="utf-8") as f:
        web_src = f.read()
    with open(req_path, encoding="utf-8") as f:
        req_src = f.read().lower()

    if "sklearn" in web_src or "scikit-learn" in web_src:
        assert "scikit-learn" in req_src or "sklearn" in req_src, (
            "service/web.py imports sklearn (e.g. as a PCA fallback for the "
            "embedding-projection endpoint) but scikit-learn is not declared "
            "anywhere in requirements.txt -- a fresh install per the README "
            "leaves this code path broken with ModuleNotFoundError (see BUG-7)."
        )


# --------------------------------------------------------------------------
# Bonus: CLI --json output on stdout should be valid JSON on its own. This
# guards against BUG-6 (third-party library prints polluting stdout before
# the JSON payload) for anyone using the natural `--json file.pdf > out.json`
# shell-redirection pattern rather than --out.
# --------------------------------------------------------------------------
def test_cli_json_stdout_is_parseable_json(tmp_path):
    import json
    import fitz  # PyMuPDF

    # Must be a real PDF: only the PDF extractor path calls find_tables(),
    # which is what pollutes stdout. A .txt fixture never exercises this.
    sample = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((36, 72), "Some narrative content for a smoke test.", fontsize=11)
    doc.save(str(sample))
    doc.close()

    proc = subprocess.run(
        [sys.executable, "-m", "docparse", "--json", str(sample)],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0
    try:
        json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        pytest.fail(
            "python -m docparse --json <file> did not print valid JSON on "
            f"stdout ({exc}). stdout began with: {proc.stdout[:120]!r}. "
            "This usually means a third-party library (e.g. PyMuPDF's "
            "find_tables() notice) printed to stdout ahead of the JSON "
            "payload (see BUG-6)."
        )


# --------------------------------------------------------------------------
# BUG-5: /extract and /chunk must not crash with a bare, uninformative 500
# on trivially bad uploads (empty file, corrupt file, mislabeled extension).
# They should return a clean 4xx with a message identifying the failure,
# the same way /ingest already does.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("endpoint", ["/extract", "/chunk"])
@pytest.mark.parametrize("filename,content", [
    ("empty.pdf", b""),
    ("empty.docx", b""),
    ("fake.pdf", b"This is not a real PDF file, just plain text."),
])
def test_bad_upload_returns_clean_error_not_bare_500(endpoint, filename, content):
    from fastapi.testclient import TestClient

    from service.app import app

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(endpoint, files={"file": (filename, content, "application/octet-stream")})

    assert response.status_code != 500 or response.text != "Internal Server Error", (
        f"{endpoint} returned a bare, uninformative 500 for {filename!r} "
        "(no exception handling around the extraction call -- see BUG-5). "
        f"status={response.status_code} body={response.text!r}"
    )
    assert response.status_code < 500, (
        f"{endpoint} should return a client error (4xx) for a bad upload "
        f"like {filename!r}, not a server error. Got {response.status_code}."
    )


# --------------------------------------------------------------------------
# BUG-8: ingesting a chunks.json containing multiple documents (e.g. produced
# by running Stage 4 over a directory of several source files) must not
# silently keep only the first document and drop the rest with no warning.
# --------------------------------------------------------------------------
def test_multi_document_chunks_are_all_ingested(tmp_path):
    from docparse.vectorization.ingest import IngestConfig, ingest

    def _doc(doc_id, n_chunks):
        return {
            "doc_id": doc_id, "source": f"{doc_id}.pdf", "stats": {},
            "chunks": [
                {"chunk_id": f"{doc_id}_c{i}", "doc_id": doc_id,
                 "chunk_type": "section_text", "text": f"Some content {i} for {doc_id}.",
                 "token_estimate": 10, "role": "body", "section_path": [],
                 "pages": [1], "metadata": {}}
                for i in range(n_chunks)
            ],
        }

    combined_path = tmp_path / "combined_chunks.json"
    import json
    json.dump([_doc("doc_one", 5), _doc("doc_two", 3)], combined_path.open("w"))

    cfg = IngestConfig(embedder="hash", db_location=":memory:")
    res, store, kg, sparse = ingest(str(combined_path), cfg)

    assert set(res.doc_ids) == {"doc_one", "doc_two"}, (
        f"expected both documents to be recorded as ingested, got {res.doc_ids!r} "
        "-- a chunks.json with multiple documents must not silently drop any "
        "of them (see BUG-8)."
    )
    total_chunks = sum(res.collections.values())
    assert total_chunks == 8, (
        f"expected all 5+3=8 chunks from both documents to be indexed, but "
        f"collections summed to {total_chunks}: {res.collections!r}"
    )
