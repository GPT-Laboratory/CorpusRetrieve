"""Pipeline units: stage tracking, page previews, packaging, and projection."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from corpusretrieve.models import STAGES
from corpusretrieve.pipeline.artifacts import safe_identifier, unique_identifier
from corpusretrieve.pipeline.previews import render_page_previews
from corpusretrieve.pipeline.progress import StageTracker
from corpusretrieve.web.projection import project_vectors


def test_stage_tracker_reports_every_stage_in_order():
    seen = []
    tracker = StageTracker(lambda stage, percent, message, stages: seen.append(stages))

    tracker.start("extract")
    tracker.update("extract", documents=2)
    tracker.finish("extract")
    tracker.report(50, "done")

    snapshot = seen[-1]
    assert [record["name"] for record in snapshot] == list(STAGES)
    assert snapshot[0]["status"] == "done"
    assert snapshot[0]["detail"]["documents"] == 2
    assert snapshot[0]["elapsed_ms"] is not None
    assert snapshot[1]["status"] == "pending"


def test_stage_tracker_records_failures():
    tracker = StageTracker(lambda *args: None)
    tracker.start("embed")
    tracker.fail("embed", "model unavailable")
    record = next(r for r in tracker.snapshot() if r["name"] == "embed")
    assert record["status"] == "error"
    assert record["detail"]["error"] == "model unavailable"


def test_identifiers_are_sanitised_and_deduplicated():
    assert safe_identifier("Report 2024/final.pdf") == "Report_2024_final_pdf"
    assert safe_identifier("///") == "document"
    taken: set[str] = set()
    assert unique_identifier("report", taken) == "report"
    assert unique_identifier("report", taken) == "report-2"
    assert unique_identifier("report", taken) == "report-3"


def test_page_previews_are_skipped_for_non_pdf_sources(tmp_path: Path):
    source = tmp_path / "notes.txt"
    source.write_text("plain text", encoding="utf-8")
    assert render_page_previews(source, tmp_path / "assets", 110, 10) == {}


def test_page_previews_describe_geometry(tmp_path: Path):
    fitz = pytest.importorskip("fitz")
    source = tmp_path / "doc.pdf"
    document = fitz.open()
    document.new_page()
    document.new_page()
    document.save(str(source))
    document.close()

    previews = render_page_previews(source, tmp_path / "assets", 90, 10)
    assert set(previews) == {"1", "2"}
    first = previews["1"]
    assert first["path"] == "pages/page-1.png"
    assert first["dpi"] == 90
    assert first["pixel_width"] > 0 and first["width"] > 0
    assert (tmp_path / "assets" / "pages" / "page-1.png").exists()


def test_page_preview_count_is_capped(tmp_path: Path):
    fitz = pytest.importorskip("fitz")
    source = tmp_path / "long.pdf"
    document = fitz.open()
    for _ in range(5):
        document.new_page()
    document.save(str(source))
    document.close()

    assert len(render_page_previews(source, tmp_path / "assets", 72, 2)) == 2


def _vector_lines(count: int, dim: int = 8) -> bytes:
    lines = []
    for index in range(count):
        vector = [float((index * (position + 1)) % 7) for position in range(dim)]
        lines.append(
            json.dumps(
                {
                    "id": f"id-{index}",
                    "vector": vector,
                    "payload": {
                        "chunk_id": f"chunk-{index}",
                        "doc_id": "doc",
                        "chunk_type": "section_text",
                        "raw_text": f"chunk number {index}",
                        "pages": [1],
                    },
                }
            )
        )
    return "\n".join(lines).encode("utf-8")


def test_projection_normalizes_into_the_unit_square():
    payload = project_vectors(_vector_lines(20).decode().splitlines(), limit=100)
    assert payload["n"] == 20
    assert payload["method"] in {"pca", "umap"}
    assert all(0.0 <= p["x"] <= 1.0 and 0.0 <= p["y"] <= 1.0 for p in payload["points"])


def test_projection_respects_the_point_limit():
    payload = project_vectors(_vector_lines(50).decode().splitlines(), limit=10)
    assert payload["n"] == 10


def test_projection_handles_an_empty_index():
    assert project_vectors([], limit=10) == {"method": "none", "n": 0, "points": []}
