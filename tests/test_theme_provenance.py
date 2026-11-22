"""Tests for provenance strictness, PDF-derived fallback and theme manifests."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from corpusretrieve.models import InputFile, JobSpec
from corpusretrieve.pipeline import CorpusProcessor, DocumentOptions
from corpusretrieve.verify import verify_bundle
from docparse.vectorization.canonical import bibliographic_record

fitz = pytest.importorskip("fitz")
ROOT = Path(__file__).resolve().parents[1]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    spec.loader.exec_module(module)
    return module


def _pdf(path: Path, with_doi: bool = True):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((48, 40), "Research Methods Review", fontsize=14)
    page.insert_text((48, 135), "A study of instrument reliability in remote laboratories", fontsize=16)
    page.insert_text((48, 185), "The study measured system behavior and reliability.", fontsize=10)
    if with_doi:
        page.insert_text((48, 760), "https://doi.org/10.1234/example.2026.001", fontsize=8)
    doc.save(str(path))
    doc.close()
    return path


def _build(path, out, metadata=None, strict=True):
    spec = JobSpec(job_id="strict", corpus_name="test", require_bibliographic=strict,
                   files=[InputFile(filename=path.name, storage_key=str(path),
                                    size_bytes=path.stat().st_size, metadata=metadata or {})])
    return CorpusProcessor(lambda *args: None, DocumentOptions(
        emit_page_previews=False, ocr_langs="eng")).run(spec, [path], out)


def test_pdf_doi_fallback_keeps_article_title_not_journal(tmp_path):
    path = _pdf(tmp_path / "journal.pdf")
    out = tmp_path / "out"
    manifest = _build(path, out)["manifest"]
    assert manifest["release_ready"], manifest["quality"]
    rec = manifest["documents"][0]["metadata"]["records"][0]
    assert rec["doi"] == "10.1234/example.2026.001"
    assert rec["origin"] == "extracted"
    assert rec["title"] == "A study of instrument reliability in remote laboratories"
    points = [json.loads(x) for x in (out / "index/vectors.jsonl").read_text().splitlines() if x]
    assert all(p["payload"]["bibliographic"]["origin"] == "extracted" for p in points)
    assert verify_bundle(out)["ok"]


def test_strict_provenance_rejects_no_record(tmp_path):
    path = _pdf(tmp_path / "x.pdf", with_doi=False)
    out = tmp_path / "out"
    manifest = _build(path, out)["manifest"]
    assert not manifest["release_ready"]
    assert "documents_without_bibliographic_record=1" in manifest["quality"]["reasons"]
    assert manifest["quality"]["unresolved_bibliographic_units"] > 0
    assert verify_bundle(out)["ok"] is False


def test_manifest_record_wins_over_extracted_doi(tmp_path):
    path = _pdf(tmp_path / "journal.pdf")
    out = tmp_path / "out"
    metadata = {"records": [{"title": "Trusted source title", "year": 2024,
                             "handle_url": "https://repository.test/123", "source": "Example Repository", "origin": "manifest"}]}
    manifest = _build(path, out, metadata)["manifest"]
    assert manifest["release_ready"]
    assert manifest["documents"][0]["metadata"]["records"] == metadata["records"]


def test_ambiguous_multi_publication_records_are_not_release_ready(tmp_path):
    path = _pdf(tmp_path / "x.pdf", with_doi=False)
    meta = {"records": [{"title": "Article A"}, {"title": "Article B"}]}
    manifest = _build(path, tmp_path / "out", meta)["manifest"]
    assert not manifest["release_ready"]
    assert manifest["quality"]["ambiguous_bibliographic_units"] > 0


def test_multi_page_range_must_cover_all_unit_pages():
    records = {"metadata": {"records": [
        {"title": "A", "page_start": 1, "page_end": 3},
        {"title": "B", "page_start": 4, "page_end": 7},
    ]}}
    assert bibliographic_record(records, [2])['title'] == 'A'
    assert bibliographic_record(records, [3, 4])['ambiguous'] is True
    assert bibliographic_record({"metadata": {"records": [{}]}}, [1]) is None


def test_themes_split_without_losing_original_global_membership(tmp_path):
    tool = _load_script("prepare_theme_jobs")
    manifest = {
        "project_name": "p", "run_id": "r", "scenarios": [{"id": "s", "themes": [
            {"theme_id": "group_a", "documents": [
                {"doc_id": "a", "file_path": "files/group_a/a.pdf", "title": "A", "handle_url": "https://repo/a"},
                {"doc_id": "b", "file_path": "files/group_a/b.pdf", "title": "B", "handle_url": "https://repo/b"},
            ]},
            {"theme_id": "group_b", "documents": [
                {"doc_id": "a", "file_path": "files/group_a/a.pdf", "title": "A", "handle_url": "https://repo/a"},
            ]},
        ]}],
    }
    report = tool.run(manifest, tmp_path / "themes", max_files=1)
    assert len(report["themes"]) == 2
    config = json.loads((tmp_path / "themes/group_a/full_theme.path_config.json").read_text())
    assert len(config) == 3  # two documents + __collection__
    assert config["files/group_a/a.pdf"]["metadata"]["themes_global"] == ["group_a", "group_b"]
    assert len(report["themes"][0]["parts"]) == 2


def test_archive_batch_member_paths_match_metadata(tmp_path):
    from zipfile import ZipFile

    tool = _load_script("prepare_theme_jobs")
    root = tmp_path / "root"
    path = root / "files/group_a/a.pdf"
    path.parent.mkdir(parents=True)
    _pdf(path)
    manifest = {"project_name": "p", "run_id": "r", "scenarios": [{
        "id": "s", "themes": [{"theme_id": "group_a", "documents": [{
            "doc_id": "a", "file_path": "files/group_a/a.pdf",
            "title": "A", "handle_url": "https://repository.test/a"}]}]}]}
    tool.run(manifest, tmp_path / "out", files_root=root, make_zips=True)
    with ZipFile(tmp_path / "out/group_a/part-001.zip") as zf:
        assert zf.namelist() == ["files/group_a/a.pdf"]
    config = json.loads((tmp_path / "out/group_a/part-001.path_config.json").read_text())
    assert config["files/group_a/a.pdf"]["document_id"] == "a"


def test_cli_can_build_only_files_listed_in_theme_config(tmp_path):
    from corpusretrieve.cli import main

    folder = tmp_path / "pdfs"
    folder.mkdir()
    _pdf(folder / "a.pdf")
    _pdf(folder / "b.pdf")
    config = {"a.pdf": {"document_id": "a", "metadata": {
        "records": [{"title": "A", "handle_url": "https://repo/a"}]}}}
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(config))
    rc = main(["build", str(folder), "--out", str(tmp_path / "out"),
               "--config", str(cfg), "--only-config-files", "--require-bibliographic"])
    assert rc == 0
    manifest = json.loads((tmp_path / "out/manifest.json").read_text())
    assert [d["source"] for d in manifest["documents"]] == ["a.pdf"]
