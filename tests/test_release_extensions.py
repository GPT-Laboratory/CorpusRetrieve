"""Regression tests for generic corpus ingestion.

* identical files are processed once and their records merged
* resumed documents take the CURRENT metadata, stale folders are pruned
* run-level provenance (exclusion list) reaches the bundle manifest
* the converter carries licence / format / Finna ID and counts records
  without a downloadable file
* ``corpusretrieve build`` is resumable and additive
* cross-lingual search through extra queries or a translator
"""
from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import pytest

from corpusretrieve.cli import main as cli
from corpusretrieve.consumer import PortableIndex, cross_lingual_risk
from corpusretrieve.models import InputFile, JobSpec
from corpusretrieve.pipeline import CorpusProcessor, DocumentOptions
from corpusretrieve.pipeline.documents import merge_records
from corpusretrieve.verify import verify_bundle
from docparse.vectorization.canonical import bibliographic_record
from docparse.vectorization.graph import KnowledgeGraph

fitz = pytest.importorskip("fitz")
ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "m2dc", ROOT / "tools" / "manifest_to_document_config.py")
m2dc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m2dc)

FI = ("Materiaalin lämpökäsittely",
      "Materiaalin lämpökäsittely paransi kestävyyttä koekappaleissa. "
      "Käsittelylämpötila oli keskimäärin 90 °C kokeissa.")
EN = ("Thermal treatment of material",
      "Thermal treatment of material improved durability in test samples. "
      "The treatment temperature averaged 90 °C during laboratory trials.")


def _pdf(path: Path, title: str, body: str) -> Path:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 90), title, fontsize=16)
    y = 130
    for sentence in body.split(". "):
        page.insert_text((72, y), sentence.strip(), fontsize=10)
        y += 16
    doc.save(str(path))
    doc.close()
    return path


def _build(out: Path, files: list[Path], metas: list[dict] | None = None) -> dict:
    out.mkdir(exist_ok=True)
    metas = metas or [{} for _ in files]
    spec = JobSpec(job_id="j", corpus_name="c",
                   files=[InputFile(filename=f.name, storage_key=str(f), size_bytes=1,
                                    metadata=m) for f, m in zip(files, metas, strict=True)])
    opts = DocumentOptions(emit_page_previews=False, ocr_langs="eng")
    return CorpusProcessor(lambda *a: None, opts).run(spec, files, out)


def _payloads(out: Path) -> list[dict]:
    return [json.loads(x)["payload"] for x in
            (out / "index/vectors.jsonl").read_text().splitlines() if x.strip()]


# ------------------------------------------------------------- duplicates

def test_identical_files_are_processed_once_and_records_merged(tmp_path):
    a = _pdf(tmp_path / "print.pdf", *EN)
    b = tmp_path / "ebook.pdf"
    b.write_bytes(a.read_bytes())
    rec_print = {"records": [{"title": EN[0], "year": 2015, "source_doc_id": "archive.1",
                              "format": "1/Book/Book/"}]}
    rec_ebook = {"records": [{"title": EN[0], "year": 2015, "source_doc_id": "archive.2",
                              "format": "1/Book/eBook/"}]}
    result = _build(tmp_path / "out", [a, b], [rec_print, rec_ebook])
    manifest = result["manifest"]
    assert len(manifest["documents"]) == 1
    dup, = manifest["duplicate_files"]
    assert dup["filename"] == "ebook.pdf" and dup["records_merged"] == 1
    bib = _payloads(tmp_path / "out")[0]["bibliographic"]
    # one publication catalogued twice: resolved, not "ambiguous"
    assert bib["source_doc_id"] == "archive.1"
    assert bib["alternate_records"][0]["format"] == "1/Book/eBook/"


def test_different_publications_in_one_file_stay_ambiguous_without_ranges():
    doc = {"metadata": {"records": [{"title": "A"}, {"title": "B"}]}}
    assert bibliographic_record(doc, [1])["ambiguous"] is True


def test_merge_records_does_not_duplicate():
    target = {"records": [{"source_doc_id": "x", "title": "T"}], "themes": ["group_a"]}
    assert merge_records(target, {"records": [{"source_doc_id": "x", "title": "T"}],
                                  "themes": ["group_b"]}) == 0
    assert target["themes"] == ["group_a", "group_b"]
    assert merge_records(target, {"title": "T2", "source_doc_id": "y"}) == 1
    assert [r["source_doc_id"] for r in target["records"]] == ["x", "y"]


def test_api_reports_duplicate_uploads(client, tmp_path):
    body = _pdf(tmp_path / "x.pdf", *EN).read_bytes()
    files = [("files", (n, io.BytesIO(body), "application/pdf")) for n in ("a.pdf", "b.pdf")]
    job = client.post("/v1/jobs", files=files).json()["job_id"]
    manifest = client.get(f"/v1/jobs/{job}/manifest").json()
    assert len(manifest["documents"]) == 1
    assert manifest["duplicate_files"][0]["duplicate_of"] == manifest["documents"][0]["doc_id"]
    assert manifest["quality"]["duplicate_files"] == 1
    assert manifest["release_ready"] is True               # informational, not degraded


# ------------------------------------------------- resume uses new metadata

def test_resumed_document_picks_up_updated_metadata(tmp_path):
    src = _pdf(tmp_path / "r.pdf", *FI)
    out = tmp_path / "out"
    _build(out, [src])
    assert _payloads(out)[0]["bibliographic"] is None
    second = _build(out, [src], [{"records": [{"title": FI[0], "doi": "10.1/x"}]}])
    assert second["manifest"]["documents"][0]["resumed"] is True
    assert all(p["bibliographic"]["doi"] == "10.1/x" for p in _payloads(out))


def test_documents_dropped_from_the_collection_leave_the_bundle(tmp_path):
    a, b = _pdf(tmp_path / "a.pdf", *FI), _pdf(tmp_path / "b.pdf", *EN)
    out = tmp_path / "out"
    _build(out, [a, b])
    assert len(list((out / "documents").iterdir())) == 2
    _build(out, [a])
    assert len(list((out / "documents").iterdir())) == 1
    assert verify_bundle(out)["ok"] is True


# ------------------------------------------------ collection-level provenance

def test_collection_block_reaches_the_manifest_and_is_not_an_unmatched_key(client, tmp_path):
    body = _pdf(tmp_path / "x.pdf", *FI).read_bytes()
    config = {"__collection__": {"manifest_date": "2026-10-30",
                                 "exclusions": [{"doc_id": "archive.9", "reason": "Africa"}]},
              "x.pdf": {"metadata": {"records": [{"title": FI[0]}]}}}
    job = client.post("/v1/jobs", files=[("files", ("x.pdf", io.BytesIO(body), "application/pdf"))],
                      data={"document_config": json.dumps(config)}).json()["job_id"]
    status = client.get(f"/v1/jobs/{job}").json()
    assert status["state"] == "completed", status
    manifest = client.get(f"/v1/jobs/{job}/manifest").json()
    assert manifest["collection"]["exclusions"][0]["reason"] == "Africa"
    assert "unmatched_config_keys" not in manifest["quality"]["intake"]


# ----------------------------------------------------------- converter

MANIFEST = {
    "manifest_date": "2026-10-30", "source": "Finna",
    "excluded": [{"doc_id": "archive.99", "title": "Maize in Kenya"}],
    "scenarios": [{"id": "s", "themes": [{"theme_id": "group_a", "documents": [
        {"doc_id": "archive.1", "file_path": "pdf/a.pdf", "title": "A", "building": "Archive",
         "format": "1/OtherText/ResearchReport/", "usage_rights": "CC BY 4.0",
         "finna_record_id": "jukuri.123"},
        {"doc_id": "library.7", "file_path": "pdf/b.pdf", "title": "B", "building": "Library"},
        {"doc_id": "archive.5", "title": "Metadata only, no PDF"},
    ]}]}],
}


def test_converter_carries_provenance_fields_and_collection():
    config, stats = m2dc.build(MANIFEST)
    rec = config["a.pdf"]["metadata"]["records"][0]
    assert rec["license"] == "CC BY 4.0" and rec["format"] == "1/OtherText/ResearchReport/"
    assert rec["finna_id"] == "jukuri.123" and rec["organization"] == "Archive"
    assert config["b.pdf"]["metadata"]["organization"] == "Library"
    collection = config["__collection__"]
    assert collection["manifest_date"] == "2026-10-30"
    assert collection["exclusions"][0]["doc_id"] == "archive.99"
    assert stats["records_without_file"] == ["archive.5"]
    assert collection["records_without_file"] == ["archive.5"]


def test_converter_finds_identical_files(tmp_path):
    (tmp_path / "pdf").mkdir()
    (tmp_path / "pdf/a.pdf").write_bytes(b"same")
    (tmp_path / "pdf/b.pdf").write_bytes(b"same")
    config, _ = m2dc.build(MANIFEST)
    assert m2dc.content_duplicates(config, tmp_path) == [["a.pdf", "b.pdf"]]


def test_new_fields_reach_payload_and_graph():
    rec = {"title": "A", "license": "CC BY 4.0", "format": "1/Book/eBook/",
           "finna_id": "jukuri.1"}
    assert bibliographic_record({"metadata": {"records": [rec]}}, [1])["license"] == "CC BY 4.0"
    kg = KnowledgeGraph()
    kg.build_from_chunks({"doc_id": "d", "chunks": [], "metadata": {"records": [rec]}},
                         "c", "m", "v1")
    paper = kg.g.nodes["paper:d"]
    assert paper["license"] == "CC BY 4.0" and paper["finna_id"] == "jukuri.1"


# ------------------------------------------------------- build CLI (additive)

def test_build_cli_is_resumable_and_additive(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _pdf(corpus / "fi.pdf", *FI)
    _pdf(corpus / "en.pdf", *EN)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "__collection__": {"manifest_date": "2026-10-30"},
        "fi.pdf": {"metadata": {"records": [{"title": FI[0]}]}},
        "en.pdf": {"metadata": {"records": [{"title": EN[0]}]}},
    }))
    out = tmp_path / "out"
    assert cli(["build", str(corpus), "--out", str(out), "--config", str(config)]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["documents"] == 2 and first["reused"] == 0 and first["release_ready"]

    _pdf(corpus / "new.pdf", "Uusi raportti", "Laitteen tehokkuus kasvoi 5 % vuonna 2020.")
    cfg = json.loads(config.read_text())
    cfg["new.pdf"] = {"metadata": {"records": [{"title": "Uusi raportti"}]}}
    config.write_text(json.dumps(cfg))
    assert cli(["build", str(corpus), "--out", str(out), "--config", str(config)]) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["documents"] == 3 and second["reused"] == 2      # only the new file parsed
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["collection"]["manifest_date"] == "2026-10-30"
    assert verify_bundle(out / "bundle.zip")["ok"] is True


def test_build_cli_gate_and_path_keys(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    (corpus / "archive").mkdir(parents=True)
    (corpus / "library").mkdir()
    _pdf(corpus / "archive/report.pdf", *FI)
    _pdf(corpus / "library/report.pdf", *EN)            # same basename, different folder
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"archive/report.pdf": {"metadata": {"title": FI[0]}},
                                  "library/report.pdf": {"metadata": {"title": EN[0]}}}))
    out = tmp_path / "out"
    assert cli(["build", str(corpus), "--out", str(out), "--config", str(config),
                "--key-by", "path"]) == 0
    capsys.readouterr()
    # basename keys collide and match nothing: degraded -> exit 3
    assert cli(["build", str(corpus), "--out", str(tmp_path / "o2"), "--config",
                str(config)]) == 3
    report = json.loads(capsys.readouterr().out)
    assert any(r.startswith("filename_collisions") for r in report["reasons"])
    assert any(r.startswith("unmatched_config_keys") for r in report["reasons"])


# --------------------------------------------------------- cross-lingual

@pytest.fixture(scope="module")
def bilingual(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("xl")
    fi, en = _pdf(tmp / "fi.pdf", *FI), _pdf(tmp / "en.pdf", *EN)
    out = tmp / "out"
    _build(out, [fi, en])
    return out


def _docs(hits) -> set[str]:
    return {h.payload["source_file"] for h in hits}


def test_monolingual_query_misses_the_other_language(bilingual):
    index = PortableIndex.load(bilingual)
    hits = [h for h in index.search("materiaalin lämpökäsittely", k=10) if h.sparse > 0]
    assert _docs(hits) == {"fi.pdf"}                  # the gap the paper did not test


def test_extra_queries_reach_both_languages(bilingual):
    index = PortableIndex.load(bilingual)
    hits = index.search("materiaalin lämpökäsittely", k=10,
                        extra_queries=["material thermal treatment"])
    top = [h for h in hits if h.sparse > 0]
    assert _docs(top) == {"fi.pdf", "en.pdf"}
    en_hit = next(h for h in top if h.payload["source_file"] == "en.pdf")
    assert en_hit.matched_query == "material thermal treatment"
    assert set(en_hit.query_scores) == {"materiaalin lämpökäsittely", "material thermal treatment"}


def test_translator_callable_and_filters(bilingual):
    index = PortableIndex.load(bilingual)
    glossary = {"materiaalin lämpökäsittely": ["material thermal treatment"]}
    hits = index.search("materiaalin lämpökäsittely", k=10, translator=lambda q: glossary.get(q, []))
    assert "en.pdf" in _docs(h for h in hits if h.sparse > 0)
    only_en = index.search("materiaalin lämpökäsittely", k=10, translator=lambda q: glossary[q],
                           where=lambda p: p["source_file"] == "en.pdf")
    assert _docs(only_en) == {"en.pdf"}


def test_language_metadata_and_risk_warning(bilingual):
    if importlib.util.find_spec("lingua") is None and importlib.util.find_spec("langdetect") is None:
        pytest.skip("language-detection dependencies are not installed in this environment")
    meta = json.loads((bilingual / "index/metadata.json").read_text())
    assert {"fi", "en"} <= set(meta["languages"])
    index = PortableIndex.load(bilingual)
    assert index.crosslingual_risk() is True
    assert any("multilingual corpus" in w for w in verify_bundle(bilingual)["warnings"])
    assert cross_lingual_risk("bge-m3", {"fi": 5, "en": 5}) is False
    fi_only = index.search("typpi", k=10, languages=["fi"])
    assert fi_only and all(h.payload["language"] == "fi" for h in fi_only)
