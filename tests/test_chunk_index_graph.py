"""Unit tests: provenance-faithful chunks, BM25 semantics, payload, graph."""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import pytest

from docparse import ExtractionConfig, chunk_result, extract
from docparse.models import Element, ExtractionResult
from docparse.postprocess import finalize
from docparse.vectorization.canonical import bibliographic_record, normalize_chunk, payload_for
from docparse.vectorization.embedding import BM25SparseEncoder, _tokens, resolve_embedder
from docparse.vectorization.graph import KnowledgeGraph
from docparse.vectorization.ids import point_composite

fitz = pytest.importorskip("fitz")
sys.path.insert(0, str(Path(__file__).parent))
from fixtures import bilingual_report  # noqa: E402


@pytest.fixture(scope="module")
def extracted(tmp_path_factory):
    path = bilingual_report(tmp_path_factory.mktemp("r") / "report.pdf")
    result = extract(str(path), ExtractionConfig(ocr_langs="eng"))
    return result, chunk_result(result, doc_id="doc")


# ------------------------------------------------------ faithful spans

def test_every_span_maps_to_text_the_unit_contains(extracted):
    result, chunks = extracted
    text = result.document_text()
    for c in chunks.chunks:
        if c.chunk_type != "section_text":
            continue
        assert c.spans, c.chunk_id
        for a, b in c.spans:
            assert text[a:b] in c.text


def test_claim_span_is_the_claim_sentence(extracted):
    result, chunks = extracted
    text = result.document_text()
    claim = next(c for c in chunks.chunks if c.chunk_type == "claim")
    (a, b), = claim.spans
    assert text[a:b] == claim.text.split("\nContext:")[0]


# ----------------------------------------------------- inline locators

def test_every_unit_carries_page_and_box_inline(extracted):
    _, chunks = extracted
    for c in chunks.chunks:
        assert c.locators, c.chunk_type
        assert all(loc["page"] for loc in c.locators)
        assert any(loc.get("bbox") for loc in c.locators), c.chunk_type


# --------------------------------------------- no cross-role/section merge

def test_tiny_chunk_is_not_merged_into_another_section_or_role():
    r = ExtractionResult(source="x.pdf")
    specs = [("heading", "1 Johdanto", 1),
             ("narrative", "Johdannon teksti on tässä riittävän pitkä lause. " * 3, None),
             ("heading", "2 Results", 1),
             ("narrative", "Yield rose 35% compared to 2010.", None)]
    for i, (k, t, lvl) in enumerate(specs):
        r.elements.append(Element(kind=k, text=t, level=lvl, page=1, order=i))
    r.stats["source_sha256"] = "1" * 64
    finalize(r, ExtractionConfig(detect_language=False))
    from docparse import scientific
    scientific.annotate(r, ExtractionConfig())
    sections = [c for c in chunk_result(r, doc_id="d").chunks if c.chunk_type == "section_text"]
    results = [c for c in sections if "Yield rose" in c.text]
    assert len(results) == 1 and results[0].role == "results"
    assert results[0].section_path == ["2 Results"]


# ---------------------------------------------- content-derived chunk IDs

def test_chunk_ids_are_content_derived_and_parent_links_remapped(extracted):
    _, chunks = extracted
    ids = {c.chunk_id for c in chunks.chunks}
    assert len(ids) == len(chunks.chunks)
    # content hashes, not positional counters such as "::0007" (a hash made
    # only of digits is valid, so test the format rather than isdigit)
    assert all(re.fullmatch(r"[0-9a-f]{12}(-\d+)?", i.split("::")[-1]) for i in ids)
    for c in chunks.chunks:
        if c.parent_chunk_id:
            assert c.parent_chunk_id in ids


# ---------------------------------- logos dropped, Finnish mentions

def test_logos_are_not_units_and_finnish_mentions_link_tables(extracted):
    _, chunks = extracted
    figures = [c for c in chunks.chunks if c.chunk_type == "figure"]
    assert len(figures) == 1 and figures[0].metadata["label"] == "Figure 1"
    sec = next(c for c in chunks.chunks if "Taulukossa 1" in c.text)
    assert sec.linked_tables == ["Table 1"]


def test_model_written_text_has_no_source_span():
    r = ExtractionResult(source="x.pdf")
    fig = Element(kind="figure", text="", page=1, order=0, bbox=[0, 0, 100, 100])
    r.elements = [fig]
    r.stats["source_sha256"] = "2" * 64
    finalize(r, ExtractionConfig(detect_language=False))
    fig.text = "A bar chart of yields"            # what describe_figures does
    fig.metadata.update(description="vlm", derived_text=True)
    c = next(c for c in chunk_result(r, doc_id="d").chunks if c.chunk_type == "figure")
    assert c.spans == [] and c.metadata["derived_text"] is True
    assert c.locators[0]["bbox"] == [0, 0, 100, 100]


# ------------------------------------------------- Unicode BM25 tokens

def test_unicode_tokenizer_keeps_finnish_words():
    assert _tokens("Määrä ja maaperän laatu") == ["määrä", "ja", "maaperän", "laatu"]

    assert _tokens("Määrä ja maaperän", "ascii-v1") == ["ja", "maaper"]


def test_unicode_tokenizer_casefolds_and_normalises():
    assert _tokens("ÄÄRIARVO") == _tokens("ääriarvo")
    assert _tokens("ﬁeld") == ["field"]            # NFKC ligature


# ------------------------------------------- query-side BM25 encoding

def test_query_dot_document_is_the_bm25_score():
    corpus = ["mittaus arvo kasvoi", "raportti arvo laski", "mittaus mittaus tulos"]
    enc = BM25SparseEncoder().fit(corpus)
    qi, qv = enc.encode_query("mittaus arvo")
    assert qv == [1.0] * len(qi)
    di, dv = enc.encode(corpus[0])
    doc = dict(zip(di, dv, strict=True))
    score = sum(doc.get(i, 0.0) * v for i, v in zip(qi, qv, strict=True))
    # reference BM25 for the same document and terms
    n, avg = 3, sum(len(t.split()) for t in corpus) / 3
    ref = 0.0
    for term in ("mittaus", "arvo"):
        d = sum(term in t.split() for t in corpus)
        idf = math.log(1 + (n - d + 0.5) / (d + 0.5))
        tf = corpus[0].split().count(term)
        ref += idf * tf * 2.4 / (tf + 1.4 * (1 - 0.75 + 0.75 * 3 / avg))
    assert score == pytest.approx(ref)


def test_encoder_records_tokenizer_and_fingerprint_and_loads_older_encoders(tmp_path):
    enc = BM25SparseEncoder().fit(["a b", "b c"])
    d = enc.to_dict()
    assert d["tokenizer"] == "unicode-v2" and d["corpus_fingerprint"]
    assert list(d["vocab"]) == sorted(d["vocab"])          # sorted vocabulary
    legacy = {"k1": 1.4, "b": 0.75, "avg_len": 2, "vocab": {"ab": 0}, "idf": {"ab": 1.0},
          "name": "bm25-local-v1"}
    (tmp_path / "legacy.json").write_text(json.dumps(legacy))
    assert BM25SparseEncoder.load(tmp_path / "legacy.json").tokenizer == "ascii-v1"


# ------------------------------------------------ embedder fallback flag

def test_auto_embedder_fallback_is_recorded():
    emb = resolve_embedder("auto")
    if emb.name.startswith("hashing"):
        assert emb.fallback_from == "auto"
    assert resolve_embedder("hash").fallback_from is None


# ------------------------------------------------- bibliographic payload

RECORDS = [{"title": "Issue article A", "year": 2015, "page_start": 1, "page_end": 2},
           {"title": "Issue article B", "year": 2015, "page_start": 3, "page_end": 9}]


def test_single_record_resolves_every_unit():
    doc = {"metadata": {"records": [{"title": "Report", "handle_url": "http://h/1"}]}}
    assert bibliographic_record(doc, [7])["title"] == "Report"


def test_multi_record_file_resolves_by_page_range():
    doc = {"metadata": {"records": RECORDS}}
    assert bibliographic_record(doc, [2])["title"] == "Issue article A"
    assert bibliographic_record(doc, [5, 6])["title"] == "Issue article B"
    assert bibliographic_record(doc, [20])["ambiguous"] is True


def test_flat_older_metadata_still_resolves():
    assert bibliographic_record({"metadata": {"title": "T", "year": 2010}}, [1])["title"] == "T"
    assert bibliographic_record({"metadata": {}}, [1]) is None


def test_payload_carries_v2_provenance_fields(extracted):
    _, chunks = extracted
    chunk = normalize_chunk(chunks.chunks[0].to_dict())
    doc = {"doc_id": "doc", "metadata": {"records": [{"title": "Report"}]}}
    p = payload_for(chunk, doc, "hashing-v1", "v1", "corpus")
    assert p["schema_version"] == "2.0.0"
    assert p["bibliographic"]["title"] == "Report"
    assert p["spans"] and p["locators"] and "language" in p


def test_point_identity_does_not_depend_on_corpus_name():
    assert "corpus" not in point_composite("d", "claim", "d::claim::ab", "m", "v1")


# --------------------------------------------------------------- graph

def _graph_doc(chunks: list[dict], **meta) -> dict:
    return {"doc_id": "d", "source": "d.pdf", "chunks": chunks, "metadata": meta}


def test_unlabelled_row_groups_link_to_their_table_by_parent():
    kg = KnowledgeGraph()
    kg.build_from_chunks(_graph_doc([
        {"chunk_id": "d::tab::1", "chunk_type": "table_summary", "metadata": {}},
        {"chunk_id": "d::tabrows::1", "chunk_type": "table_row_group",
         "parent_chunk_id": "d::tab::1", "metadata": {}},
    ]), "corpus", "m", "v1")
    assert kg.g.has_edge("table:d::tab::1", "tablerowgroup:d::tabrows::1",
                         key="TABLE_HAS_ROW_GROUP")


def test_reference_entry_and_in_text_citation_share_one_node():
    kg = KnowledgeGraph()
    kg.build_from_chunks(_graph_doc([
        {"chunk_id": "d::sec::1", "chunk_type": "section_text",
         "citations": [{"key": "smith2012"}], "metadata": {}},
        {"chunk_id": "d::ref::1", "chunk_type": "reference",
         "metadata": {"citation_key": "smith2012"}},
    ]), "corpus", "m", "v1")
    works = [n for n, d in kg.g.nodes(data=True) if d["ntype"] == "ReferenceWork"]
    assert works == ["citekey:d:smith2012"]


def test_same_titled_sections_under_different_parents_stay_distinct():
    kg = KnowledgeGraph()
    kg.build_from_chunks(_graph_doc([
        {"chunk_id": "d::sec::1", "chunk_type": "section_text",
         "section_path": ["Study 1", "Results"], "metadata": {}},
        {"chunk_id": "d::sec::2", "chunk_type": "section_text",
         "section_path": ["Study 2", "Results"], "metadata": {}},
    ]), "corpus", "m", "v1")
    results = [d for _, d in kg.g.nodes(data=True)
               if d["ntype"] == "Section" and d["title"] == "Results"]
    assert len(results) == 2


def test_paper_node_carries_bibliographic_record():
    kg = KnowledgeGraph()
    kg.build_from_chunks(_graph_doc(
        [{"chunk_id": "d::sec::1", "chunk_type": "section_text", "metadata": {}}],
        records=[{"title": "Sample report", "year": 2015, "handle_url": "http://h/1"}]),
        "corpus", "m", "v1")
    paper = kg.g.nodes["paper:d"]
    assert paper["title"] == "Sample report" and paper["handle_url"] == "http://h/1"


def test_cypher_export_uses_constraints_labels_and_batches(tmp_path):
    kg = KnowledgeGraph()
    kg.build_from_chunks(_graph_doc(
        [{"chunk_id": f"d::sec::{i}", "chunk_type": "section_text", "metadata": {}}
         for i in range(3)]), "corpus", "m", "v1")
    text = Path(kg.export(tmp_path)["cypher"]).read_text()
    assert "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Chunk) REQUIRE n.id IS UNIQUE" in text
    assert "UNWIND [" in text and "MATCH (a:Chunk {id: row.a})" in text
    assert "MATCH (a {id:" not in text


def test_one_failing_export_does_not_block_the_others(tmp_path, monkeypatch):
    kg = KnowledgeGraph()
    kg._node("paper:x", "Paper")
    monkeypatch.setattr(KnowledgeGraph, "_export_graphml",
                        lambda self, out: (_ for _ in ()).throw(ValueError("bad xml")))
    out = kg.export(tmp_path)
    assert "graphml" not in out and out["errors"]["graphml"].startswith("ValueError")
    assert Path(out["json"]).exists() and Path(out["cypher"]).exists()
