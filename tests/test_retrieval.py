"""Stage 6 tests: reranking, graph expansion, retriever composition, and the
persist->reload cycle. All offline (hash embedder + lexical reranker)."""
import json

import pytest

from docparse.vectorization import (EdgePolicy, IngestConfig, LexicalReranker,
                                    RetrieveConfig, ingest, load_retriever, rerank)
from docparse.vectorization.expand import expand_hits
from docparse.vectorization.vectorstore import SearchHit


# ---- reranking ---------------------------------------------------------------

def test_lexical_reranker_orders_by_relevance():
    rr = LexicalReranker()
    passages = ["cold days decrease sharply with warming levels",
                "the recipe calls for two cups of flour",
                "warming reduces the number of cold days each year"]
    scores = rr.score("how many fewer cold days under warming", passages)
    assert scores[0] > scores[1] and scores[2] > scores[1]   # topical > off-topic


def test_rerank_reorders_and_preserves_provenance():
    rr = LexicalReranker()
    hits = [
        SearchHit("a", "section_text", 0.9,
                  {"canonical_text": "unrelated administrative boilerplate text",
                   "raw_text": "unrelated"}),
        SearchHit("b", "claim", 0.5,
                  {"canonical_text": "cold spell frequency drops under global warming",
                   "raw_text": "cold spell frequency drops"}),
    ]
    out = rerank("cold spell frequency warming", hits, rr, alpha=1.0)
    assert out[0].chunk_id == "b"                     # relevant one wins despite lower retr
    assert out[0].payload["retrieval_score"] == 0.5   # original score preserved
    assert "rerank_score" in out[0].payload


def test_rerank_alpha_blend_respects_retrieval():
    rr = LexicalReranker()
    hits = [SearchHit("a", "x", 1.0, {"canonical_text": "off topic", "raw_text": ""}),
            SearchHit("b", "x", 0.0, {"canonical_text": "cold days warming", "raw_text": ""})]
    # alpha=0 => pure retrieval order (a first)
    assert rerank("cold days", hits[:], rr, alpha=0.0)[0].chunk_id == "a"
    # alpha=1 => pure rerank order (b first)
    assert rerank("cold days", hits[:], rr, alpha=1.0)[0].chunk_id == "b"


# ---- graph expansion (uses a small synthetic ingest) -------------------------

@pytest.fixture()
def index(tmp_path):
    doc = {"doc_id": "d1", "source": "d1.pdf", "stats": {"doc_title": "T"},
           "chunks": [
        {"chunk_id": "sec1", "doc_id": "d1", "chunk_type": "section_text",
         "text": "Method: cold days use the 10th percentile threshold.",
         "embed_text": "[S]\nMethod: cold days use the 10th percentile threshold.",
         "section_path": ["2 Methods"], "role": "methods", "token_estimate": 12},
        {"chunk_id": "cl1", "doc_id": "d1", "chunk_type": "claim",
         "text": "Cold days fall 60-80% under warming.",
         "embed_text": "[C]\nCold days fall 60-80% under warming.",
         "token_estimate": 10, "parent_chunk_id": "sec1",
         "linked_figures": ["Figure 1"], "metadata": {"claim_type": "result"}},
        {"chunk_id": "fig1", "doc_id": "d1", "chunk_type": "figure",
         "text": "Figure 1. Cold day maps.", "embed_text": "[F1]\nCold day maps.",
         "token_estimate": 8, "metadata": {"label": "Figure 1"}},
        {"chunk_id": "fig1a", "doc_id": "d1", "chunk_type": "figure_panel",
         "text": "Figure 1 panel A: counts at 0.5C.",
         "embed_text": "[F1a]\ncounts at 0.5C.", "token_estimate": 8,
         "parent_chunk_id": "fig1", "metadata": {"label": "Figure 1", "panel": "A"}},
    ]}
    p = tmp_path / "chunks.json"
    p.write_text(json.dumps([doc]), encoding="utf-8")
    cfg = IngestConfig(embedder="hash", db_location=str(tmp_path / "qdrant"),
                       graph_outdir=str(tmp_path))
    ingest(p, cfg)
    return tmp_path


def test_expand_pulls_support_and_structure(index):
    r = load_retriever(index, embedder="hash", reranker="lexical")
    # claim hit should pull its supporting section (CLAIM_SUPPORTED_BY_CHUNK)
    from docparse.vectorization.vectorstore import SearchHit
    claim_hit = SearchHit("cl1", "claim", 1.0,
                          {"chunk_id": "cl1", "raw_text": "Cold days fall"})
    bundles = expand_hits([claim_hit], r.kg, r.chunk_lookup, EdgePolicy(max_hops=1))
    rels = {n["rel"] for n in bundles[0].neighbours}
    assert "CLAIM_SUPPORTED_BY_CHUNK" in rels
    ids = {n["chunk_id"] for n in bundles[0].neighbours}
    assert "sec1" in ids


def test_expand_figure_pulls_panels_both_directions(index):
    r = load_retriever(index, embedder="hash", reranker="lexical")
    fig_hit = SearchHit("fig1", "figure", 1.0,
                        {"chunk_id": "fig1", "raw_text": "Figure 1"})
    bundles = expand_hits([fig_hit], r.kg, r.chunk_lookup, EdgePolicy(max_hops=1))
    ids = {n["chunk_id"] for n in bundles[0].neighbours}
    assert "fig1a" in ids                              # figure -> its panel


def test_expand_respects_budget(index):
    r = load_retriever(index, embedder="hash", reranker="lexical")
    hit = SearchHit("sec1", "section_text", 1.0,
                    {"chunk_id": "sec1", "raw_text": "Method"})
    pol = EdgePolicy(max_hops=2, max_neighbours_per_hit=1)
    bundles = expand_hits([hit], r.kg, r.chunk_lookup, pol)
    assert len(bundles[0].neighbours) <= 1


# ---- retriever facade + persistence -----------------------------------------

def test_retriever_full_pipeline(index):
    r = load_retriever(index, embedder="hash", reranker="lexical")
    bundles = r.retrieve("cold days under warming",
                         RetrieveConfig(limit=3, rerank=True, expand=True))
    assert bundles
    assert all(hasattr(b, "hit") for b in bundles)
    # at least one bundle carries graph-grounded neighbours
    assert any(b.neighbours for b in bundles)


def test_retriever_survives_reload(index):
    # first retriever
    r1 = load_retriever(index, embedder="hash", reranker="lexical")
    n1 = r1.kg.g.number_of_nodes()
    del r1
    # fresh load from the same persisted index (simulates process restart)
    r2 = load_retriever(index, embedder="hash", reranker="lexical")
    assert r2.kg.g.number_of_nodes() == n1            # graph restored from JSON
    assert r2.chunk_lookup                             # chunk cache restored
    assert r2.retrieve("cold days", RetrieveConfig(limit=2))


def test_retriever_rerank_off_keeps_retrieval_order(index):
    r = load_retriever(index, embedder="hash", reranker="off")
    assert r.reranker is None
    bundles = r.retrieve("cold days", RetrieveConfig(rerank=False, expand=False))
    assert bundles                                     # still returns results


def test_doc_id_filter(index):
    r = load_retriever(index, embedder="hash", reranker="lexical")
    hit_real = r.retrieve("cold days", RetrieveConfig(doc_ids=["d1"]))
    hit_none = r.retrieve("cold days", RetrieveConfig(doc_ids=["nonexistent"]))
    assert hit_real and not hit_none                   # filter actually restricts
