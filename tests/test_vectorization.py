"""Stage 5 tests: canonicalization, ingest, hybrid retrieval, KG projection,
idempotency, exports — all offline via the hashing embedder + local qdrant."""
import json

import pytest

from docparse.vectorization import (HashingEmbedder, IngestConfig, hybrid_query, ingest)
from docparse.vectorization.canonical import canonical_text, normalize_quantity
from docparse.vectorization.vectorstore import point_uuid


@pytest.fixture()
def chunks_file(tmp_path):
    doc = {"doc_id": "paper1", "source": "paper1.pdf",
           "stats": {"doc_title": "Cold Extremes Paper"},
           "chunks": [
        {"chunk_id": "sec1", "doc_id": "paper1", "chunk_type": "section_text",
         "text": "Cold days are defined via the 10th percentile threshold of daily "
                 "mean temperature in the baseline period.",
         "embed_text": "[Cold Extremes Paper > 2.3 | Cold Extreme Indices]\nCold days "
                       "are defined via the 10th percentile threshold of daily mean "
                       "temperature in the baseline period.",
         "section_path": ["2.3 | Cold Extreme Indices"], "role": "methods",
         "pages": [5], "token_estimate": 30,
         "citations": [{"key": "ipcc2021"}]},
        {"chunk_id": "cl1", "doc_id": "paper1", "chunk_type": "claim",
         "text": "Cold days decrease by 60%-80% from 0.5°C to 2.0°C warming.",
         "embed_text": "[claim/result_claim]\nCold days decrease by 60%-80% from "
                       "0.5°C to 2.0°C warming.",
         "role": "results", "pages": [1], "token_estimate": 18,
         "parent_chunk_id": "sec1", "linked_figures": ["Figure 4"],
         "quantities": [{"value": "60%–80%", "type": "percentage_change"},
                        {"value": "2.0°C", "type": "warming_level"}],
         "metadata": {"claim_type": "result_claim"}},
        {"chunk_id": "tab1", "doc_id": "paper1", "chunk_type": "table_summary",
         "text": "Table 1: models and warming years.\nColumns: Model, 2.0°C. Rows: 2.",
         "embed_text": "[Table 1]\nTable 1: models and warming years.",
         "pages": [4], "token_estimate": 15, "metadata": {"label": "Table 1"}},
        {"chunk_id": "tr1", "doc_id": "paper1", "chunk_type": "table_row_group",
         "text": "Table 1\nModel: CanESM5 — 2.0°C: 2026", "pages": [4],
         "embed_text": "[Table 1 rows 1-1]\nModel: CanESM5 — 2.0°C: 2026",
         "token_estimate": 12, "parent_chunk_id": "tab1",
         "metadata": {"label": "Table 1",
                      "facts": [{"row_key": "CanESM5", "column": "2.0°C",
                                 "value": "2026"}]}},
        {"chunk_id": "fig4", "doc_id": "paper1", "chunk_type": "figure",
         "text": "Figure 4. Caption: Cold day counts at warming levels.",
         "embed_text": "[Figure 4]\nCold day counts at warming levels.",
         "pages": [9], "token_estimate": 12, "metadata": {"label": "Figure 4"}},
        {"chunk_id": "fig4a", "doc_id": "paper1", "chunk_type": "figure_panel",
         "text": "Figure 4 panel A: counts at 0.5°C warming over Scandinavia.",
         "embed_text": "[Figure 4 panel A]\ncounts at 0.5°C warming.",
         "pages": [9], "token_estimate": 12, "parent_chunk_id": "fig4",
         "metadata": {"label": "Figure 4", "panel": "A"}},
        {"chunk_id": "eq2", "doc_id": "paper1", "chunk_type": "formula",
         "text": "Equation (2): EX = sum max(Tp10 - Tday, 0)",
         "embed_text": "[Equation (2)]\nEX = sum max(Tp10 - Tday, 0)",
         "pages": [5], "token_estimate": 14,
         "metadata": {"equation_number": 2,
                      "expression": "EX = sum max(Tp10 - Tday, 0)"}},
        {"chunk_id": "ref1", "doc_id": "paper1", "chunk_type": "reference",
         "text": "IPCC (2021). Climate Change 2021.", "role": "references",
         "embed_text": "IPCC (2021). Climate Change 2021.", "token_estimate": 9,
         "metadata": {"citation_key": "ipcc2021", "retrieval_default": "exclude"}},
        {"chunk_id": "sum1", "doc_id": "paper1", "chunk_type": "summary",
         "text": "Cold extremes weaken strongly with global warming.", "role": "abstract",
         "embed_text": "[abstract]\nCold extremes weaken strongly with warming.",
         "token_estimate": 10},
    ]}
    p = tmp_path / "chunks.json"
    p.write_text(json.dumps([doc]), encoding="utf-8")
    return p


def _run(chunks_file, tmp_path):
    cfg = IngestConfig(embedder="hash", graph_outdir=str(tmp_path / "kg"))
    return ingest(chunks_file, cfg)


def test_ingest_counts_and_validation(chunks_file, tmp_path):
    res, store, kg, sparse = _run(chunks_file, tmp_path)
    assert res.collections == {"sections": 1, "claims": 1, "tables": 2,
                               "figures": 2, "formulas": 1, "references": 1,
                               "summaries": 1}
    v = res.validation
    assert v["self_retrieval_top1"].split("/")[0] == v["self_retrieval_top1"].split("/")[1]
    assert v["references_hidden_by_default"] is True
    assert v["graph_vector_links_complete"] is True
    assert v["claims_with_support"] == "1/1"


def test_hybrid_query_finds_claim(chunks_file, tmp_path):
    res, store, kg, sparse = _run(chunks_file, tmp_path)
    emb = HashingEmbedder()
    hits = hybrid_query(store, emb, sparse,
                        "how much do cold days decrease at 2.0°C warming", limit=3)
    assert hits and any(h.chunk_id == "cl1" for h in hits[:3])
    # references never surface in default cross-type search
    assert not any(h.chunk_type == "reference" for h in hits)


def test_kg_structure(chunks_file, tmp_path):
    res, store, kg, sparse = _run(chunks_file, tmp_path)
    nt = res.graph_stats["node_types"]
    et = res.graph_stats["edge_types"]
    assert nt["TableKV"] == 1 and et["ROW_GROUP_HAS_KV"] == 1   # cell facts -> graph
    assert et["FIGURE_HAS_PANEL"] == 1
    assert et["CLAIM_SUPPORTED_BY_CHUNK"] == 1
    assert et["CLAIM_SUPPORTED_BY_FIGURE"] == 1                 # via linked_figures
    assert et["QUANTITY_HAS_UNIT"] == 2                         # percent + degC
    assert et["FORMULA_DEFINES_SYMBOL"] >= 2                    # EX, Tp10, Tday
    assert nt["Unit"] == 2
    # citation resolved: chunk cites ipcc2021 which is also the reference work
    assert et["CHUNK_CITES_WORK"] >= 1


def test_exports_written_and_cypher_wellformed(chunks_file, tmp_path):
    res, *_ = _run(chunks_file, tmp_path)
    cy = open(res.exports["cypher"], encoding="utf-8").read()
    assert cy.count("MERGE (n:") == res.graph_stats["nodes"]
    assert "CLAIM_SUPPORTED_BY_CHUNK" in cy
    assert (tmp_path / "kg" / "knowledge_graph.graphml").exists()
    assert (tmp_path / "kg" / "sparse_encoder.json").exists()


def test_idempotent_reingest(chunks_file, tmp_path):
    cfg = IngestConfig(embedder="hash", db_location=str(tmp_path / "db"))
    res1, store1, *_ = ingest(chunks_file, cfg)
    del store1                                     # release local qdrant lock
    res2, store2, *_ = ingest(chunks_file, cfg)
    assert store2.counts() == {"claims": 1, "figures": 2, "formulas": 1,
                               "references": 1, "sections": 1, "summaries": 1,
                               "tables": 2}        # upserts, no duplicates


def test_quantity_normalization():
    assert normalize_quantity({"value": "60%–80%"}) == {
        "value": "60%–80%", "unit": "percent",
        "value_norm_min": 60.0, "value_norm_max": 80.0}
    assert normalize_quantity({"value": "160°C days"})["unit"] == "degC_days"
    assert normalize_quantity({"value": "2.0°C"})["value_norm"] == 2.0
    assert normalize_quantity({"value": "once in 20 years"})["unit"] == "years"


def test_canonical_text_tagging():
    t = canonical_text({"chunk_type": "claim", "role": "results",
                        "embed_text": "Cold  days\ndecrease."})
    assert t.startswith("[CLAIM role=results]") and "Cold days decrease." in t


def test_point_uuid_stable():
    a = point_uuid("t:d:claim:c1:hash:v1")
    assert a == point_uuid("t:d:claim:c1:hash:v1") != point_uuid("t:d:claim:c1:hash:v2")
