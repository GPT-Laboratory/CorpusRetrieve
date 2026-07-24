"""Stage 6 — the retrieval facade.

This is the service's core abstraction: given a query, return grounded
evidence. It composes the three retrieval stages behind one method so callers
(the HTTP layer, or an external Stage-7 answer service) never touch qdrant,
networkx, or embedding internals directly:

    hybrid_search  (recall)  ->  rerank  (precision)  ->  graph expand (grounding)

Everything is field-agnostic. The retriever knows nothing about the documents'
subject; it operates on chunk types, edge relations, and metadata that the
ingest stage attached. A finance corpus and a genomics corpus use the identical
code path.

State (vector store, knowledge graph, encoders, rerankers) is loaded once and
reused across queries — the expensive objects are created at startup, not per
request.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from docparse.vectorization.embedding import (BM25SparseEncoder, Embedder,
                                              resolve_embedder)
from docparse.vectorization.expand import EdgePolicy, EvidenceBundle, expand_hits
from docparse.vectorization.graph import KnowledgeGraph
from docparse.vectorization.rerank import Reranker, rerank, resolve_reranker
from docparse.vectorization.vectorstore import (COLLECTION_OF, SearchHit,
                                                VectorStore)


@dataclass
class RetrieveConfig:
    limit: int = 5                       # final results returned
    candidate_multiplier: int = 4        # first-stage pulls limit*this per collection
    collections: list[str] | None = None # None = all except references
    rerank: bool = True
    rerank_alpha: float = 1.0            # 1=pure rerank, 0=retrieval order, blend between
    expand: bool = True                  # attach graph neighbours
    include_references: bool = False
    tenant: str | None = None            # filter to a tenant when multi-tenant
    doc_ids: list[str] | None = None     # restrict to specific documents
    edge_policy: EdgePolicy = field(default_factory=EdgePolicy)


class Retriever:
    """Holds loaded state; one instance serves many queries."""

    def __init__(self, store: VectorStore, kg: KnowledgeGraph,
                 embedder: Embedder, sparse: BM25SparseEncoder,
                 reranker: Reranker | None = None,
                 chunk_lookup: dict[str, dict] | None = None):
        self.store = store
        self.kg = kg
        self.emb = embedder
        self.sparse = sparse
        self.reranker = reranker
        self.chunk_lookup = chunk_lookup or {}

    # ---- the one method the service exposes --------------------------------
    def retrieve(self, query: str, cfg: RetrieveConfig | None = None
                 ) -> list[EvidenceBundle]:
        cfg = cfg or RetrieveConfig()
        dense = self.emb.encode([query])[0]
        sparse = self.sparse.encode(query)

        flt = self._build_filter(cfg)
        # first stage: recall-oriented, over-fetch for the reranker to sort
        per = cfg.limit * cfg.candidate_multiplier
        hits = self.store.search_all(
            dense, sparse, limit_per=per, collections=cfg.collections,
            include_references=cfg.include_references, flt=flt)

        # second stage: precision-oriented reordering
        if cfg.rerank and self.reranker is not None and hits:
            hits = rerank(query, hits, self.reranker,
                          top_k=cfg.limit, alpha=cfg.rerank_alpha)
        else:
            hits = hits[:cfg.limit]

        # third stage: grounding via graph neighbours
        if cfg.expand and self.kg.g.number_of_nodes():
            return expand_hits(hits, self.kg, self.chunk_lookup, cfg.edge_policy)
        return [EvidenceBundle(hit=h) for h in hits]

    def _build_filter(self, cfg: RetrieveConfig):
        from qdrant_client import models as qm
        must = []
        if cfg.tenant:
            must.append(qm.FieldCondition(key="tenant_id",
                                          match=qm.MatchValue(value=cfg.tenant)))
        if cfg.doc_ids:
            must.append(qm.FieldCondition(key="doc_id",
                                          match=qm.MatchAny(any=cfg.doc_ids)))
        return qm.Filter(must=must) if must else None

    # ---- introspection for /healthz and evaluation -------------------------
    def stats(self) -> dict:
        return {"collections": self.store.counts(),
                "graph_nodes": self.kg.g.number_of_nodes(),
                "graph_edges": self.kg.g.number_of_edges(),
                "embedder": self.emb.name,
                "reranker": self.reranker.name if self.reranker else None,
                "chunks_cached": len(self.chunk_lookup)}


# --------------------------------------------------------------------- loading

def load_retriever(index_dir: str | Path, embedder: str = "auto",
                   reranker: str = "auto",
                   qdrant_url: str | None = None) -> Retriever:
    """Reconstruct a Retriever from a persisted index directory produced by
    ingest (qdrant db + graph json + sparse encoder + chunk cache).

    This is what a long-running service calls at startup, and what lets Stage 6
    survive process restarts — the graph and sparse encoder are rebuilt from
    disk, not recomputed from source documents.
    """
    import json

    index_dir = Path(index_dir)
    emb = resolve_embedder(embedder)
    rr = resolve_reranker(reranker) if reranker != "off" else None

    store = VectorStore(qdrant_url or str(index_dir / "qdrant"), dense_dim=emb.dim)
    store.ensure_collections()

    sparse = BM25SparseEncoder.load(index_dir / "sparse_encoder.json")

    kg = KnowledgeGraph()
    graph_json = index_dir / "knowledge_graph.json"
    if graph_json.exists():
        _load_graph_json(kg, graph_json)

    chunk_lookup: dict[str, dict] = {}
    cache = index_dir / "chunk_cache.json"
    if cache.exists():
        for c in json.loads(cache.read_text(encoding="utf-8")):
            chunk_lookup[c["chunk_id"]] = c

    return Retriever(store, kg, emb, sparse, rr, chunk_lookup)


def _load_graph_json(kg: KnowledgeGraph, path) -> None:
    import json
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for n in data["nodes"]:
        attrs = {k: v for k, v in n.items() if k != "id"}
        kg.g.add_node(n["id"], **attrs)
    for e in data["edges"]:
        attrs = {k: v for k, v in e.items()
                 if k not in ("source", "target", "rel")}
        kg.g.add_edge(e["source"], e["target"], key=e["rel"], rel=e["rel"], **attrs)
