"""Vector store layer: Qdrant (local file / :memory: / server) with the
report's architecture — separate collections per element type-group, named
dense + sparse vectors, payload filters, hybrid RRF search, idempotent
UUID upserts with the composite ID preserved in payload.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Sequence

from qdrant_client import QdrantClient
from qdrant_client import models as qm

# type-group -> collection (report: claims/sections/tables/figures/references
# behave too differently to share one undifferentiated pool)
COLLECTION_OF = {
    "claim": "claims",
    "section_text": "sections",
    "summary": "summaries",
    "table_summary": "tables",
    "table_row_group": "tables",
    "table_fact": "tables",
    "figure": "figures",
    "figure_panel": "figures",
    "formula": "formulas",
    "reference": "references",
}
PAYLOAD_INDEXES = [  # created before ingest (filter-aware HNSW on server qdrant)
    ("tenant_id", "keyword"), ("doc_id", "keyword"), ("chunk_type", "keyword"),
    ("role", "keyword"), ("figure_label", "keyword"), ("table_label", "keyword"),
    ("citation_ids", "keyword"), ("retrieval_default", "keyword"),
    ("embedding_version", "keyword"), ("schema_version", "keyword"),
]


def point_uuid(composite: str) -> str:
    """Qdrant point IDs must be int/UUID; the report's composite ID pattern
    {tenant}:{doc}:{type}:{chunk}:{model}:{version} lives in payload, and the
    point ID is its stable UUID5 — same composite, same point => idempotent
    upserts across re-runs."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, composite))


@dataclass
class SearchHit:
    chunk_id: str
    chunk_type: str
    score: float
    payload: dict[str, Any]


class VectorStore:
    def __init__(self, location: str = ":memory:", dense_dim: int = 512):
        self.client = (QdrantClient(path=location) if location not in (":memory:",)
                       else QdrantClient(":memory:"))
        self.dense_dim = dense_dim

    # ---- schema -------------------------------------------------------------
    def ensure_collections(self) -> list[str]:
        made = []
        for coll in sorted(set(COLLECTION_OF.values())):
            if not self.client.collection_exists(coll):
                self.client.create_collection(
                    coll,
                    vectors_config={"text_dense": qm.VectorParams(
                        size=self.dense_dim, distance=qm.Distance.COSINE)},
                    sparse_vectors_config={"text_sparse": qm.SparseVectorParams()})
                for fieldname, schema in PAYLOAD_INDEXES:
                    try:
                        self.client.create_payload_index(coll, fieldname, field_schema=schema)
                    except Exception:
                        pass  # local mode: indexes are a no-op, server mode: real
                made.append(coll)
        return made

    # ---- ingest ---------------------------------------------------------------
    def upsert_chunks(self, records: Sequence[dict]) -> dict[str, int]:
        """records: {collection, composite_id, dense, sparse:(idx,val), payload}"""
        by_coll: dict[str, list[qm.PointStruct]] = {}
        for r in records:
            idx, val = r["sparse"]
            by_coll.setdefault(r["collection"], []).append(qm.PointStruct(
                id=point_uuid(r["composite_id"]),
                vector={"text_dense": r["dense"],
                        "text_sparse": qm.SparseVector(indices=idx, values=val)},
                payload={**r["payload"], "composite_id": r["composite_id"]}))
        counts = {}
        for coll, pts in by_coll.items():
            for i in range(0, len(pts), 128):           # batched upserts
                self.client.upsert(coll, points=pts[i:i + 128])
            counts[coll] = len(pts)
        return counts

    # ---- retrieval --------------------------------------------------------------
    def hybrid_search(self, collection: str, dense: list[float],
                      sparse: tuple[list[int], list[float]], limit: int = 10,
                      flt: qm.Filter | None = None,
                      include_references: bool = False) -> list[SearchHit]:
        must = list(flt.must) if flt and flt.must else []
        if not include_references:
            must.append(qm.FieldCondition(key="retrieval_default",
                                          match=qm.MatchValue(value="include")))
        qfilter = qm.Filter(must=must) if must else None
        idx, val = sparse
        prefetch = [qm.Prefetch(query=dense, using="text_dense",
                                limit=limit * 4, filter=qfilter)]
        if idx:
            prefetch.append(qm.Prefetch(
                query=qm.SparseVector(indices=idx, values=val),
                using="text_sparse", limit=limit * 4, filter=qfilter))
        res = self.client.query_points(
            collection, prefetch=prefetch,
            query=qm.FusionQuery(fusion=qm.Fusion.RRF), limit=limit)
        return [SearchHit(chunk_id=p.payload.get("chunk_id", ""),
                          chunk_type=p.payload.get("chunk_type", ""),
                          score=p.score, payload=p.payload) for p in res.points]

    def search_all(self, dense, sparse, limit_per: int = 5,
                   collections: Sequence[str] | None = None,
                   **kw) -> list[SearchHit]:
        """Fan out over type collections and merge by score — the simple
        cross-type router; callers with a query classifier can target instead."""
        colls = collections or sorted(set(COLLECTION_OF.values()) - {"references"})
        hits: list[SearchHit] = []
        for c in colls:
            try:
                hits += self.hybrid_search(c, dense, sparse, limit=limit_per, **kw)
            except Exception:
                continue
        return sorted(hits, key=lambda h: -h.score)

    def counts(self) -> dict[str, int]:
        return {c: self.client.count(c).count
                for c in sorted(set(COLLECTION_OF.values()))
                if self.client.collection_exists(c)}
