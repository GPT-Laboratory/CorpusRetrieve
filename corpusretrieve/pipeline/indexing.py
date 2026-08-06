from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from corpusretrieve.models import JobSpec
from corpusretrieve.pipeline.artifacts import write_json
from docparse.vectorization.canonical import normalize_chunk, payload_for
from docparse.vectorization.embedding import BM25SparseEncoder, resolve_embedder
from docparse.vectorization.ids import COLLECTION_OF, point_uuid

INDEX_FORMAT = "corpusretrieve-portable-index"
INDEX_SCHEMA_VERSION = "1.0.0"
EMBEDDING_VERSION = "v1"


def build_index(
    documents: list[dict[str, Any]], output_dir: Path, spec: JobSpec
) -> dict[str, Any]:
    """Write vectors.jsonl, the fitted sparse encoder, and index metadata."""
    embedder = resolve_embedder(spec.embedder)
    normalized = [
        {**normalize_chunk(chunk), "_document": document}
        for document in documents
        for chunk in document["chunks"]
    ]

    texts = [chunk["canonical_text"] for chunk in normalized]
    sparse = BM25SparseEncoder().fit(texts)
    vectors = embedder.encode(texts)

    index_dir = output_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    with (index_dir / "vectors.jsonl").open("w", encoding="utf-8") as handle:
        for chunk, vector in zip(normalized, vectors, strict=True):
            handle.write(
                json.dumps(
                    _record(chunk, vector, sparse, embedder.name, spec),
                    ensure_ascii=False,
                )
                + "\n"
            )
    sparse.save(index_dir / "sparse_encoder.json")

    metadata = {
        "format": INDEX_FORMAT,
        "schema_version": INDEX_SCHEMA_VERSION,
        "model": embedder.name,
        "dimension": embedder.dim,
        "metric": "cosine",
        "dense_field": "text_dense",
        "sparse_field": "text_sparse",
        "records": len(normalized),
        "documents": len(documents),
    }
    write_json(index_dir / "metadata.json", metadata)
    write_json(output_dir / "corpus" / "chunks.json", documents)
    return metadata


def _record(
    chunk: dict[str, Any],
    vector: list[float],
    sparse: BM25SparseEncoder,
    model: str,
    spec: JobSpec,
) -> dict[str, Any]:
    composite = (
        f"{spec.corpus_name}:{chunk['doc_id']}:{chunk['chunk_type']}:"
        f"{chunk['chunk_id']}:{model}:{EMBEDDING_VERSION}"
    )
    indices, values = sparse.encode(chunk["canonical_text"])
    return {
        "id": point_uuid(composite),
        "composite_id": composite,
        "collection": COLLECTION_OF.get(chunk["chunk_type"], "sections"),
        "vector": vector,
        "sparse": {"indices": indices, "values": values},
        "payload": payload_for(
            chunk,
            chunk["_document"],
            model,
            EMBEDDING_VERSION,
            spec.corpus_name,
        ),
    }
