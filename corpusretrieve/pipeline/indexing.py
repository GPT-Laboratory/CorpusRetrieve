from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from corpusretrieve.models import JobSpec
from corpusretrieve.pipeline.artifacts import write_json
from corpusretrieve.pipeline.corpus import CorpusRegistry, write_corpus_chunks
from docparse.vectorization.canonical import normalize_chunk, payload_for
from docparse.vectorization.embedding import BM25SparseEncoder, resolve_embedder
from docparse.vectorization.ids import COLLECTION_OF, point_uuid

INDEX_FORMAT = "corpusretrieve-portable-index"
INDEX_SCHEMA_VERSION = "1.0.0"
EMBEDDING_VERSION = "v1"
DEFAULT_BATCH_SIZE = 256


def build_index(
    registry: CorpusRegistry,
    output_dir: Path,
    spec: JobSpec,
    batch_size: int = DEFAULT_BATCH_SIZE,
    on_progress=None,
) -> dict[str, Any]:
    """Write vectors.jsonl, the sparse encoder, and index metadata by streaming.

    Two passes are made over the corpus, both reading documents from disk one
    at a time. The first fits the BM25 statistics, which are corpus-global and
    cannot be computed per document. The second embeds and writes records in
    fixed-size batches, so peak memory is bounded by ``batch_size`` rather than
    by the number of chunks in the corpus.
    """
    embedder = resolve_embedder(spec.embedder)

    sparse = BM25SparseEncoder().fit(_canonical_texts(registry))

    index_dir = output_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    records = 0
    with (index_dir / "vectors.jsonl").open("w", encoding="utf-8") as handle:
        for batch in _batched(_normalized_chunks(registry), batch_size):
            vectors = embedder.encode([chunk["canonical_text"] for chunk, _ in batch])
            for (chunk, document), vector in zip(batch, vectors, strict=True):
                handle.write(
                    json.dumps(
                        _record(chunk, document, vector, sparse, embedder.name, spec),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            records += len(batch)
            if on_progress is not None:
                on_progress(records)

    sparse.save(index_dir / "sparse_encoder.json")
    write_corpus_chunks(registry, output_dir / "corpus" / "chunks.json")

    metadata = {
        "format": INDEX_FORMAT,
        "schema_version": INDEX_SCHEMA_VERSION,
        "model": embedder.name,
        "dimension": embedder.dim,
        "metric": "cosine",
        "dense_field": "text_dense",
        "sparse_field": "text_sparse",
        "records": records,
        "documents": len(registry),
        "batch_size": batch_size,
    }
    write_json(index_dir / "metadata.json", metadata)
    return metadata


def _canonical_texts(registry: CorpusRegistry) -> Iterator[str]:
    for chunk, _ in registry.iter_chunks():
        yield normalize_chunk(chunk)["canonical_text"]


def _normalized_chunks(
    registry: CorpusRegistry,
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    for document in registry.iter_documents():
        for chunk in document.get("chunks", []):
            yield normalize_chunk(chunk), document


def _batched(source: Iterator, size: int) -> Iterator[list]:
    batch: list = []
    for item in source:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _record(
    chunk: dict[str, Any],
    document: dict[str, Any],
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
            chunk, document, model, EMBEDDING_VERSION, spec.corpus_name
        ),
    }
