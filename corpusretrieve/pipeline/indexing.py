from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from corpusretrieve.models import JobSpec
from corpusretrieve.pipeline.artifacts import write_json
from corpusretrieve.pipeline.corpus import CorpusRegistry, write_corpus_chunks
from docparse.vectorization.canonical import (
    PAYLOAD_SCHEMA_VERSION,
    normalize_chunk,
    payload_for,
)
from docparse.vectorization.embedding import BM25SparseEncoder, resolve_embedder
from docparse.vectorization.ids import (
    COLLECTION_OF,
    POINT_ID_SCHEME,
    point_composite,
    point_uuid,
)

INDEX_FORMAT = "corpusretrieve-portable-index"
INDEX_SCHEMA_VERSION = "2.0.0"
EMBEDDING_VERSION = "v1"
DEFAULT_BATCH_SIZE = 256
DENSE_FILE = "dense.npy"


def build_index(
    registry: CorpusRegistry,
    output_dir: Path,
    spec: JobSpec,
    batch_size: int = DEFAULT_BATCH_SIZE,
    on_progress=None,
    inline_vectors: bool = True,
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

    # dense vectors also go to a float32 .npy matrix, row i = line i of
    # vectors.jsonl, so consumers need not parse millions of JSON floats
    total = registry.total_chunks
    dense_path = index_dir / DENSE_FILE
    matrix = np.lib.format.open_memmap(
        dense_path, mode="w+", dtype=np.float32, shape=(total, embedder.dim)
    ) if total else None

    records = 0
    languages: dict[str, int] = {}
    with (index_dir / "vectors.jsonl").open("w", encoding="utf-8") as handle:
        for batch in _batched(_normalized_chunks(registry), batch_size):
            vectors = embedder.encode([chunk["canonical_text"] for chunk, _ in batch])
            for offset, ((chunk, document), vector) in enumerate(
                zip(batch, vectors, strict=True)
            ):
                record = _record(chunk, document, vector, sparse, embedder.name, spec)
                record["row"] = records + offset
                lang = record["payload"].get("language") or "unknown"
                languages[lang] = languages.get(lang, 0) + 1
                if not inline_vectors:
                    record.pop("vector")
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                if matrix is not None:
                    matrix[records + offset] = np.asarray(vector, dtype=np.float32)
            records += len(batch)
            if on_progress is not None:
                on_progress(records)
    if matrix is not None:
        matrix.flush()
        del matrix
    else:
        np.save(dense_path, np.zeros((0, embedder.dim), dtype=np.float32))

    sparse.save(index_dir / "sparse_encoder.json")
    write_corpus_chunks(registry, output_dir / "corpus" / "chunks.json")

    metadata = {
        "format": INDEX_FORMAT,
        "schema_version": INDEX_SCHEMA_VERSION,
        "model": embedder.name,
        # what a consumer needs to embed queries with the same model
        "model_id": getattr(embedder, "model_id", embedder.name),
        "query_prefix": getattr(embedder, "query_prefix", ""),
        "passage_prefix": getattr(embedder, "passage_prefix", ""),
        "dimension": embedder.dim,
        "metric": "cosine",
        "dense_field": "text_dense",
        "sparse_field": "text_sparse",
        "dense_file": f"index/{DENSE_FILE}",
        "dense_dtype": "float32",
        "inline_vectors": inline_vectors,
        "records": records,
        "documents": len(registry),
        "batch_size": batch_size,
        "payload_schema_version": PAYLOAD_SCHEMA_VERSION,
        "point_id_scheme": POINT_ID_SCHEME,
        "sparse_tokenizer": sparse.tokenizer,
        "sparse_corpus_fingerprint": sparse.corpus_fingerprint,
        "sparse_query_encoding": "BM25SparseEncoder.encode_query",
        "embedder_requested": spec.embedder,
        "embedder_fallback": getattr(embedder, "fallback_from", None),
        # units per language: with a lexical embedder, two or more languages
        # mean single-language queries miss evidence (see consumer.search)
        "languages": dict(sorted(languages.items())),
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
    composite = point_composite(
        chunk["doc_id"], chunk["chunk_type"], chunk["chunk_id"], model, EMBEDDING_VERSION
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
