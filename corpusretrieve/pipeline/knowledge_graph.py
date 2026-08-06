from __future__ import annotations

from pathlib import Path
from typing import Any

from corpusretrieve.models import JobSpec
from corpusretrieve.pipeline.indexing import EMBEDDING_VERSION
from docparse.vectorization.canonical import normalize_chunk
from docparse.vectorization.graph import KnowledgeGraph

GRAPH_DIRECTORY = "knowledge_graph"
GRAPH_ARTIFACTS = {
    "json": f"{GRAPH_DIRECTORY}/knowledge_graph.json",
    "graphml": f"{GRAPH_DIRECTORY}/knowledge_graph.graphml",
    "cypher": f"{GRAPH_DIRECTORY}/knowledge_graph.cypher",
}


def build_graph(
    documents: list[dict[str, Any]],
    output_dir: Path,
    spec: JobSpec,
    embedder_name: str,
) -> dict[str, Any]:
    """Project every document's chunks into one corpus-level graph."""
    graph = KnowledgeGraph()
    summary: dict[str, Any] = {}
    for document in documents:
        projected = dict(document)
        projected["chunks"] = [normalize_chunk(chunk) for chunk in document["chunks"]]
        summary = graph.build_from_chunks(
            projected, spec.corpus_name, embedder_name, EMBEDDING_VERSION
        )
    graph.export(output_dir / GRAPH_DIRECTORY)
    return {**summary, "artifacts": dict(GRAPH_ARTIFACTS)}
