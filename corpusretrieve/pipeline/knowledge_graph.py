from __future__ import annotations

from pathlib import Path
from typing import Any

from corpusretrieve.models import JobSpec
from corpusretrieve.pipeline.corpus import CorpusRegistry
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
    registry: CorpusRegistry,
    output_dir: Path,
    spec: JobSpec,
    embedder_name: str,
    on_progress=None,
) -> dict[str, Any]:
    """Project the corpus into one graph, reading documents one at a time.

    Each document is loaded, projected, and released before the next is read.
    The accumulated graph is the only structure that grows with corpus size,
    and it holds identifiers and short labels rather than chunk text or
    vectors.
    """
    graph = KnowledgeGraph()
    summary: dict[str, Any] = {}
    for position, document in enumerate(registry.iter_documents(), start=1):
        projected = dict(document)
        projected["chunks"] = [
            normalize_chunk(chunk) for chunk in document.get("chunks", [])
        ]
        summary = graph.build_from_chunks(
            projected, spec.corpus_name, embedder_name, EMBEDDING_VERSION
        )
        if on_progress is not None:
            on_progress(position)

    graph.export(output_dir / GRAPH_DIRECTORY)
    return {**summary, "artifacts": dict(GRAPH_ARTIFACTS)}
