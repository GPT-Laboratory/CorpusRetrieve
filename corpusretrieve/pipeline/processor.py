from __future__ import annotations

from pathlib import Path
from typing import Any

from corpusretrieve.models import JobSpec
from corpusretrieve.pipeline.documents import DocumentOptions, DocumentProcessor
from corpusretrieve.pipeline.indexing import build_index
from corpusretrieve.pipeline.knowledge_graph import build_graph
from corpusretrieve.pipeline.packaging import package_outputs
from corpusretrieve.pipeline.progress import ProgressSink, StageTracker

DOCUMENT_STAGES = ("extract", "enrich", "chunk")
DOCUMENT_PROGRESS_CEILING = 55


class CorpusProcessor:
    """Runs a job specification through the six observable pipeline stages."""

    def __init__(self, sink: ProgressSink, options: DocumentOptions | None = None) -> None:
        self.tracker = StageTracker(sink)
        self.options = options or DocumentOptions()

    def run(
        self, spec: JobSpec, input_paths: list[Path], output_dir: Path
    ) -> dict[str, Any]:
        documents = self._run_document_stages(spec, input_paths, output_dir)
        index_summary = self._run_stage_index(spec, documents, output_dir)
        graph_summary = self._run_stage_graph(
            spec, documents, output_dir, index_summary["model"]
        )
        return self._run_stage_package(
            spec, documents, index_summary, graph_summary, output_dir
        )

    def _run_document_stages(
        self, spec: JobSpec, input_paths: list[Path], output_dir: Path
    ) -> list[dict[str, Any]]:
        processor = DocumentProcessor(spec, self.options, output_dir)
        total = max(len(input_paths), 1)
        self.tracker.start(*DOCUMENT_STAGES)
        self.tracker.report(5, "Reading documents", stage="extract")

        documents = []
        for position, (item, path) in enumerate(
            zip(spec.files, input_paths, strict=True), start=1
        ):
            documents.append(processor.process(item, path))
            self._sync_document_detail(processor, spec)
            percent = 5 + round(
                (DOCUMENT_PROGRESS_CEILING - 5) * position / total
            )
            self.tracker.report(
                percent, f"Processed {position} of {total} documents", stage="extract"
            )

        self._sync_document_detail(processor, spec)
        self.tracker.finish(*DOCUMENT_STAGES)
        self.tracker.report(DOCUMENT_PROGRESS_CEILING, "Documents processed", stage="chunk")
        return documents

    def _sync_document_detail(self, processor: DocumentProcessor, spec: JobSpec) -> None:
        stats = processor.stats
        self.tracker.update("extract", **stats.extract_detail())
        self.tracker.update("enrich", **stats.enrich_detail(spec.enable_vlm))
        self.tracker.update("chunk", **stats.chunk_detail())

    def _run_stage_index(
        self, spec: JobSpec, documents: list[dict[str, Any]], output_dir: Path
    ) -> dict[str, Any]:
        self.tracker.start("embed")
        self.tracker.report(60, "Building the portable vector index", stage="embed")
        summary = build_index(documents, output_dir, spec)
        self.tracker.update(
            "embed",
            vectors=summary["records"],
            model=summary["model"],
            dim=summary["dimension"],
            metric=summary["metric"],
        )
        self.tracker.finish("embed")
        self.tracker.report(75, "Vector index complete", stage="embed")
        return summary

    def _run_stage_graph(
        self,
        spec: JobSpec,
        documents: list[dict[str, Any]],
        output_dir: Path,
        embedder_name: str,
    ) -> dict[str, Any]:
        self.tracker.start("graph")
        self.tracker.report(80, "Building the knowledge graph", stage="graph")
        summary = build_graph(documents, output_dir, spec, embedder_name)
        self.tracker.update(
            "graph",
            nodes=summary.get("nodes", 0),
            edges=summary.get("edges", 0),
            node_types=summary.get("node_types", {}),
            edge_types=summary.get("edge_types", {}),
        )
        self.tracker.finish("graph")
        self.tracker.report(88, "Knowledge graph complete", stage="graph")
        return summary

    def _run_stage_package(
        self,
        spec: JobSpec,
        documents: list[dict[str, Any]],
        index_summary: dict[str, Any],
        graph_summary: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        self.tracker.start("package")
        self.tracker.report(92, "Packaging artifacts", stage="package")
        stage_summary = {
            record["name"]: record for record in self.tracker.snapshot()
        }
        result = package_outputs(
            spec, documents, index_summary, graph_summary, stage_summary, output_dir
        )
        self.tracker.update("package", artifacts=len(result["artifacts"]))
        self.tracker.finish("package")
        self.tracker.report(97, "Artifacts packaged", stage="package")
        return result

    def stages(self) -> list[dict[str, Any]]:
        return self.tracker.snapshot()
