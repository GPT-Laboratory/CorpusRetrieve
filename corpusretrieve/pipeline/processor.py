from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from corpusretrieve.models import JobSpec
from corpusretrieve.pipeline.corpus import CorpusRegistry, DocumentFailure
from corpusretrieve.pipeline.documents import DocumentOptions, DocumentProcessor
from corpusretrieve.pipeline.indexing import DEFAULT_BATCH_SIZE, build_index
from corpusretrieve.pipeline.knowledge_graph import build_graph
from corpusretrieve.pipeline.packaging import package_outputs
from corpusretrieve.pipeline.progress import ProgressSink, StageTracker

logger = logging.getLogger(__name__)

DOCUMENT_STAGES = ("extract", "enrich", "chunk")
DOCUMENT_PROGRESS_CEILING = 55
PROGRESS_REPORT_INTERVAL = 500


class NoDocumentsProcessed(RuntimeError):
    """Raised when every document in the submission failed to process."""


class CorpusProcessor:
    """Runs a job specification through the six observable pipeline stages.

    Documents are processed one at a time and their chunks are written to
    disk. Later stages stream those files back, so peak memory is bounded by
    the largest single document and the embedding batch size rather than by
    the size of the corpus.

    A document that cannot be processed is recorded and skipped rather than
    ending the job, so a single unreadable file does not discard the work
    already completed for every other document.
    """

    def __init__(
        self,
        sink: ProgressSink,
        options: DocumentOptions | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.tracker = StageTracker(sink)
        self.options = options or DocumentOptions()
        self.batch_size = batch_size
        self.registry: CorpusRegistry | None = None

    def run(
        self, spec: JobSpec, input_paths: list[Path], output_dir: Path
    ) -> dict[str, Any]:
        registry = self._run_document_stages(spec, input_paths, output_dir)
        if not len(registry):
            raise NoDocumentsProcessed(
                "No document in the submission could be processed. "
                f"{len(registry.failures)} failed."
            )
        index_summary = self._run_stage_index(spec, registry, output_dir)
        graph_summary = self._run_stage_graph(
            spec, registry, output_dir, index_summary["model"]
        )
        return self._run_stage_package(
            spec, registry, index_summary, graph_summary, output_dir
        )

    def _run_document_stages(
        self, spec: JobSpec, input_paths: list[Path], output_dir: Path
    ) -> CorpusRegistry:
        processor = DocumentProcessor(spec, self.options, output_dir)
        registry = CorpusRegistry(output_dir)
        self.registry = registry
        total = max(len(input_paths), 1)

        self.tracker.start(*DOCUMENT_STAGES)
        self.tracker.report(5, "Reading documents", stage="extract")

        for position, (item, path) in enumerate(
            zip(spec.files, input_paths, strict=True), start=1
        ):
            try:
                registry.add(processor.process(item, path))
            except Exception as exc:  # noqa: BLE001 - one bad file must not end the job
                logger.warning("Skipping %s: %s", item.filename, exc)
                registry.fail(
                    DocumentFailure(
                        filename=item.filename,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
            self._sync_document_detail(processor, spec, registry)
            percent = 5 + round((DOCUMENT_PROGRESS_CEILING - 5) * position / total)
            self.tracker.report(
                percent,
                f"Processed {position} of {total} documents",
                stage="extract",
            )

        self._sync_document_detail(processor, spec, registry)
        self.tracker.finish(*DOCUMENT_STAGES)
        self.tracker.report(
            DOCUMENT_PROGRESS_CEILING,
            f"{len(registry)} documents processed, {len(registry.failures)} skipped",
            stage="chunk",
        )
        return registry

    def _sync_document_detail(
        self, processor: DocumentProcessor, spec: JobSpec, registry: CorpusRegistry
    ) -> None:
        stats = processor.stats
        self.tracker.update(
            "extract", **stats.extract_detail(), skipped=len(registry.failures)
        )
        self.tracker.update("enrich", **stats.enrich_detail(spec.enable_vlm))
        self.tracker.update("chunk", **stats.chunk_detail())

    def _run_stage_index(
        self, spec: JobSpec, registry: CorpusRegistry, output_dir: Path
    ) -> dict[str, Any]:
        self.tracker.start("embed")
        self.tracker.report(60, "Building the portable vector index", stage="embed")

        def report(done: int) -> None:
            if done % PROGRESS_REPORT_INTERVAL == 0:
                self.tracker.update("embed", vectors=done)
                self.tracker.report(60, f"Embedded {done} chunks", stage="embed")

        summary = build_index(
            registry, output_dir, spec, batch_size=self.batch_size, on_progress=report
        )
        self.tracker.update(
            "embed",
            vectors=summary["records"],
            model=summary["model"],
            dim=summary["dimension"],
            metric=summary["metric"],
            batch_size=summary["batch_size"],
        )
        self.tracker.finish("embed")
        self.tracker.report(75, "Vector index complete", stage="embed")
        return summary

    def _run_stage_graph(
        self,
        spec: JobSpec,
        registry: CorpusRegistry,
        output_dir: Path,
        embedder_name: str,
    ) -> dict[str, Any]:
        self.tracker.start("graph")
        self.tracker.report(80, "Building the knowledge graph", stage="graph")

        def report(done: int) -> None:
            if done % PROGRESS_REPORT_INTERVAL == 0:
                self.tracker.report(
                    80, f"Projected {done} documents into the graph", stage="graph"
                )

        summary = build_graph(
            registry, output_dir, spec, embedder_name, on_progress=report
        )
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
        registry: CorpusRegistry,
        index_summary: dict[str, Any],
        graph_summary: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        self.tracker.start("package")
        self.tracker.report(92, "Packaging artifacts", stage="package")
        stage_summary = {record["name"]: record for record in self.tracker.snapshot()}
        result = package_outputs(
            spec,
            registry,
            index_summary,
            graph_summary,
            stage_summary,
            output_dir,
        )
        self.tracker.update("package", artifacts=len(result["artifacts"]))
        self.tracker.finish("package")
        self.tracker.report(97, "Artifacts packaged", stage="package")
        return result

    def stages(self) -> list[dict[str, Any]]:
        return self.tracker.snapshot()

    def failures(self) -> list[dict[str, Any]]:
        return self.registry.failure_entries() if self.registry else []
