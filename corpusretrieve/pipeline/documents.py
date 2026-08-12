from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from corpusretrieve.models import InputFile, JobSpec
from corpusretrieve.pipeline.artifacts import (
    relative_asset_path,
    safe_identifier,
    unique_identifier,
    write_json,
)
from corpusretrieve.pipeline.corpus import DocumentRecord
from corpusretrieve.pipeline.previews import render_page_previews
from docparse import ExtractionConfig, chunk_result, extract
from docparse.enrich import describe_figures, make_anthropic_describer


@dataclass
class DocumentOptions:
    """Everything the document stage needs that is not part of the job spec."""

    anthropic_api_key: str | None = None
    emit_page_previews: bool = True
    page_preview_dpi: int = 110
    max_page_previews: int = 200


@dataclass
class DocumentStats:
    """Aggregated counters used to describe the document stages."""

    documents: int = 0
    pages: int = 0
    elements: int = 0
    elements_with_bbox: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)
    roles: dict[str, int] = field(default_factory=dict)
    claims: int = 0
    citations: int = 0
    figures_total: int = 0
    figures_described: int = 0
    page_previews: int = 0
    chunks: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    tokens: int = 0

    def extract_detail(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "pages": self.pages,
            "elements": self.elements,
            "with_bbox": self.elements_with_bbox,
            "by_kind": dict(self.by_kind),
            "page_previews": self.page_previews,
        }

    def enrich_detail(self, vlm_enabled: bool) -> dict[str, Any]:
        detail: dict[str, Any] = {
            "roles": dict(self.roles),
            "claims": self.claims,
            "citations": self.citations,
            "figures_total": self.figures_total,
        }
        if vlm_enabled:
            detail["vlm"] = "anthropic"
            detail["figures_described"] = self.figures_described
        return detail

    def chunk_detail(self) -> dict[str, Any]:
        return {
            "chunks": self.chunks,
            "by_type": dict(self.by_type),
            "tokens": self.tokens,
        }


def _increment(counter: dict[str, int], key: str, amount: int = 1) -> None:
    counter[key] = counter.get(key, 0) + amount


class DocumentProcessor:
    """Turns one input file into extraction, chunk, and asset artifacts."""

    def __init__(self, spec: JobSpec, options: DocumentOptions, output_dir: Path) -> None:
        self.spec = spec
        self.options = options
        self.output_dir = output_dir
        self.stats = DocumentStats()
        self._used_ids: set[str] = set()

    def process(self, item: InputFile, source_path: Path) -> DocumentRecord:
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        doc_id = self._document_id(item, digest)
        doc_dir = self.output_dir / "documents" / doc_id
        assets_dir = doc_dir / "assets"

        result = extract(str(source_path), self._extraction_config(assets_dir))
        result.source = item.filename

        previews = self._render_previews(source_path, assets_dir)
        vlm = self._enrich(result)
        self._rewrite_asset_paths(result)
        self._record_extraction_stats(result, previews)

        chunks = chunk_result(result, doc_id=doc_id).to_dict()
        chunks.update(
            {
                "source": item.filename,
                "sha256": digest,
                "corpus_id": self.spec.corpus_name,
                "external_id": item.document_id,
                "metadata": item.metadata,
            }
        )
        self._record_chunk_stats(chunks)

        extraction = result.to_dict()
        extraction.update(
            {
                "doc_id": doc_id,
                "source": item.filename,
                "sha256": digest,
                "metadata": item.metadata,
                "vlm": vlm,
                "page_previews": previews,
            }
        )
        write_json(doc_dir / "extraction.json", extraction)
        write_json(doc_dir / "chunks.json", chunks)
        self.stats.documents += 1
        return DocumentRecord(
            doc_id=doc_id,
            source=item.filename,
            sha256=digest,
            chunks=len(chunks.get("chunks", [])),
            metadata=item.metadata,
            external_id=item.document_id,
        )

    def _document_id(self, item: InputFile, digest: str) -> str:
        base = safe_identifier(
            item.document_id or f"{Path(item.filename).stem}-{digest[:12]}"
        )
        return unique_identifier(base, self._used_ids)

    def _extraction_config(self, assets_dir: Path) -> ExtractionConfig:
        return ExtractionConfig(
            assets_dir=str(assets_dir),
            table_serializations=("markdown", "kv"),
            scientific=True,
        )

    def _render_previews(self, source_path: Path, assets_dir: Path) -> dict[str, dict]:
        if not self.options.emit_page_previews:
            return {}
        previews = render_page_previews(
            source_path,
            assets_dir,
            self.options.page_preview_dpi,
            self.options.max_page_previews,
        )
        self.stats.page_previews += len(previews)
        return previews

    def _enrich(self, result) -> dict[str, Any]:
        figures = sum(1 for element in result.elements if element.kind == "figure")
        self.stats.figures_total += figures
        vlm: dict[str, Any] = {"enabled": self.spec.enable_vlm, "figures_described": 0}
        if not self.spec.enable_vlm:
            return vlm
        if not self.options.anthropic_api_key:
            raise ValueError("An Anthropic API key is required when VLM is enabled")
        describer = make_anthropic_describer(
            model=self.spec.vlm_model, api_key=self.options.anthropic_api_key
        )
        described = describe_figures(result, describer)
        self.stats.figures_described += described
        vlm.update({"figures_described": described, "model": self.spec.vlm_model})
        return vlm

    def _rewrite_asset_paths(self, result) -> None:
        """Replace absolute crop paths with paths relative to the output root."""
        for element in result.elements:
            reference = element.visual_ref or {}
            image_path = reference.get("image_path")
            if not image_path:
                continue
            relative = relative_asset_path(image_path, self.output_dir)
            if relative:
                reference["image_path"] = relative
            else:
                reference.pop("image_path", None)

    def _record_extraction_stats(self, result, previews: dict[str, dict]) -> None:
        self.stats.pages += result.page_count or 0
        self.stats.elements += len(result.elements)
        self.stats.elements_with_bbox += sum(1 for e in result.elements if e.bbox)
        for kind, count in result.counts().items():
            _increment(self.stats.by_kind, kind, count)
        for element in result.elements:
            role = element.metadata.get("role")
            if role:
                _increment(self.stats.roles, role)
        self.stats.claims += len(result.stats.get("claims") or [])
        self.stats.citations += len(result.stats.get("citation_graph") or {})
        if previews:
            result.stats["page_previews"] = len(previews)

    def _record_chunk_stats(self, chunks: dict[str, Any]) -> None:
        stats = chunks.get("stats", {})
        self.stats.chunks += stats.get("n_chunks", len(chunks.get("chunks", [])))
        for chunk_type, count in (stats.get("by_type") or {}).items():
            _increment(self.stats.by_type, chunk_type, count)
        self.stats.tokens += sum(
            chunk.get("token_estimate", 0) for chunk in chunks.get("chunks", [])
        )
