from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
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
from docparse.enrich import VLMBudget, VLMCircuit, describe_figures, make_anthropic_describer

# per-document counters that make a build "degraded" when non-zero
DEGRADATION_KEYS = ("ocr_failed_pages", "ocr_degraded_pages", "garbled_pages",
                    "figures_failed", "figures_skipped_budget",
                    "figures_skipped_vlm_unavailable", "low_text_coverage_pages")


@dataclass
class DocumentOptions:
    """Everything the document stage needs that is not part of the job spec."""

    anthropic_api_key: str | None = None
    emit_page_previews: bool = True
    page_preview_dpi: int = 110
    max_page_previews: int = 200
    ocr_langs: str = "eng"
    max_vlm_calls: int | None = None
    vlm_max_retries: int = 3
    vlm_timeout_seconds: int = 120
    vlm_max_consecutive_failures: int = 25
    # reuse a document's outputs when they already exist for the same bytes
    resume: bool = True


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
    # time spent in each document stage, summed over documents. The three
    # stages run per document in one loop, so a single wall clock cannot
    # separate them.
    stage_ms: dict[str, int] = field(
        default_factory=lambda: {"extract": 0, "enrich": 0, "chunk": 0})

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
        # shared across the corpus: identical figures are described once and
        # the call budget is global, not per document
        self._vlm_cache: dict[str, str] = {}
        self._vlm_budget = VLMBudget(options.max_vlm_calls or None)
        self._vlm_circuit = VLMCircuit(options.vlm_max_consecutive_failures)
        self._describer = None
        self._by_digest: dict[str, DocumentRecord] = {}
        self.duplicates: list[dict[str, Any]] = []

    def process(self, item: InputFile, source_path: Path) -> DocumentRecord | None:
        """Process one file; returns None when it duplicates an earlier file.

        Identical bytes (a print and an e-version catalogued separately, or the
        same PDF attached to two records) are processed once. The duplicate's
        bibliographic records are merged into the kept document, so no record
        is lost and no evidence is indexed twice.
        """
        digest = _sha256_file(source_path)
        original = self._by_digest.get(digest)
        if original is not None:
            merged = merge_records(original.metadata, item.metadata)
            self.duplicates.append({"filename": item.filename, "duplicate_of": original.doc_id,
                                    "source": original.source, "sha256": digest,
                                    "records_merged": merged})
            return None
        doc_id = self._document_id(item, digest)
        doc_dir = self.output_dir / "documents" / doc_id
        assets_dir = doc_dir / "assets"

        resumed = self._resume(doc_id, doc_dir, digest, item)
        if resumed is not None:
            self._by_digest[digest] = resumed
            return resumed

        started = monotonic()
        result = extract(str(source_path), self._extraction_config(assets_dir))
        result.source = item.filename
        # Fallback to a DOI present IN the file only when no manifest record was
        # supplied; never overwrite an authoritative manifest record.
        if not (item.metadata.get("records") or item.metadata.get("title")
                or item.metadata.get("handle_url") or item.metadata.get("doi")):
            paper = result.stats.get("paper") or {}
            doi = result.stats.get("source_doi") or paper.get("doi")
            if doi:
                item.metadata = dict(item.metadata)
                item.metadata["records"] = [{
                    "doi": doi,
                    **({"title": result.stats["source_title"]}
                       if result.stats.get("source_title") else {}),
                    **({"year": paper["year"]} if paper.get("year") else {}),
                    "origin": "extracted", "source": "pdf_frontmatter",
                }]
                result.stats["bibliographic_fallback"] = "extracted_from_pdf"

        previews = self._render_previews(source_path, assets_dir)
        extracted = monotonic()
        vlm = self._enrich(result)
        self._rewrite_asset_paths(result)
        self._record_extraction_stats(result, previews)
        enriched = monotonic()

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

        quality = document_quality(result, chunks, self.spec.enable_vlm, item.metadata)
        timing = {"extract": int((extracted - started) * 1000),
                  "enrich": int((enriched - extracted) * 1000),
                  "chunk": int((monotonic() - enriched) * 1000)}
        for stage, ms in timing.items():
            self.stats.stage_ms[stage] += ms
        chunks["quality"] = quality

        extraction = result.to_dict()
        extraction.update(
            {
                "doc_id": doc_id,
                "source": item.filename,
                "sha256": digest,
                "metadata": item.metadata,
                "vlm": vlm,
                "page_previews": previews,
                "quality": quality,
            }
        )
        write_json(doc_dir / "extraction.json", extraction)
        write_json(doc_dir / "chunks.json", chunks)
        self.stats.documents += 1
        record = DocumentRecord(
            doc_id=doc_id,
            source=item.filename,
            sha256=digest,
            chunks=len(chunks.get("chunks", [])),
            metadata=item.metadata,
            external_id=item.document_id,
            quality=quality,
            timing_ms=timing,
        )
        self._by_digest[digest] = record
        return record

    def _resume(self, doc_id: str, doc_dir: Path, digest: str,
                item: InputFile) -> DocumentRecord | None:
        """Return the existing record when this exact file was already processed."""
        path = doc_dir / "chunks.json"
        if not self.options.resume or not path.exists():
            return None
        try:
            chunks = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if chunks.get("sha256") != digest:
            return None
        if not (item.metadata.get("records") or item.metadata.get("title")
                or item.metadata.get("handle_url") or item.metadata.get("doi")):
            extraction_path = doc_dir / "extraction.json"
            if extraction_path.exists():
                extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
                stats = extraction.get("stats") or {}
                paper = stats.get("paper") or {}
                doi = stats.get("source_doi") or paper.get("doi")
                if doi:
                    item.metadata = dict(item.metadata)
                    item.metadata["records"] = [{
                        "doi": doi,
                        **({"title": stats["source_title"]}
                           if stats.get("source_title") else {}),
                        **({"year": paper["year"]} if paper.get("year") else {}),
                        "origin": "extracted", "source": "pdf_frontmatter",
                    }]
        self.stats.documents += 1
        self._record_chunk_stats(chunks)
        from docparse.vectorization.canonical import bibliographic_record
        q = dict(chunks.get("quality") or {})
        resolutions = [bibliographic_record({"metadata": item.metadata}, c.get("pages"))
                       for c in chunks.get("chunks", [])]
        q["unresolved_bibliographic_units"] = sum(
            b is None or bool(b.get("ambiguous")) for b in resolutions)
        q["ambiguous_bibliographic_units"] = sum(
            bool(b and b.get("ambiguous")) for b in resolutions)
        q["has_bibliographic_record"] = bool(resolutions) and not q["unresolved_bibliographic_units"]
        q["bibliographic_origin"] = ("none" if not resolutions or all(b is None for b in resolutions)
                                     else "extracted" if any(
            b and b.get("origin") == "extracted" for b in resolutions) else "manifest")
        return DocumentRecord(
            doc_id=doc_id, source=item.filename, sha256=digest,
            chunks=len(chunks.get("chunks", [])), metadata=item.metadata,
            external_id=item.document_id, quality=q,
            resumed=True,
        )

    def _document_id(self, item: InputFile, digest: str) -> str:
        # content-addressed by default. The file name is provenance, not
        # identity, so a renamed or case-folded copy keeps its ID and two
        # different files can never share one.
        base = safe_identifier(item.document_id or f"doc-{digest[:16]}")
        return unique_identifier(base, self._used_ids)

    def _extraction_config(self, assets_dir: Path) -> ExtractionConfig:
        return ExtractionConfig(
            assets_dir=str(assets_dir),
            table_serializations=("markdown", "kv"),
            scientific=True,
            ocr_langs=self.options.ocr_langs,
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
        if self._describer is None:
            self._describer = make_anthropic_describer(
                model=self.spec.vlm_model, api_key=self.options.anthropic_api_key,
                timeout_seconds=self.options.vlm_timeout_seconds,
            )
        described = describe_figures(result, self._describer, cache=self._vlm_cache,
                                     budget=self._vlm_budget,
                                     max_retries=self.options.vlm_max_retries,
                                     circuit=self._vlm_circuit)
        self.stats.figures_described += described
        vlm.update({"figures_described": described, "model": self.spec.vlm_model,
                    "budget_exhausted": self._vlm_budget.exhausted,
                    "stopped": self._vlm_circuit.open,
                    "stopped_reason": self._vlm_circuit.reason})
        return vlm

    def vlm_status(self) -> dict[str, Any]:
        """Build-level outcome of figure descriptions, for the manifest."""
        return {"calls": self._vlm_budget.used,
                "budget_exhausted": self._vlm_budget.exhausted,
                "stopped": self._vlm_circuit.open,
                "stopped_reason": self._vlm_circuit.reason}

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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def document_quality(result, chunks: dict[str, Any], vlm_enabled: bool,
                     metadata: dict[str, Any]) -> dict[str, Any]:
    """Every way this document could be silently worse than it looks.

    Each counter is a way content can be missing from the outputs while the
    build still looks complete, so each one is counted rather than left in a
    free-text warning.
    """
    stats = result.stats
    figures = [e for e in result.elements if e.kind == "figure"
               and not e.metadata.get("repeated_image")]
    q: dict[str, Any] = {k: int(stats.get(k, 0) or 0) for k in DEGRADATION_KEYS}
    q["pages"] = result.page_count or 0
    q["figures"] = len(figures)
    q["repeated_images"] = int(stats.get("repeated_images", 0) or 0)
    if vlm_enabled:
        described = sum(1 for e in figures if e.metadata.get("description") == "vlm")
        q["figures_undescribed"] = len(figures) - described
    q["orphan_warnings"] = sum(1 for w in chunks.get("warnings", []) if "not covered" in w)
    q["warnings"] = len(result.warnings)
    from docparse.vectorization.canonical import bibliographic_record
    resolutions = [bibliographic_record({"metadata": metadata}, c.get("pages"))
                   for c in chunks.get("chunks", [])]
    ambiguous = sum(bool(b and b.get("ambiguous")) for b in resolutions)
    unresolved = sum(b is None or bool(b.get("ambiguous")) for b in resolutions)
    q["has_bibliographic_record"] = bool(resolutions) and unresolved == 0
    q["unresolved_bibliographic_units"] = unresolved
    q["ambiguous_bibliographic_units"] = ambiguous
    q["bibliographic_origin"] = ("none" if not resolutions or all(b is None for b in resolutions)
                                 else "extracted" if any(
        isinstance(b, dict) and b.get("origin") == "extracted" for b in resolutions)
                                 else "manifest")
    q["uncertain_tables"] = int(result.stats.get("uncertain_tables", 0) or 0)
    q["units"] = len(chunks.get("chunks", []))
    q["degraded"] = any(q[k] for k in DEGRADATION_KEYS) or (
        vlm_enabled and q.get("figures_undescribed", 0) > 0) or (
        q["pages"] > 0 and q["units"] == 0)
    return q


def _records_of(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    if metadata.get("records"):
        return list(metadata["records"])
    if any(metadata.get(k) for k in ("title", "handle_url", "doi", "source_doc_id")):
        return [{k: v for k, v in metadata.items()
                 if k not in ("records", "scenarios", "themes", "keywords_matched")}]
    return []


def merge_records(target: dict[str, Any], incoming: dict[str, Any]) -> int:
    """Append ``incoming``'s bibliographic records to ``target`` (in place).

    Records already present (same source_doc_id, or same title and year) are
    not added twice. Returns how many records were added.
    """
    have = _records_of(target)
    if not have and not _records_of(incoming):
        return 0

    def key(rec: dict[str, Any]) -> tuple:
        if rec.get("source_doc_id"):
            return ("id", rec["source_doc_id"])
        return ("ty", str(rec.get("title", "")).casefold().strip(), rec.get("year"))

    seen = {key(r) for r in have}
    added = 0
    for rec in _records_of(incoming):
        if key(rec) not in seen:
            have.append(rec)
            seen.add(key(rec))
            added += 1
    target["records"] = have
    for list_key in ("scenarios", "themes"):
        values = list(target.get(list_key) or [])
        values += [v for v in incoming.get(list_key) or [] if v not in values]
        if values:
            target[list_key] = values
    return added
