from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import platform
import subprocess
import time
import zipfile
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from corpusretrieve.config import VERSION, get_settings
from corpusretrieve.models import JobSpec
from corpusretrieve.pipeline.artifacts import write_json
from corpusretrieve.pipeline.corpus import CorpusRegistry, DocumentRecord
from corpusretrieve.pipeline.documents import DEGRADATION_KEYS

BUNDLE_NAME = "bundle.zip"
BUNDLE_HASH_NAME = "bundle.zip.sha256"
CATALOG_NAME = "artifact_catalog.json"
MANIFEST_NAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = "2.0.0"
_HASH_BLOCK = 1024 * 1024


def package_outputs(
    spec: JobSpec,
    registry: CorpusRegistry,
    index_summary: dict[str, Any],
    graph_summary: dict[str, Any],
    stage_summary: dict[str, Any],
    output_dir: Path,
    complete: bool = True,
    error: str | None = None,
    vlm_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the manifest, hash every artifact, and produce the ZIP bundle."""
    manifest = _manifest(
        spec, registry, index_summary, graph_summary, stage_summary, complete, error,
        vlm_status,
    )
    write_json(output_dir / MANIFEST_NAME, manifest)
    return _seal(spec.job_id, output_dir, manifest)


def _seal(job_id: str, output_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Catalog every file, zip the bundle, and record the bundle's own hash."""
    catalog = _catalog(job_id, output_dir)
    write_json(output_dir / CATALOG_NAME, catalog)
    _write_bundle(output_dir)
    bundle_sha = sha256_file(output_dir / BUNDLE_NAME)
    (output_dir / BUNDLE_HASH_NAME).write_text(f"{bundle_sha}  {BUNDLE_NAME}\n",
                                                encoding="utf-8")
    return {"manifest": manifest, "artifacts": catalog["artifacts"],
            "bundle_sha256": bundle_sha}


def sha256_file(path: Path) -> str:
    """Streaming SHA-256, so a multi-gigabyte artifact is never held in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_HASH_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest(
    spec: JobSpec,
    registry: CorpusRegistry,
    index_summary: dict[str, Any],
    graph_summary: dict[str, Any],
    stage_summary: dict[str, Any],
    complete: bool,
    error: str | None,
    vlm_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    quality = corpus_quality(registry, index_summary, graph_summary, spec, complete)
    return {
        "service": "CorpusRetrieve",
        "version": VERSION,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "release_ready": quality["release_ready"],
        "quality": quality,
        "environment": build_environment(),
        "repairs": [],
        "job_id": spec.job_id,
        "corpus_name": spec.corpus_name,
        "metadata": spec.metadata,
        "submission": {"mode": spec.submission_mode, "warnings": spec.warnings},
        "complete": complete,
        "error": error,
        "documents": registry.manifest_entries(),
        "skipped_documents": registry.failure_entries(),
        # identical files submitted more than once: processed once, their
        # bibliographic records merged into the kept document
        "duplicate_files": registry.duplicates,
        # collection-level provenance (run date, source, exclusion list)
        "collection": (getattr(spec, "collection", None) or {}),
        "index": index_summary,
        "knowledge_graph": graph_summary,
        "stages": stage_summary,
        "vlm": {
            "enabled": spec.enable_vlm,
            "model": spec.vlm_model if spec.enable_vlm else None,
            **(vlm_status or {}),
        },
    }


def _catalog(job_id: str, output_dir: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name in {CATALOG_NAME, BUNDLE_NAME, BUNDLE_HASH_NAME}:
            continue
        relative = path.relative_to(output_dir).as_posix()
        artifacts.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "media_type": mimetypes.guess_type(relative)[0]
                or "application/octet-stream",
            }
        )
    return {"job_id": job_id, "artifacts": artifacts}


def _write_bundle(output_dir: Path) -> None:
    bundle = output_dir / BUNDLE_NAME
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file() and path != bundle and path.name != BUNDLE_HASH_NAME:
                archive.write(path, path.relative_to(output_dir).as_posix())


# --------------------------------------------------------------------- quality

def corpus_quality(registry: CorpusRegistry, index_summary: dict[str, Any],
                   graph_summary: dict[str, Any], spec: JobSpec | None,
                   complete: bool = True) -> dict[str, Any]:
    """Aggregate every degradation into one release decision.

    A build is ``release_ready`` only if nothing was silently lost: no OCR
    page went unread, no figure description failed or was skipped for budget,
    no graph export failed, the requested embedder was the one used, no two
    uploads collided, and no document_config entry went unmatched.
    """
    totals = {k: 0 for k in DEGRADATION_KEYS}
    undescribed = 0
    degraded_docs: list[str] = []
    without_record = 0
    unresolved_units = 0
    ambiguous_units = 0
    uncertain_tables = 0
    for record in registry.records:
        q = record.quality or {}
        for key in totals:
            totals[key] += int(q.get(key, 0) or 0)
        undescribed += int(q.get("figures_undescribed", 0) or 0)
        if q.get("degraded"):
            degraded_docs.append(record.doc_id)
        if not q.get("has_bibliographic_record"):
            without_record += 1
        unresolved_units += int(q.get("unresolved_bibliographic_units", 0) or 0)
        ambiguous_units += int(q.get("ambiguous_bibliographic_units", 0) or 0)
        uncertain_tables += int(q.get("uncertain_tables", 0) or 0)
    intake = (getattr(spec, "intake", None) or {}) if spec is not None else {}
    reasons: list[str] = []
    for key, value in totals.items():
        if value:
            reasons.append(f"{key}={value}")
    # figures without a description that no counter above explains (for
    # example, a crop that could not be written); the explained ones are
    # already reasons, so they are not counted twice
    unexplained = undescribed - sum(totals[k] for k in (
        "figures_failed", "figures_skipped_budget", "figures_skipped_vlm_unavailable"))
    if unexplained > 0:
        reasons.append(f"figures_undescribed_unexplained={unexplained}")
    no_units = [r.doc_id for r in registry.records
                if (r.quality or {}).get("pages") and not (r.quality or {}).get("units")]
    if no_units:
        reasons.append(f"documents_without_units={len(no_units)}")
    if index_summary.get("embedder_fallback"):
        reasons.append(f"embedder_fallback={index_summary['embedder_fallback']}->"
                       f"{index_summary.get('model')}")
    if graph_summary.get("export_errors"):
        reasons.append("graph_export_errors=" + ",".join(sorted(graph_summary["export_errors"])))
    if intake.get("filename_collisions"):
        reasons.append(f"filename_collisions={len(intake['filename_collisions'])}")
    if intake.get("unmatched_config_keys"):
        reasons.append(f"unmatched_config_keys={len(intake['unmatched_config_keys'])}")
    if intake.get("files_without_document_config"):
        reasons.append("files_without_document_config="
                       f"{len(intake['files_without_document_config'])}")
    strict_bibliographic = bool((getattr(spec, "require_bibliographic", False)
                                 if spec is not None else False)
                                or get_settings().require_bibliographic)
    if strict_bibliographic:
        if without_record:
            reasons.append(f"documents_without_bibliographic_record={without_record}")
        if unresolved_units:
            reasons.append(f"unresolved_bibliographic_units={unresolved_units}")
        if ambiguous_units:
            reasons.append(f"ambiguous_bibliographic_units={ambiguous_units}")
    if registry.failures:
        reasons.append(f"skipped_documents={len(registry.failures)}")
    if not complete:
        reasons.append("incomplete_build")
    return {
        "release_ready": not reasons,
        "degraded": bool(reasons),
        "reasons": reasons,
        "totals": {**totals, "figures_undescribed": undescribed},
        "degraded_documents": degraded_docs,
        "documents_without_units": no_units,
        "documents_without_bibliographic_record": without_record,
        "unresolved_bibliographic_units": unresolved_units,
        "ambiguous_bibliographic_units": ambiguous_units,
        "uncertain_tables": uncertain_tables,
        "require_bibliographic": strict_bibliographic,
        "skipped_documents": len(registry.failures),
        "duplicate_files": len(registry.duplicates),
        "graph_export_errors": graph_summary.get("export_errors") or {},
        "embedder_fallback": index_summary.get("embedder_fallback"),
        "intake": intake,
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    if os.getenv("GIT_COMMIT"):
        return os.getenv("GIT_COMMIT")
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=5, cwd=Path(__file__).resolve().parent)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - provenance is best effort
        return None


def build_environment(ocr_langs: str | None = None) -> dict[str, Any]:
    """Everything needed to say which software produced this bundle."""
    from docparse.ocr import ocr_status

    langs = ocr_langs or get_settings().ocr_langs
    ocr = ocr_status(langs)
    return {
        "corpusretrieve": VERSION,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pythonhashseed": os.getenv("PYTHONHASHSEED"),
        "packages": {name: _package_version(name) for name in
                     ("pymupdf", "pdfplumber", "numpy", "networkx", "pytesseract",
                      "anthropic", "FlagEmbedding", "sentence-transformers")},
        "tesseract": {"binary": ocr.get("binary"), "version": ocr.get("version"),
                      "installed_langs": ocr.get("installed"),
                      "requested_langs": ocr.get("requested"), "missing": ocr.get("missing")},
    }


# --------------------------------------------------------------------- repair

REPAIR_TOOL = "corpusretrieve.repair"


class _RebuildSpec:
    """The JobSpec fields the index and graph builders read, from a manifest."""

    def __init__(self, manifest: dict[str, Any], embedder: str) -> None:
        self.job_id = manifest.get("job_id", "repaired")
        self.corpus_name = manifest.get("corpus_name", "corpus")
        self.embedder = embedder
        self.require_bibliographic = bool(
            (manifest.get("quality") or {}).get("require_bibliographic"))
        self.intake = (manifest.get("quality") or {}).get("intake") or {}


def repair_outputs(output_dir: Path, actions: tuple[str, ...] = ("graph",),
                   embedder: str | None = None,
                   on_progress=None) -> dict[str, Any]:
    """Regenerate derived artifacts from the stored chunks, then re-seal.

    ``index`` re-embeds every unit with ``embedder`` (for example a semantic
    model in place of the hashing embedder) without extracting the documents
    again; the graph is rebuilt with it, because vector point IDs include the
    model name. ``graph`` alone rebuilds the graph exports. Either way the
    catalog and bundle are rewritten and the run is appended to
    ``manifest.repairs``, so the integrity record always matches the files.
    """
    from corpusretrieve.pipeline.indexing import build_index
    from corpusretrieve.pipeline.knowledge_graph import build_graph

    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    registry = CorpusRegistry(output_dir)
    for entry in manifest.get("documents", []):
        registry.add(DocumentRecord(
            doc_id=entry["doc_id"], source=entry.get("source", ""),
            sha256=entry.get("sha256", ""), chunks=entry.get("chunks", 0),
            metadata=entry.get("metadata") or {}, quality=entry.get("quality") or {},
            resumed=bool(entry.get("resumed")), timing_ms=entry.get("timing_ms") or {}))
    index_meta = manifest.get("index") or {}
    if "index" in actions and not embedder:
        raise ValueError("re-indexing needs an embedder spec")
    spec = _RebuildSpec(manifest, embedder or index_meta.get("embedder_requested") or "hash")
    done: list[str] = []
    if "index" in actions:
        manifest["index"] = index_meta = build_index(
            registry, output_dir, spec, inline_vectors=bool(index_meta.get("inline_vectors", True)),
            on_progress=on_progress)
        done.append("index")
    if "graph" in actions or "index" in actions:
        summary = build_graph(registry, output_dir, spec, index_meta.get("model", "hashing-v1"))
        manifest["knowledge_graph"] = summary
        done.append("graph")
    quality = manifest.get("quality")
    if quality is not None:
        reasons = [r for r in quality.get("reasons", [])
                   if not r.startswith(("graph_export_errors=", "embedder_fallback="))]
        if (manifest.get("knowledge_graph") or {}).get("export_errors"):
            reasons.append("graph_export_errors=" + ",".join(
                sorted(manifest["knowledge_graph"]["export_errors"])))
        if index_meta.get("embedder_fallback"):
            reasons.append(f"embedder_fallback={index_meta['embedder_fallback']}->"
                           f"{index_meta.get('model')}")
        quality["reasons"] = reasons
        quality["graph_export_errors"] = (manifest.get("knowledge_graph") or {}).get(
            "export_errors") or {}
        quality["embedder_fallback"] = index_meta.get("embedder_fallback")
        quality["release_ready"] = not reasons
        quality["degraded"] = bool(reasons)
        manifest["release_ready"] = quality["release_ready"]
    manifest.setdefault("repairs", []).append({
        "tool": REPAIR_TOOL, "version": VERSION, "actions": done,
        **({"embedder": embedder} if "index" in actions else {}),
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
    })
    write_json(output_dir / MANIFEST_NAME, manifest)
    return _seal(manifest.get("job_id", "repaired"), output_dir, manifest)
