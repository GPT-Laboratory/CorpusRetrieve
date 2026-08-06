from __future__ import annotations

import hashlib
import mimetypes
import zipfile
from pathlib import Path
from typing import Any

from corpusretrieve.config import VERSION
from corpusretrieve.models import JobSpec
from corpusretrieve.pipeline.artifacts import write_json

BUNDLE_NAME = "bundle.zip"
CATALOG_NAME = "artifact_catalog.json"
MANIFEST_NAME = "manifest.json"


def package_outputs(
    spec: JobSpec,
    documents: list[dict[str, Any]],
    index_summary: dict[str, Any],
    graph_summary: dict[str, Any],
    stage_summary: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Write the manifest, hash every artifact, and produce the ZIP bundle."""
    manifest = _manifest(spec, documents, index_summary, graph_summary, stage_summary)
    write_json(output_dir / MANIFEST_NAME, manifest)

    catalog = _catalog(spec.job_id, output_dir)
    write_json(output_dir / CATALOG_NAME, catalog)
    _write_bundle(output_dir)
    return {"manifest": manifest, "artifacts": catalog["artifacts"]}


def _manifest(
    spec: JobSpec,
    documents: list[dict[str, Any]],
    index_summary: dict[str, Any],
    graph_summary: dict[str, Any],
    stage_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "service": "CorpusRetrieve",
        "version": VERSION,
        "job_id": spec.job_id,
        "corpus_name": spec.corpus_name,
        "metadata": spec.metadata,
        "submission": {"mode": spec.submission_mode, "warnings": spec.warnings},
        "documents": [
            {
                "doc_id": document["doc_id"],
                "source": document["source"],
                "sha256": document["sha256"],
                "chunks": len(document["chunks"]),
                "metadata": document.get("metadata", {}),
            }
            for document in documents
        ],
        "index": index_summary,
        "knowledge_graph": graph_summary,
        "stages": stage_summary,
        "vlm": {
            "enabled": spec.enable_vlm,
            "model": spec.vlm_model if spec.enable_vlm else None,
        },
    }


def _catalog(job_id: str, output_dir: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name in {CATALOG_NAME, BUNDLE_NAME}:
            continue
        relative = path.relative_to(output_dir).as_posix()
        data = path.read_bytes()
        artifacts.append(
            {
                "path": relative,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "media_type": mimetypes.guess_type(relative)[0]
                or "application/octet-stream",
            }
        )
    return {"job_id": job_id, "artifacts": artifacts}


def _write_bundle(output_dir: Path) -> None:
    bundle = output_dir / BUNDLE_NAME
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file() and path != bundle:
                archive.write(path, path.relative_to(output_dir).as_posix())
