from __future__ import annotations

import logging
import mimetypes
import tempfile
from pathlib import Path
from typing import Any

from corpusretrieve.config import get_settings
from corpusretrieve.models import JobSpec
from corpusretrieve.pipeline import CorpusProcessor, DocumentOptions
from corpusretrieve.pipeline.packaging import package_outputs
from corpusretrieve.security import get_credential_store
from corpusretrieve.state import get_job_store
from corpusretrieve.storage import get_artifact_store

logger = logging.getLogger(__name__)


def process_job(job_id: str) -> None:
    """Execute one job end to end and publish its artifacts."""
    settings = get_settings()
    store = get_artifact_store()
    jobs = get_job_store()
    credentials = get_credential_store()

    spec = jobs.get_spec(job_id)
    if spec is None:
        logger.warning("Job %s no longer exists; nothing to process", job_id)
        return

    api_key = credentials.take(job_id) or settings.anthropic_api_key or None

    def sink(stage: str, percent: int, message: str, stages: list[dict[str, Any]]) -> None:
        jobs.update_state(
            job_id,
            state="processing",
            stage=stage,
            progress=percent,
            message=message,
            stages=stages,
            error=None,
        )

    processor = CorpusProcessor(
        sink,
        DocumentOptions(
            anthropic_api_key=api_key,
            emit_page_previews=settings.emit_page_previews,
            page_preview_dpi=settings.page_preview_dpi,
            max_page_previews=settings.max_page_previews_per_document,
        ),
        batch_size=settings.embedding_batch_size,
    )

    try:
        with tempfile.TemporaryDirectory(prefix=f"corpusretrieve-{job_id}-") as temp:
            workspace = Path(temp)
            inputs_dir = workspace / "inputs"
            outputs_dir = workspace / "outputs"
            inputs_dir.mkdir()
            outputs_dir.mkdir()

            try:
                input_paths = _materialize_inputs(store, spec, inputs_dir)
                processor.run(spec, input_paths, outputs_dir)
            except Exception as exc:  # noqa: BLE001 - salvage before reporting
                reason = f"{type(exc).__name__}: {_redact(str(exc), api_key)}"
                logger.exception("Job %s failed; publishing partial outputs", job_id)
                _package_partial(spec, processor, outputs_dir, reason)
                artifacts = _publish_artifacts(store, job_id, outputs_dir)
                _report_failure(jobs, job_id, processor, artifacts, reason)
                return

            artifacts = _publish_artifacts(store, job_id, outputs_dir)

        failures = processor.failures()
        jobs.update_state(
            job_id,
            state="completed",
            stage="completed",
            progress=100,
            message=(
                "Artifacts are ready"
                if not failures
                else f"Artifacts are ready; {len(failures)} document(s) were skipped"
            ),
            manifest_key=f"jobs/{job_id}/artifacts/manifest.json",
            bundle_key=f"jobs/{job_id}/artifacts/bundle.zip",
            artifacts=artifacts,
            stages=processor.stages(),
            warnings=spec.warnings + [_skip_warning(failures)] if failures else spec.warnings,
            error=None,
        )
    finally:
        credentials.delete(job_id)
        if settings.delete_inputs_after_processing:
            store.delete_prefix(f"jobs/{job_id}/inputs/")


def _skip_warning(failures: list[dict[str, Any]]) -> str:
    names = ", ".join(item["filename"] for item in failures[:5])
    more = f" and {len(failures) - 5} more" if len(failures) > 5 else ""
    return f"Skipped {len(failures)} unreadable document(s): {names}{more}."


def _package_partial(
    spec: JobSpec, processor: CorpusProcessor, outputs_dir: Path, reason: str
) -> None:
    """Write a manifest, catalog, and bundle describing whatever completed.

    A job that fails partway has usually produced usable per-document output
    already. Packaging it means the caller can download and keep that work
    instead of losing the entire run.
    """
    registry = processor.registry
    if registry is None:
        return
    try:
        package_outputs(
            spec,
            registry,
            _summary_from_stage(processor, "embed"),
            _summary_from_stage(processor, "graph"),
            {record["name"]: record for record in processor.stages()},
            outputs_dir,
            complete=False,
            error=reason,
        )
    except Exception:  # noqa: BLE001 - salvage must never mask the original failure
        logger.exception("Could not package partial outputs")


def _summary_from_stage(processor: CorpusProcessor, name: str) -> dict[str, Any]:
    record = next((r for r in processor.stages() if r["name"] == name), None)
    return dict(record["detail"]) if record else {}


def _report_failure(
    jobs, job_id: str, processor: CorpusProcessor, artifacts: list[dict[str, Any]], reason: str
) -> None:
    partial = bool(artifacts)
    jobs.update_state(
        job_id,
        state="failed",
        stage="failed",
        message=(
            f"Processing failed; {len(artifacts)} partial artifact(s) are available"
            if partial
            else "Processing failed"
        ),
        manifest_key=(
            f"jobs/{job_id}/artifacts/manifest.json"
            if any(a["path"] == "manifest.json" for a in artifacts)
            else None
        ),
        bundle_key=(
            f"jobs/{job_id}/artifacts/bundle.zip"
            if any(a["path"] == "bundle.zip" for a in artifacts)
            else None
        ),
        artifacts=artifacts,
        stages=processor.stages(),
        error=reason,
    )


def _materialize_inputs(store, spec: JobSpec, inputs_dir: Path) -> list[Path]:
    paths = []
    for index, item in enumerate(spec.files):
        path = inputs_dir / f"{index:04d}-{Path(item.filename).name}"
        path.write_bytes(store.get_bytes(item.storage_key))
        paths.append(path)
    return paths


def _publish_artifacts(store, job_id: str, outputs_dir: Path) -> list[dict[str, Any]]:
    artifacts = []
    for path in sorted(outputs_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(outputs_dir).as_posix()
        key = f"jobs/{job_id}/artifacts/{relative}"
        media_type = mimetypes.guess_type(relative)[0] or "application/octet-stream"
        with path.open("rb") as handle:
            store.put_fileobj(key, handle, media_type)
        artifacts.append(
            {
                "path": relative,
                "key": key,
                "size_bytes": path.stat().st_size,
                "media_type": media_type,
            }
        )
    return artifacts


def _redact(message: str, secret: str | None) -> str:
    return message.replace(secret, "[redacted]") if secret else message
