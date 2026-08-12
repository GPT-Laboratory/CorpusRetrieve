"""Corpus-level streaming and partial-artifact recovery."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from corpusretrieve.pipeline.corpus import (
    CorpusRegistry,
    DocumentFailure,
    DocumentRecord,
    write_corpus_chunks,
)
from corpusretrieve.pipeline.indexing import _batched


def _seed(tmp_path: Path, count: int) -> CorpusRegistry:
    registry = CorpusRegistry(tmp_path)
    for index in range(count):
        doc_id = f"doc-{index}"
        payload = {
            "doc_id": doc_id,
            "source": f"{doc_id}.pdf",
            "chunks": [
                {
                    "chunk_id": f"{doc_id}::c{n}",
                    "doc_id": doc_id,
                    "chunk_type": "section_text",
                    "text": f"Body {n} of document {index}.",
                }
                for n in range(3)
            ],
        }
        path = registry.document_path(doc_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        registry.add(
            DocumentRecord(
                doc_id=doc_id, source=f"{doc_id}.pdf", sha256="0" * 64, chunks=3
            )
        )
    return registry


def test_registry_streams_documents_from_disk(tmp_path: Path):
    registry = _seed(tmp_path, 4)
    assert len(registry) == 4
    assert registry.total_chunks == 12
    assert sum(1 for _ in registry.iter_documents()) == 4
    assert sum(1 for _ in registry.iter_chunks()) == 12


def test_registry_holds_no_chunk_text_in_memory(tmp_path: Path):
    """Records are handles, not payloads: the corpus lives on disk."""
    registry = _seed(tmp_path, 3)
    for record in registry.records:
        assert not hasattr(record, "chunks_data")
        assert isinstance(record.chunks, int)


def test_combined_chunks_file_is_valid_json(tmp_path: Path):
    registry = _seed(tmp_path, 5)
    destination = tmp_path / "corpus" / "chunks.json"
    assert write_corpus_chunks(registry, destination) == 5
    loaded = json.loads(destination.read_text(encoding="utf-8"))
    assert isinstance(loaded, list) and len(loaded) == 5
    assert loaded[0]["doc_id"] == "doc-0"


def test_combined_chunks_file_handles_an_empty_corpus(tmp_path: Path):
    registry = CorpusRegistry(tmp_path)
    destination = tmp_path / "corpus" / "chunks.json"
    assert write_corpus_chunks(registry, destination) == 0
    assert json.loads(destination.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize(
    ("total", "size", "expected"), [(10, 4, [4, 4, 2]), (4, 4, [4]), (0, 4, [])]
)
def test_batching_covers_every_item(total, size, expected):
    batches = list(_batched(iter(range(total)), size))
    assert [len(b) for b in batches] == expected
    assert [item for batch in batches for item in batch] == list(range(total))


def test_failures_are_recorded_not_raised(tmp_path: Path):
    registry = _seed(tmp_path, 2)
    registry.fail(DocumentFailure(filename="broken.pdf", error="FileDataError: bad"))
    assert len(registry) == 2
    entries = registry.failure_entries()
    assert entries[0]["filename"] == "broken.pdf"
    assert entries[0]["stage"] == "extract"


def test_unreadable_document_is_skipped_and_the_job_completes(client, tmp_path):
    import io

    fitz = pytest.importorskip("fitz")
    document = fitz.open()
    document.new_page().insert_text((72, 90), "Readable report", fontsize=12)
    good = document.tobytes()
    document.close()

    files = [
        ("files", ("good.pdf", io.BytesIO(good), "application/pdf")),
        ("files", ("broken.pdf", io.BytesIO(b"%PDF-1.4 not a pdf"), "application/pdf")),
    ]
    job_id = client.post("/v1/jobs", files=files).json()["job_id"]

    status = client.get(f"/v1/jobs/{job_id}").json()
    assert status["state"] == "completed"
    assert any("broken.pdf" in w for w in status["warnings"])

    manifest = client.get(f"/v1/jobs/{job_id}/manifest").json()
    assert manifest["complete"] is True
    assert len(manifest["documents"]) == 1
    assert manifest["skipped_documents"][0]["filename"] == "broken.pdf"


def test_a_corpus_of_only_unreadable_documents_fails(client):
    import io

    files = [
        ("files", (f"broken{n}.pdf", io.BytesIO(b"not a pdf"), "application/pdf"))
        for n in range(2)
    ]
    job_id = client.post("/v1/jobs", files=files).json()["job_id"]
    status = client.get(f"/v1/jobs/{job_id}").json()
    assert status["state"] == "failed"
    assert "NoDocumentsProcessed" in status["error"]


def test_partial_artifacts_survive_a_fatal_stage_failure(client, monkeypatch):
    """A hard failure after the document stages must not discard completed work."""
    import io

    import corpusretrieve.pipeline.processor as processor

    fitz = pytest.importorskip("fitz")
    document = fitz.open()
    document.new_page().insert_text((72, 90), "Recoverable report", fontsize=12)
    body = document.tobytes()
    document.close()

    def explode(*args, **kwargs):
        raise MemoryError("simulated failure while embedding")

    monkeypatch.setattr(processor, "build_index", explode)

    files = [
        ("files", (f"doc{n}.pdf", io.BytesIO(body), "application/pdf")) for n in range(3)
    ]
    job_id = client.post("/v1/jobs", files=files).json()["job_id"]

    status = client.get(f"/v1/jobs/{job_id}").json()
    assert status["state"] == "failed"
    assert "MemoryError" in status["error"]
    assert "partial artifact" in status["message"]

    artifacts = client.get(f"/v1/jobs/{job_id}/artifacts").json()["artifacts"]
    paths = {item["path"] for item in artifacts}
    assert "manifest.json" in paths
    assert "bundle.zip" in paths
    assert any(p.endswith("chunks.json") for p in paths)

    manifest = client.get(f"/v1/jobs/{job_id}/manifest").json()
    assert manifest["complete"] is False
    assert "MemoryError" in manifest["error"]
    assert len(manifest["documents"]) == 3

    assert client.get(f"/v1/jobs/{job_id}/bundle").status_code == 200
    doc_id = manifest["documents"][0]["doc_id"]
    assert client.get(f"/v1/jobs/{job_id}/documents/{doc_id}/chunks").status_code == 200
