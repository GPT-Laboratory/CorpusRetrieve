from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CHUNKS_FILENAME = "chunks.json"


@dataclass
class DocumentRecord:
    """A lightweight handle to one processed document.

    The document stages write each document's chunks to disk. Only this
    summary is retained in memory, so peak usage is bounded by the largest
    single document rather than by the size of the corpus.
    """

    doc_id: str
    source: str
    sha256: str
    chunks: int
    metadata: dict[str, Any] = field(default_factory=dict)
    external_id: str | None = None

    def to_manifest_entry(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "sha256": self.sha256,
            "chunks": self.chunks,
            "metadata": self.metadata,
        }


@dataclass
class DocumentFailure:
    """A document that could not be processed, recorded rather than raised."""

    filename: str
    error: str
    stage: str = "extract"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CorpusRegistry:
    """Tracks processed documents and streams their chunks back from disk."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.records: list[DocumentRecord] = []
        self.failures: list[DocumentFailure] = []

    def add(self, record: DocumentRecord) -> None:
        self.records.append(record)

    def fail(self, failure: DocumentFailure) -> None:
        self.failures.append(failure)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def total_chunks(self) -> int:
        return sum(record.chunks for record in self.records)

    def document_path(self, doc_id: str) -> Path:
        return self.output_dir / "documents" / doc_id / CHUNKS_FILENAME

    def load(self, record: DocumentRecord) -> dict[str, Any]:
        """Read one document's chunk file back into memory."""
        return json.loads(self.document_path(record.doc_id).read_text(encoding="utf-8"))

    def iter_documents(self) -> Iterator[dict[str, Any]]:
        """Yield each document's chunk payload one at a time."""
        for record in self.records:
            yield self.load(record)

    def iter_chunks(self) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
        """Yield ``(chunk, document)`` pairs across the whole corpus."""
        for document in self.iter_documents():
            for chunk in document.get("chunks", []):
                yield chunk, document

    def manifest_entries(self) -> list[dict[str, Any]]:
        return [record.to_manifest_entry() for record in self.records]

    def failure_entries(self) -> list[dict[str, Any]]:
        return [failure.to_dict() for failure in self.failures]


def write_corpus_chunks(registry: CorpusRegistry, destination: Path) -> int:
    """Stream every document into one JSON array without buffering the corpus.

    The combined file keeps the array shape the API already returns, but it is
    written incrementally so a large corpus never exists in memory as a single
    serialised structure.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with destination.open("w", encoding="utf-8") as handle:
        handle.write("[")
        for index, document in enumerate(registry.iter_documents()):
            if index:
                handle.write(",")
            handle.write("\n")
            json.dump(document, handle, ensure_ascii=False)
            written += 1
        handle.write("\n]" if written else "]")
    return written
