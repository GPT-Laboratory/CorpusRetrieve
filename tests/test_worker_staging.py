"""Short internal staging paths must preserve external source provenance."""
from __future__ import annotations

from pathlib import Path

from corpusretrieve.models import InputFile, JobSpec
from corpusretrieve.worker import _materialize_inputs


class _Store:
    def get_bytes(self, key: str) -> bytes:
        return f"PDF contents for {key}".encode()


def test_long_source_filename_has_short_internal_path(tmp_path: Path) -> None:
    filename = "long_research_publication_filename_" + "measurement_" * 22 + ".pdf"
    spec = JobSpec(
        job_id="sample-job",
        corpus_name="neutral-corpus",
        files=[InputFile(filename=filename, storage_key="source/a", size_bytes=11)],
    )
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    staged = _materialize_inputs(_Store(), spec, inputs)
    assert len(staged) == 1
    assert staged[0].name == "0000.pdf"
    assert staged[0].read_bytes() == b"PDF contents for source/a"
    assert spec.files[0].filename == filename  # original identity remains intact


def test_staging_many_same_basename_documents_without_collision(tmp_path: Path) -> None:
    files = [InputFile(filename="same.pdf", storage_key=f"source/{n}", size_bytes=1)
             for n in range(2)]
    spec = JobSpec(job_id="sample-job", corpus_name="neutral-corpus", files=files)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    paths = _materialize_inputs(_Store(), spec, inputs)
    assert [p.name for p in paths] == ["0000.pdf", "0001.pdf"]
    assert paths[0].read_bytes() != paths[1].read_bytes()
