"""Operating a build: persistence, VLM failure handling, timing, re-indexing."""
from __future__ import annotations

import io
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from corpusretrieve.config import Settings, valid_embedder_spec
from corpusretrieve.models import InputFile, JobSpec, JobState
from corpusretrieve.pipeline import CorpusProcessor, DocumentOptions
from corpusretrieve.pipeline.packaging import repair_outputs
from corpusretrieve.state.local import LocalJobStore
from corpusretrieve.storage.local import LocalArtifactStore
from docparse.enrich import VLMCircuit, describe_figures
from docparse.models import Element, ExtractionResult

# ------------------------------------------------------------ local storage


def test_local_artifact_store_round_trip_and_rejects_escaping_keys(tmp_path):
    store = LocalArtifactStore(Settings(local_data_dir=str(tmp_path)))
    store.put_fileobj("jobs/a/artifacts/x.bin", io.BytesIO(b"abc" * 1000))
    assert store.exists("jobs/a/artifacts/x.bin")
    assert b"".join(store.stream("jobs/a/artifacts/x.bin", chunk_size=7)) == b"abc" * 1000
    assert [item["key"] for item in store.list("jobs/a/")] == ["jobs/a/artifacts/x.bin"]
    store.delete_prefix("jobs/a/")
    assert not store.exists("jobs/a/artifacts/x.bin")
    with pytest.raises(ValueError):
        store.put_bytes("../outside", b"x")


def test_local_job_store_survives_a_restart_and_fails_interrupted_jobs(tmp_path):
    settings = Settings(local_data_dir=str(tmp_path))
    first = LocalJobStore(settings)
    spec = JobSpec(job_id="a" * 24, corpus_name="c", files=[])
    first.create(spec, JobState(job_id=spec.job_id, corpus_name="c"))
    done = JobSpec(job_id="b" * 24, corpus_name="c", files=[])
    first.create(done, JobState(job_id=done.job_id, state="completed"))
    first.update_state(spec.job_id, state="processing", progress=40)

    second = LocalJobStore(settings)                    # the server restarted
    assert second.get_spec(spec.job_id).corpus_name == "c"
    interrupted = second.get_state(spec.job_id)
    assert interrupted.state == "failed" and "Interrupted" in interrupted.message
    assert second.get_state(done.job_id).state == "completed"
    assert [s.job_id for s in second.list_states()][0] in {spec.job_id, done.job_id}


def test_storage_backend_local_is_accepted_by_validation():
    Settings(storage_backend="local").validate()
    with pytest.raises(ValueError):
        Settings(storage_backend="disk").validate()


# ------------------------------------------------------- VLM failure handling


class _ProviderError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


def _figures(tmp_path: Path, count: int) -> ExtractionResult:
    result = ExtractionResult(source="doc.pdf")
    for index in range(count):
        image = tmp_path / f"f{index}.png"
        image.write_bytes(f"image-{index}".encode())
        result.elements.append(Element(kind="figure", text="", order=index,
                                       visual_ref={"image_path": str(image)}))
    return result


def test_an_authentication_error_stops_figure_descriptions_at_once(tmp_path):
    calls = []

    def describe(path, context):
        calls.append(path)
        raise _ProviderError(401, "invalid x-api-key")

    circuit = VLMCircuit(max_consecutive=25)
    result = _figures(tmp_path, 10)
    describe_figures(result, describe, circuit=circuit, sleep=lambda s: None)
    assert len(calls) == 1                                  # no doomed calls
    assert result.stats["figures_failed"] == 1
    assert result.stats["figures_skipped_vlm_unavailable"] == 9
    assert circuit.open and circuit.reason.startswith("_ProviderError")
    # one warning per distinct error, not one per figure
    assert sum("figure description failed" in w for w in result.warnings) == 1
    assert any("stopped for this build" in w for w in result.warnings)


def test_repeated_transient_failures_open_the_circuit(tmp_path):
    sleeps = []

    def describe(path, context):
        raise _ProviderError(529, "overloaded")

    circuit = VLMCircuit(max_consecutive=3)
    result = _figures(tmp_path, 8)
    describe_figures(result, describe, circuit=circuit, max_retries=1,
                     backoff_seconds=0.01, sleep=sleeps.append)
    assert result.stats["figures_failed"] == 3
    assert result.stats["figures_skipped_vlm_unavailable"] == 5
    assert len(sleeps) == 3                                 # one retry per sent figure


def test_a_success_resets_the_consecutive_failure_count(tmp_path):
    outcomes = iter([_ProviderError(500, "x"), "ok", _ProviderError(500, "x"), "ok"])

    def describe(path, context):
        value = next(outcomes)
        if isinstance(value, Exception):
            raise value
        return value

    circuit = VLMCircuit(max_consecutive=2)
    describe_figures(_figures(tmp_path, 4), describe, circuit=circuit, max_retries=0)
    assert not circuit.open


def test_gate_does_not_count_failed_figures_twice(tmp_path):
    from corpusretrieve.pipeline.corpus import CorpusRegistry, DocumentRecord
    from corpusretrieve.pipeline.packaging import corpus_quality

    registry = CorpusRegistry(tmp_path)
    registry.add(DocumentRecord(doc_id="d", source="d.pdf", sha256="0", chunks=1, quality={
        "figures_failed": 4, "figures_skipped_vlm_unavailable": 6, "figures_undescribed": 10,
        "pages": 1, "units": 1}))
    quality = corpus_quality(registry, {}, {}, None)
    assert quality["reasons"] == ["figures_failed=4", "figures_skipped_vlm_unavailable=6"]
    assert quality["totals"]["figures_undescribed"] == 10


# ---------------------------------------------------------- stage timing


def _build(tmp_path: Path, text_document: Path, embedder: str = "hash") -> Path:
    out = tmp_path / "out"
    out.mkdir()
    spec = JobSpec(job_id="job", corpus_name="corpus", embedder=embedder, files=[
        InputFile(filename=text_document.name, storage_key=str(text_document), size_bytes=1)])
    CorpusProcessor(lambda *a: None, DocumentOptions(emit_page_previews=False)).run(
        spec, [text_document], out)
    return out


def test_each_document_stage_reports_its_own_time(tmp_path, text_document):
    out = _build(tmp_path, text_document)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["documents"][0]
    assert set(entry["timing_ms"]) == {"extract", "enrich", "chunk"}
    stages = manifest["stages"]
    assert stages["extract"]["elapsed_ms"] == entry["timing_ms"]["extract"]
    assert "loop_wall_ms" in stages["chunk"]["detail"]
    # timing stays out of the per-document artifacts, which must be reproducible
    chunks = json.loads(next((out / "documents").glob("*/chunks.json")).read_text("utf-8"))
    assert "timing_ms" not in chunks["quality"]


# ------------------------------------------------------- re-indexing


def test_reindex_swaps_the_dense_model_without_re_extracting(tmp_path, text_document):
    out = _build(tmp_path, text_document, embedder="hash")
    before = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    chunks_before = sorted(p.read_bytes() for p in (out / "documents").glob("*/chunks.json"))

    result = repair_outputs(out, actions=("index",), embedder="hash-v2")
    manifest = result["manifest"]
    assert manifest["index"]["model"] == "hashing-v2"
    assert manifest["repairs"][-1]["actions"] == ["index", "graph"]
    assert manifest["repairs"][-1]["embedder"] == "hash-v2"
    assert manifest["index"]["records"] == before["index"]["records"]
    assert sorted(p.read_bytes() for p in (out / "documents").glob("*/chunks.json")) == chunks_before
    graph = (out / "knowledge_graph" / "knowledge_graph.json").read_text(encoding="utf-8")
    assert "hashing-v2" in graph and "hashing-v1" not in graph

    from corpusretrieve.verify import verify_bundle
    assert verify_bundle(out)["ok"]


def test_reindex_cli_extracts_a_bundle_zip(tmp_path, text_document):
    from corpusretrieve.cli import main

    out = _build(tmp_path, text_document)
    target = tmp_path / "reindexed"
    assert main(["reindex", str(out / "bundle.zip"), "--embedder", "hash-v2",
                 "--out", str(target)]) == 0
    assert json.loads((target / "index" / "metadata.json").read_text())["model"] == "hashing-v2"
    assert main(["reindex", str(out / "bundle.zip"), "--embedder", "auto",
                 "--out", str(tmp_path / "x")]) == 2


# ------------------------------------------------------- semantic embedders


class _FakeModel:
    def __init__(self, name):
        self.seen: list[str] = []

    def get_sentence_embedding_dimension(self):
        return 4

    def encode(self, texts, normalize_embeddings=True):
        self.seen.extend(texts)
        return np.ones((len(texts), 4), dtype=np.float32) / 2


@pytest.fixture
def fake_sentence_transformers(monkeypatch):
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _FakeModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_e5_models_get_their_passage_and_query_prefixes(fake_sentence_transformers):
    from docparse.vectorization.embedding import resolve_embedder

    embedder = resolve_embedder("st:intfloat/multilingual-e5-small")
    assert embedder.model_id == "intfloat/multilingual-e5-small"
    assert embedder.name == "multilingual-e5-small"
    embedder.encode(["a passage"])
    embedder.encode_queries(["a question"])
    assert embedder._m.seen == ["passage: a passage", "query: a question"]


def test_consumer_reloads_the_full_model_id():
    from corpusretrieve.consumer import embedder_spec_for

    assert embedder_spec_for({"model": "hashing-v2"}) == "hash-v2"
    assert embedder_spec_for({"model": "multilingual-e5-base",
                              "model_id": "intfloat/multilingual-e5-base"}) == \
        "st:intfloat/multilingual-e5-base"
    with pytest.raises(ValueError):
        embedder_spec_for({"model": "multilingual-e5-base"})


def test_embedder_specs_accept_sentence_transformer_ids():
    assert valid_embedder_spec("st:intfloat/multilingual-e5-base")
    assert not valid_embedder_spec("st:")
    assert not valid_embedder_spec("word2vec")


# ------------------------------------------------------------- OCR binary


def test_tesseract_cmd_points_at_a_binary_off_path(tmp_path, monkeypatch):
    from docparse import ocr

    fake = tmp_path / "tesseract.exe"
    fake.write_text("")
    monkeypatch.setenv("TESSERACT_CMD", str(fake))
    assert ocr.tesseract_binary() == str(fake)
    monkeypatch.setenv("TESSERACT_CMD", str(tmp_path / "missing.exe"))
    assert ocr.tesseract_binary() is None


# --------------------------------------------------------- console sampling


def test_projection_samples_the_whole_index_evenly(client, tmp_path):
    from corpusretrieve.state import get_job_store
    from corpusretrieve.storage import get_artifact_store
    from corpusretrieve.web.routes import project_sample

    store = get_artifact_store()
    rows = 50
    lines = [json.dumps({"id": str(i), "payload": {"chunk_id": f"c{i}", "doc_id": f"d{i // 10}"}})
             for i in range(rows)]
    store.put_bytes("t/vectors.jsonl", ("\n".join(lines) + "\n").encode())
    buffer = io.BytesIO()
    np.save(buffer, np.random.default_rng(0).normal(size=(rows, 8)).astype("float32"))
    store.put_bytes("t/dense.npy", buffer.getvalue())
    store.put_bytes("t/metadata.json", json.dumps({"records": rows}).encode())
    state = JobState(job_id="c" * 24, artifacts=[
        {"path": "index/vectors.jsonl", "key": "t/vectors.jsonl"},
        {"path": "index/dense.npy", "key": "t/dense.npy"},
        {"path": "index/metadata.json", "key": "t/metadata.json"}])
    get_job_store()                                     # warm the cache like the app does
    result = project_sample(state, limit=10)
    docs = {point["doc_id"] for point in result["points"]}
    assert result["n"] == 10 and docs == {f"d{i}" for i in range(5)}   # every part of the corpus


@pytest.mark.parametrize("embedder", ["hash", "hash-v2"])
def test_every_listed_embedder_is_accepted_by_the_upload_form(client, text_document, embedder):
    with text_document.open("rb") as handle:
        response = client.post("/v1/jobs", files={"files": (text_document.name, handle, "text/plain")},
                               data={"embedder": embedder})
    assert response.status_code == 202, response.text
    status = client.get(f"/v1/jobs/{response.json()['job_id']}").json()
    assert status["state"] == "completed"


def test_an_uninstalled_embedder_is_reported_by_name(client, text_document):
    from importlib.util import find_spec

    if find_spec("sentence_transformers") is not None:
        pytest.skip("sentence-transformers is installed here")
    with text_document.open("rb") as handle:
        response = client.post("/v1/jobs", files={"files": (text_document.name, handle, "text/plain")},
                               data={"embedder": "st:no-such/model-xyz"})
    assert response.status_code == 422
    assert response.json()["code"] in {"embedder-unavailable"}


def test_a_model_that_cannot_load_stops_the_job_before_extraction(tmp_path, text_document,
                                                                   monkeypatch):
    from corpusretrieve.pipeline.processor import EmbedderUnavailable

    def broken(name):
        raise OSError("model not found")

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = broken
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    spec = JobSpec(job_id="job", corpus_name="c", embedder="st:nope/missing", files=[
        InputFile(filename=text_document.name, storage_key=str(text_document), size_bytes=1)])
    stages = []
    with pytest.raises(EmbedderUnavailable, match="nope/missing"):
        CorpusProcessor(lambda *a: stages.append(a), DocumentOptions(emit_page_previews=False)).run(
            spec, [text_document], tmp_path)
    assert not (tmp_path / "documents").exists()
