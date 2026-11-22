"""Service and end-to-end regression tests.

These reproduce the paper's failures at the level where they happened: across
stages, through the API, and in the bundle a consumer receives.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from corpusretrieve.config import VERSION
from corpusretrieve.consumer import PortableIndex, recall_at_k
from corpusretrieve.models import InputFile, JobSpec
from corpusretrieve.pipeline import CorpusProcessor, DocumentOptions
from corpusretrieve.pipeline.packaging import repair_outputs
from corpusretrieve.verify import verify_bundle

fitz = pytest.importorskip("fitz")
sys.path.insert(0, str(Path(__file__).parent))
from fixtures import bilingual_report, blank_pdf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RECORD = {"title": "Measurement methods in a laboratory", "authors": ["Virtanen, A."],
          "year": 2015, "handle_url": "http://urn.fi/URN:NBN:fi-fe0001"}


# ------------------------------------------------------------------ helpers

def _build(tmp: Path, files: list[Path], metadata: list[dict] | None = None,
           **options) -> tuple[dict, Path]:
    out = tmp / "out"
    out.mkdir(exist_ok=True)
    metadata = metadata or [{} for _ in files]
    spec = JobSpec(job_id="job", corpus_name="corpus",
                   files=[InputFile(filename=f.name, storage_key=str(f), size_bytes=1,
                                    metadata=m) for f, m in zip(files, metadata, strict=True)])
    opts = DocumentOptions(**{"emit_page_previews": False, "ocr_langs": "eng", **options})
    result = CorpusProcessor(lambda *a: None, opts).run(spec, files, out)
    return result, out


def _vectors(path: Path) -> list[dict]:
    if path.is_file():
        lines = zipfile.ZipFile(path).read("index/vectors.jsonl").decode().splitlines()
    else:
        lines = (path / "index/vectors.jsonl").read_text().splitlines()
    return [json.loads(x) for x in lines if x.strip()]


def _upload(client, files, **data):
    payload = [("files", (name, io.BytesIO(body), "application/pdf")) for name, body in files]
    return client.post("/v1/jobs", files=payload, data=data)


@pytest.fixture(scope="module")
def report_bytes(tmp_path_factory) -> bytes:
    return bilingual_report(tmp_path_factory.mktemp("r") / "r.pdf").read_bytes()


# ------------------------------------- golden end-to-end provenance

def test_golden_provenance_chain_from_record_to_payload(client, report_bytes):
    config = {"sample_2015.pdf": {"document_id": "sample-2015", "metadata": {"records": [RECORD]}}}
    job = _upload(client, [("sample_2015.pdf", report_bytes)],
                  document_config=json.dumps(config)).json()["job_id"]
    status = client.get(f"/v1/jobs/{job}").json()
    assert status["state"] == "completed", status

    manifest = client.get(f"/v1/jobs/{job}/manifest").json()
    assert manifest["release_ready"] is True and manifest["quality"]["reasons"] == []

    extraction = client.get(f"/v1/jobs/{job}/documents/sample-2015/extraction").json()
    text = "\n\n".join(e["text"] for e in extraction["elements"] if e.get("text"))
    records = [json.loads(x) for x in
               client.get(f"/v1/jobs/{job}/index/vectors").text.splitlines() if x.strip()]
    assert records
    for rec in records:
        p = rec["payload"]
        # bibliographic provenance: every unit resolves to its record
        assert p["bibliographic"]["title"] == RECORD["title"]
        assert p["bibliographic"]["handle_url"] == RECORD["handle_url"]
        # element provenance inline: page + box without reading element files
        assert p["pages"] and any(loc.get("bbox") for loc in p["locators"])
        if p["chunk_type"] == "section_text":
            for a, b in p["spans"]:
                assert text[a:b] in p["raw_text"]
    claims = [r for r in records if r["payload"]["chunk_type"] == "claim"]
    assert len(claims) == 2


# ----------------------------------------------- intake and release gate

def test_unmatched_document_config_is_rejected_before_processing(client, report_bytes):
    config = {"SAMPLE_2015.pdf": {"metadata": {"records": [RECORD]}}}     # wrong case
    response = _upload(client, [("sample_2015.pdf", report_bytes)],
                       document_config=json.dumps(config))
    assert response.status_code == 422
    problem = response.json()
    assert problem["code"] == "intake-check-failed"
    checks = {e["check"]: e for e in problem["errors"]}
    assert checks["unmatched_config_keys"]["examples"] == ["SAMPLE_2015.pdf"]
    assert checks["files_without_document_config"]["count"] == 1

    # accepted on request: the build runs and the gate records the same findings
    job = _upload(client, [("sample_2015.pdf", report_bytes)], allow_degraded="true",
                  document_config=json.dumps(config)).json()["job_id"]
    status = client.get(f"/v1/jobs/{job}").json()
    assert status["state"] == "completed" and status["release_ready"] is False
    assert "unmatched_config_keys=1" in status["quality_reasons"]
    assert any("matched no file" in w for w in status["warnings"])


def test_case_insensitive_filename_collision_is_detected(client, report_bytes):
    files = [("Report.pdf", report_bytes), ("report.pdf", report_bytes + b"\n%x")]
    response = _upload(client, files)
    assert response.status_code == 422
    assert response.json()["errors"][0]["check"] == "filename_collisions"

    accepted = _upload(client, files, allow_degraded="true")
    assert any("collision" in w for w in accepted.json()["warnings"])
    manifest = client.get(f"/v1/jobs/{accepted.json()['job_id']}/manifest").json()
    ids = [d["doc_id"] for d in manifest["documents"]]
    assert len(ids) == 2 and len(set(ids)) == 2         # both kept, distinct IDs


def test_missing_ocr_language_makes_the_build_degraded(tmp_path):
    path = blank_pdf(tmp_path / "scan.pdf", pages=2)
    result, _ = _build(tmp_path, [path], ocr_langs="eng+zzz")
    quality = result["manifest"]["quality"]
    assert quality["release_ready"] is False
    assert "ocr_degraded_pages=2" in quality["reasons"]
    assert result["manifest"]["documents"][0]["quality"]["degraded"] is True


# ----------------------------------------------------------- auth & CORS

def test_api_keys_protect_every_job_route():
    from fastapi.testclient import TestClient

    from corpusretrieve.api.app import create_app
    from corpusretrieve.config import Settings

    app = create_app(Settings(api_keys=("s3cret",)))
    with TestClient(app) as c:
        assert c.get("/v1/health").status_code == 200
        assert c.get("/v1/capabilities").status_code == 200
        denied = c.get("/v1/jobs/abc")
        assert denied.status_code == 401
        assert denied.headers["content-type"].startswith("application/problem+json")
        assert denied.json()["code"] == "authentication-required"
        assert c.delete("/v1/jobs/abc").status_code == 401
        assert c.get("/v1/jobs/abc", headers={"Authorization": "Bearer nope"}).status_code == 401
        assert c.get("/v1/jobs/abc", headers={"Authorization": "Bearer s3cret"}).status_code == 404
        assert c.get("/v1/jobs/abc", headers={"X-API-Key": "s3cret"}).status_code == 404


def test_cors_allows_no_foreign_origin_by_default():
    from corpusretrieve.config import Settings
    assert Settings().cors_origins == ()


# ------------------------------------------ bundle sealing and verification

@pytest.fixture
def built(tmp_path, report_bytes):
    src = tmp_path / "report.pdf"
    src.write_bytes(report_bytes)
    return _build(tmp_path, [src], [{"records": [RECORD]}])


def test_manifest_records_environment_quality_and_bundle_hash(built):
    result, out = built
    m = result["manifest"]
    assert m["schema_version"] == "2.0.0" and m["version"] == VERSION
    assert m["environment"]["tesseract"]["installed_langs"]
    assert "pymupdf" in m["environment"]["packages"]
    sidecar = (out / "bundle.zip.sha256").read_text().split()[0]
    assert sidecar == hashlib.sha256((out / "bundle.zip").read_bytes()).hexdigest()
    assert result["bundle_sha256"] == sidecar


def test_dense_matrix_is_row_aligned_with_vector_records(built):
    _, out = built
    dense = np.load(out / "index/dense.npy")
    records = _vectors(out)
    assert dense.shape == (len(records), 512) and dense.dtype == np.float32
    for rec in records:
        assert np.allclose(dense[rec["row"]], rec["vector"], atol=1e-6)


def test_verify_accepts_a_sealed_bundle_and_detects_tampering(built):
    _, out = built
    assert verify_bundle(out / "bundle.zip")["ok"] is True
    assert verify_bundle(out)["ok"] is True
    (out / "index/sparse_encoder.json").write_text("{}")
    report = verify_bundle(out)
    assert not report["ok"] and "hash mismatch: index/sparse_encoder.json" in report["errors"]


def test_uncatalogued_file_is_an_error(built):
    _, out = built
    (out / "knowledge_graph/extra.json").write_text("{}")
    assert "file not in catalog: knowledge_graph/extra.json" in verify_bundle(out)["errors"]


def test_repair_regenerates_exports_and_reseals_the_catalog(built):
    """Regenerated graph exports must end up inside the catalog."""
    _, out = built
    (out / "knowledge_graph/knowledge_graph.graphml").unlink()
    assert not verify_bundle(out)["ok"]
    result = repair_outputs(out)
    assert (out / "knowledge_graph/knowledge_graph.graphml").exists()
    assert verify_bundle(out)["ok"] is True
    assert verify_bundle(out / "bundle.zip")["ok"] is True
    repair = result["manifest"]["repairs"][-1]
    assert repair["tool"] == "corpusretrieve.repair" and repair["version"] == VERSION


def test_cli_verify_exit_codes(built):
    from corpusretrieve.cli import main
    _, out = built
    assert main(["verify", str(out / "bundle.zip")]) == 0
    (out / "manifest.json").write_text("{}")
    assert main(["verify", str(out)]) == 1


# --------------------------------------------------------- resume

def test_rebuilding_into_the_same_directory_reuses_documents(tmp_path, report_bytes):
    src = tmp_path / "report.pdf"
    src.write_bytes(report_bytes)
    first, _ = _build(tmp_path, [src])
    second, _ = _build(tmp_path, [src])
    assert first["manifest"]["documents"][0]["resumed"] is False
    assert second["manifest"]["documents"][0]["resumed"] is True
    assert first["manifest"]["documents"][0]["doc_id"] == second["manifest"]["documents"][0]["doc_id"]


# ---------------------------------------------------- consumer

def test_reference_consumer_finds_the_finnish_table(built):
    _, out = built
    index = PortableIndex.load(out / "bundle.zip")
    hits = index.search("mittaukset toimipaikoittain", k=3)
    assert hits and hits[0].payload["chunk_type"].startswith("table")
    assert all(h.payload["retrieval_default"] != "exclude" for h in hits)
    only_tables = index.search("mittaus", k=5, collections=["tables"])
    assert {h.payload["chunk_type"] for h in only_tables} <= {"table_summary", "table_row_group"}


def test_consumer_loads_older_bundles_without_dense_matrix(built, tmp_path):
    _, out = built
    legacy = tmp_path / "legacy"
    shutil.copytree(out, legacy, ignore=shutil.ignore_patterns("bundle.zip*", "dense.npy"))
    meta = json.loads((legacy / "index/metadata.json").read_text())
    meta.pop("dense_file")
    (legacy / "index/metadata.json").write_text(json.dumps(meta))
    a = PortableIndex.load(out).search("mittaukset", k=5)
    b = PortableIndex.load(legacy).search("mittaukset", k=5)
    assert [h.id for h in a] == [h.id for h in b]
    assert recall_at_k([h.id for h in a], [h.id for h in b]) == 1.0


# ------------------------------------------- determinism end to end

_BUILD_SCRIPT = """
import sys
from pathlib import Path
from corpusretrieve.models import InputFile, JobSpec
from corpusretrieve.pipeline import CorpusProcessor, DocumentOptions
src, out = Path(sys.argv[1]), Path(sys.argv[2])
spec = JobSpec(job_id="j", corpus_name="c",
               files=[InputFile(filename=src.name, storage_key="k", size_bytes=1)])
CorpusProcessor(lambda *a: None, DocumentOptions(emit_page_previews=False,
                ocr_langs="eng")).run(spec, [src], out)
"""


def test_two_builds_under_different_hash_seeds_are_identical(tmp_path, report_bytes):
    src = tmp_path / "report.pdf"
    src.write_bytes(report_bytes)
    digests = []
    for seed in ("0", "4242"):
        out = tmp_path / f"out{seed}"
        out.mkdir()
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(ROOT),
               "OCR_LANGS": "eng"}
        subprocess.run([sys.executable, "-c", _BUILD_SCRIPT, str(src), str(out)],
                       env=env, check=True, capture_output=True)
        files = ["index/vectors.jsonl", "index/sparse_encoder.json", "index/dense.npy",
                 "knowledge_graph/knowledge_graph.json"]
        docs = sorted((out / "documents").glob("*/chunks.json"))
        digests.append({f: hashlib.sha256((out / f).read_bytes()).hexdigest() for f in files}
                       | {d.parent.name: hashlib.sha256(d.read_bytes()).hexdigest()
                          for d in docs})
    assert digests[0] == digests[1]
