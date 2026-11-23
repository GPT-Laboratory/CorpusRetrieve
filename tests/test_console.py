"""The browser console serves view models derived from the same stored artifacts."""
from __future__ import annotations

import pytest


def test_settings_expose_only_the_key_presence_flag(client):
    payload = client.get("/console/settings").json()
    assert payload["vlm_api_key_configured"] is False
    assert payload["stages"] == ["extract", "enrich", "chunk", "embed", "graph", "package"]
    assert "anthropic_api_key" not in payload
    assert not any("sk-ant" in str(value) for value in payload.values())


def test_documents_view_lists_every_processed_document(client, completed_job):
    payload = client.get(f"/console/jobs/{completed_job}/documents").json()
    assert payload["job_id"] == completed_job
    assert payload["documents"]
    assert {"doc_id", "source"} <= set(payload["documents"][0])


def test_chunk_view_carries_provenance_fields(client, completed_job):
    documents = client.get(f"/console/jobs/{completed_job}/documents").json()["documents"]
    doc_id = documents[0]["doc_id"]
    payload = client.get(
        f"/console/jobs/{completed_job}/documents/{doc_id}/chunks"
    ).json()
    assert payload["chunks"]
    chunk = payload["chunks"][0]
    for field in ("chunk_id", "chunk_type", "pages", "element_ids", "token_estimate"):
        assert field in chunk


def test_element_view_supports_page_and_kind_filters(client, completed_job):
    documents = client.get(f"/console/jobs/{completed_job}/documents").json()["documents"]
    doc_id = documents[0]["doc_id"]
    base = f"/console/jobs/{completed_job}/documents/{doc_id}/elements"

    everything = client.get(base).json()
    assert everything["count"] == len(everything["elements"])

    filtered = client.get(base, params={"kind": "narrative"}).json()
    assert all(item["kind"] == "narrative" for item in filtered["elements"])
    assert filtered["count"] <= everything["count"]


def test_graph_view_draws_one_document_with_a_stable_layout(client, completed_job):
    documents = client.get(f"/console/jobs/{completed_job}/documents").json()["documents"]
    doc_id = documents[0]["doc_id"]
    first = client.get(f"/console/jobs/{completed_job}/graph", params={"doc_id": doc_id}).json()
    again = client.get(f"/console/jobs/{completed_job}/graph", params={"doc_id": doc_id}).json()
    assert first["doc_id"] == doc_id
    assert first["n_nodes"] == len(first["nodes"]) and first["nodes"]
    assert [(n["x"], n["y"]) for n in first["nodes"]] == [(n["x"], n["y"]) for n in again["nodes"]]
    paper = next(n for n in first["nodes"] if n["type"] == "Paper")
    assert (paper["x"], paper["y"]) == (0.0, 0.0) and paper["depth"] == 0
    # every node id belongs to this document or is a shared unit/vector node
    ids = {n["id"] for n in first["nodes"]}
    assert all(e["source"] in ids and e["target"] in ids for e in first["edges"])
    # chunk nodes carry their chunk id so the workbench can highlight them
    chunk_ids = {n["chunk_id"] for n in first["nodes"] if n["chunk_id"]}
    chunks = client.get(f"/console/jobs/{completed_job}/documents/{doc_id}/chunks").json()
    assert chunk_ids and chunk_ids <= {c["chunk_id"] for c in chunks["chunks"]}
    # labels are readable text, never a bare hash
    for node in first["nodes"]:
        if node["type"] == "Chunk":
            assert " " in node["label"] or len(node["label"]) < 12
    assert "VectorPoint" in first["default_hidden"]
    assert first["corpus"]["nodes"] >= first["total_nodes"]


def test_graph_view_rejects_an_unknown_document(client, completed_job):
    response = client.get(f"/console/jobs/{completed_job}/graph", params={"doc_id": "nope"})
    assert response.status_code == 404


def test_recent_jobs_are_listed_for_reopening(client, completed_job):
    jobs = client.get("/console/jobs").json()["jobs"]
    entry = next(job for job in jobs if job["job_id"] == completed_job)
    assert entry["state"] == "completed" and entry["artifacts_available"] is True


def test_projection_returns_normalized_coordinates(client, completed_job):
    payload = client.get(f"/console/jobs/{completed_job}/projection").json()
    assert payload["n"] == len(payload["points"])
    for point in payload["points"]:
        assert 0.0 <= point["x"] <= 1.0
        assert 0.0 <= point["y"] <= 1.0


def test_artifacts_are_grouped_for_the_download_panel(client, completed_job):
    payload = client.get(f"/console/jobs/{completed_job}/artifacts").json()
    names = [group["name"] for group in payload["artifact_groups"]]
    assert "Index" in names and "Knowledge graph" in names
    assert payload["bundle"] == f"/v1/jobs/{completed_job}/bundle"
    assert payload["bundle_size_bytes"] > 0
    for group in payload["artifact_groups"]:
        assert group["count"] >= len(group["items"])
        for item in group["items"]:
            assert item["download"].startswith("/v1/jobs/")
    # the corpus-level files the download panel features are listed in full
    paths = {item["path"] for item in payload["artifacts"]}
    assert {"index/vectors.jsonl", "manifest.json"} <= paths
    assert not any(path.startswith("documents/") for path in paths)


@pytest.mark.parametrize("job_id", ["not-a-job", "0" * 24])
def test_unknown_jobs_return_a_problem_document(client, job_id):
    response = client.get(f"/console/jobs/{job_id}/documents")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/problem+json")


def test_interface_files_are_revalidated_after_an_upgrade(client):
    for path in ("/", "/static/app.jsx", "/static/styles.css"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-cache"


def test_a_job_that_fails_partway_still_shows_and_downloads_its_outputs(
        client, text_document, monkeypatch):
    """A crash after the documents are processed must not hide them."""
    import corpusretrieve.pipeline.processor as processor

    def broken_index(*args, **kwargs):
        raise RuntimeError("simulated failure while embedding")

    monkeypatch.setattr(processor, "build_index", broken_index)
    with text_document.open("rb") as handle:
        job = client.post("/v1/jobs", files={"files": (text_document.name, handle,
                                                        "text/plain")}).json()["job_id"]
    status = client.get(f"/v1/jobs/{job}").json()
    assert status["state"] == "failed" and "simulated failure" in status["error"]
    assert status["artifacts_available"] is True and status["release_ready"] is False

    overview = client.get(f"/console/jobs/{job}/documents").json()
    assert overview["complete"] is False and overview["documents"]
    doc_id = overview["documents"][0]["doc_id"]
    chunks = client.get(f"/console/jobs/{job}/documents/{doc_id}/chunks").json()
    assert chunks["chunks"]
    graph = client.get(f"/console/jobs/{job}/graph", params={"doc_id": doc_id}).json()
    assert graph["nodes"]
    artifacts = client.get(f"/console/jobs/{job}/artifacts").json()
    assert artifacts["bundle"] == f"/v1/jobs/{job}/bundle"
    bundle = client.get(f"/v1/jobs/{job}/bundle")
    assert bundle.status_code == 200 and bundle.content[:2] == b"PK"
