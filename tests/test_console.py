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


def test_graph_view_hides_bookkeeping_nodes_by_default(client, completed_job):
    payload = client.get(f"/console/jobs/{completed_job}/graph").json()
    assert "VectorPoint" in payload["hidden_types"]
    assert all(node["type"] != "VectorPoint" for node in payload["nodes"])
    assert payload["n_nodes"] == len(payload["nodes"])

    unfiltered = client.get(
        f"/console/jobs/{completed_job}/graph", params={"hide": ""}
    ).json()
    assert unfiltered["n_nodes"] >= payload["n_nodes"]


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
    for group in payload["artifact_groups"]:
        for item in group["items"]:
            assert item["download"].startswith("/v1/jobs/")


@pytest.mark.parametrize("job_id", ["not-a-job", "0" * 24])
def test_unknown_jobs_return_a_problem_document(client, job_id):
    response = client.get(f"/console/jobs/{job_id}/documents")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/problem+json")
