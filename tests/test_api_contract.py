"""Every endpoint published by the integration API must stay reachable."""
from __future__ import annotations

import pytest

EXPECTED_ROUTES = {
    ("GET", "/v1"),
    ("POST", "/v1/jobs"),
    ("GET", "/v1/jobs/{job_id}"),
    ("DELETE", "/v1/jobs/{job_id}"),
    ("GET", "/v1/jobs/{job_id}/events"),
    ("GET", "/v1/jobs/{job_id}/manifest"),
    ("GET", "/v1/jobs/{job_id}/documents"),
    ("GET", "/v1/jobs/{job_id}/documents/{document_id}/extraction"),
    ("GET", "/v1/jobs/{job_id}/documents/{document_id}/chunks"),
    ("GET", "/v1/jobs/{job_id}/documents/{document_id}/assets"),
    ("GET", "/v1/jobs/{job_id}/documents/{document_id}/assets/{asset_path}"),
    ("GET", "/v1/jobs/{job_id}/corpus/chunks"),
    ("GET", "/v1/jobs/{job_id}/index"),
    ("GET", "/v1/jobs/{job_id}/index/vectors"),
    ("GET", "/v1/jobs/{job_id}/index/sparse-encoder"),
    ("GET", "/v1/jobs/{job_id}/graph"),
    ("GET", "/v1/jobs/{job_id}/graph/graphml"),
    ("GET", "/v1/jobs/{job_id}/graph/cypher"),
    ("GET", "/v1/jobs/{job_id}/artifacts"),
    ("GET", "/v1/jobs/{job_id}/artifacts/download"),
    ("GET", "/v1/jobs/{job_id}/bundle"),
    ("GET", "/v1/capabilities"),
    ("GET", "/v1/health"),
}


def test_every_documented_route_is_registered(client):
    """The published OpenAPI document is the contract consumers integrate against."""
    schema = client.get("/openapi.json").json()
    registered = {
        (method.upper(), path)
        for path, operations in schema["paths"].items()
        for method in operations
        if method.lower() in {"get", "post", "put", "patch", "delete"}
    }
    assert registered >= EXPECTED_ROUTES, sorted(EXPECTED_ROUTES - registered)


def test_openapi_document_is_valid(client):
    schema = client.get("/openapi.json").json()
    assert schema["info"]["version"] == "1.0.0"
    assert "ProblemDetail" in schema["components"]["schemas"]
    upload = schema["paths"]["/v1/jobs"]["post"]
    assert upload["responses"]["415"]["content"] == {
        "application/problem+json": {
            "schema": {"$ref": "#/components/schemas/ProblemDetail"}
        }
    }


def test_health_reports_dependencies(client):
    body = client.get("/v1/health").json()
    assert body["status"] == "ok"
    assert body["checks"] == {"artifact_store": True, "job_store": True}
    assert body["vlm_api_key_configured"] is False
    assert body["console_enabled"] is True


def test_capabilities_lists_stages_and_formats(client):
    body = client.get("/v1/capabilities").json()
    assert body["stages"] == ["extract", "enrich", "chunk", "embed", "graph", "package"]
    assert ".pdf" in body["inputs"]["formats"]
    assert "hash" in body["embedders"]


def test_api_guide_points_at_the_upload_endpoint(client):
    body = client.get("/v1").json()
    assert body["how_to_upload"]["all_uploads"]["endpoint"] == "/v1/jobs"


@pytest.mark.parametrize("job_id", ["not-a-job", "0" * 23, "z" * 24])
def test_malformed_job_ids_are_rejected(client, job_id):
    response = client.get(f"/v1/jobs/{job_id}")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["code"] == "job-not-found"


def test_problem_responses_carry_a_request_id(client):
    response = client.get("/v1/jobs/" + "0" * 24)
    assert response.json()["request_id"] == response.headers["X-Request-ID"]
