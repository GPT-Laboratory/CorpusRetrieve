from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("STORAGE_BACKEND", "memory")
os.environ.setdefault("JOB_MODE", "inline")
os.environ.setdefault("DEFAULT_EMBEDDER", "hash")
os.environ.setdefault("EMIT_PAGE_PREVIEWS", "true")
os.environ.pop("ANTHROPIC_API_KEY", None)


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from corpusretrieve.api.app import create_app
    from corpusretrieve.state import get_job_store
    from corpusretrieve.storage import get_artifact_store

    get_job_store.cache_clear()
    get_artifact_store.cache_clear()
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def text_document(tmp_path: Path) -> Path:
    path = tmp_path / "report.txt"
    path.write_text(
        "Overview\n\n"
        "This document describes a measurement procedure and its outcome.\n\n"
        "Method\n\n"
        "Samples were processed in three batches of 40 units each.\n\n"
        "Results\n\n"
        "The observed throughput increased by 18 percent relative to the baseline.\n\n"
        "Conclusion\n\n"
        "The procedure is reproducible across all three batches.\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def completed_job(client, text_document):
    with text_document.open("rb") as handle:
        response = client.post(
            "/v1/jobs",
            files={"files": (text_document.name, handle, "text/plain")},
            data={"corpus_name": "test-corpus"},
        )
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    state = client.get(f"/v1/jobs/{job_id}").json()
    assert state["state"] == "completed", state.get("error")
    return job_id
