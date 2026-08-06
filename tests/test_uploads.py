"""Upload validation, archive safety, and credential handling."""
from __future__ import annotations

import io
import zipfile


def upload(client, files, **data):
    return client.post("/v1/jobs", files=files, data=data)


def test_single_document_is_a_document_job(client, text_document):
    with text_document.open("rb") as handle:
        response = upload(client, {"files": (text_document.name, handle, "text/plain")})
    assert response.status_code == 202
    body = response.json()
    assert body["submission_mode"] == "document"
    assert body["files_accepted"] == 1
    assert response.headers["Location"] == body["links"]["status"]
    assert response.headers["Retry-After"] == "1"


def test_multiple_documents_form_a_corpus_job(client):
    files = [
        ("files", ("a.txt", io.BytesIO(b"First document body text."), "text/plain")),
        ("files", ("b.txt", io.BytesIO(b"Second document body text."), "text/plain")),
    ]
    body = client.post("/v1/jobs", files=files).json()
    assert body["submission_mode"] == "corpus"
    assert body["files_accepted"] == 2


def test_unsupported_extension_is_rejected(client):
    response = upload(client, {"files": ("payload.exe", io.BytesIO(b"MZ"), "application/octet-stream")})
    assert response.status_code == 415
    assert response.json()["code"] == "unsupported-file-type"


def test_empty_file_is_rejected(client):
    response = upload(client, {"files": ("empty.txt", io.BytesIO(b""), "text/plain")})
    assert response.status_code == 422
    assert response.json()["code"] == "empty-file"


def test_invalid_metadata_json_is_reported(client, text_document):
    with text_document.open("rb") as handle:
        response = upload(
            client,
            {"files": (text_document.name, handle, "text/plain")},
            metadata="not json",
        )
    assert response.status_code == 422
    assert response.json()["code"] == "invalid-json-field"


def test_blank_corpus_name_is_rejected(client, text_document):
    with text_document.open("rb") as handle:
        response = upload(
            client,
            {"files": (text_document.name, handle, "text/plain")},
            corpus_name="   ",
        )
    assert response.status_code == 422
    assert response.json()["code"] == "invalid-corpus-name"


def test_vlm_without_any_key_is_rejected(client, text_document):
    with text_document.open("rb") as handle:
        response = upload(
            client,
            {"files": (text_document.name, handle, "text/plain")},
            enable_vlm="true",
        )
    assert response.status_code == 422
    assert response.json()["code"] == "vlm-api-key-required"


def test_request_header_key_satisfies_the_vlm_requirement(client, text_document, monkeypatch):
    calls = {}

    def fake_describer(model, api_key, **kwargs):
        calls["model"] = model
        calls["api_key"] = api_key
        return lambda image_path, context="": ""

    monkeypatch.setattr(
        "corpusretrieve.pipeline.documents.make_anthropic_describer", fake_describer
    )
    with text_document.open("rb") as handle:
        response = client.post(
            "/v1/jobs",
            files={"files": (text_document.name, handle, "text/plain")},
            data={"enable_vlm": "true", "vlm_model": "claude-sonnet-4-6"},
            headers={"X-Anthropic-API-Key": "sk-ant-test-key"},
        )
    assert response.status_code == 202
    assert calls["api_key"] == "sk-ant-test-key"
    assert calls["model"] == "claude-sonnet-4-6"


def test_environment_key_satisfies_the_vlm_requirement(client, text_document, monkeypatch):
    from corpusretrieve.config import get_settings

    monkeypatch.setattr(
        "corpusretrieve.api.uploads.get_settings",
        lambda: _with_key(get_settings(), "sk-ant-env"),
    )
    monkeypatch.setattr(
        "corpusretrieve.worker.get_settings",
        lambda: _with_key(get_settings(), "sk-ant-env"),
    )
    seen = {}

    def fake_describer(model, api_key, **kwargs):
        seen["api_key"] = api_key
        return lambda image_path, context="": ""

    monkeypatch.setattr(
        "corpusretrieve.pipeline.documents.make_anthropic_describer", fake_describer
    )
    with text_document.open("rb") as handle:
        response = client.post(
            "/v1/jobs",
            files={"files": (text_document.name, handle, "text/plain")},
            data={"enable_vlm": "true"},
        )
    assert response.status_code == 202
    assert seen["api_key"] == "sk-ant-env"


def _with_key(settings, key):
    import dataclasses

    return dataclasses.replace(settings, anthropic_api_key=key)


def _archive(entries) -> io.BytesIO:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, payload in entries:
            archive.writestr(name, payload)
    buffer.seek(0)
    return buffer


def test_archive_upload_expands_supported_members(client):
    archive = _archive(
        [
            ("corpus/alpha.txt", "Alpha document body."),
            ("corpus/beta.md", "# Beta\n\nBeta document body."),
            ("corpus/notes.bin", "skip me"),
        ]
    )
    body = client.post(
        "/v1/jobs", files={"files": ("corpus.zip", archive, "application/zip")}
    ).json()
    assert body["submission_mode"] == "archive"
    assert body["files_accepted"] == 2
    assert body["warnings"] and "notes.bin" in body["warnings"][0]


def test_archive_cannot_be_combined_with_other_files(client):
    files = [
        ("files", ("corpus.zip", _archive([("a.txt", "body")]), "application/zip")),
        ("files", ("extra.txt", io.BytesIO(b"body"), "text/plain")),
    ]
    response = client.post("/v1/jobs", files=files)
    assert response.status_code == 422
    assert response.json()["code"] == "zip-must-be-uploaded-alone"


def test_archive_path_traversal_is_rejected(client):
    archive = _archive([("../escape.txt", "body")])
    response = client.post(
        "/v1/jobs", files={"files": ("corpus.zip", archive, "application/zip")}
    )
    assert response.status_code == 400
    assert response.json()["code"] == "unsafe-archive-path"


def test_archive_without_supported_documents_is_rejected(client):
    archive = _archive([("readme.bin", "body")])
    response = client.post(
        "/v1/jobs", files={"files": ("corpus.zip", archive, "application/zip")}
    )
    assert response.status_code == 422
    assert response.json()["code"] == "archive-has-no-supported-documents"


def test_corrupt_archive_is_rejected(client):
    response = client.post(
        "/v1/jobs",
        files={"files": ("corpus.zip", io.BytesIO(b"not a zip"), "application/zip")},
    )
    assert response.status_code == 400
    assert response.json()["code"] == "invalid-zip-archive"


def test_document_config_assigns_identifiers(client, text_document):
    with text_document.open("rb") as handle:
        response = client.post(
            "/v1/jobs",
            files={"files": (text_document.name, handle, "text/plain")},
            data={
                "document_config": '{"report.txt": {"document_id": "chosen-id",'
                ' "metadata": {"language": "en"}}}'
            },
        )
    job_id = response.json()["job_id"]
    documents = client.get(f"/v1/jobs/{job_id}/documents").json()["documents"]
    assert documents[0]["doc_id"] == "chosen-id"
    assert documents[0]["metadata"]["language"] == "en"
