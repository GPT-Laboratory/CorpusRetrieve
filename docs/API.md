# API reference

Base URL in the examples: `http://127.0.0.1:8000`.
Interactive documentation: `/docs` (Swagger UI) and `/redoc`.

## Job lifecycle

```text
POST /v1/jobs                 -> 202 Accepted, Location: /v1/jobs/{job_id}
GET  /v1/jobs/{job_id}        -> poll until state == "completed"
GET  /v1/jobs/{job_id}/events -> or subscribe to server-sent events
GET  /v1/jobs/{job_id}/...    -> read typed outputs
DELETE /v1/jobs/{job_id}      -> remove everything stored for the job
```

`state` is one of `queued`, `processing`, `completed`, `failed`.

## POST /v1/jobs

`multipart/form-data`. One endpoint for every submission shape.

| Field | Type | Required | Default |
|---|---|---|---|
| `files` | one or more documents, or exactly one `.zip` | yes | — |
| `corpus_name` | string, 1–200 characters | no | `corpus` |
| `metadata` | JSON object encoded as text | no | `{}` |
| `document_config` | JSON object keyed by filename or archive path | no | `{}` |
| `embedder` | `hash` or `bge-m3` | no | `DEFAULT_EMBEDDER` |
| `enable_vlm` | boolean | no | `false` |
| `vlm_model` | Anthropic model ID | no | `DEFAULT_VLM_MODEL` |

Header `X-Anthropic-API-Key` is accepted and required only when `enable_vlm` is
true and no service key is configured.

`document_config` values may carry `document_id` and `metadata`:

```json
{
  "report.pdf": {
    "document_id": "report",
    "metadata": { "language": "en", "source_system": "intake" }
  }
}
```

Duplicate filenames in a direct upload are addressed as `report.pdf#2`,
`report.pdf#3`, and so on. ZIP entries may be keyed by their full archive path.

### Response

```json
{
  "job_id": "a19c25fd48a24e149b091935",
  "state": "queued",
  "submission_mode": "corpus",
  "files_received": 2,
  "files_accepted": 2,
  "warnings": [],
  "links": {
    "status": "/v1/jobs/a19c25fd48a24e149b091935",
    "events": "/v1/jobs/a19c25fd48a24e149b091935/events",
    "manifest": "/v1/jobs/a19c25fd48a24e149b091935/manifest",
    "documents": "/v1/jobs/a19c25fd48a24e149b091935/documents",
    "corpus_chunks": "/v1/jobs/a19c25fd48a24e149b091935/corpus/chunks",
    "index": "/v1/jobs/a19c25fd48a24e149b091935/index",
    "graph": "/v1/jobs/a19c25fd48a24e149b091935/graph",
    "artifacts": "/v1/jobs/a19c25fd48a24e149b091935/artifacts",
    "bundle": "/v1/jobs/a19c25fd48a24e149b091935/bundle"
  }
}
```

Follow `links` rather than constructing URLs.

## GET /v1/jobs/{job_id}

```json
{
  "job_id": "a19c25fd48a24e149b091935",
  "state": "completed",
  "stage": "completed",
  "progress": 100,
  "message": "Artifacts are ready",
  "error": null,
  "warnings": [],
  "stages": [
    {
      "name": "extract",
      "index": 0,
      "status": "done",
      "elapsed_ms": 1180,
      "detail": {
        "documents": 2,
        "pages": 24,
        "elements": 412,
        "with_bbox": 407,
        "by_kind": { "narrative": 300, "table": 6, "figure": 9 },
        "page_previews": 24
      }
    }
  ],
  "outputs_ready": true,
  "links": {}
}
```

`stages` always contains six entries in pipeline order, even before processing
starts. Stage detail keys vary by stage and are additive across releases; treat
unknown keys as informational.

## GET /v1/jobs/{job_id}/events

Server-sent events. Each `event: job` carries the same payload as the status
endpoint. The stream ends when the job completes or fails, and emits
`event: deleted` if the job disappears.

```bash
curl -N http://127.0.0.1:8000/v1/jobs/<job-id>/events
```

## Reading outputs

| Endpoint | Returns |
|---|---|
| `/v1/jobs/{id}/manifest` | Manifest JSON |
| `/v1/jobs/{id}/documents` | Document list with per-document links |
| `/v1/jobs/{id}/documents/{doc}/extraction` | Elements, provenance, page previews |
| `/v1/jobs/{id}/documents/{doc}/chunks` | Chunks for one document |
| `/v1/jobs/{id}/documents/{doc}/assets` | Asset list with download URLs |
| `/v1/jobs/{id}/documents/{doc}/assets/{path}` | One asset |
| `/v1/jobs/{id}/corpus/chunks` | Combined chunks across the corpus |
| `/v1/jobs/{id}/index` | Index metadata and file links |
| `/v1/jobs/{id}/index/vectors` | `application/x-ndjson` vector records |
| `/v1/jobs/{id}/index/sparse-encoder` | Fitted sparse encoder |
| `/v1/jobs/{id}/graph` | Knowledge graph JSON |
| `/v1/jobs/{id}/graph/graphml` | GraphML |
| `/v1/jobs/{id}/graph/cypher` | Cypher import statements |
| `/v1/jobs/{id}/artifacts` | Every generated file |
| `/v1/jobs/{id}/artifacts/download?path=` | One file by artifact path |
| `/v1/jobs/{id}/bundle` | Complete ZIP |

### A vector record

```json
{
  "id": "eff63218-220c-55e8-98eb-9e7f09dfa882",
  "composite_id": "corpus:report-ab12:section_text:report-ab12::sec::0001:hashing-v1:v1",
  "collection": "sections",
  "vector": [0.031, -0.118],
  "sparse": { "indices": [41, 902], "values": [0.62, 0.41] },
  "payload": {
    "chunk_id": "report-ab12::sec::0001",
    "doc_id": "report-ab12",
    "corpus_id": "corpus",
    "chunk_type": "section_text",
    "pages": [1, 2],
    "section_path": ["Method"],
    "raw_text": "..."
  }
}
```

`id` is a UUID5 of `composite_id`, so re-processing the same content yields the
same point ID and upserts cleanly.

## Partial results

A job that fails partway still publishes everything it produced. The state is
`failed` and `error` explains why, but `GET /v1/jobs/{id}/artifacts`,
`/manifest`, `/bundle`, and the per-document endpoints all return what
completed.

The manifest distinguishes the two cases:

| Field | Meaning |
|---|---|
| `complete` | `false` when the run ended early; the outputs are a subset |
| `error` | The failure that ended the run, or `null` |
| `documents` | Documents that produced output |
| `skipped_documents` | Documents that could not be read, with the reason |

An unreadable document does not end the job. It is recorded in
`skipped_documents`, surfaced in the job's `warnings`, and counted in the
`extract` stage's `skipped` detail. A job ends as `failed` only when no
document at all could be processed.

## Errors

`application/problem+json` with a stable `code` and a `request_id` echoed in
`X-Request-ID`.

| Status | Representative codes |
|---|---|
| 400 | `unsafe-archive-path`, `encrypted-archive-entry`, `invalid-zip-archive` |
| 404 | `job-not-found`, `document-not-found`, `artifact-not-found` |
| 409 | `artifact-not-ready`, `manifest-not-ready` |
| 410 | `artifact-expired` |
| 413 | `file-too-large`, `too-many-files`, `archive-uncompressed-limit`, `archive-entry-limit` |
| 415 | `unsupported-file-type`, `zip-required` |
| 422 | `validation-error`, `invalid-corpus-name`, `vlm-api-key-required`, `embedder-unavailable`, `zip-must-be-uploaded-alone`, `empty-file` |
| 503 | `artifact-storage-unavailable`, `job-queue-unavailable` |

A `409` means the output is not ready yet, so poll. A `410` means retention
expired and the job must be resubmitted.

## Service endpoints

`GET /v1` returns a short orientation guide.

`GET /v1/capabilities` returns supported formats, configured limits, available
embedders, VLM status, and the stage list.

`GET /v1/health` returns `200` when the artifact store and job store both
respond, `503` otherwise. Suitable as a container health check.

## Console layer

`/console` serves the browser interface. It is excluded from the OpenAPI
document, is read-only, and is disabled by `CONSOLE_ENABLED=false`. It exposes
no data that `/v1` does not already provide and is not a supported integration
surface; its shapes may change between releases.

| Endpoint | Purpose |
|---|---|
| `GET /console/settings` | Bootstrap values, including whether a service key exists |
| `GET /console/jobs/{id}` | Job status for the interface |
| `GET /console/jobs/{id}/events` | Progress stream for the interface |
| `GET /console/jobs/{id}/documents` | Corpus overview and document list |
| `GET /console/jobs/{id}/documents/{doc}/elements` | Elements with page geometry; `page` and `kind` filters |
| `GET /console/jobs/{id}/documents/{doc}/chunks` | Chunks trimmed for rendering |
| `GET /console/jobs/{id}/graph` | Graph shaped for exploration; `hide` filter |
| `GET /console/jobs/{id}/projection` | 2D projection of the vectors |
| `GET /console/jobs/{id}/artifacts` | Artifacts grouped by output family |

`GET /console/settings` returns only a boolean for the Anthropic key. The key
value is never sent to the browser.
