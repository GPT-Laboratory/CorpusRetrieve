# CorpusRetrieve

**Version 2.0.0 — consolidated production baseline**

CorpusRetrieve converts one document or an entire corpus into portable,
provenance-preserving retrieval artifacts: typed chunks, dense and sparse
vectors, a knowledge graph, and a manifest that ties every output back to its
source region.

It is domain-neutral. Nothing in the pipeline is specialised for a particular
subject area, industry, or document collection.

CorpusRetrieve produces evidence. It does not generate answers. A downstream
application consumes these artifacts, builds its own prompt or query, and calls
whatever model it chooses.

---

## Contents

- [What you get](#what-you-get)
- [Quick start](#quick-start)
- [Web interface](#web-interface)
- [Anthropic API key](#anthropic-api-key)
- [Integration API](#integration-api)
- [Deployment](#deployment)
- [Configuration](#configuration)
- [Repository layout](#repository-layout)
- [Testing](#testing)
- [Security defaults](#security-defaults)
- [Documentation](#documentation)

---

## What you get

Every completed job produces a downloadable ZIP with this structure:

```text
manifest.json
artifact_catalog.json
corpus/
  chunks.json
documents/<document-id>/
  extraction.json
  chunks.json
  assets/
    <figure crops>
    pages/page-1.png
index/
  vectors.jsonl
  dense.npy
  sparse_encoder.json
  metadata.json
knowledge_graph/
  knowledge_graph.json
  knowledge_graph.graphml
  knowledge_graph.cypher
```

`index/vectors.jsonl` holds a stable point ID, a dense vector, a sparse vector,
a logical collection, raw and canonical text, document identity, and source
provenance on every line. It is database-neutral: load it into any vector store
without editing it by hand.

### The six stages

Processing runs as six observable stages. Each reports its own status, timing,
and counters, which the API returns in `stages[]` and the web interface renders
as a live spine.

| # | Stage | Produces |
|---|---|---|
| 1 | `extract` | Typed elements with page, bounding box, and character offsets |
| 2 | `enrich` | Roles, claims, citations, and optional figure descriptions |
| 3 | `chunk` | Structure-aware typed chunks with provenance |
| 4 | `embed` | Dense and sparse vectors plus a fitted sparse encoder |
| 5 | `graph` | Knowledge graph in JSON, GraphML, and Cypher |
| 6 | `package` | Manifest, hashed artifact catalog, and ZIP bundle |

---

## Building a large corpus (resumable, additive)

```bash
python tools/manifest_to_document_config.py manifest.json -o config.json \
    --files-root downloads/ --report conversion_report.json
corpusretrieve build downloads/ --out build/ --config config.json
# add files or update the manifest, then run the same command again:
# unchanged files are reused, only new ones are parsed
```

## Verifying and consuming a bundle

```bash
corpusretrieve verify bundle.zip --require-release-ready   # 0 ok, 1 invalid, 2 not release-ready
corpusretrieve repair outputs/                              # rebuild graph exports, re-seal catalog
```

```python
from corpusretrieve.consumer import PortableIndex
index = PortableIndex.load("bundle.zip")
for hit in index.search("material performance", k=5):
    print(hit.score, hit.payload["bibliographic"], hit.payload["locators"][0])
```

For a bilingual corpus with a lexical index, pass the question in the other
language too, from any translator you choose:

```python
hits = index.search("material performance", k=10,
                    extra_queries=["materials performance"])
# or: index.search(q, translator=my_translate)  # returns a list of query strings
```

Every manifest carries a `quality` block and `release_ready`. With the default
`DEGRADED_POLICY=fail`, a build with missing OCR, failed figure descriptions,
filename collisions or unmatched `document_config` keys is published but marked
failed, and the web interface still shows and offers every output. Problems
that would fail the gate whatever the build produces (unmatched
`document_config` keys, name collisions) are rejected at upload. See
`CHANGELOG.md` for the 2.0.0 changes.

### Long builds on one machine

```bash
STORAGE_BACKEND=local LOCAL_DATA_DIR=./corpusretrieve-data \
  python -m uvicorn service.app:app --port 8000
```

Outputs and job status are written to disk, survive restarts, and reopen from
the start page or `http://localhost:8000/#job/<job id>`. On Windows, set
`TESSERACT_CMD` to `tesseract.exe` if Tesseract is not on `PATH`.

### Semantic embeddings

`pip install -r requirements-embeddings.txt` enables multilingual E5 and
BGE-M3. `corpusretrieve reindex <bundle.zip> --embedder
st:intfloat/multilingual-e5-small --out <dir>` adds them to a finished build
without re-extracting it. See [docs/EMBEDDINGS.md](docs/EMBEDDINGS.md).

---

## Quick start

Local development, no external services:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate
python -m pip install --upgrade pip 
pip install -e .

cp .env.example .env               # Windows: Copy-Item .env.example .env
```

Set these three values in `.env` for a local run:

```dotenv
STORAGE_BACKEND=memory
JOB_MODE=inline
DEFAULT_EMBEDDER=hash
```

Run it:

```bash
python -m uvicorn service.app:app --reload --port 8000
```

| Interface | URL |
|---|---|
| Web interface | `http://127.0.0.1:8000` |
| Swagger UI | `http://127.0.0.1:8000/docs` |
| ReDoc | `http://127.0.0.1:8000/redoc` |
| Health | `http://127.0.0.1:8000/v1/health` |

The memory backends hold everything in the API process. They are intended for
development and tests only; use S3 and Redis for deployment.

Verify a running service end to end:

```bash
python tools/api_smoke_test.py
python tools/api_smoke_test.py mydoc.pdf --delete-job
```

---

## Web interface

The interface is served from the same origin as the API and reads the same
stored artifacts through a read-only console layer. It has three modes:

**Inspect** is the workbench. A live six-stage spine, a filterable chunk list,
and a centre panel with four views:

- *Source* renders the page and draws bounding boxes over it. Select a chunk to
  highlight exactly the regions it came from.
- *Chunks* is a typed span map where bar width is token count and colour is
  chunk type.
- *Graph* is an interactive knowledge-graph explorer with type legends, node
  inspection, and PNG export.
- *Embedding* projects the vectors into 2D so semantic neighbourhoods are
  visible.

A document bar switches between documents in a multi-document corpus.

**Explain** walks through what happened to the corpus in plain language, with
the real numbers from the run.

**Learn** is a scrollytelling narrative with an animated visualization and a
chunk-size sandbox.

The interface uses vendored React and Babel under
`corpusretrieve/web/static/vendor/`. There is no Node build step and no
outbound request at runtime.

---

## Anthropic API key

Figure description with a vision model is optional and off by default. When it
is enabled, the key is resolved in this order:

1. The `X-Anthropic-API-Key` request header, which the web interface sends when
   you type a key into the configuration panel.
2. `ANTHROPIC_API_KEY` from the environment or the `.env` file.

### Setting the key in `.env`

```dotenv
ANTHROPIC_API_KEY=sk-ant-...
DEFAULT_VLM_MODEL=claude-sonnet-4-6
```

Docker Compose passes this to both the API and the worker through `env_file`.

### Setting the key in the web interface

Enable **Describe figures with a vision model** in the configuration panel. A
password field appears, with a show/hide toggle and a model override. The hint
below the field states whether a service key is already configured, so you know
whether the field is required or an override.

The browser only ever receives a boolean saying whether a service key exists.
The key itself is never sent to the browser, written to artifacts, recorded in
manifests, or included in logs. In queue mode, a per-request key is encrypted
with Fernet, stored in Redis under a short TTL, read exactly once by the worker,
and deleted immediately. Error messages are scanned and the key redacted before
being stored on the job.

Enabling figure description without any key available returns a `422` with code
`vlm-api-key-required` rather than failing later during processing.

---

## Integration API

Every endpoint is versioned under `/v1` and documented at `/docs`.

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/v1` | API quick-start guide |
| `POST` | `/v1/jobs` | Upload one document, several documents, or one ZIP |
| `GET` | `/v1/jobs/{job_id}` | Job status, progress, and stage detail |
| `GET` | `/v1/jobs/{job_id}/events` | Server-sent progress events |
| `GET` | `/v1/jobs/{job_id}/manifest` | Output manifest |
| `GET` | `/v1/jobs/{job_id}/documents` | Processed documents and resource links |
| `GET` | `/v1/jobs/{job_id}/documents/{doc_id}/extraction` | Elements and provenance |
| `GET` | `/v1/jobs/{job_id}/documents/{doc_id}/chunks` | Chunks for one document |
| `GET` | `/v1/jobs/{job_id}/documents/{doc_id}/assets` | Generated assets |
| `GET` | `/v1/jobs/{job_id}/documents/{doc_id}/assets/{path}` | Download one asset |
| `GET` | `/v1/jobs/{job_id}/corpus/chunks` | Combined corpus chunks |
| `GET` | `/v1/jobs/{job_id}/index` | Index metadata and file links |
| `GET` | `/v1/jobs/{job_id}/index/vectors` | Dense and sparse vector records |
| `GET` | `/v1/jobs/{job_id}/index/sparse-encoder` | Fitted sparse encoder |
| `GET` | `/v1/jobs/{job_id}/graph` | Knowledge graph as JSON |
| `GET` | `/v1/jobs/{job_id}/graph/graphml` | Knowledge graph as GraphML |
| `GET` | `/v1/jobs/{job_id}/graph/cypher` | Cypher import statements |
| `GET` | `/v1/jobs/{job_id}/artifacts` | List every generated file |
| `GET` | `/v1/jobs/{job_id}/artifacts/download` | Download one file by path |
| `GET` | `/v1/jobs/{job_id}/bundle` | Download all outputs as ZIP |
| `DELETE` | `/v1/jobs/{job_id}` | Delete the job and its stored objects |
| `GET` | `/v1/capabilities` | Formats, limits, embedders, VLM status, stages |
| `GET` | `/v1/health` | Dependency health and release version |

A separate read-only console layer under `/console` serves the browser-specific
view models. It is excluded from the OpenAPI document, can be disabled with
`CONSOLE_ENABLED=false`, and adds no capability that `/v1` does not already
expose. See [docs/API.md](docs/API.md).

### Upload

One multipart endpoint handles every submission shape. `submission_mode` is set
automatically to `document`, `corpus`, or `archive`.

```bash
curl -X POST http://127.0.0.1:8000/v1/jobs \
  -F "files=@report.pdf" \
  -F "corpus_name=release-corpus" \
  -F 'metadata={"owner":"platform-team"}' \
  -F 'document_config={"report.pdf":{"document_id":"report","metadata":{"language":"en"}}}' \
  -F "embedder=hash"
```

Several documents: repeat `files`. A whole folder: send its files together, or
ZIP it and send the ZIP by itself. Do not mix a ZIP with other files.

| Field | Type | Required | Default |
|---|---|---|---|
| `files` | one or more documents, or exactly one ZIP | yes | — |
| `corpus_name` | string, 1–200 characters | no | `corpus` |
| `metadata` | corpus-level JSON object as text | no | `{}` |
| `document_config` | JSON object keyed by filename or archive path | no | `{}` |
| `embedder` | `hash`, or `bge-m3` when installed | no | configured default |
| `enable_vlm` | boolean | no | `false` |
| `vlm_model` | Anthropic model ID | no | configured default |
| `X-Anthropic-API-Key` | header | only with VLM and no service key | — |

The response is `202 Accepted` with a `Location` header and a `links` object
pointing at every output resource.

### Errors

Errors use `application/problem+json` with a stable machine-readable `code` and
a request ID echoed in the `X-Request-ID` header:

```json
{
  "type": "urn:corpusretrieve:problem:unsupported-file-type",
  "title": "Unsupported Media Type",
  "status": 415,
  "detail": "example.exe is not supported.",
  "instance": "/v1/jobs",
  "code": "unsupported-file-type",
  "request_id": "e7bd7dd0b6944cbcac13a4076c239d91"
}
```

Validation failures add an `errors` array of `field`, `message`, and `code`.

### Supported inputs

```text
.pdf .docx .xlsx .xlsm .csv .tsv .json .jsonl .html .htm
.txt .md .log .png .jpg .jpeg .webp .gif .tif .tiff .bmp
```

Call `GET /v1/capabilities` for the authoritative runtime list and the
configured upload limits. OCR requires Tesseract; the image ships English data.

---

## Deployment

Requirements: Docker with Compose, and roughly 4 GB of memory for the offline
hashing embedder. BGE-M3 needs more memory and model storage.

```bash
cp .env.example .env
make key          # generate CREDENTIAL_ENCRYPTION_KEY
```

Set in `.env`:

```dotenv
STORAGE_BACKEND=s3
JOB_MODE=rq
S3_SECRET_KEY=<strong-random-secret>
MINIO_ROOT_PASSWORD=<the same secret>
CREDENTIAL_ENCRYPTION_KEY=<generated Fernet key>
ANTHROPIC_API_KEY=<optional>
```

```bash
docker compose up --build -d
docker compose ps
docker compose logs -f api worker
```

| Interface | URL |
|---|---|
| Web interface and API | `http://127.0.0.1:8000` |
| Object storage console | `http://127.0.0.1:9001` |

The API never keeps indexes on its local disk. Workers use an isolated
temporary directory and delete it when the job ends. Artifacts live in the
configured object store under a seven-day lifecycle by default.

For BGE-M3, set `INSTALL_BGE=true` and `DEFAULT_EMBEDDER=bge-m3` before
building. See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

---

## Configuration

All configuration is environment-driven and loaded from `.env` at startup.
Invalid combinations fail fast at boot rather than at first request.

| Variable | Purpose | Default |
|---|---|---|
| `STORAGE_BACKEND` | `memory` or `s3` | `memory` |
| `JOB_MODE` | `inline` or `rq` | `inline` |
| `REDIS_URL` | Redis connection for queue and state | `redis://localhost:6379/0` |
| `DEFAULT_EMBEDDER` | `hash` or `bge-m3` | `hash` |
| `ANTHROPIC_API_KEY` | Deployment-wide VLM key | empty |
| `DEFAULT_VLM_MODEL` | Anthropic model ID | `claude-sonnet-4-6` |
| `CREDENTIAL_ENCRYPTION_KEY` | Fernet key; required when `JOB_MODE=rq` | empty |
| `CREDENTIAL_TTL_SECONDS` | Lifetime of a per-request key | `1800` |
| `MAX_UPLOAD_BYTES` | Per-file and per-ZIP limit | `104857600` |
| `MAX_FILES_PER_JOB` | Documents per corpus | `2000` |
| `MAX_CORPUS_UNCOMPRESSED_BYTES` | ZIP expansion limit | `524288000` |
| `ARTIFACT_RETENTION_DAYS` | Object lifecycle for outputs | `7` |
| `DELETE_INPUTS_AFTER_PROCESSING` | Remove uploads after the worker finishes | `true` |
| `CONSOLE_ENABLED` | Serve the web interface and `/console` | `true` |
| `EMIT_PAGE_PREVIEWS` | Render PDF pages for the overlay | `true` |
| `PAGE_PREVIEW_DPI` | Preview resolution | `110` |
| `MAX_PAGE_PREVIEWS_PER_DOCUMENT` | Preview cap per document | `200` |
| `CORS_ORIGINS` | Comma-separated allowed origins | `*` |

Full reference: [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

---

## Repository layout

| Path | Responsibility |
|---|---|
| `corpusretrieve/api/` | HTTP contract, validation, errors, downloads |
| `corpusretrieve/pipeline/` | Stage orchestration, extraction, indexing, graph, packaging |
| `corpusretrieve/storage/` | Memory and S3-compatible artifact stores |
| `corpusretrieve/state/` | Memory and Redis job-state stores |
| `corpusretrieve/web/` | Console view models and browser assets |
| `corpusretrieve/config.py` | Environment-driven settings and validation |
| `corpusretrieve/security.py` | Transient VLM credential handling |
| `corpusretrieve/queueing.py` | Inline and Redis Queue dispatch |
| `corpusretrieve/worker.py` | Isolated workspaces and artifact publication |
| `docparse/` | Format-independent extraction and chunking core |
| `service/` | ASGI entry point |
| `tools/` | Operational scripts |
| `tests/` | Test suite |

Each layer depends only on the one beneath it. The API never touches the
filesystem, the pipeline never touches HTTP, and the console never reaches past
the artifact store.

---

## Testing

```bash
pip install -r requirements-dev.txt
make test
make lint
```

The suite covers the endpoint contract, upload validation and archive safety,
credential handling, stage tracking, console view models, and the projection
maths. It runs entirely on the memory backends with no external services.

---

## Security defaults

- VLM credentials are transient, encrypted in queue mode, used once, and never
  written to artifacts, manifests, or logs.
- Uploads are removed from object storage after processing by default.
- ZIP handling rejects absolute paths, traversal, and encrypted entries, and
  enforces entry-count, per-file, and total-uncompressed limits.
- Containers run as an unprivileged user.
- `.env` is excluded from version control and Docker build context.
- Deleting a job removes its entire object prefix, job state, and credentials.

A deployment reachable from outside a trusted network should add an
authentication gateway, TLS, rate limiting, malware scanning, and tenant
authorization. CorpusRetrieve does not provide these.

---

## Documentation

| Document | Contents |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Layers, data flow, and design decisions |
| [docs/API.md](docs/API.md) | Endpoint reference, error codes, console layer |
| [docs/WEB_INTERFACE.md](docs/WEB_INTERFACE.md) | Modes, views, and asset structure |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Compose, scaling, operations |
| [docs/CONFIGURATION.md](docs/CONFIGURATION.md) | Every environment variable |
| [docs/EMBEDDINGS.md](docs/EMBEDDINGS.md) | Lexical and semantic models, re-indexing a build |
| [docs/RELEASE_2.0.0.md](docs/RELEASE_2.0.0.md) | Release contract and upgrading from 1.0.x |
| [CHANGELOG.md](CHANGELOG.md) | Release history |

---

## Provenance and group-oriented ingestion

The per-document configuration JSON holds publication metadata and stable IDs.
The optional collection-level `__collection__` block carries grouping and provenance context.
A strict provenance gate is enabled with `REQUIRE_BIBLIOGRAPHIC=true`, the
web upload checkbox, or `corpusretrieve build --require-bibliographic`.
When no authoritative record is supplied, PDF-extracted DOI/title hints are labelled
`origin: "extracted"`; they are not externally validated.

For generic manifests with nested `scenarios[].themes[].documents[]` structure,
use `tools/prepare_theme_jobs.py` to create one configuration per logical group:

```bash
python tools/prepare_theme_jobs.py external_manifest.json --out group_jobs --max-files 150
corpusretrieve build ./downloaded_files --config group_jobs/example_group/full_theme.path_config.json \
  --key-by path --only-config-files --require-bibliographic \
  --out ./outputs/example_group --corpus-name example_group
```

`external_manifest.json`, PDFs, output bundles, and group configurations are
**consumer-owned data**. No project, sector, vocabulary, or theme list is built
into CorpusRetrieve. The importer handles a generic documented manifest shape,
not a particular external collection.

See [Document configuration](docs/DOCUMENT_CONFIG.md),
[Manifest and group workflows](docs/MANIFEST_WORKFLOWS.md),
[Production deployment](docs/DEPLOYMENT.md), and
[Release contract](docs/RELEASE_2.0.0.md).
