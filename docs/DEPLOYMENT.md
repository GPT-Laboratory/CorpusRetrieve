# Deployment

## Stack

`docker compose up` starts four services:

| Service | Role |
|---|---|
| `api` | FastAPI application, web interface, upload handling |
| `worker` | Redis Queue worker running the pipeline |
| `redis` | Job state, queue, and encrypted transient credentials |
| `minio` | S3-compatible artifact storage |

Replace `minio` with any S3-compatible service by pointing `S3_ENDPOINT_URL` at
it and removing the container.

## Prepare

```bash
cp .env.example .env
make key          # prints a Fernet key
```

Required edits in `.env`:

```dotenv
STORAGE_BACKEND=s3
JOB_MODE=rq
S3_SECRET_KEY=<strong-random-secret>
MINIO_ROOT_PASSWORD=<the same secret>
CREDENTIAL_ENCRYPTION_KEY=<generated Fernet key>
```

`S3_SECRET_KEY` and `MINIO_ROOT_PASSWORD` must match, as must `S3_ACCESS_KEY`
and `MINIO_ROOT_USER`. Optionally set `ANTHROPIC_API_KEY` for a
deployment-wide figure-description key.

## Start

```bash
docker compose up --build -d
docker compose ps
docker compose logs -f api worker
```

| Interface | URL |
|---|---|
| Web interface and API | `http://127.0.0.1:8000` |
| Swagger UI | `http://127.0.0.1:8000/docs` |
| Health | `http://127.0.0.1:8000/v1/health` |
| Object storage console | `http://127.0.0.1:9001` |

Verify:

```bash
python tools/api_smoke_test.py --base-url http://127.0.0.1:8000 --delete-job
```

Stop with `docker compose down`, or `docker compose down -v` to discard stored
artifacts and Redis state.

## Production embeddings

```dotenv
INSTALL_BGE=true
DEFAULT_EMBEDDER=bge-m3
```

Rebuild after changing `INSTALL_BGE`; it is a build argument. Weights download
on first use, so mount a model cache or bake them into the image if workers
start in a network-restricted environment. Expect noticeably higher memory use.

## Scaling

Workers are stateless. Add capacity with:

```bash
docker compose up -d --scale worker=4
```

The API is also stateless when `STORAGE_BACKEND=s3` and `JOB_MODE=rq`; run
several replicas behind a load balancer. Do not scale the API beyond one replica
with memory backends, because job state would not be shared.

Sizing notes: a worker needs enough memory for the largest single document plus
its rendered previews and one embedding batch. Every stage streams documents
from disk one at a time, so peak memory is largely independent of corpus size;
`EMBEDDING_BATCH_SIZE` is the lever that bounds it. The knowledge graph is the
one structure that still grows with the corpus, holding identifiers and short
labels rather than chunk text.

## Operations

**Health.** The image health check polls `/v1/health` every 30 seconds. It
returns `503` when the artifact store or job store is unreachable, which is the
right signal for an orchestrator to restart or drain the container.

**Retention.** `ARTIFACT_RETENTION_DAYS` installs a bucket lifecycle rule on the
`jobs/` prefix. Outputs are removed automatically; `DELETE /v1/jobs/{id}`
removes them immediately.

**Logs.** Failures are logged with the exception and the job ID. Client errors
carry a request ID returned in `X-Request-ID` and in the problem document, which
is the value to search for when correlating a report with server logs.

**Backups.** Job state in Redis is derived and expendable; the artifact bucket
is the only thing worth backing up. A lost Redis makes completed jobs
unreadable through the API even though the objects still exist, so treat job
state as a cache with a retention window, not a record.

## Hardening

CorpusRetrieve ships no authentication. Before exposing it beyond a trusted
network, add:

- an authentication gateway in front of the API
- TLS termination
- rate limiting on `POST /v1/jobs`
- malware scanning of uploads
- tenant authorization, since any caller holding a job ID can read that job
- a restricted `CORS_ORIGINS` value instead of `*`

Also review `MAX_UPLOAD_BYTES`, `MAX_FILES_PER_JOB`, and
`MAX_CORPUS_UNCOMPRESSED_BYTES` against available worker memory and disk.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Boot fails with `CREDENTIAL_ENCRYPTION_KEY is required` | `JOB_MODE=rq` without a Fernet key. Run `make key` |
| Jobs stay `queued` | No worker running, or worker and API disagree on `REDIS_URL` |
| `503 artifact-storage-unavailable` | Object store unreachable or credentials mismatched |
| Uploads succeed but outputs 404 | API and worker are using different `STORAGE_BACKEND` values |
| `410 artifact-expired` | Retention lifecycle removed the objects. Resubmit |
| Job `failed` but artifacts exist | Expected. A failed run publishes whatever completed; check `complete` and `error` in the manifest |
| Documents missing from the manifest | They were unreadable. See `skipped_documents` in the manifest and the job warnings |
| Source view shows no page | Not a PDF, or `EMIT_PAGE_PREVIEWS=false` |
| `bge-m3` missing from capabilities | Image built without `INSTALL_BGE=true` |
| OCR produces nothing | Language pack absent. Add it to the Dockerfile and set `ocr_langs` |
