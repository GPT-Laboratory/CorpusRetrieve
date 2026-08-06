# Configuration

Every setting is read from the environment, with `.env` loaded at startup.
`Settings` is a frozen dataclass validated once during boot, so an invalid
combination fails immediately with a specific message.

## Identity

| Variable | Default | Notes |
|---|---|---|
| `APP_NAME` | `CorpusRetrieve` | Reported by `/v1/health` and `/v1/capabilities` |

## Artifact storage

| Variable | Default | Notes |
|---|---|---|
| `STORAGE_BACKEND` | `memory` | `memory` or `s3`. Use `s3` for deployment |
| `S3_ENDPOINT_URL` | `http://localhost:9000` | Any S3-compatible endpoint |
| `S3_ACCESS_KEY` | `corpusretrieve` | |
| `S3_SECRET_KEY` | `change-this-secret` | Change before deploying |
| `S3_BUCKET` | `corpusretrieve` | Created if absent |
| `S3_REGION` | `us-east-1` | |
| `ARTIFACT_RETENTION_DAYS` | `7` | Bucket lifecycle for `jobs/`. `0` disables |

`memory` holds artifacts in the API process. They vanish on restart and are
invisible to a separate worker, so it suits development and tests only.

## Job dispatch

| Variable | Default | Notes |
|---|---|---|
| `JOB_MODE` | `inline` | `inline` runs in the API process; `rq` uses workers |
| `REDIS_URL` | `redis://localhost:6379/0` | Queue, job state, and credentials |
| `JOB_TTL_SECONDS` | `604800` | Redis lifetime for job records |

`inline` blocks a background task in the API process. It is fine for local use
and small corpora; use `rq` when uploads are large or concurrent.

## Upload limits

| Variable | Default | Notes |
|---|---|---|
| `MAX_UPLOAD_BYTES` | `104857600` | Per file, and for a submitted ZIP |
| `MAX_FILES_PER_JOB` | `100` | Documents per corpus |
| `MAX_CORPUS_UNCOMPRESSED_BYTES` | `524288000` | Total expanded ZIP size |

`MAX_CORPUS_UNCOMPRESSED_BYTES` must be at least `MAX_UPLOAD_BYTES`. Archive
inspection also refuses more than ten times `MAX_FILES_PER_JOB` entries.

## Embedding

| Variable | Default | Notes |
|---|---|---|
| `DEFAULT_EMBEDDER` | `hash` | `hash` or `bge-m3` |
| `INSTALL_BGE` | `false` | Build argument; installs the optional dependency |

`hash` is deterministic, offline, and downloads nothing. It validates the
pipeline but is not a substitute for a trained model in production relevance.
`bge-m3` requires an image built with `INSTALL_BGE=true`; `/v1/capabilities`
reports what is actually available at runtime.

## Figure description

| Variable | Default | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | empty | Deployment-wide key. Leave empty to require per-request keys |
| `DEFAULT_VLM_MODEL` | `claude-sonnet-4-6` | Overridable per request |
| `CREDENTIAL_ENCRYPTION_KEY` | empty | Fernet key. Required when `JOB_MODE=rq` |
| `CREDENTIAL_TTL_SECONDS` | `1800` | Lifetime of a stored per-request key |

Generate a Fernet key:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

A per-request `X-Anthropic-API-Key` takes precedence over `ANTHROPIC_API_KEY`.
Neither is written to job metadata, manifests, artifacts, logs, or object
storage.

## Web interface

| Variable | Default | Notes |
|---|---|---|
| `CONSOLE_ENABLED` | `true` | Serves `/` and `/console` |
| `EMIT_PAGE_PREVIEWS` | `true` | Renders PDF pages for the provenance overlay |
| `PAGE_PREVIEW_DPI` | `110` | Between 40 and 400 |
| `MAX_PAGE_PREVIEWS_PER_DOCUMENT` | `200` | Cap per document |
| `CONSOLE_MAX_PROJECTION_POINTS` | `5000` | Vectors read for the embedding map |

Page previews are stored as ordinary artifacts and count toward job size. At the
default DPI, a page costs roughly 20–60 KB. Set `EMIT_PAGE_PREVIEWS=false` for
API-only deployments; the source view then reports that no preview exists rather
than failing.

Preview DPI is independent of the extraction DPI used for figure crops, so
reducing it does not degrade the images sent to a vision model.

## Privacy and access

| Variable | Default | Notes |
|---|---|---|
| `DELETE_INPUTS_AFTER_PROCESSING` | `true` | Removes uploads once the worker finishes |
| `CORS_ORIGINS` | `*` | Comma-separated. Restrict for deployment |

## Validation rules

Startup fails when:

- `STORAGE_BACKEND` is not `memory` or `s3`
- `JOB_MODE` is not `inline` or `rq`
- `DEFAULT_EMBEDDER` is not `hash` or `bge-m3`
- `MAX_FILES_PER_JOB` or `MAX_UPLOAD_BYTES` is not positive
- `MAX_CORPUS_UNCOMPRESSED_BYTES` is below `MAX_UPLOAD_BYTES`
- `PAGE_PREVIEW_DPI` is outside 40–400
- `JOB_MODE=rq` and `CREDENTIAL_ENCRYPTION_KEY` is missing or not a Fernet key
