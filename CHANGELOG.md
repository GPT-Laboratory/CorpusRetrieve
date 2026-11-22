# Changelog

## 2.0.0

The 2.0.0 release replaces 1.0.x. The REST API path stays `/v1`; the bundle,
index, payload and manifest schemas are `2.0.0`. Rebuild corpora from their
source files: 1.0.x bundles are not converted in place.

### Running long builds

- `STORAGE_BACKEND=local` keeps artifacts and job status on disk under
  `LOCAL_DATA_DIR`. Results survive a server restart and large outputs are not
  held in memory. Jobs that were running when the server stopped are marked
  "interrupted" instead of appearing to run forever.
- Problems that guarantee a failed quality gate are reported at upload, before
  any processing: `document_config` keys that match no file, files without a
  configuration entry (when a configuration is given), and file names that
  differ only in letter case. The upload is rejected with HTTP 422
  `intake-check-failed`, listing the files, unless `allow_degraded=true`.
- Each upload is checked for Tesseract and its language data; a problem is
  reported as a warning before the build starts. `TESSERACT_CMD` points at the
  executable when it is not on `PATH` (the usual case on Windows).
- Figure descriptions stop for the rest of a build at the first
  authentication, permission, missing-model or billing error, or after
  `VLM_MAX_CONSECUTIVE_FAILURES` failures in a row. The remaining figures are
  counted as `figures_skipped_vlm_unavailable` instead of being sent. The
  Anthropic SDK's own retries are turned off (CorpusRetrieve retries), each
  request times out after `VLM_TIMEOUT_SECONDS`, and repeated errors become one
  warning with a count.
- A model-based embedder (`bge-m3`, `st:<model id>`) is loaded before
  extraction starts, so a missing package or a misspelt model id stops the job
  at once instead of after the documents are processed.
- Extract, enrich and chunk each report their own processing time; each
  document's timing is recorded in the manifest.
- `corpusretrieve reindex` re-embeds a finished build (a bundle ZIP or an
  output directory) with another embedder, rebuilds the graph and re-seals the
  bundle, without extracting the documents again.

### Web interface

- A job that fails its quality gate, or fails partway, shows everything it
  produced: documents, chunks, provenance overlay, graph, embedding map, a
  "not release-ready" notice with each reason, and the downloads.
- A download button for the bundle, with its size, whenever outputs exist.
- The job ID is kept in the address (`#job/<id>`); the start page lists
  previous jobs and opens any job by ID.
- The knowledge graph view shows one document at a time with a stable radial
  layout of its outline, readable labels, every node type in the legend with
  its count, edges coloured by kind, and chunk selection highlighting.
- The embedding map samples the whole index evenly from `index/dense.npy`.
- The download panel lists the corpus-level files with their sizes.
- Large corpora: a searchable document picker, and per-request work that no
  longer scales with the number of documents times the number of artifacts.
- New controls: accept a build that is not release-ready; the strict
  publication-metadata default follows `REQUIRE_BIBLIOGRAPHIC`; the embedder
  list shows every built-in model and whether it is installed.

### Embeddings

- Built-in choices: `hash` (hashing-v1, the default), `hash-v2` (Unicode word
  features), `st:intfloat/multilingual-e5-small`,
  `st:intfloat/multilingual-e5-base`, `bge-m3`, and any
  `st:<sentence-transformers model id>`.
- E5 models get their `passage:` and `query:` prefixes. Index metadata records
  the full model id and the prefixes, and the reference consumer reloads the
  same model for queries. BGE-M3 uses half precision only on a GPU.

### Core pipeline

- Stable source-derived element and chunk IDs, typed retrieval units, source
  spans and bounding boxes in every payload, optional figure descriptions.
- PDF, DOCX, XLSX, HTML and text processing; OCR and text-layer accounting.
- Publication records from per-document configuration, a labelled DOI
  fallback from the PDF, a strict mode and a multi-publication ambiguity check.
- Dense and BM25 sparse indexing with a query-side encoder, `index/dense.npy`,
  a portable reference consumer, and knowledge-graph exports (JSON, GraphML,
  Cypher with labelled batches).
- Manifest with a `quality` block and `release_ready`, artifact checksums,
  `bundle.zip.sha256`, `corpusretrieve verify` and `corpusretrieve repair`.
- Identical files are processed once and their records merged; resumable,
  additive CLI builds; generic manifest-to-configuration and group tools.
- Uploaded files are staged under short internal names, so long source file
  names do not exceed the Windows path limit.

### Security and deployment

- Optional API keys (`API_KEYS`), no cross-origin callers unless configured.
- `DEPLOYMENT_MODE=production` checks API keys, S3, RQ, CORS and the object
  store secret on start-up.

### Defaults kept for compatibility

- `DEFAULT_EMBEDDER=hash` and `INDEX_INLINE_VECTORS=true`, so consumers of
  1.0.x bundles keep working until they opt in to `hash-v2`, a semantic model,
  or `dense.npy` only.

### Known limits

- Multi-column and multi-page table fidelity and reference-list coverage need
  corpus-specific spot checks; checksums establish integrity, not accuracy.
- The web interface has no API-key sign-in; use it on trusted networks or
  behind a gateway that adds the key.
- The corpus knowledge graph links papers only through shared units: citation
  keys and entities are scoped to their document.
- Bundles are not byte-identical across environments (timestamps, build
  metadata).
