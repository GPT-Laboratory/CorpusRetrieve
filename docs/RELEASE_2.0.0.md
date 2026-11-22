# CorpusRetrieve 2.0.0 — release contract

## Version

The application, the Python package and the Docker image are 2.0.0. The REST
API path is `/v1` (an API path, not the application version). The portable
payload, index and manifest schemas are `2.0.0`. Tag the reviewed commit
`v2.0.0` once tests, security review and integration acceptance pass.

## Separation of responsibilities

- **Engine:** document extraction, OCR, element typing, provenance, metadata
  validation, typed chunks, indexing, graph exports, jobs, storage,
  verification, repair, re-indexing, and the web interface.
- **Data (external):** document lists, groups, source IDs, titles, authors,
  years, DOI or repository links, subjects, language labels and local paths.
- **Consumers (external):** prompt construction, query routing, generation,
  citation rendering and domain-specific evaluation.

## Artifact contract

A build writes `manifest.json`, `artifact_catalog.json`, `bundle.zip`,
`bundle.zip.sha256`, `corpus/chunks.json`, per-document `extraction.json`
and `chunks.json`, `index/vectors.jsonl`, `index/dense.npy`,
`index/sparse_encoder.json`, `index/metadata.json`, and the knowledge-graph
exports. `index/metadata.json` names the embedding model (`model`,
`model_id`) and the query and passage prefixes a consumer must use.

Point IDs depend on the document, the unit and the embedding model, not on the
corpus name. Do not mix point IDs from 1.0.x indexes with 2.0.0 records.

## Release decision

Every manifest has a `quality` block and `release_ready`. With
`DEGRADED_POLICY=fail`, a build with unread OCR pages, failed or skipped
figure descriptions, documents without units, an embedder fallback, graph
export errors, skipped documents, or (in strict mode) unresolved publication
records is published but the job is marked failed. Intake problems that
guarantee this outcome are rejected at upload instead. `allow_degraded=true`
accepts such a build as completed; the reasons stay in the manifest.

## Claims and non-claims

Checksum verification establishes file integrity relative to the catalog, not
correctness. Bounding boxes may be present without being exact. Complex
tables, reference-list coverage, figure descriptions and answer support need
representative evaluation. No large-corpus re-evaluation or production load
test is implied by the test suite.

## Upgrading from 1.0.x

1. Keep the 1.0.x code, `.env`, metadata and bundles as evidence.
2. Create a fresh Python environment for 2.0.0.
3. Review `STORAGE_BACKEND`, `OCR_LANGS`, `TESSERACT_CMD`, `API_KEYS`,
   `REQUIRE_BIBLIOGRAPHIC` and the embedder choice ([Embeddings](EMBEDDINGS.md)).
4. Rebuild from the source files and run `corpusretrieve verify` on the bundle.
5. Test consumers against the 2.0.0 `bibliographic` payload, stable IDs and
   `index/metadata.json` before switching production traffic.
