# Changelog

## 1.0.0

First unified release. The production integration API and the interactive web
workbench, previously separate lines of development, are merged into one
deployable project.

### Added

- Six-stage pipeline tracking. `extract`, `enrich`, `chunk`, `embed`, `graph`,
  and `package` each report status, elapsed time, and counters. Exposed in
  `stages[]` on the job status endpoint and the event stream.
- Interactive web interface with Inspect, Explain, and Learn modes: provenance
  overlay on rendered pages, typed-chunk span map, knowledge-graph explorer, and
  2D embedding map.
- Anthropic API key input in the web interface, sent per request as
  `X-Anthropic-API-Key`, with a show/hide toggle and an indicator showing
  whether a service key is already configured.
- `.env` support for `ANTHROPIC_API_KEY` and every other setting, loaded at
  startup and passed to the API and worker containers.
- Page preview rendering for PDF documents, published as ordinary artifacts and
  controlled by `EMIT_PAGE_PREVIEWS`, `PAGE_PREVIEW_DPI`, and
  `MAX_PAGE_PREVIEWS_PER_DOCUMENT`.
- Read-only `/console` layer serving browser view models, excluded from the
  OpenAPI document and disableable with `CONSOLE_ENABLED`.
- 2D projection of chunk embeddings using an SVD-based PCA on NumPy, with UMAP
  used when it is installed.
- `tools/api_smoke_test.py`, which exercises health, capabilities, upload,
  polling, the event stream, every typed output, and the bundle download.
- Test suite covering the endpoint contract, upload validation, archive safety,
  credential handling, stage tracking, console view models, and projection.
- `Makefile`, `requirements-dev.txt`, and expanded documentation under `docs/`.

### Changed

- The pipeline is decomposed into `progress`, `documents`, `previews`,
  `indexing`, `knowledge_graph`, `packaging`, and `artifacts` modules
  orchestrated by a thin processor, replacing a single monolithic module.
- Job state carries stage records, so the interface renders server-reported
  facts rather than a client-side animation.
- Redis and boto3 are imported lazily by the store factories, so a
  memory-backed development or test run installs neither.
- Artifact grouping for the downloads panel moved into the console view layer.
- Documentation rewritten around the merged project.

### Removed

- The filesystem-backed job service, which kept one directory per job and could
  not be deployed across replicas.
- The Qdrant-dependent evidence service and its retrieval, reranking, and graph
  expansion modules. CorpusRetrieve produces portable artifacts; a retrieval
  service consumes them.
- Client-side batch orchestration. A multi-document upload is now one corpus
  job, and the interface switches between documents within it.
- Domain-specific wording and development commentary throughout the source.
