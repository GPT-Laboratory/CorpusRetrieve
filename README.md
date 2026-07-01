# Multimodal Knowledge Base (MKB)

> A generic, provenance-first, multimodal retrieval + knowledge-graph **microservice**. It turns diverse, messy, multilingual, visually-rich source material into **citable evidence units** that any agentic system can retrieve over a clean HTTP API. Domain-neutral by design; domain meaning rides in metadata and configuration, never in the service.

---

## Repository description (short)

**This is the AGIME knowledge layer, extracted as a standalone microservice.** It ingests documents from many kinds of sources, converts them into modality-aware, provenance-bearing chunks, indexes them for hybrid (dense + sparse) retrieval with an optional knowledge graph, and serves retrieval with **citations**. Every capability that touches the outside world — extraction, embedding, vector store, lexical index, reranker, graph store, source connectors, vision retrieval — sits behind a swappable adapter, so the same service runs zero-dependency on a laptop and scales to a self-hosted, EU-data-resident stack without code changes.

## What this is (and why it exists)

Agentic systems are only as good as the evidence they can retrieve and *cite*. Most RAG stacks treat "the corpus" as an afterthought: a pile of text chunks behind an embedding model. That breaks the moment real sources arrive — PDFs where the decisive information lives in charts, tables, and maps; multiple languages in one document; repository APIs instead of file uploads; sources of wildly different trustworthiness; and outputs that must be auditable and license-aware.

The knowlwdge base reframes the corpus as the product. The unit of value is not a "text chunk" but an **evidence unit**: a piece of content that knows *what it is* (its modality), *where it came from* (document, page, region, extraction method, confidence), *what language it is in*, and *how much it can be trusted* (source-quality metadata). Retrieval returns evidence units plus **citations** that resolve back to the exact source location. Enforcement of downstream rules — "every claim must cite a source," "state the uncertainty," "attach the disclaimer" — is the *consumer's* job (e.g. an AGIME workflow), but it provides the provenance that makes such enforcement possible in the first place.

The design keeps one principle above all: **the service is generic.** No domain vocabulary, no use-case logic. A use case configures MKB (which connectors, which extraction adapters, which embedding model, which metadata schema) and consumes its API; it does not modify it.

## Why BioAI is the motivating example (not part of the service)

The BioAI / UC-T5 use case — regional agronomic foresight for the Satakunta region of Finland — is a near-perfect stress test for how diverse real source material can be, which is why it's referenced throughout the design as an *example*. None of it ships in MKB. It simply illustrates why each generic abstraction earns its place:

- **Heterogeneous acquisition** — an institutional repository (Jukuri, via DSpace REST / OAI-PMH), a journal, government-institute publications (FMI, SYKE), economic policy briefs (PTT), and practitioner magazines. → motivates **source connectors** as adapters, not "upload a file."
- **Visually-rich content** — the highest-value numbers live in charts, maps, and complex tables, much of it vector graphics invisible to naive text extraction. → motivates a **modality-aware chunk model** and a **pluggable extraction pipeline** (layout / table / figure / chart-to-table / map), plus an optional **page-image (vision) retrieval** track.
- **Multilinguality** — Finnish, English, and Swedish, often mixed in one document, where Finnish morphology defeats generic multilingual encoders. → motivates a **swappable embedding adapter** (default multilingual, with a language-specialized track benchmarked against it) and a **language** field on every unit.
- **Trust heterogeneity** — peer-reviewed vs. institute report vs. policy brief vs. practitioner note carry very different evidential weight. → motivates **provenance + multidimensional trust metadata** carried on every unit, never collapsed into a hidden score.
- **Legal reality** — reuse rights differ per record. → motivates a **per-record license/attribution registry** and storage-policy metadata.

If MKB handles this corpus cleanly while staying domain-neutral, it handles almost anything: legal, biomedical, financial filings, internal wikis, scanned archives.

## Design principles

1. **Generic and provenance-first.** Every retrievable unit carries source, page/locator, modality, language, confidence, license, and open metadata. Citations are a first-class return type, resolvable to the exact chunk.
2. **Everything behind an adapter.** Extraction, chunking, embedding, vector store, lexical index, reranker, graph store, source connectors, and vision retrieval are all pluggable via a uniform contract (a `provider` selector + `endpoint_url` + `parameters`). The default of every adapter is dependency-free so the service boots with zero external infrastructure; a config change points it at a production backend.
3. **Hybrid by default, graph and vision when they earn it.** Dense + sparse retrieval with reciprocal-rank / weighted fusion and an optional reranker is the backbone. A knowledge graph and a page-image retrieval track are optional augmentations, gated on measured benefit rather than assumed.
4. **Staged and measurable ("highly optimizable").** A minimal viable pipeline (acquire → extract → chunk → embed → hybrid retrieve → cite) is separated from research extensions (chart-to-table numeric grounding, map georeferencing, vision retrieval, graph-augmented retrieval). Extensions are promoted only when a gold-set evaluation shows they help — each is a flag, not a fork.
5. **Self-hostable, offline-capable, residency-friendly.** Runs fully in-memory with no network egress for local/dev/demo; the self-hosted profile keeps all data in your own infrastructure (e.g. Postgres/pgvector, Neo4j, local embedding + extraction services), which matters for regulated / EU-data-residency deployments. Managed cloud backends are supported but never required.
6. **Coherent with AGIME.** MKB is the knowledge substrate AGIME's agentic framework already assumes: the same corpus / document / chunk / citation model, the same adapter contracts. AGIME (or any agent) calls MKB over HTTP; MKB never depends on the caller.

## How it fits with AGIME

```
        ┌────────────────────────┐        HTTP        ┌──────────────────────────────┐
        │   Agentic consumers     │  ───────────────▶ │            MKB                │
        │  (AGIME workflows,      │   retrieve /       │  multimodal knowledge base    │
        │   chatbots, pipelines)  │   ingest /         │  (this repo)                  │
        │                         │ ◀───────────────  │                               │
        │  own the constraints:   │   citations +      │  owns: evidence units,        │
        │  cite / uncertainty /   │   evidence units   │  provenance, retrieval,       │
        │  disclaimer enforcement │                    │  graph, adapters              │
        └────────────────────────┘                    └──────────────┬───────────────┘
                                                                      │ adapters (external_http)
                              ┌───────────────────────────────────────┼───────────────────────────────┐
                              ▼                     ▼                  ▼                ▼                ▼
                        extraction            embedding          vector store      graph store      source
                     (layout/table/           (multilingual)    (pgvector/Qdrant)  (Neo4j/...)     connectors
                      figure/chart/map)                                                            (DSpace/OAI/http/S3)
```

AGIME keeps enforcement (a claim must cite; uncertainty must be disclosed; a domain disclaimer must be present). MKB keeps evidence and provenance. Clean separation, coherent contracts.

## Core concepts / data model (high level)

- **Corpus** — a named, independently-configured collection. Holds its own adapter configuration (which embedding model, which vector backend, which graph backend, retrieval mode). You can run many corpora side by side.
- **Document** — a source item with provenance: title, source URI, language, page count, source type, license, attribution, and per-modality counts.
- **Evidence unit (chunk)** — the retrievable unit. Carries text (or serialized table / figure description / chart-derived table), a `content_modality` (`narrative`, `abstract`, `caption`, `table`, `figure_description`, `chart_derived`, `page_image`, `reference`, …), `language`, page/bbox locator, `confidence`, an optional link to a parent unit, the extraction provider that produced it, and open `metadata` (where domain tags, themes, region, trust dimensions live).
- **Citation** — a first-class object linking a retrieved unit back to its document, page, modality, and score; resolvable to the full unit for audit.
- **Graph node / edge** — optional entities and relations (e.g. concepts, sources, and their links) with properties, linked to documents and chunks; enables relational / multi-hop queries and explainability alongside vector retrieval.
- **Ingestion job** — an async record for long-running ingestion (repository harvests, heavy extraction), with progress and per-item errors.

Domain meaning (themes, regions, crops, scenarios, trust scores, …) is expressed entirely through `metadata` and graph nodes — never through service code.

## Adapter matrix (the swappable seams)

| Seam | Default (zero-dep) | Production options (via `external_http` / providers) |
|---|---|---|
| **Extraction** | `text` (plain text / basic PDF/DOCX) | layout, OCR, table structure, figure/caption, chart-to-table, map georeferencing (e.g. Docling, Table Transformer, DePlot-style, map pipelines) |
| **Chunking** | `fixed` window | `section_aware`, `semantic`; promote-leading-summary; drop-boilerplate patterns |
| **Embedding** | deterministic `hashing` (dev only) | multilingual / domain models (e.g. BGE-M3), managed embeddings, or a language-specialized track |
| **Vector store** | `in_memory` | pgvector, Qdrant, Weaviate, Milvus |
| **Lexical / sparse** | `in_memory` BM25 | external search service |
| **Reranker** | none | cross-encoder / LLM reranker |
| **Graph store** | `disabled` / `in_memory` | Neo4j, Memgraph, RDF store |
| **Source connectors** | `http` | DSpace REST, OAI-PMH, S3, custom repositories |
| **Vision retrieval** | off | page-image / late-interaction retriever (ColPali / VisRAG-style) |

Every adapter shares one contract shape — a `provider` selector plus an `endpoint_url` and free-form `parameters` — so swapping one is a configuration change, not a code change.

## API surface (high level)

- **Corpora** — create / list / get / patch / delete (each carries its own adapter config).
- **Ingestion** — from text, from an uploaded file, or **from a source connector** (repository harvest); long jobs return an ingestion-job handle you can poll.
- **Retrieval** — hybrid search over a corpus (dense / sparse / hybrid, with optional rerank), returning evidence units with scores and component scores.
- **Citations** — resolve any retrieved unit back to its full chunk for audit.
- **Graph** — upsert nodes/edges, search, neighborhood, stats.
- **Stats & health** — corpus statistics, backend info, health.

Exact request/response schemas will live in `docs/API.md` and the service's OpenAPI once the code lands; they mirror the AGIME knowledge-layer contracts so existing AGIME consumers work unchanged.

## Deployment profiles

- **Embedded / demo (default).** Zero external dependencies — in-memory vector + lexical + graph, hashing embeddings, text extraction. Boots in seconds; ideal for local dev, CI, and offline demos.
- **Self-hosted (production).** Docker Compose brings up the service plus the backends you choose (e.g. Postgres/pgvector for vectors + lexical, Neo4j for the graph, a local embedding service, a local extraction service). All data stays in your infrastructure. GPU is optional and only needed for heavier extraction / vision models.
- **Managed backends (optional).** Any adapter can point at a managed cloud service if residency and cost allow. Never required.

## Data residency & governance stance

MKB is built so that a privacy-sensitive, EU-data-resident deployment is the easy path, not the hard one: the default and recommended posture is self-hosted, with every byte staying in your infrastructure. The service records per-record **license and attribution**, tracks how each unit was produced (**extraction provenance and confidence**), and carries **source-quality metadata** so consumers can make and *audit* trust decisions. MKB deliberately does not decide "the one best source" for you — it keeps the dimensions separate and lets the consuming policy do the weighting, transparently.

## Roadmap (staged capabilities)

**MVP (the backbone).** Source connectors → modality-aware extraction (text + tables) → section-aware chunking → hybrid dense+sparse retrieval with fusion → citations with full provenance → a small optional knowledge graph → self-hosted profile.

**Research extensions (flagged, gated on measured benefit).** Chart-to-table numeric grounding with confidence and human-verification hooks; map-to-data georeferencing with residual-error reporting and descriptive fallback; page-image (vision) retrieval; graph-augmented retrieval; a language-specialized embedding track; and a reproducible **gold-set evaluation harness** (stratified by language, visual density, and document condition) that decides which extensions are promoted to defaults.

## Proposed repository layout

```
agentic-knowledge-base/
├── README.md                  # this file
├── docs/
│   ├── ARCHITECTURE.md        # evidence-unit model, adapter contracts, data flow
│   ├── API.md                 # endpoint reference (mirrors OpenAPI)
│   └── ADAPTERS.md            # how to write a new adapter
├── MKB/                       # the service (FastAPI): models, services, routes
│   ├── models/                # corpus, document, evidence-unit, citation, graph, jobs
│   ├── services/              # extraction, chunking, embedding, vector, lexical,
│   │                          #   rerank, graph, source-connectors, vision (all adapters)
│   ├── routes/                # corpora, ingest, search, citations, graph, stats
│   └── app.py                 # assembles the service
├── adapters/                  # optional external adapter microservices
│   ├── extraction_*/          # e.g. Docling / table / chart-to-table / map
│   ├── embeddings_*/          # e.g. a multilingual embedding server
│   ├── vector_*/              # e.g. pgvector service (dense + lexical)
│   └── graph_*/               # e.g. Neo4j service
├── infra/
│   └── docker-compose.yml     # self-hosted profile
├── examples/
│   └── bioai/                 # BioAI as a *configuration* example — no service code
└── tests/
```

The `MKB/` service stays generic. `examples/bioai/` shows how a real, diverse use case *configures* MKB (which connectors, which extraction adapters, which metadata schema) — as documentation, not as a dependency.

## Status & non-goals

**Status:** design/README stage. This document defines the contract and shape; the service and adapters follow, extracted from and kept coherent with the AGIME knowledge layer.

**Non-goals:** MKB is not an agent framework, not an LLM orchestrator, and not a domain application. It does not enforce answer-level rules (that's the consumer's job) and it does not decide trust for you. It is the evidence substrate underneath all of that.

## License & attribution

To be set by the project. Because MKB is intended for use in regulated and publicly-funded settings, prefer a permissive open-source license and keep third-party model/tool licenses documented per adapter in `docs/ADAPTERS.md`.

---

*MKB keeps one job and does it well: turn diverse, messy, real-world sources into citable evidence — generically, auditably, and on infrastructure you control.*
