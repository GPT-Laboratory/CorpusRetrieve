# Web interface

Served at `/` from the same origin as the API. Disable it with
`CONSOLE_ENABLED=false`, which also removes the `/console` routes.

## Configuration panel

| Control | Effect |
|---|---|
| Corpus name | Used in point IDs, the manifest, and payload `corpus_id` |
| Embedding model | `hash` offline, or `bge-m3` when installed |
| Describe figures with a vision model | Reveals the Anthropic controls |
| Anthropic API key | Sent as `X-Anthropic-API-Key` for this request only |
| Anthropic model | Overrides `DEFAULT_VLM_MODEL` |
| Corpus metadata | JSON object attached to the whole corpus |
| Per-document configuration | JSON object keyed by filename or archive path; includes bibliographic records and a JSON file picker |
| Require resolvable publication metadata | Reject unresolved/ambiguous publication records from release-ready bundles |

The key field has a show/hide toggle and is cleared after submission. The hint
beneath it reports whether a service key is already configured, so it is clear
whether the field is required or an override. The browser only receives a
boolean; the configured key value never leaves the server.

Files can be chosen individually, as a folder, or dropped onto the panel. One
ZIP may be uploaded by itself. Both JSON fields are validated before upload, and
the VLM key requirement is checked client-side so the failure is immediate.

## Modes

### Inspect

The workbench, and the default mode.

- **Stage spine** shows all six stages with status, elapsed time, and counters
  taken from the job's `stages[]`. These are server-reported facts, not an
  animation.
- **Document bar** switches between documents in a multi-document corpus.
- **Chunk list** is filterable by chunk type. Selecting a chunk drives every
  other view.
- **Inspector** shows the selected chunk's identity, role, section path, pages,
  token estimate, character span, and source element IDs. Clicking an element ID
  highlights that region.

The centre panel has four views:

| View | Shows |
|---|---|
| Source | The rendered page with bounding boxes over every element |
| Chunks | A typed span map; width is token count, colour is chunk type |
| Graph | The selected document's knowledge graph, with a type legend, edge kinds, and PNG export |
| Embedding | Vectors projected into 2D; nearby points are semantically similar |

The source view needs page previews, which are rendered for PDF documents when
`EMIT_PAGE_PREVIEWS` is true. Other formats have no page geometry to draw on,
and the view says so rather than failing.

The graph shows one document at a time: the paper in the centre, its sections
on the first ring, and their units (text, claims, figures, tables, references)
further out, laid out the same way every time. Edges are coloured by kind:
outline, evidence, citation, values, vector. `VectorPoint`, `Quantity`, `Unit`
and `TableKV` start hidden; every type is in the legend with its count, and a
click shows or hides it. Selecting a chunk highlights its node and neighbours.
The whole-corpus graph is too large to draw in a browser for big builds; the
caption gives its totals, and the JSON, GraphML and Cypher downloads load it
into Gephi or Neo4j.

## Finished, failed and earlier jobs

When a job finishes, the status bar shows a download button for the bundle with
its size. A job that fails its quality gate, or fails partway, still shows every
output it produced, with a "not release-ready" notice listing each reason. The
job ID is kept in the address (`#job/<id>`), so reloading reopens it, and the
start page lists previous jobs and opens any job by ID. With
`STORAGE_BACKEND=local` this works after a server restart too.

Before an upload, the configuration panel warns when Tesseract or a requested
OCR language is missing and when results are held only in memory. "Accept a
build that is not release-ready" sends `allow_degraded=true`.

### Explain

A plain-language walkthrough of what happened to the corpus, with the real
counters from the run and direct download links.

### Learn

A scrollytelling narrative with an animated visualization that transforms at
each stage, ending in a chunk-size sandbox that shows the precision and context
trade-off against this corpus's actual token count.

The selected mode persists in `localStorage`.

## Assets

```text
corpusretrieve/web/static/
  index.html      markup and script tags
  styles.css      all styling
  app.jsx         the React application
  api-docs.css    Swagger UI theme
  vendor/         React, ReactDOM, Babel, Graphology, Sigma
```

Markup, styling, and behaviour are separate files. `app.jsx` is transformed in
the browser by the vendored Babel, so there is no Node build step and no
outbound request at runtime.

The cost is the 2.8 MB Babel transformer on first load. For deployments where
that matters, precompile `app.jsx` to plain JavaScript, reference the compiled
file from `index.html`, and drop `vendor/babel.min.js`. No other file changes.

## Endpoints used

Reads go to `/console` for browser-specific view models. Uploads and every
download go directly to `/v1`, so a download URL copied from the interface is a
valid API URL.

## Workflow for external group data

Prepare a `*.direct_config.json` file for direct document selection, or a
`*.path_config.json` file for a ZIP upload with relative file paths. Use the
**Per-document configuration** picker and check the matching preview. The
**Corpus metadata** field does not populate publication fields on each chunk.
The browser can upload a batch but does not automatically merge separate jobs
into one index. Use the CLI for a single persistent group index.

In protected production deployments, the browser UI does not automatically
send an `API_KEYS` bearer token; deploy behind a properly configured proxy
or use an authorized API client with `CONSOLE_ENABLED=false`.

See [Generic manifest workflows](MANIFEST_WORKFLOWS.md).
