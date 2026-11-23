const { useState, useRef, useEffect, useCallback } = React;

const STAGES = ["extract", "enrich", "chunk", "embed", "graph", "package"];

const CHUNK_COLORS = {
  section_text: "#60a5fa", claim: "#f59e0b", table_summary: "#a78bfa",
  table_row_group: "#c084fc", table_fact: "#d8b4fe", figure: "#34d399",
  figure_panel: "#6ee7b7", formula: "#fbbf24", reference: "#9ca3af",
  summary: "#fb7185", heading: "#60a5fa", narrative: "#7dd3fc",
  caption: "#34d399", title: "#f472b6",
};

const NODE_COLORS = {
  Paper: "#e63946", Section: "#457b9d", Chunk: "#a8dadc", Claim: "#f59e0b",
  Figure: "#34d399", FigurePanel: "#6ee7b7", Table: "#a78bfa",
  TableRowGroup: "#c084fc", Formula: "#fbbf24", ReferenceWork: "#9ca3af",
  Quantity: "#cdb4db", Unit: "#ffc8dd", Entity: "#bde0fe",
  VectorPoint: "#dee2e6", TableKV: "#e0afa0",
};

const chunkColor = (type) => CHUNK_COLORS[type] || "#9aa4b2";
const nodeColor = (type) => NODE_COLORS[type] || "#8899a6";

const STAGE_INFO = {
  extract: {
    icon: "📄", color: "#60a5fa", title: "1 · Read every document",
    plain: "Each file is opened, its format identified, and every page read — including scanned pages, using OCR when the page carries no text layer. Headings, paragraphs, tables, figures, captions and formulas are located and their exact position on the page is recorded.",
    summary: (d) => `Read ${d.documents || 0} document(s), ${d.pages || 0} page(s), found ${d.elements || 0} elements (${d.with_bbox || 0} with coordinates).`,
  },
  enrich: {
    icon: "🏷️", color: "#f59e0b", title: "2 · Understand the pieces",
    plain: "Each piece is labelled with the part it plays in the document, and numbers, citations and claims are pulled out. When vision enrichment is enabled, figures are described so they can be searched as text.",
    summary: (d) => `Labelled ${Object.keys(d.roles || {}).length} role type(s), ${d.claims || 0} claim(s), ${d.citations || 0} citation link(s)${d.vlm ? `, described ${d.figures_described || 0}/${d.figures_total || 0} figures` : ""}.`,
  },
  chunk: {
    icon: "✂️", color: "#a78bfa", title: "3 · Cut into retrievable pieces",
    plain: "The corpus is split along meaningful boundaries — a whole claim, a table row group, a figure — instead of arbitrary character counts, so every chunk stands on its own and keeps a path back to its source.",
    summary: (d) => `Created ${d.chunks || 0} typed chunks across ${Object.keys(d.by_type || {}).length} type(s).`,
  },
  embed: {
    icon: "🔢", color: "#7dd3fc", title: "4 · Turn text into vectors",
    plain: "Every chunk becomes a dense vector capturing its meaning and a sparse vector capturing its exact terms. Both are written to a portable file that loads into any vector database.",
    summary: (d) => `Encoded ${d.vectors || 0} vectors (${d.dim || "?"}-dimensional, ${d.model || "?"}).`,
  },
  graph: {
    icon: "🕸️", color: "#fb7185", title: "5 · Connect the pieces",
    plain: "Relationships become explicit: which claim is supported by which figure, which chunk belongs to which section, what cites what. The result is exported as JSON, GraphML and Cypher.",
    summary: (d) => `Built a graph with ${d.nodes || 0} nodes and ${d.edges || 0} edges.`,
  },
  package: {
    icon: "📦", color: "#34d399", title: "6 · Package the results",
    plain: "The manifest, an artifact catalogue with a checksum for every file, and a single ZIP bundle are written so the whole corpus can be handed to another system.",
    summary: (d) => `Catalogued ${d.artifacts || 0} artifacts.`,
  },
};

const SNIPPETS = {
  qdrant: `import json
from qdrant_client import QdrantClient, models

client = QdrantClient(":memory:")
records = [json.loads(line) for line in open("index/vectors.jsonl")]
dimension = len(records[0]["vector"])

for collection in {record["collection"] for record in records}:
    client.recreate_collection(
        collection,
        vectors_config=models.VectorParams(
            size=dimension, distance=models.Distance.COSINE
        ),
    )

for record in records:
    client.upsert(record["collection"], points=[models.PointStruct(
        id=record["id"], vector=record["vector"], payload=record["payload"],
    )])`,
  neo4j: `# Idempotent MERGE statements; safe to re-run.
cypher-shell -u neo4j -p <password> \\
  < knowledge_graph/knowledge_graph.cypher`,
  pgvector: `CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE chunks (
    id          text PRIMARY KEY,
    collection  text NOT NULL,
    embedding   vector(512),
    payload     jsonb
);

-- Load index/vectors.jsonl with COPY or a client-side batch insert.
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);`,
};

const formatBytes = (n) =>
  n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1048576).toFixed(1)} MB`;

const countLine = (counts) =>
  Object.entries(counts || {}).map(([k, v]) => `${k}:${v}`).join(" ");

async function readJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch (error) { /* non-JSON error body */ }
    throw new Error(detail);
  }
  return response.json();
}

function Pill({ type }) {
  const color = chunkColor(type);
  return (
    <span className="pill" style={{ background: color + "22", color, border: `1px solid ${color}55` }}>
      {type}
    </span>
  );
}

/* ------------------------------------------------------------------ setup */

function SetupPanel({ settings, onSubmit, busy, error, onOpenJob }) {
  const [files, setFiles] = useState([]);
  const [corpusName, setCorpusName] = useState("corpus");
  const [embedder, setEmbedder] = useState(settings.default_embedder || "hash");
  const [metadata, setMetadata] = useState("{}");
  const [documentConfig, setDocumentConfig] = useState("{}");
  const [requireBibliographic, setRequireBibliographic] = useState(Boolean(settings.require_bibliographic));
  const [allowDegraded, setAllowDegraded] = useState(false);
  const [useVlm, setUseVlm] = useState(false);
  const [apiKey, setApiKey] = useState("");
  const [showKey, setShowKey] = useState(false);
  const [vlmModel, setVlmModel] = useState(settings.default_vlm_model || "");
  const [hot, setHot] = useState(false);
  const [notice, setNotice] = useState("");
  const fileRef = useRef();
  const folderRef = useRef();

  const fileLimit = settings.max_files_per_job || 2000;
  const configPreview = (() => {
    try {
      const parsed = JSON.parse(documentConfig || "{}");
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        return { error: "Document configuration must be a JSON object." };
      }
      if (files.length === 1 && files[0].name.toLowerCase().endsWith(".zip")) {
        return { archive: true, keys: Object.keys(parsed).filter(k => k !== "__collection__").length };
      }
      const seen = {};
      const names = files.map(file => {
        const index = (seen[file.name] || 0) + 1;
        seen[file.name] = index;
        return index === 1 ? file.name : `${file.name}#${index}`;
      });
      const keys = Object.keys(parsed).filter(k => k !== "__collection__");
      const missing = names.filter(name => !Object.prototype.hasOwnProperty.call(parsed, name));
      const unexpected = keys.filter(name => !names.includes(name));
      const malformed = keys.filter(name => !parsed[name] || typeof parsed[name] !== "object" ||
        Array.isArray(parsed[name]) || (parsed[name].metadata != null &&
        (typeof parsed[name].metadata !== "object" || Array.isArray(parsed[name].metadata))));
      return { total: names.length, matched: names.length - missing.length,
               missing, unexpected, malformed, empty: keys.length === 0 };
    } catch (error) { return { error: error.message }; }
  })();

  const addFiles = (list) => {
    const incoming = Array.from(list || []);
    setNotice("");
    setFiles((previous) => {
      const keys = new Set(previous.map((f) => `${f.name}:${f.size}:${f.lastModified}`));
      const merged = [...previous];
      let rejected = 0;
      incoming.forEach((file) => {
        const key = `${file.name}:${file.size}:${file.lastModified}`;
        if (keys.has(key)) return;
        if (merged.length >= fileLimit) { rejected += 1; return; }
        merged.push(file);
        keys.add(key);
      });
      if (rejected) {
        setNotice(
          `This job accepts at most ${fileLimit} documents. ` +
          `${rejected} file${rejected === 1 ? " was" : "s were"} not added.`
        );
      }
      return merged;
    });
  };

  const removeFile = (index) => {
    setNotice("");
    setFiles((previous) => previous.filter((_, i) => i !== index));
  };

  const submit = () =>
    onSubmit({
      files, corpusName, embedder, metadata, documentConfig, requireBibliographic, allowDegraded,
      useVlm, apiKey: apiKey.trim(), vlmModel: vlmModel.trim(),
      clearKey: () => setApiKey(""),
    });

  const keyHint = settings.vlm_api_key_configured
    ? "A service key is configured in the environment. Enter a key only to override it for this job."
    : "No service key is configured. Enter a key for this job, or set ANTHROPIC_API_KEY in the .env file.";

  return (
    <div className="setup">
      <section className="card">
        <div className="card-head"><h2>Documents</h2></div>
        <div className="card-body">
          <div
            className={"drop" + (hot ? " hot" : "")}
            onDragOver={(e) => { e.preventDefault(); setHot(true); }}
            onDragLeave={() => setHot(false)}
            onDrop={(e) => { e.preventDefault(); setHot(false); addFiles(e.dataTransfer.files); }}
          >
            <div className="lead">Drop one document, several files, a folder, or one .zip</div>
            <div>Everything you drop is processed as a single corpus.</div>
            <div className="formats">
              PDF &middot; DOCX &middot; XLSX &middot; CSV &middot; TSV &middot; JSON &middot; JSONL &middot; HTML &middot; TXT &middot; MD &middot; LOG &middot; images
              <br />Up to {fileLimit.toLocaleString()} files, {formatBytes(settings.max_bytes_per_file)} each.
            </div>
            <div className="actions">
              <button type="button" className="btn" onClick={() => fileRef.current.click()}>Choose files</button>
              <button type="button" className="btn" onClick={() => folderRef.current.click()}>Choose folder</button>
            </div>
          </div>
          <input ref={fileRef} type="file" multiple hidden
            onChange={(e) => { addFiles(e.target.files); e.target.value = ""; }} />
          <input ref={folderRef} type="file" webkitdirectory="" directory="" multiple hidden
            onChange={(e) => { addFiles(e.target.files); e.target.value = ""; }} />

          {notice && <div className="hint warn">{notice}</div>}
          {files.length > 0 && (
            <div className="hint">
              {files.length.toLocaleString()} of {fileLimit.toLocaleString()} files selected
            </div>
          )}

          <div className="filelist">
            {files.map((file, index) => (
              <div className="filerow" key={`${file.name}-${index}`}>
                <span className="nm">{file.webkitRelativePath || file.name}</span>
                <span className="sz">{formatBytes(file.size)}</span>
                <button type="button" title="Remove" onClick={() => removeFile(index)}>&times;</button>
              </div>
            ))}
          </div>
        </div>
      </section>

      <aside className="card">
        <div className="card-head"><h2>Configuration</h2></div>
        <div className="card-body">
          <div className="field">
            <label htmlFor="corpusName">Corpus name</label>
            <input id="corpusName" maxLength={200} value={corpusName}
              onChange={(e) => setCorpusName(e.target.value)} />
          </div>

          <div className="field">
            <label htmlFor="embedder">Embedding model</label>
            <select id="embedder" value={embedder} onChange={(e) => setEmbedder(e.target.value)}>
              {(settings.embedders || [{ spec: "hash", label: "Hashing", available: true }]).map((option) => (
                <option key={option.spec} value={option.spec} disabled={!option.available}>
                  {option.label}{option.available ? "" : " (not installed)"}
                </option>
              ))}
            </select>
            {(() => {
              const chosen = (settings.embedders || []).find((option) => option.spec === embedder);
              if (chosen && chosen.kind === "lexical") {
                return <div className="hint">Lexical: finds passages that share words with the question, not passages that mean the same thing. A semantic model can be added later with <code>corpusretrieve reindex</code>, without processing the documents again.</div>;
              }
              if (chosen && chosen.kind === "semantic") {
                return <div className="hint">Semantic: the model is downloaded on first use. Without a GPU, embedding a large corpus can take hours.</div>;
              }
              return null;
            })()}
            {(settings.embedders || []).some((option) => !option.available) && (
              <div className="hint">Semantic models need <code>pip install -r requirements-embeddings.txt</code>.</div>
            )}
          </div>

          <div className="switch">
            <input id="vlm" type="checkbox" checked={useVlm}
              onChange={(e) => setUseVlm(e.target.checked)} />
            <label htmlFor="vlm">Describe figures with a vision model</label>
          </div>

          {useVlm && (
            <div className="vlmbox">
              <div className="field">
                <label htmlFor="apiKey">Anthropic API key</label>
                <div className="keywrap">
                  <input id="apiKey" type={showKey ? "text" : "password"} autoComplete="off"
                    placeholder="sk-ant-..." value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)} />
                  <button type="button" className="btn reveal" onClick={() => setShowKey(!showKey)}>
                    {showKey ? "Hide" : "Show"}
                  </button>
                </div>
                <div className={"hint " + (settings.vlm_api_key_configured ? "ok" : "warn")}>{keyHint}</div>
                <div className="hint">
                  The key is sent once with this upload, held only for the duration of the job,
                  and never written to artifacts, logs, or manifests.
                </div>
              </div>
              <div className="field" style={{ marginBottom: 0 }}>
                <label htmlFor="vlmModel">Anthropic model</label>
                <input id="vlmModel" value={vlmModel} onChange={(e) => setVlmModel(e.target.value)} />
              </div>
            </div>
          )}

          <div className="field">
            <label htmlFor="metadata">Corpus metadata (JSON)</label>
            <textarea id="metadata" value={metadata} onChange={(e) => setMetadata(e.target.value)} />
          </div>

          <div className="field">
            <label htmlFor="documentConfig">Per-document configuration (JSON)</label>
            <textarea id="documentConfig" value={documentConfig}
              onChange={(e) => setDocumentConfig(e.target.value)} />
            <div className="hint">This field supplies per-document publication metadata (title, authors, year, DOI, and source links). Generate it from any supported external manifest, or provide it manually. Collection metadata above is separate.</div>
            <label className="btn" style={{ display: "inline-block", marginTop: 8, cursor: "pointer" }}>
              Load document_config.json
              <input type="file" accept=".json,application/json" hidden
                onChange={async (e) => {
                  const file = e.target.files && e.target.files[0];
                  if (file) setDocumentConfig(await file.text());
                  e.target.value = "";
                }} />
            </label>
            <details style={{ marginTop: 8 }}>
              <summary>Example for one PDF</summary>
              <pre className="config-example">{JSON.stringify({
                "article.pdf": {document_id: "source-123", metadata: {records: [
                  {title: "Example publication title", year: 2025,
                   handle_url: "https://repository.example/123"}
                ]}}
              }, null, 2)}</pre>
            </details>
            {configPreview.error && <div className="hint warn">Invalid configuration: {configPreview.error}</div>}
            {configPreview.archive && <div className="hint">ZIP archive: {configPreview.keys} configuration keys. Matching is verified after archive expansion on the server; keys should be archive paths or exact filenames.</div>}
            {!configPreview.archive && !configPreview.error && files.length > 0 && (
              <div className="hint">
                {configPreview.matched}/{configPreview.total} selected files matched by exact filename.
                {configPreview.missing.length > 0 && (
                  <div className="warn">No config for: {configPreview.missing.slice(0, 5).join(", ")}{configPreview.missing.length > 5 ? " ..." : ""}. PDF DOI fallback may supply an extracted record; strict mode will reject unresolved files.</div>
                )}
                {configPreview.unexpected.length > 0 && (
                  <div className="warn">Config keys not selected: {configPreview.unexpected.slice(0, 5).join(", ")}{configPreview.unexpected.length > 5 ? " ..." : ""}</div>
                )}
                {configPreview.malformed.length > 0 && (
                  <div className="warn">Malformed metadata entries: {configPreview.malformed.slice(0, 5).join(", ")}</div>
                )}
              </div>
            )}
          </div>
          <div className="switch">
            <input id="requireBibliographic" type="checkbox" checked={requireBibliographic}
              onChange={(e) => setRequireBibliographic(e.target.checked)} />
            <label htmlFor="requireBibliographic">Require resolvable publication metadata (strict provenance)</label>
          </div>
          <div className="hint">If no manifest record matches, the extractor attempts a labelled DOI fallback. Ambiguous or missing publication identities keep the release from being marked ready.</div>

          <div className="switch">
            <input id="allowDegraded" type="checkbox" checked={allowDegraded}
              onChange={(e) => setAllowDegraded(e.target.checked)} />
            <label htmlFor="allowDegraded">Accept a build that is not release-ready</label>
          </div>
          <div className="hint">
            Off: mismatched configuration stops the upload before processing, and a build with
            quality problems finishes as failed (its outputs stay downloadable). On: the build
            finishes as completed and the problems are listed in the manifest.
          </div>

          {settings.ocr && !settings.ocr.ready && (
            <div className="preflight warn">{settings.ocr.message}</div>
          )}
          {settings.storage_backend === "memory" && (
            <div className="preflight">
              Results are kept in server memory and are lost when the server stops. Download the
              bundle when a job finishes, or set <code>STORAGE_BACKEND=local</code> to keep them on disk.
            </div>
          )}

          <button className="btn primary" style={{ width: "100%" }}
            disabled={busy || files.length === 0} onClick={submit}>
            {busy ? "Processing…" : `Process ${files.length || ""} file(s)`}
          </button>
          {error && (
            <div className="formerror">
              {error.message || String(error)}
              {error.errors && error.errors.length > 0 && (
                <ul>
                  {error.errors.map((item, index) => (
                    item.check ? (
                      <li key={item.check}>
                        <b>{item.check.replace(/_/g, " ")}</b>: {item.count}
                        {item.examples && item.examples.length > 0 && ` (${item.examples.slice(0, 5).join(", ")}${item.count > 5 ? ", …" : ""})`}
                      </li>
                    ) : (
                      <li key={index}><b>{item.field || "request"}</b>: {item.message || JSON.stringify(item)}</li>
                    )
                  ))}
                </ul>
              )}
            </div>
          )}
        </div>
      </aside>
      <RecentJobs onOpen={onOpenJob} />
    </div>
  );
}

function RecentJobs({ onOpen }) {
  const [jobs, setJobs] = useState(null);
  const [typed, setTyped] = useState("");
  useEffect(() => {
    readJson("/console/jobs").then((data) => setJobs(data.jobs)).catch(() => setJobs([]));
  }, []);
  const when = (seconds) => new Date(seconds * 1000).toLocaleString();
  return (
    <section className="card recent">
      <div className="card-head"><h2>Previous jobs</h2></div>
      <div className="card-body">
        <div className="openjob">
          <input placeholder="Job ID" value={typed} maxLength={24}
            onChange={(e) => setTyped(e.target.value.trim())}
            onKeyDown={(e) => { if (e.key === "Enter" && typed) onOpen(typed); }} />
          <button className="btn" disabled={!/^[0-9a-f]{24}$/.test(typed)} onClick={() => onOpen(typed)}>Open</button>
        </div>
        {jobs === null && <div className="hint">Loading…</div>}
        {jobs && jobs.length === 0 && (
          <div className="hint">No jobs on this server yet. Jobs started here appear in this list until the server stops (or permanently with STORAGE_BACKEND=local).</div>
        )}
        {jobs && jobs.length > 0 && (
          <div className="joblist">
            {jobs.map((item) => (
              <button key={item.job_id} className="jobrow" onClick={() => onOpen(item.job_id)}>
                <span className={"state " + item.state}>{item.state}</span>
                <span className="nm">{item.corpus_name || "corpus"}</span>
                <span className="meta">
                  {item.release_ready === false && item.artifacts_available ? "not release-ready · " : ""}
                  {when(item.created_at)}
                </span>
                <span className="id">{item.job_id}</span>
              </button>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

/* ------------------------------------------------------------ stage spine */

function stageDetailText(name, detail) {
  if (!detail) return "";
  if (name === "extract") {
    return `${detail.documents || 0} docs · ${detail.pages || 0}p · ${detail.elements || 0} elements\n${countLine(detail.by_kind)}`;
  }
  if (name === "enrich") {
    let text = `claims:${detail.claims || 0} · cites:${detail.citations || 0}\n${countLine(detail.roles)}`;
    if (detail.vlm) text += `\nVLM: ${detail.figures_described || 0}/${detail.figures_total || 0} figures`;
    return text;
  }
  if (name === "chunk") return `${detail.chunks || 0} chunks\n${countLine(detail.by_type)}`;
  if (name === "embed") return `${detail.vectors || 0} vectors · ${detail.model || ""}\ndim ${detail.dim || "?"} · ${detail.metric || ""}`;
  if (name === "graph") return `${detail.nodes || 0} nodes · ${detail.edges || 0} edges`;
  if (name === "package") return `${detail.artifacts || 0} artifacts`;
  return "";
}

function StageSpine({ stages }) {
  const byName = Object.fromEntries((stages || []).map((s) => [s.name, s]));
  return (
    <div className="spine">
      {STAGES.map((name, index) => {
        const stage = byName[name] || { status: "pending" };
        return (
          <div key={name} className={"stage " + stage.status}>
            <div className="ms">{stage.elapsed_ms != null ? `${stage.elapsed_ms}ms` : ""}</div>
            <div className="num">STAGE {index + 1}</div>
            <div className="name">{name}</div>
            <div className="st">{stage.status}</div>
            {stage.detail && Object.keys(stage.detail).length > 0 && (
              <div className="detail">{stageDetailText(name, stage.detail)}</div>
            )}
            <div className="bar"></div>
          </div>
        );
      })}
    </div>
  );
}

function JobStatus({ job, jobId, bundleSize }) {
  if (!job) return null;
  const finished = job.state === "completed" || job.state === "failed";
  const reasons = job.quality_reasons || [];
  return (
    <div className="jobstatus">
      <div className="row">
        <span style={{ fontFamily: "var(--mono)", textTransform: "capitalize" }}>
          {job.state} &middot; {job.stage}
        </span>
        <div className="track"><div className="fill" style={{ width: `${job.progress || 0}%` }} /></div>
        <span style={{ fontFamily: "var(--mono)" }}>{job.progress || 0}%</span>
        <span style={{ color: "var(--dim)" }}>{job.message}</span>
        {finished && job.artifacts_available && (
          <a className="btn primary download" href={`/v1/jobs/${jobId}/bundle`}>
            Download bundle{bundleSize ? ` (${formatBytes(bundleSize)})` : ""}
          </a>
        )}
      </div>
      <div className="jobid">Job {jobId}</div>
      {finished && job.artifacts_available && job.release_ready === false && (
        <div className="outcome">
          <b>Not release-ready.</b> Every output was kept and can be inspected and downloaded below.
          {reasons.length > 0 && (
            <div className="reasons">
              {reasons.map((reason) => <span key={reason} className="reason">{reason}</span>)}
            </div>
          )}
        </div>
      )}
      {job.error && !(job.error.startsWith("QualityGateFailed") && reasons.length) && (
        <div className="err">{job.error}</div>
      )}
    </div>
  );
}

function DocumentBar({ overview, activeDoc, onSelect, jobId }) {
  const [query, setQuery] = useState("");
  if (!overview) return null;
  const many = overview.documents.length > 30;
  const needle = query.trim().toLowerCase();
  const matches = many
    ? overview.documents.filter((document) => !needle
        || document.source.toLowerCase().includes(needle)
        || document.doc_id.toLowerCase().includes(needle)).slice(0, 40)
    : overview.documents;
  const active = overview.documents.find((document) => document.doc_id === activeDoc);
  return (
    <div className="jobbar">
      <span className="label">
        {overview.corpus_name} &middot; {overview.documents.length} document(s)
      </span>
      {many && (
        <input className="docsearch" placeholder={active ? active.source : "Find a document"}
          value={query} onChange={(e) => setQuery(e.target.value)} />
      )}
      <div className="docs">
        {matches.map((document) => (
          <button
            key={document.doc_id}
            className={"docchip" + (document.doc_id === activeDoc ? " on" : "")}
            title={`${document.source} · ${document.chunks} chunks`}
            onClick={() => onSelect(document.doc_id)}
          >
            <span className={"dot" + (document.degraded ? " degraded" : "")}></span>
            {document.source.length > 28 ? `${document.source.slice(0, 26)}…` : document.source}
          </button>
        ))}
      </div>
      {overview.warnings.length > 0 && (
        <span className="warn" title={overview.warnings.join("\n")}>
          {overview.warnings.length} warning(s)
        </span>
      )}
      <a className="get" href={`/v1/jobs/${jobId}/bundle`}>Download bundle</a>
    </div>
  );
}

/* -------------------------------------------------------------- workbench */

function PageOverlay({ jobId, docId, page, preview, elements, highlighted, onPick }) {
  const [natural, setNatural] = useState(null);
  const url = `/v1/jobs/${jobId}/documents/${docId}/assets/${preview.path}`;
  return (
    <div className="pagewrap">
      <img src={url} alt={`Page ${page}`}
        onLoad={(e) => setNatural({ w: e.target.clientWidth, h: e.target.clientHeight })}
        onError={() => setNatural(null)} />
      {natural && elements.filter((element) => element.bbox).map((element) => {
        const sx = natural.w / preview.width;
        const sy = natural.h / preview.height;
        const [x0, y0, x1, y1] = element.bbox;
        const on = highlighted.includes(element.id);
        return (
          <div key={element.id} className={"ov" + (on ? " hi" : "")}
            title={`${element.kind}${element.role ? ` · ${element.role}` : ""}\n${(element.text || "").slice(0, 120)}`}
            onClick={() => onPick(element.id)}
            style={{
              left: x0 * sx, top: y0 * sy,
              width: (x1 - x0) * sx, height: (y1 - y0) * sy,
              borderColor: on ? undefined : chunkColor(element.kind) + "88",
            }} />
        );
      })}
    </div>
  );
}

function SpanMap({ chunks, selected, onSelect }) {
  const [tip, setTip] = useState(null);
  const ordered = [...chunks].sort((a, b) => {
    const pa = (a.pages && a.pages[0]) || 999;
    const pb = (b.pages && b.pages[0]) || 999;
    return pa !== pb ? pa - pb : (a.char_start || 0) - (b.char_start || 0);
  });
  const types = [...new Set(chunks.map((c) => c.chunk_type))];
  return (
    <div>
      <div className="legend" style={{ padding: "6px 14px" }}>
        {types.map((type) => (
          <span key={type} className="lg" style={{ color: chunkColor(type) }}>
            <span className="swatch" style={{ background: chunkColor(type), borderRadius: 2 }}></span>{type}
          </span>
        ))}
      </div>
      <div className="spanmap">
        {ordered.map((chunk) => (
          <span key={chunk.chunk_id}
            className={"span" + (selected && selected.chunk_id === chunk.chunk_id ? " sel" : "")}
            style={{
              background: chunkColor(chunk.chunk_type) + "33",
              borderBottom: `2px solid ${chunkColor(chunk.chunk_type)}`,
              width: Math.max(30, Math.min(320, (chunk.token_estimate || 20) * 1.1)),
            }}
            onMouseEnter={(e) => setTip({ chunk, x: e.clientX, y: e.clientY })}
            onMouseMove={(e) => setTip({ chunk, x: e.clientX, y: e.clientY })}
            onMouseLeave={() => setTip(null)}
            onClick={() => onSelect(chunk)} />
        ))}
      </div>
      <div className="caption">
        each bar = one chunk &middot; width proportional to token count &middot; colour = type &middot; click to open
      </div>
      {tip && (
        <div className="tip" style={{ left: Math.min(tip.x + 14, window.innerWidth - 340), top: tip.y + 14 }}>
          <div>
            <b style={{ color: chunkColor(tip.chunk.chunk_type) }}>{tip.chunk.chunk_type}</b>
            {" · "}{tip.chunk.token_estimate} tok
            {tip.chunk.pages && tip.chunk.pages.length ? ` · p${tip.chunk.pages.join(",")}` : ""}
          </div>
          {tip.chunk.section_path && tip.chunk.section_path.length > 0 && (
            <div style={{ color: "var(--dim)", marginTop: 3 }}>{tip.chunk.section_path.join(" › ")}</div>
          )}
          <div style={{ color: "var(--dim)", marginTop: 3 }}>{tip.chunk.text.slice(0, 140)}…</div>
        </div>
      )}
    </div>
  );
}

const EDGE_FAMILIES = [
  [/^(PAPER_HAS|SECTION_HAS|TABLE_HAS|FIGURE_HAS)/, "#3a4250", "outline"],
  [/^CLAIM_SUPPORTED_BY/, "#b7791f", "evidence"],
  [/CITES/, "#3b6ea5", "citation"],
  [/(QUANTITY|UNIT|_KV|SYMBOL)/, "#7c5aa6", "values"],
  [/VECTOR_POINT/, "#4b5563", "vector"],
];
const edgeFamily = (rel) => EDGE_FAMILIES.find(([pattern]) => pattern.test(rel || "")) || [null, "#3a4250", "other"];
// cross-links are drawn on request, so the outline stays readable
const DEFAULT_HIDDEN_EDGES = ["citation", "values", "vector"];
const NODE_SIZE = { Paper: 12, Section: 7, Table: 4.5, Figure: 4.5, Claim: 3, Chunk: 2.5 };

function SigmaGraph({ graph, selected, jobId }) {
  const container = useRef();
  const renderer = useRef();
  const graphRef = useRef();
  const [picked, setPicked] = useState(null);
  const [hidden, setHidden] = useState(new Set());
  const [hiddenEdges, setHiddenEdges] = useState(new Set(DEFAULT_HIDDEN_EDGES));

  useEffect(() => {
    setHidden(new Set((graph && graph.default_hidden) || []));
    setPicked(null);
  }, [graph]);

  useEffect(() => {
    if (!container.current || !graph || graph.error) return undefined;
    const Graph = window.graphology;
    const SigmaCtor = window.Sigma && (window.Sigma.Sigma || window.Sigma);
    if (!Graph || !SigmaCtor) return undefined;

    const model = new Graph({ multi: true });
    graph.nodes.filter((node) => !hidden.has(node.type)).forEach((node) => {
      model.addNode(node.id, {
        label: String(node.label || "").slice(0, 48),
        size: NODE_SIZE[node.type] || 2.5,
        color: nodeColor(node.type),
        x: node.x, y: -node.y,
        _type: node.type, _role: node.role, _text: node.text, _label: node.label,
        _pages: node.pages, _degree: node.degree, _chunk: node.chunk_id,
      });
    });
    graph.edges.forEach((edge, index) => {
      const [, color, family] = edgeFamily(edge.rel);
      if (!hiddenEdges.has(family) && model.hasNode(edge.source) && model.hasNode(edge.target)) {
        model.addEdgeWithKey(`e${index}`, edge.source, edge.target,
          { size: 0.7, color, _rel: edge.rel });
      }
    });

    if (renderer.current) { renderer.current.kill(); renderer.current = null; }
    const sigma = new SigmaCtor(model, container.current, {
      // only the paper and its sections are labelled; hover or click for the rest
      labelRenderedSizeThreshold: 6,
      labelColor: { color: "#9aa4b2" },
      defaultEdgeColor: "#3a4250",
    });
    renderer.current = sigma;
    graphRef.current = model;
    requestAnimationFrame(() => { try { sigma.resize(); sigma.refresh(); } catch (error) { /* not visible yet */ } });

    sigma.on("clickNode", ({ node }) => {
      const attributes = model.getNodeAttributes(node);
      setPicked({
        id: node, type: attributes._type, role: attributes._role, label: attributes._label,
        text: attributes._text, pages: attributes._pages, degree: attributes._degree,
        neighbours: model.neighbors(node).map((other) => ({
          id: other,
          type: model.getNodeAttribute(other, "_type"),
          label: model.getNodeAttribute(other, "label"),
          rel: model.edges(node, other).concat(model.edges(other, node))
            .map((key) => model.getEdgeAttribute(key, "_rel"))[0],
          outgoing: model.edges(node, other).length > 0,
        })),
      });
    });
    sigma.on("clickStage", () => setPicked(null));

    return () => { if (renderer.current) { renderer.current.kill(); renderer.current = null; } };
  }, [graph, hidden, hiddenEdges]);

  useEffect(() => {
    const model = graphRef.current;
    const sigma = renderer.current;
    if (!model || !sigma) return;
    let target = null;
    if (selected) {
      model.forEachNode((node, attributes) => { if (attributes._chunk === selected.chunk_id) target = node; });
    }
    const neighbourhood = new Set(target ? [target, ...model.neighbors(target)] : []);
    model.forEachNode((node) =>
      model.setNodeAttribute(node, "color",
        target && !neighbourhood.has(node) ? "#262c36" : nodeColor(model.getNodeAttribute(node, "_type"))));
    sigma.refresh();
  }, [selected, graph, hidden, hiddenEdges]);

  if (graph && graph.error) return <div className="empty">Knowledge graph unavailable: {graph.error}</div>;
  const counts = {};
  ((graph && graph.nodes) || []).forEach((node) => { counts[node.type] = (counts[node.type] || 0) + 1; });
  const types = Object.keys(counts).sort((a, b) => counts[b] - counts[a]);
  const shown = ((graph && graph.nodes) || []).filter((node) => !hidden.has(node.type)).length;

  const exportPng = () => {
    const canvases = container.current.querySelectorAll("canvas");
    const output = document.createElement("canvas");
    output.width = container.current.clientWidth;
    output.height = container.current.clientHeight;
    const context = output.getContext("2d");
    context.fillStyle = "#0a0e14";
    context.fillRect(0, 0, output.width, output.height);
    canvases.forEach((canvas) => context.drawImage(canvas, 0, 0, output.width, output.height));
    const link = document.createElement("a");
    link.href = output.toDataURL("image/png");
    link.download = `knowledge_graph_${graph.doc_id}.png`;
    link.click();
  };

  return (
    <div style={{ padding: "6px 14px" }}>
      <div className="legend">
        {types.map((type) => (
          <span key={type} className={"lg" + (hidden.has(type) ? " off" : "")}
            style={{ color: nodeColor(type) }}
            title={hidden.has(type) ? "Show this type" : "Hide this type"}
            onClick={() => setHidden((previous) => {
              const next = new Set(previous);
              if (next.has(type)) next.delete(type); else next.add(type);
              return next;
            })}>
            <span className="swatch" style={{ background: nodeColor(type) }}></span>{type} {counts[type]}
          </span>
        ))}
        <span style={{ marginLeft: "auto", display: "flex", gap: 6 }}>
          <button className="gbtn" onClick={exportPng}>Export PNG</button>
          <a className="gbtn" href={`/v1/jobs/${jobId}/graph`}>Corpus JSON</a>
          <a className="gbtn" href={`/v1/jobs/${jobId}/graph/cypher`}>Cypher</a>
        </span>
      </div>
      <div className="legend edges">
        {EDGE_FAMILIES.map(([, color, name]) => (
          <span key={name} className={"lg" + (hiddenEdges.has(name) ? " off" : "")} style={{ color }}
            title={hiddenEdges.has(name) ? "Show these links" : "Hide these links"}
            onClick={() => setHiddenEdges((previous) => {
              const next = new Set(previous);
              if (next.has(name)) next.delete(name); else next.add(name);
              return next;
            })}>
            <span className="line" style={{ background: color }}></span>{name}
          </span>
        ))}
      </div>
      <div id="sigma" ref={container}></div>
      {picked ? (
        <div className="nodecard">
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span className="swatch" style={{ background: nodeColor(picked.type), width: 10, height: 10 }}></span>
            <b>{picked.type}</b>
            <span style={{ color: "var(--faint)", fontSize: 11, fontFamily: "var(--mono)", marginLeft: "auto" }}>
              {picked.degree} connections{picked.pages ? ` · page ${picked.pages}` : ""}
            </span>
          </div>
          <div style={{ fontSize: 12, margin: "6px 0" }}>{picked.label}</div>
          <div style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--faint)", marginBottom: 6 }}>{picked.id}</div>
          {picked.text && picked.text !== picked.label && (
            <div style={{ fontSize: 12, color: "var(--dim)", marginBottom: 6 }}>{picked.text}</div>
          )}
          {picked.neighbours && picked.neighbours.length > 0 && (
            <div style={{ display: "flex", flexDirection: "column", gap: 3, maxHeight: 150, overflow: "auto" }}>
              {picked.neighbours.slice(0, 20).map((neighbour, index) => (
                <div key={index} style={{ fontSize: 11, fontFamily: "var(--mono)" }}>
                  <span style={{ color: edgeFamily(neighbour.rel)[1] }}>
                    {neighbour.outgoing ? "" : "← "}{neighbour.rel || "—"}{neighbour.outgoing ? " →" : ""}
                  </span>{" "}
                  <span style={{ color: nodeColor(neighbour.type) }}>{neighbour.type}</span>{" "}
                  <span style={{ color: "var(--dim)" }}>{String(neighbour.label).slice(0, 40)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      ) : (
        <div className="caption">
          {graph ? (
            <>
              This document: {shown} of {graph.total_nodes} nodes shown{graph.truncated ? ` (largest ${graph.n_nodes} kept)` : ""}, {graph.n_edges} edges.
              {graph.corpus && graph.corpus.nodes ? ` Whole corpus: ${graph.corpus.nodes.toLocaleString()} nodes, ${graph.corpus.edges.toLocaleString()} edges (download it to explore it in Neo4j or Gephi).` : ""}
              {" "}The paper is at the centre; each ring is one level deeper in its outline. Click a node to inspect it, a legend entry to show or hide a node type or a kind of link (citations and values start hidden).
            </>
          ) : "Loading this document's graph…"}
        </div>
      )}
    </div>
  );
}

function EmbeddingMap({ jobId, selected, onSelect }) {
  const [projection, setProjection] = useState(null);
  const [error, setError] = useState(null);
  const [tip, setTip] = useState(null);

  useEffect(() => {
    setProjection(null);
    setError(null);
    readJson(`/console/jobs/${jobId}/projection`).then(setProjection).catch((e) => setError(e.message));
  }, [jobId]);

  if (error) return <div className="empty">Embedding map unavailable: {error}</div>;
  if (!projection) return <div className="empty">Projecting embeddings into two dimensions…</div>;

  const types = [...new Set(projection.points.map((point) => point.type))];
  const width = 760;
  const height = 560;
  const pad = 30;

  return (
    <div style={{ padding: "6px 14px" }}>
      <div className="legend">
        {types.map((type) => (
          <span key={type} className="lg" style={{ color: chunkColor(type) }}>
            <span className="swatch" style={{ background: chunkColor(type) }}></span>{type}
          </span>
        ))}
        <span style={{ marginLeft: "auto", color: "var(--faint)", fontFamily: "var(--mono)", fontSize: 11 }}>
          {projection.method.toUpperCase()} &middot; {projection.n} chunks
        </span>
      </div>
      <svg width="100%" viewBox={`0 0 ${width} ${height}`}
        style={{ background: "#0a0e14", borderRadius: 8, border: "1px solid var(--border)" }}>
        {projection.points.map((point) => {
          const on = selected && selected.chunk_id === point.chunk_id;
          return (
            <circle key={point.id}
              cx={pad + point.x * (width - 2 * pad)}
              cy={pad + (1 - point.y) * (height - 2 * pad)}
              r={on ? 6 : 3.5}
              fill={chunkColor(point.type)}
              stroke={on ? "#fff" : "none"} strokeWidth={on ? 1.5 : 0}
              style={{ cursor: "pointer", transition: ".15s" }}
              onMouseEnter={(e) => setTip({ point, x: e.clientX, y: e.clientY })}
              onMouseMove={(e) => setTip({ point, x: e.clientX, y: e.clientY })}
              onMouseLeave={() => setTip(null)}
              onClick={() => onSelect && onSelect(point)} />
          );
        })}
      </svg>
      <div className="caption">
        each dot = one chunk &middot; nearby dots have similar meaning &middot; colour = type &middot; distances are approximate
      </div>
      {tip && (
        <div className="tip" style={{ left: Math.min(tip.x + 14, window.innerWidth - 340), top: tip.y + 14 }}>
          <div>
            <b style={{ color: chunkColor(tip.point.type) }}>{tip.point.type}</b>
            {tip.point.pages && tip.point.pages.length ? ` · p${tip.point.pages.join(",")}` : ""}
          </div>
          <div style={{ color: "var(--dim)", marginTop: 3 }}>{tip.point.text}…</div>
        </div>
      )}
    </div>
  );
}

function Inspector({ chunk, onHighlight }) {
  if (!chunk) {
    return (
      <div style={{ padding: 14, color: "var(--faint)", fontSize: 12 }}>
        Select a chunk to inspect its metadata, provenance, and source elements.
      </div>
    );
  }
  return (
    <div className="insp">
      <Pill type={chunk.chunk_type} />
      <div className="k">chunk id</div><div className="v">{chunk.chunk_id}</div>
      {chunk.role && (<><div className="k">role</div><div className="v">{chunk.role}</div></>)}
      {chunk.section_path && chunk.section_path.length > 0 && (
        <><div className="k">section path</div><div className="v">{chunk.section_path.join("  ›  ")}</div></>
      )}
      <div className="k">pages</div>
      <div className="v">{chunk.pages && chunk.pages.length ? chunk.pages.join(", ") : "—"}</div>
      {chunk.label && (<><div className="k">label</div><div className="v">{chunk.label}</div></>)}
      <div className="k">token estimate</div><div className="v">{chunk.token_estimate}</div>
      {chunk.char_start != null && (
        <><div className="k">char span</div><div className="v">{chunk.char_start} – {chunk.char_end}</div></>
      )}
      <div className="k">source elements ({(chunk.element_ids || []).length}) — click to highlight</div>
      <div className="trace">
        {(chunk.element_ids || []).map((id) => (
          <span key={id} className="e" onClick={() => onHighlight([id])}>{id.slice(0, 8)}</span>
        ))}
      </div>
      <div className="k">text</div><div className="txt">{chunk.text}</div>
    </div>
  );
}

function ArtifactPanel({ artifacts, jobId }) {
  const [tab, setTab] = useState("qdrant");
  const featured = [
    ["index/vectors.jsonl", "dense + sparse vectors, any vector database"],
    ["knowledge_graph/knowledge_graph.json", "nodes and edges"],
    ["knowledge_graph/knowledge_graph.cypher", "Neo4j or Memgraph import"],
    ["knowledge_graph/knowledge_graph.graphml", "Gephi or yEd"],
    ["index/dense.npy", "dense vectors as a float32 matrix"],
    ["index/metadata.json", "embedding model, dimension, and query settings"],
    ["corpus/chunks.json", "every chunk in the corpus"],
    ["manifest.json", "job manifest, quality report, and document inventory"],
  ];
  const items = (artifacts && artifacts.artifacts) || [];
  const sizes = Object.fromEntries(items.map((item) => [item.path, item.size_bytes]));
  const groups = (artifacts && artifacts.artifact_groups) || [];
  return (
    <>
      <h2 style={{ marginTop: 10 }}>Portable artifacts</h2>
      <div className="dl">
        {featured.filter(([path]) => path in sizes).map(([path, description]) => (
          <div className="item" key={path}>
            <span className="nm">{path.split("/").pop()}</span>
            <span className="ds">{description} &middot; {formatBytes(sizes[path])}</span>
            <a className="get" href={`/v1/jobs/${jobId}/artifacts/download?path=${encodeURIComponent(path)}`}>
              download
            </a>
          </div>
        ))}
        {artifacts && artifacts.bundle && (
          <div className="item">
            <span className="nm">bundle.zip</span>
            <span className="ds">every artifact in one archive &middot; {formatBytes(artifacts.bundle_size_bytes || 0)}</span>
            <a className="get" href={artifacts.bundle}>download</a>
          </div>
        )}
      </div>
      {groups.length > 0 && (
        <div className="groups">
          {groups.map((group) => `${group.name}: ${group.count.toLocaleString()} file(s), ${formatBytes(group.size_bytes)}`).join(" · ")}
        </div>
      )}
      <h2>Load into your stack</h2>
      <div className="tabs">
        {Object.keys(SNIPPETS).map((name) => (
          <span key={name} className={"tab" + (tab === name ? " on" : "")} onClick={() => setTab(name)}>{name}</span>
        ))}
      </div>
      <div className="snippet">{SNIPPETS[tab]}</div>
    </>
  );
}

function Workbench({ jobId, overview, activeDoc, chunks, elements, page, setPage, selected, setSelected, highlighted, setHighlighted, graph, artifacts }) {
  const [view, setView] = useState("source");
  const [typeFilter, setTypeFilter] = useState(null);

  const pageData = elements[page];
  const preview = pageData && pageData.page_previews && pageData.page_previews[String(page)];
  const filtered = typeFilter ? chunks.filter((c) => c.chunk_type === typeFilter) : chunks;
  const types = [...new Set(chunks.map((c) => c.chunk_type))];

  const titles = {
    source: `Source · provenance overlay${preview ? ` · page ${page}` : ""}`,
    chunks: "Typed-chunk map",
    graph: "Knowledge graph",
    embedding: "Embedding map",
  };

  return (
    <div className="main">
      <div className="col">
        <h2>Chunks {chunks.length ? `(${filtered.length})` : ""}</h2>
        {types.length > 0 && (
          <div className="filt">
            <span className={"f" + (!typeFilter ? " on" : "")} onClick={() => setTypeFilter(null)}>all</span>
            {types.map((type) => (
              <span key={type} className={"f" + (typeFilter === type ? " on" : "")}
                style={typeFilter === type ? { borderColor: chunkColor(type), color: chunkColor(type) } : {}}
                onClick={() => setTypeFilter(type)}>{type}</span>
            ))}
          </div>
        )}
        {filtered.map((chunk) => (
          <div key={chunk.chunk_id}
            className={"chunk" + (selected && selected.chunk_id === chunk.chunk_id ? " sel" : "")}
            onClick={() => setSelected(chunk)}>
            <div className="row">
              <Pill type={chunk.chunk_type} />
              <span className="cid">{chunk.chunk_id.split("::").slice(-2).join("::")}</span>
            </div>
            <div className="prev">{chunk.text.slice(0, 140)}</div>
            <div className="meta">
              {chunk.pages && chunk.pages.length ? `p${chunk.pages.join(",")}` : "—"}
              {" · "}{chunk.token_estimate || 0} tok
              {" · "}{(chunk.element_ids || []).length} elts
              {chunk.section_path && chunk.section_path.length ? ` · ${chunk.section_path[0]}` : ""}
            </div>
          </div>
        ))}
        {!chunks.length && (
          <div style={{ padding: 14, color: "var(--faint)", fontSize: 12 }}>No chunks in this document.</div>
        )}
      </div>

      <div className="col viewer">
        <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "0 14px" }}>
          <h2 style={{ padding: "12px 0 8px" }}>{titles[view]}</h2>
          <div style={{ marginLeft: "auto", display: "flex", gap: 4 }}>
            {["source", "chunks", "graph", "embedding"].map((name) => (
              <span key={name} className={"tab" + (view === name ? " on" : "")} onClick={() => setView(name)}>
                {name === "graph" && graph && !graph.error ? `graph (${graph.n_nodes})` : name}
              </span>
            ))}
          </div>
        </div>

        {view === "embedding" ? (
          <EmbeddingMap jobId={jobId} selected={selected}
            onSelect={(point) => {
              const match = chunks.find((c) => c.chunk_id === point.chunk_id);
              if (match) setSelected(match);
            }} />
        ) : view === "graph" ? (
          graph ? <SigmaGraph graph={graph} selected={selected} jobId={jobId} />
                : <div className="empty">Loading the knowledge graph…</div>
        ) : view === "chunks" ? (
          chunks.length ? <SpanMap chunks={chunks} selected={selected} onSelect={setSelected} />
                        : <div className="empty">No chunks to map.</div>
        ) : preview ? (
          <div>
            <PageOverlay jobId={jobId} docId={activeDoc} page={page} preview={preview}
              elements={pageData.elements} highlighted={highlighted}
              onPick={(id) => setHighlighted([id])} />
            <div className="pageno">
              <button className="btn" disabled={page <= 1} onClick={() => setPage(page - 1)}>&lsaquo;</button>
              &nbsp; page {page} / {pageData.page_count} &nbsp;
              <button className="btn" disabled={page >= pageData.page_count} onClick={() => setPage(page + 1)}>&rsaquo;</button>
              &nbsp;&middot;&nbsp; select a chunk to highlight its source region
            </div>
          </div>
        ) : (
          <div className="empty">
            No page preview for this document.
            <br /><span style={{ fontSize: 11 }}>
              Page previews are rendered for PDF sources when EMIT_PAGE_PREVIEWS is enabled.
              Use the chunk map, graph, and embedding views instead.
            </span>
          </div>
        )}
      </div>

      <div className="col">
        <h2>Inspector</h2>
        <Inspector chunk={selected} onHighlight={setHighlighted} />
        {artifacts && <ArtifactPanel artifacts={artifacts} jobId={jobId} />}
      </div>
    </div>
  );
}

/* ----------------------------------------------------------- explain mode */

function ExplainMode({ job, overview, jobId, onInspect }) {
  const byName = Object.fromEntries((job.stages || []).map((s) => [s.name, s]));
  return (
    <div style={{ overflow: "auto" }}>
      <div className="banner">
        <h2>What happened to your documents</h2>
        <p>
          Every step below ran on <b>{overview ? overview.corpus_name : "your corpus"}</b>.
          The results are portable files you can take anywhere.
        </p>
      </div>
      <div className="explain">
        {STAGES.map((name, index) => {
          const stage = byName[name] || { status: "pending" };
          const info = STAGE_INFO[name];
          const done = stage.status === "done";
          const running = stage.status === "running";
          return (
            <div key={name}>
              <div className="step" style={{
                opacity: done ? 1 : running ? 0.85 : 0.45,
                borderColor: running ? info.color : "var(--border)",
              }}>
                <div className="icon" style={{ background: info.color + "22" }}>{info.icon}</div>
                <div style={{ flex: 1 }}>
                  <h3>
                    {info.title}
                    {running && <span style={{ fontSize: 11, color: info.color }}> · working…</span>}
                  </h3>
                  <p>{info.plain}</p>
                  {done && <div className="nums">✓ {info.summary(stage.detail || {})}</div>}
                </div>
              </div>
              {index < STAGES.length - 1 && <div className="arrow">↓</div>}
            </div>
          );
        })}

        {job.outputs_ready && (
          <>
            <div className="banner" style={{ marginTop: 20 }}>
              <h2>Take the results anywhere</h2>
              <p>These files work in any vector database or graph store. Nothing is locked to this service.</p>
            </div>
            <div className="quicklinks">
              <a href={`/v1/jobs/${jobId}/corpus/chunks`}>Chunks<span className="cap">JSON</span></a>
              <a href={`/v1/jobs/${jobId}/index/vectors`}>Vectors<span className="cap">JSONL</span></a>
              <a href={`/v1/jobs/${jobId}/graph`}>Knowledge graph<span className="cap">JSON</span></a>
              <a href={`/v1/jobs/${jobId}/bundle`}>Complete bundle<span className="cap">ZIP</span></a>
            </div>
            <div style={{ textAlign: "center", marginTop: 14 }}>
              <button className="btn" onClick={onInspect}>Open the inspector for full detail →</button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------- learn mode */

function EvidenceAnim({ step }) {
  const count = 60;
  const dots = useRef([...Array(count)].map((_, i) => ({ i, seed: Math.random() }))).current;
  const layouts = {
    0: (d, i) => ({ x: 50 + (d.seed - 0.5) * 8, y: 8 + (i / count) * 84, c: "#3a4150" }),
    1: (d, i) => {
      const palette = ["#60a5fa", "#34d399", "#a78bfa", "#fbbf24", "#fb7185", "#7dd3fc"];
      const column = i % 6;
      return { x: 16 + column * 12, y: 14 + Math.floor(i / 6) * 8, c: palette[column] };
    },
    2: (d, i) => ({
      x: 20 + (i % 5) * 15, y: 16 + Math.floor(i / 5) * 7,
      c: ["#f59e0b", "#60a5fa", "#34d399", "#a78bfa", "#fb7185"][Math.floor(i / 12) % 5],
    }),
    3: (d, i) => {
      const group = Math.floor(i / 8);
      return {
        x: 20 + (group % 3) * 30 + ((i % 8) % 3) * 5,
        y: 20 + Math.floor(group / 3) * 22 + Math.floor((i % 8) / 3) * 5,
        c: ["#60a5fa", "#a78bfa", "#34d399", "#fbbf24", "#fb7185", "#c084fc", "#7dd3fc"][group % 7],
      };
    },
    4: (d) => {
      const angle = d.seed * 2 * Math.PI;
      const radius = 8 + d.seed * 34;
      return { x: 50 + Math.cos(angle) * radius, y: 50 + Math.sin(angle) * radius * 0.9, c: "#7dd3fc" };
    },
    5: (d, i) => {
      const angle = (2 * Math.PI * i) / count;
      return {
        x: 50 + (i === 0 ? 0 : Math.cos(angle) * 30),
        y: 50 + (i === 0 ? 0 : Math.sin(angle) * 30),
        c: i === 0 ? "#e63946" : "#457b9d",
      };
    },
    6: (d, i) => ({
      x: 50 + Math.cos((2 * Math.PI * i) / count) * 32,
      y: 50 + Math.sin((2 * Math.PI * i) / count) * 32, c: "#2dd4bf",
    }),
    7: (d, i) => ({
      x: 50 + Math.cos((2 * Math.PI * i) / count) * 32,
      y: 50 + Math.sin((2 * Math.PI * i) / count) * 32, c: "#34d399",
    }),
  };
  const captions = [
    "Reading every document, page by page.",
    "Labelling roles and pulling out claims, numbers and citations.",
    "Grouping content into meaning-preserving chunks.",
    "Turning each chunk into a vector — meaning becomes geometry.",
    "Connecting everything into a knowledge graph.",
    "Packaging portable artifacts.",
    "Your turn — experiment below.",
    "Portable results, ready to leave with you.",
  ];
  const index = Math.min(step, 7);
  const layout = layouts[index] || layouts[0];
  const edges = index === 5 ? dots.slice(1).map((d) => layout(d, d.i)) : [];

  return (
    <div className="anim-wrap">
      <svg viewBox="0 0 100 100" className="anim-svg">
        {edges.map((point, i) => (
          <line key={i} x1="50" y1="50" x2={point.x} y2={point.y} stroke="#2b313b" strokeWidth="0.3" />
        ))}
        {dots.map((dot) => {
          const point = layout(dot, dot.i);
          return (
            <circle key={dot.i} cx={point.x} cy={point.y}
              r={index === 5 && dot.i === 0 ? 2.6 : 1.3} fill={point.c}
              style={{ transition: "cx .8s cubic-bezier(.4,0,.2,1), cy .8s cubic-bezier(.4,0,.2,1), fill .6s" }} />
          );
        })}
      </svg>
      <div className="anim-cap">{captions[index]}</div>
    </div>
  );
}

function Sandbox({ chunks }) {
  const [size, setSize] = useState(400);
  const total = chunks.reduce((sum, chunk) => sum + (chunk.token_estimate || 0), 0);
  const produced = Math.max(1, Math.round(total / size));
  const average = Math.round(total / produced);
  const effective = Math.round(produced * 1.15);
  const precision = size <= 250 ? "high" : size <= 500 ? "balanced" : "lower";
  const context = size <= 250 ? "fragmented" : size <= 500 ? "good" : "very broad";
  const note = size <= 250
    ? "Small chunks give precise matches, but each hit carries little context, so you need more of them."
    : size <= 500
      ? "This is the usual sweet spot: enough context to be useful, tight enough to stay relevant."
      : "Large chunks carry plenty of context, but retrieval gets fuzzier and wastes tokens on irrelevant text.";

  return (
    <div className="sandbox">
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <span style={{ fontFamily: "var(--mono)", fontSize: 12, color: "var(--dim)" }}>chunk size</span>
        <input type="range" min="120" max="900" step="20" value={size}
          onChange={(e) => setSize(Number(e.target.value))} style={{ flex: 1 }} />
        <span style={{ fontFamily: "var(--mono)", fontSize: 13, color: "var(--teal)", width: 70 }}>{size} tok</span>
      </div>
      <div className="sandbox-grid">
        <div><div className="sv">{effective}</div><div className="sl">chunks produced</div></div>
        <div><div className="sv">{average}</div><div className="sl">avg tokens/chunk</div></div>
        <div><div className="sv">{precision}</div><div className="sl">retrieval precision</div></div>
        <div><div className="sv">{context}</div><div className="sl">context per hit</div></div>
      </div>
      <div className="sandbox-bar">
        {[...Array(Math.min(effective, 60))].map((_, i) => (
          <span key={i} style={{
            flex: 1, height: Math.max(6, Math.min(28, average / 20)),
            background: "#60a5fa55", borderBottom: "2px solid #60a5fa", borderRadius: 2,
          }} />
        ))}
      </div>
      <div style={{ fontSize: 12, color: "var(--dim)", marginTop: 8, lineHeight: 1.5 }}>{note}</div>
      <div style={{ fontSize: 11, color: "var(--faint)", marginTop: 6 }}>
        Estimated from this corpus’s {total.toLocaleString()} tokens.
      </div>
    </div>
  );
}

function LearnMode({ job, chunks, jobId, onInspect }) {
  const [active, setActive] = useState(0);
  const stepRefs = useRef([]);
  const byName = Object.fromEntries((job.stages || []).map((s) => [s.name, s]));

  useEffect(() => {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) setActive(Number(entry.target.dataset.step));
      });
    }, { rootMargin: "-45% 0px -45% 0px", threshold: 0 });
    stepRefs.current.forEach((element) => element && observer.observe(element));
    return () => observer.disconnect();
  }, []);

  return (
    <div style={{ overflow: "auto", position: "relative" }}>
      <div className="banner" style={{ paddingTop: 30 }}>
        <h2>How a corpus becomes retrievable evidence</h2>
        <p>Scroll down. The visualization on the right transforms at each stage.</p>
      </div>
      <div className="scrolly">
        <div className="scrolly-steps">
          {STAGES.map((name, index) => {
            const info = STAGE_INFO[name];
            const stage = byName[name];
            return (
              <div key={name} ref={(el) => { stepRefs.current[index] = el; }} data-step={index}
                className={"scrolly-step" + (active === index ? " on" : "")}>
                <div className="icon" style={{ background: info.color + "22" }}>{info.icon}</div>
                <h3>{info.title}</h3>
                <p>{info.plain}</p>
                {stage && stage.status === "done" && (
                  <div className="nums">✓ {info.summary(stage.detail || {})}</div>
                )}
              </div>
            );
          })}
          <div ref={(el) => { stepRefs.current[6] = el; }} data-step={6}
            className={"scrolly-step" + (active === 6 ? " on" : "")}>
            <div className="icon" style={{ background: "#2dd4bf22" }}>{"🎛️"}</div>
            <h3>Now try it yourself</h3>
            <p>
              Chunk size is the single biggest lever in a retrieval pipeline. Drag the slider to see how it
              changes the number of chunks and the trade-off between precision and context.
            </p>
            {chunks.length > 0 && <Sandbox chunks={chunks} />}
          </div>
          <div ref={(el) => { stepRefs.current[7] = el; }} data-step={7} className="scrolly-step">
            <div className="icon" style={{ background: "#34d39922" }}>{"📦"}</div>
            <h3>Take it anywhere</h3>
            <p>Everything you just watched is downloadable and vendor-neutral.</p>
            {job.outputs_ready && (
              <div className="quicklinks" style={{ justifyContent: "flex-start" }}>
                <a href={`/v1/jobs/${jobId}/corpus/chunks`}>Chunks<span className="cap">JSON</span></a>
                <a href={`/v1/jobs/${jobId}/index/vectors`}>Vectors<span className="cap">JSONL</span></a>
                <a href={`/v1/jobs/${jobId}/bundle`}>Bundle<span className="cap">ZIP</span></a>
              </div>
            )}
            <button className="btn" style={{ marginTop: 12 }} onClick={onInspect}>
              Open the full inspector →
            </button>
          </div>
        </div>
        <div className="scrolly-sticky"><EvidenceAnim step={active} /></div>
      </div>
    </div>
  );
}

/* --------------------------------------------------------------------- app */

function App() {
  const [settings, setSettings] = useState(null);
  const [mode, setMode] = useState(() => {
    try { return localStorage.getItem("corpusretrieve.mode") || "inspect"; }
    catch (error) { return "inspect"; }
  });
  const [jobId, setJobId] = useState(null);
  const [job, setJob] = useState(null);
  const [busy, setBusy] = useState(false);
  const [formError, setFormError] = useState(null);

  const [overview, setOverview] = useState(null);
  const [activeDoc, setActiveDoc] = useState(null);
  const [chunks, setChunks] = useState([]);
  const [elements, setElements] = useState({});
  const [page, setPage] = useState(1);
  const [selected, setSelected] = useState(null);
  const [highlighted, setHighlighted] = useState([]);
  const [graph, setGraph] = useState(null);
  const [artifacts, setArtifacts] = useState(null);

  useEffect(() => {
    const fallback = [{ spec: "hash", label: "Hashing (ASCII words)", kind: "lexical", available: true }];
    readJson("/console/settings")
      .then((loaded) => setSettings({ ...loaded, embedders: loaded.embedders || fallback }))
      .catch(() => setSettings({ embedders: fallback, stages: STAGES, version: "" }));
  }, []);

  useEffect(() => {
    try { localStorage.setItem("corpusretrieve.mode", mode); } catch (error) { /* storage disabled */ }
  }, [mode]);

  const resetResults = () => {
    setOverview(null); setActiveDoc(null); setChunks([]); setElements({});
    setPage(1); setSelected(null); setHighlighted([]); setGraph(null); setArtifacts(null);
  };

  const submit = async (form) => {
    setFormError(null);
    let metadata;
    let documentConfig;
    try {
      metadata = JSON.parse(form.metadata || "{}");
      documentConfig = JSON.parse(form.documentConfig || "{}");
      if (typeof metadata !== "object" || Array.isArray(metadata)) throw new Error("metadata must be an object");
      if (typeof documentConfig !== "object" || Array.isArray(documentConfig)) {
        throw new Error("document configuration must be an object");
      }
    } catch (error) {
      setFormError({ message: error.message });
      return;
    }

    const archives = form.files.filter((file) => file.name.toLowerCase().endsWith(".zip"));
    if (archives.length && form.files.length !== 1) {
      setFormError({ message: "Upload one ZIP by itself, or upload documents without a ZIP." });
      return;
    }
    if (form.useVlm && !form.apiKey && !settings.vlm_api_key_configured) {
      setFormError({ message: "Enter an Anthropic API key, set ANTHROPIC_API_KEY in the .env file, or turn off figure description." });
      return;
    }

    const body = new FormData();
    form.files.forEach((file) => body.append("files", file, file.name));
    body.append("corpus_name", form.corpusName.trim() || "corpus");
    body.append("metadata", JSON.stringify(metadata));
    body.append("document_config", JSON.stringify(documentConfig));
    body.append("require_bibliographic", form.requireBibliographic ? "true" : "false");
    body.append("allow_degraded", form.allowDegraded ? "true" : "false");
    body.append("embedder", form.embedder);
    body.append("enable_vlm", form.useVlm ? "true" : "false");
    if (form.vlmModel) body.append("vlm_model", form.vlmModel);

    const headers = {};
    if (form.useVlm && form.apiKey) headers["X-Anthropic-API-Key"] = form.apiKey;

    setBusy(true);
    resetResults();
    try {
      const response = await fetch("/v1/jobs", { method: "POST", body, headers });
      const payload = await response.json();
      if (!response.ok) {
        setFormError({ message: payload.detail || `Upload failed (HTTP ${response.status})`,
                       errors: payload.errors || [] });
        setBusy(false);
        return;
      }
      setJobId(payload.job_id);
      setJob({ ...payload, progress: 0, stages: [], message: "Job accepted" });
      watch(payload.job_id);
    } catch (error) {
      setFormError({ message: error.message });
      setBusy(false);
    } finally {
      form.clearKey();
    }
  };

  const finish = (id, state) => {
    // a failed job still publishes what it produced (quality gate, or a
    // partial build): show it instead of an empty page
    if (state.state === "completed" || (state.state === "failed" && state.artifacts_available)) {
      loadResults(id);
    }
  };

  const watch = (id) => {
    const source = new EventSource(`/console/jobs/${id}/events`);
    source.addEventListener("job", (event) => {
      const state = JSON.parse(event.data);
      setJob(state);
      if (state.state === "completed" || state.state === "failed") {
        source.close(); setBusy(false); finish(id, state);
      }
    });
    source.addEventListener("error", () => { source.close(); setBusy(false); });
  };

  const openJob = async (id) => {
    setFormError(null);
    try {
      const state = await readJson(`/console/jobs/${id}`);
      resetResults();
      setJobId(id);
      setJob(state);
      setMode("inspect");
      if (state.state === "queued" || state.state === "processing") {
        setBusy(true);
        watch(id);
      } else {
        finish(id, state);
      }
    } catch (error) {
      setFormError({ message: `Could not open job ${id}: ${error.message}` });
      window.history.replaceState(null, "", window.location.pathname);
    }
  };

  // the job id lives in the address (#job/<id>), so a reload reopens the job
  useEffect(() => {
    const match = /^#job\/([0-9a-f]{24})$/.exec(window.location.hash);
    if (settings && match && !jobId) openJob(match[1]);
  }, [settings]);

  useEffect(() => {
    if (!settings) return;
    const target = jobId ? `#job/${jobId}` : "";
    if (window.location.hash !== target) {
      window.history.replaceState(null, "", window.location.pathname + target);
    }
  }, [jobId, settings]);

  const loadResults = async (id) => {
    try {
      const documents = await readJson(`/console/jobs/${id}/documents`);
      setOverview(documents);
      readJson(`/console/jobs/${id}/artifacts`).then(setArtifacts).catch(() => {});
      if (documents.documents.length) selectDocument(id, documents.documents[0].doc_id);
    } catch (error) {
      setFormError({ message: error.message });
    }
  };

  const selectDocument = async (id, docId) => {
    setActiveDoc(docId); setSelected(null); setHighlighted([]); setElements({}); setPage(1);
    setGraph(null);
    readJson(`/console/jobs/${id}/graph?doc_id=${encodeURIComponent(docId)}`)
      .then(setGraph).catch((error) => setGraph({ error: error.message }));
    const document = await readJson(`/console/jobs/${id}/documents/${docId}/chunks`);
    setChunks(document.chunks || []);
    loadPage(id, docId, 1);
  };

  const loadPage = useCallback(async (id, docId, number) => {
    setPage(number);
    let cached = false;
    setElements((previous) => { cached = Boolean(previous[number]); return previous; });
    if (cached) return;
    try {
      const data = await readJson(`/console/jobs/${id}/documents/${docId}/elements?page=${number}`);
      setElements((previous) => ({ ...previous, [number]: data }));
    } catch (error) { /* the document may have no page previews */ }
  }, []);

  const chooseChunk = (chunk) => {
    setSelected(chunk);
    setHighlighted(chunk.element_ids || []);
    if (chunk.pages && chunk.pages.length) loadPage(jobId, activeDoc, chunk.pages[0]);
  };

  if (!settings) return <div className="empty">Loading…</div>;

  const header = (
    <header>
      <h1>CorpusRetrieve</h1>
      <span className="tag">v{settings.version}</span>
      <span className="sub">Documents in &middot; portable chunks, vectors, and a knowledge graph out</span>
      <div className="toolbar">
        {overview && (
          <span className="sub" style={{ fontFamily: "var(--mono)", fontSize: 11 }}>
            {overview.documents.length} docs &middot; {overview.index.records || 0} vectors &middot;{" "}
            {overview.knowledge_graph.nodes || 0} nodes
          </span>
        )}
        <div className="modeswitch">
          {[["learn", "Learn"], ["explain", "Explain"], ["inspect", "Inspect"]].map(([value, label]) => (
            <button key={value} className={mode === value ? "on" : ""} onClick={() => setMode(value)}>{label}</button>
          ))}
        </div>
        {jobId && (
          <button className="btn" disabled={busy} onClick={() => { setJobId(null); setJob(null); resetResults(); }}>
            New job
          </button>
        )}
      </div>
    </header>
  );

  if (!jobId) {
    return (
      <div className="app">
        {header}
        <SetupPanel settings={settings} onSubmit={submit} busy={busy} error={formError}
          onOpenJob={openJob} />
      </div>
    );
  }

  if (mode === "learn") {
    return (
      <div className="app">
        {header}
        <LearnMode job={job || {}} chunks={chunks} jobId={jobId} onInspect={() => setMode("inspect")} />
      </div>
    );
  }

  if (mode === "explain") {
    return (
      <div className="app">
        {header}
        <ExplainMode job={job || {}} overview={overview} jobId={jobId} onInspect={() => setMode("inspect")} />
      </div>
    );
  }

  return (
    <div className="app">
      {header}
      {overview && (
        <DocumentBar overview={overview} activeDoc={activeDoc} jobId={jobId}
          onSelect={(docId) => selectDocument(jobId, docId)} />
      )}
      <StageSpine stages={(job && job.stages) || []} />
      <JobStatus job={job} jobId={jobId} bundleSize={artifacts && artifacts.bundle_size_bytes} />
      {overview ? (
        <Workbench
          jobId={jobId} overview={overview} activeDoc={activeDoc} chunks={chunks}
          elements={elements} page={page} setPage={(n) => loadPage(jobId, activeDoc, n)}
          selected={selected} setSelected={chooseChunk}
          highlighted={highlighted} setHighlighted={setHighlighted}
          graph={graph} artifacts={artifacts} />
      ) : (
        <div className="empty">
          {job && job.state === "failed"
            ? (job.artifacts_available
                ? "Loading the outputs this job produced…"
                : "The job failed before it produced any output. The message above says why.")
            : "Processing the corpus… results appear here when every stage completes."}
        </div>
      )}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
