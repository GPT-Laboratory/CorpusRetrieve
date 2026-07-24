"""Stage-1 web-app tests. Cover the full developer-mode workflow offline:
upload -> SSE pipeline -> provenance elements -> chunk tracing -> page image ->
graph -> portable artifact downloads, plus the two go/no-go portability
benchmarks (Qdrant load, Cypher validity)."""
import io
import json
import os

import pytest

os.environ.setdefault("EMBEDDER", "hash")

def _find_pdf():
    import glob
    cands = ["tests/fixtures/sample.pdf"]
    cands += sorted(glob.glob("/mnt/user-data/uploads/*.pdf"))
    cands += sorted(glob.glob("*.pdf")) + sorted(glob.glob("**/*.pdf", recursive=True))
    for p in cands:
        if os.path.exists(p):
            return p
    return None

PDF = _find_pdf()


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["JOBS_DIR"] = str(tmp_path_factory.mktemp("jobs"))
    os.environ["EMBEDDER"] = "hash"
    from fastapi.testclient import TestClient
    import importlib
    import service.web as web
    importlib.reload(web)
    return TestClient(web.app)


@pytest.fixture(scope="module")
def finished_job(client):
    if PDF is None:
        pytest.skip("no sample PDF available")
    with open(PDF, "rb") as fh:
        job = client.post("/api/jobs",
                          files={"file": ("p.pdf", fh.read(), "application/pdf")}
                          ).json()["job_id"]
    stages = []
    with client.stream("GET", f"/api/jobs/{job}/stream") as r:
        for line in r.iter_lines():
            if line.startswith("data:"):
                d = json.loads(line[5:])
                if d.get("status") == "done":
                    stages.append(d["stage"])
    return job, stages


def test_frontend_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "CorpusRetrieve" in r.text


def test_healthz(client):
    assert client.get("/api/healthz").json()["status"] == "ok"


def test_all_six_stages_stream(finished_job):
    _, stages = finished_job
    assert stages == ["parse", "extract", "enrich", "chunk", "embed", "graph"]


def test_manifest_summary(client, finished_job):
    job, _ = finished_job
    m = client.get(f"/api/jobs/{job}/manifest").json()
    s = m["summary"]
    assert s["pages"] > 0 and s["elements"] > 0 and s["chunks"] > 0
    assert s["vectors"] == s["chunks"]           # one vector per chunk
    assert s["graph_nodes"] > 0


def test_elements_have_bbox_and_page_dims(client, finished_job):
    job, _ = finished_job
    e = client.get(f"/api/jobs/{job}/elements", params={"page": 1}).json()
    assert e["page_dims"].get("1", {}).get("width", 0) > 0
    assert any(el["bbox"] for el in e["elements"])


def test_bbox_within_page_bounds(client, finished_job):
    """The provenance overlay math: every bbox must scale to within the page."""
    job, _ = finished_job
    e = client.get(f"/api/jobs/{job}/elements", params={"page": 1}).json()
    dims = e["page_dims"]["1"]
    for el in e["elements"]:
        if not el["bbox"]:
            continue
        x0, y0, x1, y1 = el["bbox"]
        assert -2 <= x0 <= x1 <= dims["width"] + 2
        assert -2 <= y0 <= y1 <= dims["height"] + 2


def test_chunk_traces_to_elements(client, finished_job):
    job, _ = finished_job
    ch = client.get(f"/api/jobs/{job}/chunks").json()
    traced = [c for c in ch["chunks"] if c["element_ids"]]
    assert traced
    # element ids referenced by chunks must exist in the extraction
    ex = client.get(f"/api/jobs/{job}/elements").json()
    known = {el["id"] for el in ex["elements"]}
    sample = traced[0]["element_ids"][0]
    assert sample in known                       # provenance link is real


def test_page_image_renders(client, finished_job):
    job, _ = finished_job
    r = client.get(f"/api/jobs/{job}/page/1.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert len(r.content) > 1000


def test_graph_hides_bookkeeping_by_default(client, finished_job):
    job, _ = finished_job
    g = client.get(f"/api/jobs/{job}/graph").json()
    assert g["n_nodes"] > 0 and g["n_edges"] > 0
    types = {n["type"] for n in g["nodes"]}
    assert "VectorPoint" not in types            # hidden by default


def test_downloads_present(client, finished_job):
    job, _ = finished_job
    for art in ("chunks", "embeddings", "graph_json", "graph_cypher",
                "graph_graphml", "extraction"):
        r = client.get(f"/api/jobs/{job}/download/{art}")
        assert r.status_code == 200 and len(r.content) > 0
        assert "attachment" in r.headers.get("content-disposition", "")


# ---- the two go/no-go portability benchmarks --------------------------------

def test_benchmark_embeddings_load_into_qdrant(client, finished_job):
    job, _ = finished_job
    raw = client.get(f"/api/jobs/{job}/download/embeddings").content.decode()
    recs = [json.loads(l) for l in raw.splitlines() if l.strip()]
    assert recs
    from qdrant_client import QdrantClient, models as m
    qc = QdrantClient(":memory:")
    dim = len(recs[0]["vector"])
    for coll in {r["collection"] for r in recs}:
        qc.recreate_collection(coll, vectors_config=m.VectorParams(
            size=dim, distance=m.Distance.COSINE))
    for r in recs:
        qc.upsert(r["collection"], points=[m.PointStruct(
            id=r["id"], vector=r["vector"], payload=r["payload"])])
    total = sum(qc.count(c).count for c in {r["collection"] for r in recs})
    assert total == len(recs)                    # loaded with zero edits
    # payload carries provenance for downstream grounding
    hit = qc.query_points(recs[0]["collection"], query=recs[0]["vector"],
                          limit=1).points[0]
    assert {"doc_id", "pages", "chunk_type"} <= set(hit.payload)


def test_benchmark_cypher_is_valid_line_per_statement(client, finished_job):
    job, _ = finished_job
    cy = client.get(f"/api/jobs/{job}/download/graph_cypher").content.decode()
    lines = [l for l in cy.split("\n")
             if l.strip() and not l.strip().startswith("//")]
    assert lines
    for l in lines:
        assert l.rstrip().endswith(";")          # complete statement per line
        assert l.count("{") == l.count("}")      # balanced property maps
        # balanced unescaped quotes
        n, i = 0, 0
        while i < len(l):
            if l[i] == "\\":
                i += 2
                continue
            if l[i] == '"':
                n += 1
            i += 1
        assert n % 2 == 0
