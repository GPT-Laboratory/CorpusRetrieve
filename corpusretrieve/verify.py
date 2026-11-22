"""Bundle verification: the artifact contract, checked from the outside.

``verify_bundle`` answers "can a consumer trust this bundle?" without running
the pipeline. It catches, for example, graph exports regenerated after a build
out of the catalog and nobody noticed.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

REQUIRED_FILES = ("manifest.json", "artifact_catalog.json", "index/vectors.jsonl",
                  "index/sparse_encoder.json", "index/metadata.json")
REQUIRED_PAYLOAD = ("doc_id", "source_file", "chunk_id", "chunk_type", "pages", "spans",
                    "locators", "bibliographic", "schema_version")


class _Bundle:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.zip = zipfile.ZipFile(path) if path.is_file() else None

    def names(self) -> set[str]:
        if self.zip is not None:
            return {n for n in self.zip.namelist() if not n.endswith("/")}
        return {p.relative_to(self.path).as_posix() for p in self.path.rglob("*")
                if p.is_file() and p.name not in {"bundle.zip", "bundle.zip.sha256"}}

    def open(self, name: str):
        if self.zip is not None:
            return self.zip.open(name)
        return (self.path / name).open("rb")

    def read(self, name: str) -> bytes:
        with self.open(name) as fh:
            return fh.read()


def _sha(handle) -> str:
    h = hashlib.sha256()
    for block in iter(lambda: handle.read(1 << 20), b""):
        h.update(block)
    return h.hexdigest()


def verify_bundle(path: str | Path, check_graph: bool = True) -> dict[str, Any]:
    bundle = _Bundle(Path(path))
    errors: list[str] = []
    warnings: list[str] = []
    names = bundle.names()

    for name in REQUIRED_FILES:
        if name not in names:
            errors.append(f"missing required file: {name}")
    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings}

    # 1) integrity: every catalogued file matches, every file is catalogued
    catalog = json.loads(bundle.read("artifact_catalog.json"))
    listed = {a["path"]: a for a in catalog.get("artifacts", [])}
    for rel, entry in listed.items():
        if rel not in names:
            errors.append(f"catalogued file missing: {rel}")
            continue
        with bundle.open(rel) as fh:
            if _sha(fh) != entry["sha256"]:
                errors.append(f"hash mismatch: {rel}")
    for rel in sorted(names - set(listed) - {"artifact_catalog.json"}):
        errors.append(f"file not in catalog: {rel}")

    # 2) contract: payload fields, unique ids, dense rows line up
    manifest = json.loads(bundle.read("manifest.json"))
    metadata = json.loads(bundle.read("index/metadata.json"))
    ids: set[str] = set()
    n = 0
    without_box = without_record = ambiguous_record = 0
    for line in bundle.read("index/vectors.jsonl").decode("utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        n += 1
        if rec["id"] in ids:
            errors.append(f"duplicate point id: {rec['id']}")
        ids.add(rec["id"])
        payload = rec.get("payload") or {}
        missing = [f for f in REQUIRED_PAYLOAD if f not in payload]
        if missing:
            errors.append(f"{rec['id']}: payload missing {missing}")
            continue
        if not any(loc.get("bbox") for loc in payload["locators"]):
            without_box += 1
        if not payload["bibliographic"]:
            without_record += 1
        elif payload["bibliographic"].get("ambiguous"):
            ambiguous_record += 1
        for a, b in payload["spans"]:
            if a < 0 or b < a:
                errors.append(f"{rec['id']}: invalid span {a}-{b}")
    dense_name = metadata.get("dense_file")
    if dense_name:
        if dense_name not in names:
            errors.append(f"dense matrix missing: {dense_name}")
        else:
            import numpy as np
            rows = np.load(io.BytesIO(bundle.read(dense_name)), allow_pickle=False).shape[0]
            if rows != n:
                errors.append(f"dense matrix has {rows} rows for {n} records")
    if without_box:
        warnings.append(f"{without_box} unit(s) have no bounding box")
    if without_record:
        warnings.append(f"{without_record} unit(s) resolve to no bibliographic record")
    if ambiguous_record:
        warnings.append(f"{ambiguous_record} unit(s) have ambiguous bibliographic records")
    if (manifest.get("quality") or {}).get("require_bibliographic"):
        if without_record:
            errors.append(f"strict provenance: {without_record} unit(s) lack bibliographic records")
        if ambiguous_record:
            errors.append(f"strict provenance: {ambiguous_record} unit(s) have ambiguous records")

    # 3) graph joins: every VectorPoint node is a real point
    if check_graph and "knowledge_graph/knowledge_graph.json" in names:
        graph = json.loads(bundle.read("knowledge_graph/knowledge_graph.json"))
        points = {nd["id"][4:] for nd in graph.get("nodes", [])
                  if nd.get("ntype") == "VectorPoint"}
        dangling = points - ids
        if dangling:
            errors.append(f"{len(dangling)} graph VectorPoint node(s) join no vector")

    from corpusretrieve.consumer import cross_lingual_risk
    if cross_lingual_risk(metadata.get("model", ""), metadata.get("languages") or {}):
        warnings.append(
            f"multilingual corpus {metadata.get('languages')} indexed with lexical model "
            f"{metadata.get('model')}: single-language queries will miss evidence in the "
            "other language; use PortableIndex.search(extra_queries=/translator=) or a "
            "multilingual embedder")
    if not manifest.get("release_ready", False):
        warnings.append("manifest says the build is not release-ready: "
                        + "; ".join((manifest.get("quality") or {}).get("reasons", [])))
    return {"ok": not errors, "errors": errors, "warnings": warnings,
            "records": n, "release_ready": bool(manifest.get("release_ready"))}
