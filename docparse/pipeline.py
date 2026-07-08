"""Dispatch layer: turn a path (file, directory, or archive) into ExtractionResult(s)."""
from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path

from docparse.router import get_extractor
from docparse.detect import detect_type
from docparse.postprocess import finalize
from docparse import scientific
from docparse.models import ExtractionConfig, ExtractionResult

_SKIP = ("__MACOSX", ".DS_Store")


def extract(path: str, config: ExtractionConfig | None = None) -> ExtractionResult:
    """Extract a single file, dispatching on detected type."""
    cfg = config or ExtractionConfig()
    doc_type, mime = detect_type(path)
    extractor = get_extractor(doc_type)
    if extractor is None:
        # last resort: treat as plain text
        extractor = get_extractor("text")
        if extractor is None:
            r = ExtractionResult(source=path, doc_type=doc_type, mime=mime)
            r.warnings.append(f"no extractor for type '{doc_type}'")
            return r
    result = extractor.extract(path, cfg)
    result.mime = result.mime or mime
    if cfg.scientific:
        scientific.clean(result, cfg)
    finalize(result, cfg)
    if cfg.scientific:
        scientific.annotate(result, cfg)
    return result


def extract_zip(path: str, config: ExtractionConfig | None = None,
                members: list[str] | None = None) -> list[ExtractionResult]:
    """Extract each member of a zip archive (optionally a subset)."""
    cfg = config or ExtractionConfig()
    out: list[ExtractionResult] = []
    with zipfile.ZipFile(path) as zf:
        names = members or [n for n in zf.namelist() if not n.endswith("/")]
        for name in names:
            if any(s in name for s in _SKIP):
                continue
            data = zf.read(name)
            suffix = Path(name).suffix or ".bin"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
                tf.write(data)
                tmp = tf.name
            try:
                res = extract(tmp, cfg)
                res.source = f"{os.path.basename(path)}::{name}"
                out.append(res)
            except Exception as exc:  # keep going through the archive
                r = ExtractionResult(source=f"{os.path.basename(path)}::{name}")
                r.warnings.append(f"extraction failed: {exc}")
                out.append(r)
            finally:
                os.unlink(tmp)
    return out


def extract_path(path: str, config: ExtractionConfig | None = None) -> list[ExtractionResult]:
    """Extract a file, every file in a directory (recursively), or every archive member."""
    p = Path(path)
    if p.is_dir():
        results: list[ExtractionResult] = []
        for fp in sorted(p.rglob("*")):
            if fp.is_file() and not any(s in str(fp) for s in _SKIP):
                results.append(extract(str(fp), config))
        return results
    if p.suffix.lower() == ".zip":
        return extract_zip(str(p), config)
    return [extract(str(p), config)]
