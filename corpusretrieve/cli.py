"""Command line: ``build``, ``verify`` and ``repair``.

``corpusretrieve build`` is the resumable, additive way to build a large
corpus: it writes into a persistent output directory, reuses every document
whose bytes have not changed, and rebuilds only the corpus-wide stages
(index, graph, package). The HTTP service processes each job in a temporary
directory and therefore always starts from scratch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="corpusretrieve")
    sub = parser.add_subparsers(dest="command", required=True)
    v = sub.add_parser("verify", help="check a bundle ZIP or output directory")
    v.add_argument("bundle", type=Path)
    v.add_argument("--require-release-ready", action="store_true",
                   help="exit non-zero when the manifest is not release-ready")
    r = sub.add_parser("repair", help="rebuild graph exports and re-seal the catalog")
    r.add_argument("output_dir", type=Path)
    x = sub.add_parser("reindex", help="re-embed an existing build with another embedder "
                       "(no re-extraction), rebuild the graph, and re-seal")
    x.add_argument("source", type=Path, help="a bundle ZIP or an extracted output directory")
    x.add_argument("--embedder", required=True,
                   help="hash | hash-v2 | bge-m3 | st:<sentence-transformers model id>")
    x.add_argument("--out", type=Path,
                   help="directory to extract a bundle ZIP into (required for a ZIP)")
    b = sub.add_parser("build", help="build or extend a corpus in a persistent directory")
    b.add_argument("inputs", nargs="+", type=Path, help="files and/or directories")
    b.add_argument("--out", type=Path, required=True, help="persistent output directory")
    b.add_argument("--config", type=Path, help="document_config JSON")
    b.add_argument("--require-bibliographic", action="store_true",
                   help="fail the release gate for missing/ambiguous publication records")
    b.add_argument("--key-by", choices=["basename", "path"], default="basename",
                   help="how document_config keys name files (path = relative to input dir)")
    b.add_argument("--only-config-files", action="store_true",
                   help="process only files listed in --config; useful for theme-level builds")
    b.add_argument("--corpus-name", default="corpus")
    b.add_argument("--embedder", default=None, help="hash | hash-v2 | bge-m3 | auto | st:<model>")
    b.add_argument("--allow-degraded", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "build":
        return build(args)

    if args.command == "verify":
        from corpusretrieve.verify import verify_bundle
        report = verify_bundle(args.bundle)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if not report["ok"]:
            return 1
        return 2 if args.require_release_ready and not report["release_ready"] else 0

    if args.command == "reindex":
        return reindex(args)

    from corpusretrieve.pipeline.packaging import repair_outputs
    result = repair_outputs(args.output_dir)
    print(json.dumps({"repaired": result["manifest"]["repairs"][-1],
                      "bundle_sha256": result["bundle_sha256"]}, indent=2))
    return 0


def _collect(inputs: list[Path], key_by: str) -> list[tuple[str, Path]]:
    from corpusretrieve.api.formats import SUPPORTED_EXTENSIONS

    found: list[tuple[str, Path]] = []
    for item in inputs:
        if item.is_dir():
            for path in sorted(item.rglob("*")):
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    name = (path.relative_to(item).as_posix() if key_by == "path"
                            else path.name)
                    found.append((name, path))
        elif item.is_file():
            found.append((item.name, item))
        else:
            raise SystemExit(f"not found: {item}")
    return found


def build(args) -> int:
    from corpusretrieve.config import get_settings
    from corpusretrieve.intake import COLLECTION_KEY, intake_report, intake_warnings
    from corpusretrieve.models import InputFile, JobSpec
    from corpusretrieve.pipeline import CorpusProcessor, DocumentOptions

    settings = get_settings()
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    collection = config.pop(COLLECTION_KEY, None) or {}
    files = _collect(args.inputs, args.key_by)
    if args.only_config_files:
        if not args.config:
            raise SystemExit("--only-config-files requires --config")
        files = [(name, path) for name, path in files if name in config]
    if not files:
        print("no supported files found", file=sys.stderr)
        return 1

    inputs, used, unconfigured = [], set(), []
    for name, path in files:
        entry = config.get(name)
        if entry is not None:
            used.add(name)
        elif config:
            unconfigured.append(name)
        entry = entry or {}
        inputs.append(InputFile(filename=name, storage_key=str(path),
                                size_bytes=path.stat().st_size,
                                document_id=entry.get("document_id"),
                                metadata=dict(entry.get("metadata") or {})))
    intake = intake_report([n for n, _ in files], set(config), used,
                           unconfigured if config else [])
    for warning in intake_warnings(intake):
        print("warning:", warning, file=sys.stderr)

    spec = JobSpec(job_id=f"build-{args.out.name}", corpus_name=args.corpus_name,
                   files=inputs, embedder=args.embedder or settings.default_embedder,
                   allow_degraded=args.allow_degraded,
                   require_bibliographic=args.require_bibliographic or settings.require_bibliographic,
                   intake=intake,
                   collection=collection)
    options = DocumentOptions(
        emit_page_previews=settings.emit_page_previews,
        page_preview_dpi=settings.page_preview_dpi,
        max_page_previews=settings.max_page_previews_per_document,
        ocr_langs=settings.ocr_langs,
        max_vlm_calls=settings.max_vlm_calls or None,
        vlm_max_retries=settings.vlm_max_retries, resume=True)

    def progress(*event):
        if event and isinstance(event[-1], str):
            print(" ", event[-1], file=sys.stderr)

    args.out.mkdir(parents=True, exist_ok=True)
    result = CorpusProcessor(progress, options, batch_size=settings.embedding_batch_size,
                             inline_vectors=settings.index_inline_vectors
                             ).run(spec, [p for _, p in files], args.out)
    manifest = result["manifest"]
    docs = manifest["documents"]
    summary = {
        "output": str(args.out), "documents": len(docs),
        "reused": sum(1 for d in docs if d.get("resumed")),
        "duplicate_files": len(manifest.get("duplicate_files", [])),
        "skipped": len(manifest.get("skipped_documents", [])),
        "release_ready": manifest["release_ready"],
        "reasons": manifest["quality"]["reasons"],
        "bundle_sha256": result["bundle_sha256"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if manifest["quality"]["degraded"] and settings.degraded_policy == "fail" \
            and not args.allow_degraded:
        return 3
    return 0


def reindex(args) -> int:
    """Swap the dense model of a finished build in place, keeping its chunks."""
    import zipfile

    from corpusretrieve.config import valid_embedder_spec
    from corpusretrieve.pipeline.packaging import repair_outputs

    if not valid_embedder_spec(args.embedder) or args.embedder == "auto":
        print("error: --embedder must be hash, hash-v2, bge-m3 or st:<model id>",
              file=sys.stderr)
        return 2
    target = args.source
    if args.source.is_file():
        if args.out is None:
            print("error: --out is required when the source is a ZIP", file=sys.stderr)
            return 2
        if args.out.exists() and any(args.out.iterdir()):
            print(f"error: {args.out} is not empty", file=sys.stderr)
            return 2
        print(f"extracting {args.source} to {args.out} ...", file=sys.stderr)
        with zipfile.ZipFile(args.source) as archive:
            archive.extractall(args.out)
        target = args.out
    if not (target / "manifest.json").is_file():
        print(f"error: {target} has no manifest.json", file=sys.stderr)
        return 2

    reported = [0]

    def progress(done: int) -> None:
        if done // 5000 > reported[0]:
            reported[0] = done // 5000
            print(f"  embedded {done} units", file=sys.stderr)

    result = repair_outputs(target, actions=("index",), embedder=args.embedder,
                            on_progress=progress)
    index = result["manifest"]["index"]
    print(json.dumps({"output": str(target), "model": index["model"],
                      "model_id": index.get("model_id"), "dimension": index["dimension"],
                      "records": index["records"],
                      "release_ready": result["manifest"]["release_ready"],
                      "bundle_sha256": result["bundle_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
