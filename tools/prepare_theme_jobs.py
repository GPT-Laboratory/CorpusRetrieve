"""Prepare per-theme CorpusRetrieve upload configurations and optional ZIP batches.

Usage (metadata only):
    python tools/prepare_theme_jobs.py manifest.json --out theme_jobs

Usage (when the source documents are available):
    python tools/prepare_theme_jobs.py manifest.json --out theme_jobs \
        --files-root C:/Data --make-zips --max-files 150 --max-zip-mb 80

With --make-zips, upload each batch ZIP with its same-numbered path_config.json.
ZIP members preserve the original input manifest file_path. A ZIP
must stay below MAX_UPLOAD_BYTES (default 100 MiB) and its uncompressed
contents below MAX_CORPUS_UNCOMPRESSED_BYTES (default 500 MiB).
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

from manifest_to_document_config import COLLECTION_KEY, build, safe_id


def _dump(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _batch_manifest(original: dict, scenario: dict, theme: dict,
                    docs: list[dict], memberships: dict[str, list[str]]) -> dict:
    subset = {k: copy.deepcopy(v) for k, v in original.items() if k != "scenarios"}
    subset["scenarios"] = [{k: copy.deepcopy(v) for k, v in scenario.items() if k != "themes"}]
    subset["scenarios"][0]["themes"] = [{**{k: copy.deepcopy(v) for k, v in theme.items()
                                            if k != "documents"}, "documents": docs}]
    subset["theme_id"] = theme["theme_id"]
    subset["selected_documents"] = len(docs)
    return subset


def _write_batch(original, scenario, theme, docs, output, memberships, files_root,
                 make_zips, part_index, max_zip_bytes) -> dict:
    key = f"part-{part_index:03d}"
    path_cfg, stats = build(_batch_manifest(original, scenario, theme, docs, memberships),
                            key_by="path")
    name_cfg, name_stats = build(_batch_manifest(original, scenario, theme, docs, memberships),
                                key_by="basename")
    for cfg in (path_cfg, name_cfg):
        for entry in cfg.values():
            if isinstance(entry, dict) and "metadata" in entry:
                did = entry["metadata"].get("source_doc_id")
                entry["metadata"]["themes_global"] = memberships.get(did, [])
        cfg[COLLECTION_KEY].update({
            "source_manifest_run_id": original.get("run_id"),
            "source_manifest_generated_at": original.get("generated_at"),
            "scenario_id": scenario["id"], "theme_id": theme["theme_id"],
            "batch": key, "theme_job": True,
        })
    _dump(output / f"{key}.path_config.json", path_cfg)
    if name_stats["key_collisions"] or name_stats["case_collisions"]:
        _dump(output / f"{key}.basename_conflicts.json", {
            "key_collisions": name_stats["key_collisions"],
            "case_collisions": name_stats["case_collisions"],
            "hint": "Upload the ZIP and use the path-based config."
        })
    else:
        _dump(output / f"{key}.direct_config.json", name_cfg)

    rows = []
    not_found = []
    input_bytes = 0
    for doc in docs:
        name = doc.get("file_path") or ""
        pdf = files_root / name if files_root and name else None
        exists = pdf is not None and pdf.is_file()
        if not exists:
            not_found.append(name)
        if exists:
            input_bytes += pdf.stat().st_size
        rows.append({
            "file_path": name,
            "doc_id": doc.get("doc_id"),
            "title": doc.get("title"),
            "source": doc.get("source"),
            "language": doc.get("language"),
            "year": doc.get("year"),
            "handle_url": doc.get("handle_url"),
            "exists": exists if files_root else "not checked",
        })
    with (output / f"{key}.files.csv").open("w", newline="", encoding="utf-8-sig") as fp:
        wr = csv.DictWriter(fp, fieldnames=list(rows[0]) if rows else ["file_path"])
        wr.writeheader()
        wr.writerows(rows)

    archive_path = None
    if make_zips:
        if not_found:
            # Never create a ZIP containing fewer files than its config claims.
            _dump(output / f"{key}.missing_files.json", not_found)
        else:
            archive_path = output / f"{key}.zip"
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=6) as zf:
                for doc in docs:
                    rel = doc["file_path"].replace("\\", "/")
                    if PurePosixPath(rel).is_absolute() or ".." in PurePosixPath(rel).parts:
                        raise ValueError(f"Unsafe manifest path: {rel}")
                    zf.write(files_root / rel, arcname=rel)
            if archive_path.stat().st_size > max_zip_bytes:
                raise ValueError(f"Created ZIP exceeds --max-zip-mb: {archive_path}")
    return {"part": key, "records": stats["records"], "documents": len(docs),
            "direct_filename_collisions": len(name_stats["key_collisions"]),
            "files_missing": len(not_found) if files_root else None,
            "files_checked": bool(files_root), "source_bytes": input_bytes,
            "archive": archive_path.name if archive_path else None}


def run(manifest: dict, output: Path, files_root: Path | None = None,
        make_zips: bool = False, max_files: int = 150,
        max_zip_bytes: int = 80 * 1024 * 1024, only_themes: set[str] | None = None) -> dict:
    if make_zips and files_root is None:
        raise ValueError("--make-zips requires --files-root")
    output.mkdir(parents=True, exist_ok=True)
    memberships: dict[str, list[str]] = defaultdict(list)
    for scenario in manifest.get("scenarios", []):
        for theme in scenario.get("themes", []):
            for doc in theme.get("documents", []):
                id_ = doc.get("doc_id")
                if id_ and theme["theme_id"] not in memberships[id_]:
                    memberships[id_].append(theme["theme_id"])
    summary = {"project_name": manifest.get("project_name"),
               "run_id": manifest.get("run_id"), "manifest_entries": 0,
               "unique_doc_ids": len(memberships), "themes": []}
    for scenario in manifest.get("scenarios", []):
        for theme in scenario.get("themes", []):
            theme_id = theme["theme_id"]
            if only_themes and theme_id not in only_themes:
                continue
            folder = output / safe_id(theme_id)
            folder.mkdir(parents=True, exist_ok=True)
            # Each theme gets its own source metadata, but use global membership
            # for downstream cross-theme filtering and deduplication.
            entries = theme.get("documents", [])
            unique: dict[str, dict] = {}
            for doc in entries:
                path = doc.get("file_path")
                if path and path not in unique:
                    unique[path] = doc
            docs = list(unique.values())
            batches = []
            current = []
            current_bytes = 0
            for doc in docs:
                path = doc["file_path"]
                doc_bytes = (files_root / path).stat().st_size if (
                    files_root and (files_root / path).is_file()) else 0
                if make_zips and doc_bytes > max_zip_bytes:
                    raise ValueError(f"Source file too large for one ZIP: {path}")
                if current and (len(current) >= max_files or (
                    make_zips and current_bytes + doc_bytes > max_zip_bytes * 0.9)):
                    batches.append(current)
                    current = []
                    current_bytes = 0
                current.append(doc)
                current_bytes += doc_bytes
            if current:
                batches.append(current)
            # Full theme-specific config for `corpusretrieve build --only-config-files`.
            full_cfg, full_stats = build(_batch_manifest(manifest, scenario, theme, docs,
                                                        memberships), key_by="path")
            for entry in full_cfg.values():
                if isinstance(entry, dict) and "metadata" in entry:
                    did = entry["metadata"].get("source_doc_id")
                    entry["metadata"]["themes_global"] = memberships.get(did, [])
            full_cfg[COLLECTION_KEY].update({
                "source_manifest_run_id": manifest.get("run_id"),
                "scenario_id": scenario["id"], "theme_id": theme_id,
                "theme_job": True,
            })
            _dump(folder / "full_theme.path_config.json", full_cfg)
            stats = [_write_batch(manifest, scenario, theme, batch, folder, memberships,
                                  files_root, make_zips, i, max_zip_bytes)
                     for i, batch in enumerate(batches, 1)]
            report = {"scenario": scenario["id"], "theme_id": theme_id,
                      "theme_label": theme.get("theme_name", theme_id),
                      "documents": len(docs), "parts": stats,
                      "recommended_corpus_name": f"{scenario['id']}-{theme_id}"}
            _dump(folder / "theme_report.json", report)
            summary["themes"].append(report)
            summary["manifest_entries"] += len(entries)
    _dump(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, default=Path("theme_jobs"))
    parser.add_argument("--theme", action="append", help="Only generate this theme, may be repeated")
    parser.add_argument("--files-root", type=Path, help="Directory relative to manifest file_path values")
    parser.add_argument("--make-zips", action="store_true")
    parser.add_argument("--max-files", type=int, default=150)
    parser.add_argument("--max-zip-mb", type=int, default=80)
    args = parser.parse_args()
    if args.max_files < 1 or args.max_zip_mb < 1:
        parser.error("Limits must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = run(manifest, args.out, args.files_root, args.make_zips,
                 args.max_files, args.max_zip_mb * 1024 * 1024,
                 set(args.theme) if args.theme else None)
    print(json.dumps({"unique_doc_ids": report["unique_doc_ids"],
                      "themes": [{"id": t["theme_id"], "documents": t["documents"],
                                  "parts": len(t["parts"]),
                                  "missing": (sum(p["files_missing"] for p in t["parts"])
                                              if args.files_root else "not checked") }
                                 for t in report["themes"]]}, indent=2))


if __name__ == "__main__":
    main()
