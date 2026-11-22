"""Convert a nested external collection manifest into a CorpusRetrieve document_config.

The manifest lists documents per scenario and theme, so one PDF can appear
several times when it matches more than one theme. Entries are merged by file
so each document is configured once, with its themes collected into a list.

Usage:
    python manifest_to_document_config.py manifest.json -o document_config.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import OrderedDict
from pathlib import Path, PurePosixPath

FIELDS = (
    "title", "source", "authors", "year", "language", "subjects",
    "doi", "online_url", "handle_url", "relevance_score",
)
# per-record fields kept in metadata.records (the payload's bibliographic view)
RECORD_FIELDS = ("title", "authors", "year", "language", "doi", "online_url",
                 "handle_url", "source", "page_start", "page_end", "urn",
                 "series", "isbn", "issn", "publisher")
# provenance fields whose name varies between different manifest providers.
# canonical name -> accepted manifest keys, first match wins
FIELD_ALIASES = {
    "finna_id": ("finna_id", "finna_record_id", "record_id", "id"),
    "format": ("format", "formats", "finna_format"),
    "organization": ("organization", "organisation", "building", "institution"),
    "license": ("license", "licence", "usage_rights", "rights_license"),
    "rights": ("rights", "copyright", "access_rights"),
}
# manifest keys that may hold the list of documents removed from the collection
EXCLUSION_KEYS = ("excluded", "exclusions", "excluded_documents", "removed",
                  "removed_documents")
COLLECTION_KEY = "__collection__"


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")[:80]


def _fold(name: str) -> str:
    return unicodedata.normalize("NFC", name).casefold()


def _extend_unique(target: list, values) -> None:
    for value in values or ():
        if value is not None and value not in target:
            target.append(value)


def _accumulate(metadata: dict, scenario: dict, theme: dict, doc: dict) -> None:
    """Merge one manifest entry into the document's accumulated metadata."""
    _extend_unique(metadata["scenarios"], [scenario.get("id")])
    _extend_unique(metadata["themes"], [theme.get("theme_id")])
    _extend_unique(metadata["keywords_matched"], doc.get("keywords_matched"))


def _aliased(doc: dict) -> dict:
    out = {}
    for name, keys in FIELD_ALIASES.items():
        for key in keys:
            if key == "id" and "doc_id" in doc:
                continue                       # "id" only when nothing better exists
            if doc.get(key) not in (None, "", []):
                out[name] = doc[key]
                break
    return out


def _record(doc: dict) -> dict:
    rec = {f: doc.get(f) for f in RECORD_FIELDS if doc.get(f) not in (None, "", [])}
    rec.update(_aliased(doc))
    rec["source_doc_id"] = doc.get("doc_id")
    return rec


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def collection_block(manifest: dict, stats: dict) -> dict:
    """Run-level provenance carried into the bundle manifest ."""
    block = {k: v for k, v in manifest.items()
             if not isinstance(v, (dict, list)) and v not in (None, "")}
    for key in EXCLUSION_KEYS:
        if isinstance(manifest.get(key), list):
            block["exclusions"] = manifest[key]
            break
    block["records_without_file"] = stats["records_without_file"]
    block["converter"] = {"tool": "manifest_to_document_config", "version": "2.0.0"}
    return block


def content_duplicates(config: dict, files_root: Path) -> list[list[str]]:
    """Groups of config keys whose files have identical bytes."""
    by_hash: dict[str, list[str]] = {}
    for key, entry in config.items():
        if key == COLLECTION_KEY:
            continue
        path = files_root / entry["metadata"].get("file_path", key)
        if path.is_file():
            by_hash.setdefault(_sha256(path), []).append(key)
    return [keys for keys in by_hash.values() if len(keys) > 1]


def build(manifest: dict, key_by: str = "basename") -> tuple[dict, dict]:
    """Return (document_config, stats).

    Keeping only the FIRST record's bibliographic fields when several records
    shared a file (34 files held 154 records), and keyed by basename without
    detecting collisions. Every record is kept in ``metadata.records`` and
    reports both exact and case-insensitive key collisions.
    """
    merged: OrderedDict[str, dict] = OrderedDict()
    stats = {"entries": 0, "documents": 0, "merged_duplicates": 0,
             "multi_record_files": 0, "records": 0,
             "key_collisions": [], "case_collisions": [], "records_without_file": []}
    paths_by_key: dict[str, set[str]] = {}

    for scenario in manifest.get("scenarios", []):
        for theme in scenario.get("themes", []):
            for doc in theme.get("documents", []):
                stats["entries"] += 1
                path = doc.get("file_path")
                if not path:
                    # not skipped silently: they are records with no
                    # downloadable full text and must be visible in the report
                    rid = doc.get("doc_id") or doc.get("title")
                    if rid not in stats["records_without_file"]:
                        stats["records_without_file"].append(rid)
                    continue
                key = PurePosixPath(path).name if key_by == "basename" else path
                paths_by_key.setdefault(key, set()).add(path)
                if key in merged:
                    existing = merged[key]
                    if doc.get("doc_id") not in existing["_doc_ids"]:
                        existing["_doc_ids"].add(doc.get("doc_id"))
                        existing["metadata"]["records"].append(_record(doc))
                    _accumulate(existing["metadata"], scenario, theme, doc)
                    stats["merged_duplicates"] += 1
                    continue

                metadata = {
                    field: doc.get(field)
                    for field in FIELDS
                    if doc.get(field) not in (None, "", [])
                }
                metadata.update(_aliased(doc))
                metadata["source_doc_id"] = doc.get("doc_id")
                metadata["file_path"] = path
                metadata["records"] = [_record(doc)]
                metadata["scenarios"] = []
                metadata["themes"] = []
                metadata["keywords_matched"] = []
                _accumulate(metadata, scenario, theme, doc)
                merged[key] = {
                    "_doc_ids": {doc.get("doc_id")},
                    "document_id": safe_id(doc.get("doc_id") or Path(path).stem),
                    "metadata": metadata,
                }
                stats["documents"] += 1

    folded: dict[str, list[str]] = {}
    for key in merged:
        folded.setdefault(_fold(key), []).append(key)
    stats["case_collisions"] = [keys for keys in folded.values() if len(keys) > 1]
    stats["key_collisions"] = sorted(k for k, paths in paths_by_key.items() if len(paths) > 1)

    config = {}
    for key, value in merged.items():
        meta = value["metadata"]
        stats["records"] += len(meta["records"])
        if len(meta["records"]) > 1:
            stats["multi_record_files"] += 1
            # the file-level title would describe only one of its publications
            for f in ("title", "authors", "year", "doi", "handle_url", "online_url"):
                meta.pop(f, None)
        config[key] = {"document_id": value["document_id"], "metadata": meta}
    config[COLLECTION_KEY] = collection_block(manifest, stats)
    return config, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--theme", action="append", metavar="THEME_ID",
                        help="limit conversion to a theme ID; repeat for several themes")
    parser.add_argument("-o", "--output", type=Path, default=Path("document_config.json"))
    parser.add_argument("--key-by", choices=["basename", "path"], default="basename",
                        help="basename for direct upload; path for ZIP upload")
    parser.add_argument("--slim", action="store_true",
                        help="drop keywords_matched and subjects to reduce payload size")
    parser.add_argument("--allow-collisions", action="store_true",
                        help="write the config even if keys collide (not recommended)")
    parser.add_argument("--files-root", type=Path,
                        help="directory holding the downloaded files; enables "
                             "content-duplicate detection")
    parser.add_argument("--report", type=Path,
                        help="write the full conversion statistics as JSON")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.theme:
        selected = set(args.theme)
        manifest["scenarios"] = [{**s, "themes": [t for t in s.get("themes", [])
                                                 if t.get("theme_id") in selected]}
                                 for s in manifest.get("scenarios", [])]
        manifest["scenarios"] = [s for s in manifest["scenarios"] if s["themes"]]
        if not manifest["scenarios"]:
            parser.error(f"No matching theme IDs: {', '.join(sorted(selected))}")
    config, stats = build(manifest, args.key_by)

    if args.files_root:
        stats["content_duplicates"] = content_duplicates(config, args.files_root)
    if args.slim:
        for key, entry in config.items():
            if key == COLLECTION_KEY:
                continue
            entry["metadata"].pop("keywords_matched", None)
            entry["metadata"].pop("subjects", None)

    print(f"entries read        : {stats['entries']}")
    print(f"unique documents    : {stats['documents']}")
    print(f"records kept        : {stats['records']}")
    print(f"multi-record files  : {stats['multi_record_files']}")
    print(f"duplicates merged   : {stats['merged_duplicates']}")
    print(f"key collisions      : {len(stats['key_collisions'])}")
    print(f"case collisions     : {len(stats['case_collisions'])}")
    print(f"records without file: {len(stats['records_without_file'])}")
    if "content_duplicates" in stats:
        print(f"identical files     : {len(stats['content_duplicates'])} group(s)")
    if args.report:
        args.report.write_text(json.dumps(stats, ensure_ascii=False, indent=1),
                               encoding="utf-8")
    if (stats["key_collisions"] or stats["case_collisions"]) and not args.allow_collisions:
        for keys in stats["case_collisions"]:
            print("  case-insensitive collision:", ", ".join(keys), file=sys.stderr)
        for key in stats["key_collisions"]:
            print("  same key, different paths:", key, file=sys.stderr)
        print("Refusing to write an ambiguous config; rename the files, use "
              "--key-by path, or pass --allow-collisions.", file=sys.stderr)
        sys.exit(2)

    args.output.write_text(
        json.dumps(config, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    print(f"written             : {args.output} "
          f"({args.output.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
