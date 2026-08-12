"""Convert a source-collector manifest into a CorpusRetrieve document_config.

The manifest lists documents per scenario and theme, so one PDF can appear
several times when it matches more than one theme. Entries are merged by file
so each document is configured once, with its themes collected into a list.

Usage:
    python manifest_to_document_config.py manifest.json -o document_config.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path, PurePosixPath

FIELDS = (
    "title", "source", "authors", "year", "language", "subjects",
    "doi", "online_url", "handle_url", "relevance_score",
)


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")[:80]


def _extend_unique(target: list, values) -> None:
    for value in values or ():
        if value is not None and value not in target:
            target.append(value)


def _accumulate(metadata: dict, scenario: dict, theme: dict, doc: dict) -> None:
    """Merge one manifest entry into the document's accumulated metadata.

    A document can be listed under several scenarios and themes, and the same
    file can be referenced by more than one source doc_id, so every repeated
    field is collected as a deduplicated list.
    """
    _extend_unique(metadata["scenarios"], [scenario.get("id")])
    _extend_unique(metadata["themes"], [theme.get("theme_id")])
    _extend_unique(metadata["keywords_matched"], doc.get("keywords_matched"))


def build(manifest: dict, key_by: str = "basename") -> tuple[dict, dict]:
    merged: OrderedDict[str, dict] = OrderedDict()
    stats = {"entries": 0, "documents": 0, "merged_duplicates": 0, "collisions": 0}

    for scenario in manifest.get("scenarios", []):
        for theme in scenario.get("themes", []):
            for doc in theme.get("documents", []):
                stats["entries"] += 1
                path = doc.get("file_path")
                if not path:
                    continue
                key = (
                    PurePosixPath(path).name if key_by == "basename" else path
                )
                if key in merged:
                    existing = merged[key]
                    if doc.get("doc_id") not in existing["_doc_ids"]:
                        existing["_doc_ids"].add(doc.get("doc_id"))
                        stats["collisions"] += 1
                    _accumulate(existing["metadata"], scenario, theme, doc)
                    stats["merged_duplicates"] += 1
                    continue

                metadata = {
                    field: doc.get(field)
                    for field in FIELDS
                    if doc.get(field) not in (None, "", [])
                }
                metadata["source_doc_id"] = doc.get("doc_id")
                metadata["file_path"] = path
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

    config = {
        key: {"document_id": value["document_id"], "metadata": value["metadata"]}
        for key, value in merged.items()
    }
    return config, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("document_config.json"))
    parser.add_argument("--key-by", choices=["basename", "path"], default="basename",
                        help="basename for direct upload; path for ZIP upload")
    parser.add_argument("--slim", action="store_true",
                        help="drop keywords_matched and subjects to reduce payload size")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    config, stats = build(manifest, args.key_by)

    if args.slim:
        for entry in config.values():
            entry["metadata"].pop("keywords_matched", None)
            entry["metadata"].pop("subjects", None)

    args.output.write_text(
        json.dumps(config, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    print(f"entries read        : {stats['entries']}")
    print(f"unique documents    : {stats['documents']}")
    print(f"duplicates merged   : {stats['merged_duplicates']}")
    print(f"files with >1 doc_id: {stats['collisions']}")
    print(f"written             : {args.output} "
          f"({args.output.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
