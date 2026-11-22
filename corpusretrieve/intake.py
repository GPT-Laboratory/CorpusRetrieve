"""Intake checks shared by the HTTP API and the ``corpusretrieve build`` CLI."""
from __future__ import annotations

import unicodedata

# reserved document_config key: collection-level provenance (run date, source,
# exclusion list), carried into the bundle manifest instead of matched to a file
COLLECTION_KEY = "__collection__"


def _fold(name: str) -> str:
    """The identity a case-insensitive, normalising file system gives a name."""
    return unicodedata.normalize("NFC", name).casefold()


def intake_report(filenames: list[str], config_keys: set[str], used_keys: set[str],
                  configured: list[str]) -> dict:
    """Name collisions and document_config mismatches, found before processing.

    Two files whose names differ only in letter case collapse into one on a
    case-insensitive file system, and a document_config entry that matches no
    file silently drops that file's publication record. Both are found here,
    before any processing starts.
    """
    groups: dict[str, list[str]] = {}
    for name in filenames:
        groups.setdefault(_fold(name), []).append(name)
    collisions = [names for names in groups.values() if len(names) > 1]
    report: dict = {}
    if collisions:
        report["filename_collisions"] = collisions
    if config_keys:
        unmatched = sorted(config_keys - used_keys)
        if unmatched:
            report["unmatched_config_keys"] = unmatched
        missing = [f for f in configured if f]
        if missing:
            report["files_without_document_config"] = missing
    return report


def intake_warnings(report: dict) -> list[str]:
    out = []
    for names in report.get("filename_collisions", []):
        out.append("Filename collision (case/Unicode-insensitive): " + ", ".join(names))
    if report.get("unmatched_config_keys"):
        keys = report["unmatched_config_keys"]
        out.append(f"{len(keys)} document_config key(s) matched no file: "
                   + ", ".join(keys[:10]) + (" ..." if len(keys) > 10 else ""))
    if report.get("files_without_document_config"):
        files = report["files_without_document_config"]
        out.append(f"{len(files)} file(s) have no document_config entry: "
                   + ", ".join(files[:10]) + (" ..." if len(files) > 10 else ""))
    return out


# intake findings that make the quality gate fail, whatever the build produces
GATE_FAILING_KEYS = ("filename_collisions", "unmatched_config_keys",
                     "files_without_document_config")


def gate_failures(report: dict) -> list[dict]:
    """Intake findings that would fail the quality gate, one entry per kind."""
    out = []
    for key in GATE_FAILING_KEYS:
        values = report.get(key) or []
        if values:
            examples = [", ".join(v) if isinstance(v, list) else str(v) for v in values[:10]]
            out.append({"check": key, "count": len(values), "examples": examples})
    return out
