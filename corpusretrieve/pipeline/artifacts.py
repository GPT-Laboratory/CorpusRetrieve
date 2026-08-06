from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def write_json(path: Path, value: Any, indent: int | None = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=indent), encoding="utf-8"
    )


def safe_identifier(value: str, fallback: str = "document") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return cleaned[:80] or fallback


def unique_identifier(candidate: str, taken: set[str]) -> str:
    identifier = candidate
    suffix = 2
    while identifier in taken:
        identifier = f"{candidate[:75]}-{suffix}"
        suffix += 1
    taken.add(identifier)
    return identifier


def relative_asset_path(image_path: str, output_dir: Path) -> str | None:
    try:
        return Path(image_path).resolve().relative_to(output_dir.resolve()).as_posix()
    except (ValueError, OSError):
        return None
