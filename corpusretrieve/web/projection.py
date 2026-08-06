from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import numpy as np

MIN_POINTS_FOR_PROJECTION = 4
MAX_POINT_TEXT = 160


def project_vectors(lines: Iterable[str], limit: int) -> dict[str, Any]:
    """Reduce dense chunk vectors to two dimensions for the embedding map.

    UMAP is used when it is installed because it preserves neighbourhood
    structure better; otherwise a deterministic PCA over the covariance's
    singular vectors is used, which needs no additional dependency.
    """
    records = _load_records(lines, limit)
    if not records:
        return {"method": "none", "n": 0, "points": []}

    matrix = np.asarray([record["vector"] for record in records], dtype="float32")
    coordinates, method = _reduce(matrix)
    normalized = _normalize(coordinates)

    points = []
    for record, (x, y) in zip(records, normalized, strict=True):
        payload = record.get("payload", {})
        points.append(
            {
                "id": record["id"],
                "chunk_id": payload.get("chunk_id"),
                "doc_id": payload.get("doc_id"),
                "type": payload.get("chunk_type"),
                "x": round(float(x), 4),
                "y": round(float(y), 4),
                "pages": payload.get("pages"),
                "text": (payload.get("raw_text") or "")[:MAX_POINT_TEXT],
            }
        )
    return {"method": method, "n": len(points), "points": points}


def _load_records(lines: Iterable[str], limit: int) -> list[dict[str, Any]]:
    records = []
    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("vector"):
            records.append(record)
        if len(records) >= limit:
            break
    return records


def _reduce(matrix: np.ndarray) -> tuple[np.ndarray, str]:
    if len(matrix) < MIN_POINTS_FOR_PROJECTION:
        return np.zeros((len(matrix), 2), dtype="float32"), "none"
    try:
        import umap

        reducer = umap.UMAP(
            n_neighbors=min(15, len(matrix) - 1),
            min_dist=0.1,
            n_components=2,
            random_state=42,
        )
        return np.asarray(reducer.fit_transform(matrix), dtype="float32"), "umap"
    except Exception:  # noqa: BLE001 - PCA is the supported default
        return _pca(matrix), "pca"


def _pca(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    return np.asarray(centered @ components[:2].T, dtype="float32")


def _normalize(coordinates: np.ndarray) -> np.ndarray:
    low = coordinates.min(axis=0)
    high = coordinates.max(axis=0)
    span = np.where((high - low) == 0, 1.0, high - low)
    return (coordinates - low) / span
