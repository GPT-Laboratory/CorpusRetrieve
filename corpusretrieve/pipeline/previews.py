from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PREVIEW_DIRECTORY = "pages"


def render_page_previews(
    source_path: Path,
    assets_dir: Path,
    dpi: int,
    max_pages: int,
) -> dict[str, dict]:
    """Rasterize PDF pages so the browser can draw provenance overlays.

    Returns a mapping of page number to preview descriptor. Non-PDF sources and
    rendering failures yield an empty mapping; previews are never required for
    the retrieval artifacts to be complete.
    """
    if source_path.suffix.lower() != ".pdf":
        return {}
    try:
        import fitz
    except ImportError:
        logger.warning("PyMuPDF is unavailable; page previews were skipped")
        return {}

    previews: dict[str, dict] = {}
    target = assets_dir / PREVIEW_DIRECTORY
    target.mkdir(parents=True, exist_ok=True)
    try:
        with fitz.open(str(source_path)) as document:
            for number in range(min(document.page_count, max_pages)):
                page = document[number]
                pixmap = page.get_pixmap(dpi=dpi)
                name = f"page-{number + 1}.png"
                pixmap.save(str(target / name))
                previews[str(number + 1)] = {
                    "path": f"{PREVIEW_DIRECTORY}/{name}",
                    "dpi": dpi,
                    "pixel_width": pixmap.width,
                    "pixel_height": pixmap.height,
                    "width": page.rect.width,
                    "height": page.rect.height,
                    "unit": "pt",
                    "origin": "top-left",
                }
    except Exception as exc:  # noqa: BLE001 - previews are best effort
        logger.warning("Page preview rendering failed for %s: %s", source_path.name, exc)
    return previews
