from __future__ import annotations

import uuid

COLLECTION_OF = {
    "claim": "claims",
    "section_text": "sections",
    "summary": "summaries",
    "table_summary": "tables",
    "table_row_group": "tables",
    "table_fact": "tables",
    "figure": "figures",
    "figure_panel": "figures",
    "formula": "formulas",
    "reference": "references",
}


def point_uuid(composite: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, composite))
