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


POINT_ID_SCHEME = "v2"


def point_composite(doc_id: str, chunk_type: str, chunk_id: str, embed_model: str,
                    embed_version: str) -> str:
    """The single definition of a vector point's identity (index and graph share it).

    The corpus name is not part of it, so renaming a corpus keeps every point ID.
    The corpus is recorded in the payload instead.
    """
    return f"{doc_id}:{chunk_type}:{chunk_id}:{embed_model}:{embed_version}"
