"""
Standard in-memory representation of a loaded clinical guideline document.

Downstream stages (preprocessing, extraction, embedding) should only ever
depend on this shape — not on the raw JSON structure — so that if you later
add PDFs or DOCX guidelines, only the loader needs to change.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass
class Document:
    doc_id: str                     # stable id, e.g. filename stem
    source_path: str                # absolute path the file was loaded from
    title: Optional[str]            # from document_metadata.title
    category: Optional[str]         # from document_metadata.category
    target_audience: Optional[str]  # from document_metadata.target_audience

    workflow_steps: list = field(default_factory=list)
    conditions_registry: list = field(default_factory=list)
    global_red_flags: list = field(default_factory=list)
    clinical_best_practices: list = field(default_factory=list)
    disclaimer: Optional[str] = None

    raw: dict[str, Any] = field(default_factory=dict)   # full original JSON
    loaded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def n_conditions(self) -> int:
        return len(self.conditions_registry)

    def __repr__(self) -> str:
        return (
            f"Document(doc_id={self.doc_id!r}, title={self.title!r}, "
            f"conditions={self.n_conditions})"
        )
