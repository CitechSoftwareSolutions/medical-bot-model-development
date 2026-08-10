"""
Chunk schema for the Medical Guiding System.

A Chunk is one retrievable unit: everything a medical student would want
returned together for a single condition. This is the object that gets
embedded and stored in FAISS, and the object returned by the search API.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    chunk_id: str            # f"{doc_id}::{condition_slug}" - stable & unique
    doc_id: str               # which source file this came from
    source_title: str
    category: str | None
    condition_name: str
    text: str                 # the exact string that gets embedded
    key_features: list[str] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)   # this document's global red flags
    metadata: dict[str, Any] = field(default_factory=dict)  # anything extra, kept for display/debug

    def to_dict(self) -> dict[str, Any]:
        """Flat dict for JSON storage alongside the FAISS index."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source_title": self.source_title,
            "category": self.category,
            "condition_name": self.condition_name,
            "text": self.text,
            "key_features": self.key_features,
            "red_flags": self.red_flags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=data["chunk_id"],
            doc_id=data["doc_id"],
            source_title=data.get("source_title"),
            category=data.get("category"),
            condition_name=data["condition_name"],
            text=data["text"],
            key_features=data.get("key_features", []),
            red_flags=data.get("red_flags", []),
            metadata=data.get("metadata", {}),
        )
