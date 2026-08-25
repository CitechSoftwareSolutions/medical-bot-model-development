"""
Extraction stage for the Medical Guiding System.

Responsibility of this module: turn (Document, [PreprocessedCondition]) pairs
into a flat list of Chunk objects — one per condition — with a single text
blob assembled for embedding, plus structured metadata kept alongside for
display and filtering.

Chunking granularity: one chunk per condition_name. This is the natural
retrieval unit for this data — a medical student asking "how do I manage
migraines" should get back one complete, coherent answer, not fragments
split mid-management-plan.
"""

import re

from src.document_loader.schema import Document
from src.extraction.schema import Chunk
from src.preprocessing.preprocessor import PreprocessedCondition, preprocess_document


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "condition"


def build_chunk_text(condition: PreprocessedCondition) -> str:
    """
    Assemble the single string that gets embedded. Order matters a little:
    condition name and key features first (most discriminative for semantic
    search), management plan next (what the student actually needs), red
    flags last (present but not dominant in the embedding).
    """
    parts = [f"Condition: {condition.condition_name}"]

    if condition.key_features:
        parts.append("Key features: " + "; ".join(condition.key_features))

    if condition.investigations:
        parts.append("Investigations: " + "; ".join(condition.investigations))

    if condition.management_conservative:
        parts.append("Initial/conservative management: " + "; ".join(condition.management_conservative))

    if condition.management_advanced:
        parts.append("Advanced/specific management: " + "; ".join(condition.management_advanced))

    if condition.management_follow_up:
        parts.append("Follow-up: " + "; ".join(condition.management_follow_up))

    if condition.differential_red_flags:
        parts.append("Differential red flags: " + "; ".join(condition.differential_red_flags))

    return "\n".join(parts)


def extract_chunks(document: Document) -> list[Chunk]:
    """
    Preprocess and extract every condition in a Document into Chunks.
    This is the single entry point the pipeline calls per document.
    """
    conditions = preprocess_document(document)
    chunks: list[Chunk] = []

    for condition in conditions:
        chunk_id = f"{document.doc_id}::{_slugify(condition.condition_name)}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=document.doc_id,
                source_title=document.title,
                category=document.category,
                condition_name=condition.condition_name,
                text=build_chunk_text(condition),
                key_features=condition.key_features,
                red_flags=document.global_red_flags,
                metadata={
                    "investigations": condition.investigations,
                    "management_conservative": condition.management_conservative,
                    "management_advanced": condition.management_advanced,
                    "management_follow_up": condition.management_follow_up,
                    "differential_red_flags": condition.differential_red_flags,
                },
            )
        )

    return chunks


def extract_all(documents: list[Document]) -> list[Chunk]:
    """Extract chunks across a batch of documents."""
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(extract_chunks(document))
    return chunks
