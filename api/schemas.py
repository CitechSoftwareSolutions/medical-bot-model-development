"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    doc_id: str
    title: str | None
    conditions_found: int
    chunks_added: int
    total_vectors_in_store: int


class DocumentSummary(BaseModel):
    doc_id: str
    title: str | None
    category: str | None
    conditions_count: int
    global_red_flags_count: int


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language query, e.g. a symptom description")
    top_k: int = Field(5, ge=1, le=20)


class SearchHit(BaseModel):
    score: float
    chunk_id: str
    doc_id: str
    source_title: str | None
    condition_name: str
    text: str
    red_flags: list[str]


class SearchResponse(BaseModel):
    query: str
    results: list[SearchHit]


class RebuildResponse(BaseModel):
    documents_loaded: int
    chunks_added: int
    total_vectors_in_store: int


class HealthResponse(BaseModel):
    status: str
    embedding_backend: str
    embedding_dimension: int
    documents_indexed: int
    vectors_in_store: int
