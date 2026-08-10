"""
Search endpoints: query the FAISS vector store, rebuild it from scratch,
and a health check that reports which embedding backend is actually active.
"""

from fastapi import APIRouter

from src import config
from src.embedding import get_embedder
from src.pipeline import get_vector_store, rebuild_index
from api.schemas import HealthResponse, RebuildResponse, SearchHit, SearchRequest, SearchResponse

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    """
    Embed the query with the same embedder used to build the index, then
    run a similarity search against FAISS. This is the endpoint that
    connects the API to the vector store for testing retrieval quality.
    """
    embedder = get_embedder()
    store = get_vector_store()

    query_vector = embedder.embed([request.query])[0]
    raw_hits = store.search(query_vector, top_k=request.top_k)

    results = [
        SearchHit(
            score=hit["score"],
            chunk_id=hit["chunk_id"],
            doc_id=hit["doc_id"],
            source_title=hit.get("source_title"),
            condition_name=hit["condition_name"],
            text=hit["text"],
            red_flags=hit.get("red_flags", []),
        )
        for hit in raw_hits
    ]
    return SearchResponse(query=request.query, results=results)


@router.post("/vector-store/rebuild", response_model=RebuildResponse)
def rebuild():
    """
    Wipe and rebuild the entire index from every file in data/raw/.
    Useful for testing: after manually adding/editing/removing files
    directly in data/raw/ without going through the upload endpoint.
    """
    result = rebuild_index()
    return RebuildResponse(
        documents_loaded=result.documents_loaded,
        chunks_added=result.chunks_added,
        total_vectors_in_store=result.total_vectors_in_store,
    )


@router.get("/health", response_model=HealthResponse)
def health():
    embedder = get_embedder()
    store = get_vector_store()
    return HealthResponse(
        status="ok",
        embedding_backend=embedder.name,
        embedding_dimension=embedder.dimension,
        documents_indexed=len(store.doc_ids()),
        vectors_in_store=len(store),
    )
