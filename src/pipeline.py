"""
Pipeline orchestration for the Medical Guiding System.

This is the only module that knows about every stage. Everything else
(the API, CLI scripts, tests) should call through here rather than
wiring the stages together themselves.
"""

import logging
from dataclasses import dataclass

from src import config
from src.document_loader.loader import DocumentLoader
from src.document_loader.schema import Document
from src.embedding import get_embedder
from src.extraction.extractor import extract_chunks
from src.vector_store.faiss_store import FAISSVectorStore

logger = logging.getLogger(__name__)

_store_instance: FAISSVectorStore | None = None


def get_vector_store() -> FAISSVectorStore:
    """
    Singleton store for the process. Tries to load a previously saved index
    from disk first; only builds fresh from data/raw/ if none exists yet.
    """
    global _store_instance
    if _store_instance is not None:
        return _store_instance

    embedder = get_embedder()
    store = FAISSVectorStore(
        dimension=embedder.dimension,
        index_path=config.FAISS_INDEX_PATH,
        metadata_path=config.FAISS_METADATA_PATH,
    )

    if not store.load():
        logger.info("No existing index found — building from %s", config.RAW_DATA_DIR)
        rebuild_index(store=store)

    _store_instance = store
    return store


@dataclass
class IngestResult:
    documents_loaded: int
    chunks_added: int
    total_vectors_in_store: int


def rebuild_index(store: FAISSVectorStore | None = None) -> IngestResult:
    """
    Wipes and rebuilds the entire vector store from every file currently
    in data/raw/. Use after manually editing files in data/raw/, or when
    the embedding backend has changed.
    """
    # NOTE: must be an explicit `is None` check, not `store or get_vector_store()`.
    # FAISSVectorStore defines __len__, so a freshly created (empty) store is
    # falsy in Python's eyes — `or` would treat a valid empty store as "missing"
    # and call get_vector_store() again, which calls back into rebuild_index()
    # with no store yet, infinitely.
    if store is None:
        store = get_vector_store()
    embedder = get_embedder()

    store.clear()
    documents = DocumentLoader().load_all()

    chunks_added = 0
    for document in documents:
        chunks = extract_chunks(document)
        if not chunks:
            continue
        vectors = embedder.embed([c.text for c in chunks])
        store.add(vectors, chunks)
        chunks_added += len(chunks)

    store.save()
    logger.info(
        "Rebuilt index: %d documents, %d chunks, %d vectors total",
        len(documents), chunks_added, len(store),
    )
    return IngestResult(
        documents_loaded=len(documents),
        chunks_added=chunks_added,
        total_vectors_in_store=len(store),
    )


def ingest_document(document: Document) -> int:
    """
    Add a single already-loaded Document's chunks into the existing store
    and persist it. Used by the upload API for incremental ingestion,
    so uploading one file doesn't require rebuilding the whole index.

    Returns the number of chunks added.
    """
    store = get_vector_store()
    embedder = get_embedder()

    chunks = extract_chunks(document)
    if not chunks:
        logger.warning("%s produced zero chunks — nothing to add", document.doc_id)
        return 0

    vectors = embedder.embed([c.text for c in chunks])
    store.add(vectors, chunks)
    store.save()

    logger.info("Ingested %s: +%d chunks (%d total in store)", document.doc_id, len(chunks), len(store))
    return len(chunks)
