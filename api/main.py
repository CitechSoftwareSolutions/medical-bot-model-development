"""
Medical Guiding System API.

Run:
    uvicorn api.main:app --reload --port 8000

Then browse to http://localhost:8000/docs for interactive Swagger docs.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.pipeline import get_vector_store
from api.routers import documents, search

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up the embedder + vector store once at startup rather than on
    # the first request, so the first real request isn't the slow one.
    logger.info("Starting up: initializing embedder and vector store...")
    store = get_vector_store()
    logger.info("Ready: %d vectors loaded across %d documents", len(store), len(store.doc_ids()))
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Documents Handeling API",
    description="Sprint 1: document upload, preprocessing, extraction, embedding, and FAISS vector search.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(documents.router)
app.include_router(search.router)


@app.get("/")
def root():
    return {"message": "Medical Guiding System API. See /docs for endpoints."}
