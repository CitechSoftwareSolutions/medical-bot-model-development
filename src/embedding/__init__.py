"""
Embedder factory. This is the only thing the rest of the codebase should
import from src.embedding — never instantiate a backend class directly,
so the fallback logic below is always applied consistently.
"""

import logging

from src import config
from src.embedding.base import BaseEmbedder
from src.embedding.hashing_embedder import HashingEmbedder

logger = logging.getLogger(__name__)

_embedder_instance: BaseEmbedder | None = None


def get_embedder(force_backend: str | None = None) -> BaseEmbedder:
    """
    Returns a singleton embedder for the process, chosen by
    config.EMBEDDING_BACKEND ("auto" / "sentence-transformers" / "hashing"),
    optionally overridden by force_backend (mainly for tests).
    """
    global _embedder_instance
    backend = force_backend or config.EMBEDDING_BACKEND

    if _embedder_instance is not None and force_backend is None:
        return _embedder_instance

    if backend == "hashing":
        embedder = HashingEmbedder(dimension=config.EMBEDDING_DIM)

    elif backend == "sentence-transformers":
        from src.embedding.sentence_transformer_embedder import SentenceTransformerEmbedder
        embedder = SentenceTransformerEmbedder(config.EMBEDDING_MODEL_NAME, token=config.HF_TOKEN)

    elif backend == "auto":
        try:
            from src.embedding.sentence_transformer_embedder import SentenceTransformerEmbedder
            embedder = SentenceTransformerEmbedder(config.EMBEDDING_MODEL_NAME, token=config.HF_TOKEN)
            logger.info("Using sentence-transformers model '%s'", config.EMBEDDING_MODEL_NAME)
        except Exception as e:
            logger.warning(
                "Could not load sentence-transformers model (%s: %s). "
                "Falling back to the offline HashingEmbedder — semantic search "
                "quality will be degraded until this is resolved (usually a "
                "network/model-download issue).",
                type(e).__name__, e,
            )
            embedder = HashingEmbedder(dimension=config.EMBEDDING_DIM)

    else:
        raise ValueError(f"Unknown EMBEDDING_BACKEND: {backend!r}")

    if force_backend is None:
        _embedder_instance = embedder
    return embedder
