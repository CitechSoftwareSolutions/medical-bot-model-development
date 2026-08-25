"""
Deterministic offline embedder using feature hashing.

This is NOT a semantic embedding model — it has no notion of word meaning.
It exists for two reasons:
  1. So the pipeline is fully testable with zero external dependencies or
     network access (this is what the test suite and this sandbox use).
  2. So the API keeps working (in degraded form) if the real
     sentence-transformers model can't be downloaded at runtime.

Two texts sharing more tokens will hash to more similar vectors, so search
"works" in a shallow keyword-overlap sense, but for real semantic quality
you want SentenceTransformerEmbedder — see sentence_transformer_embedder.py.
"""

import hashlib
import re

import numpy as np

from src.embedding.base import BaseEmbedder


class HashingEmbedder(BaseEmbedder):
    name = "hashing"

    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self._dimension), dtype="float32")
        for i, text in enumerate(texts):
            vectors[i] = self._embed_one(text)
        return vectors

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dimension, dtype="float32")
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self._dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[index] += sign

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec
