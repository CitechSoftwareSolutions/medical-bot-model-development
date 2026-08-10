"""
Real semantic embedder, backed by sentence-transformers.

Requires internet access on first run to download the model from
Hugging Face (cached locally afterwards). If that download fails — e.g. in
a network-restricted environment — the factory in __init__.py catches it
and falls back to HashingEmbedder automatically.
"""

import numpy as np

from src.embedding.base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    name = "sentence-transformers"

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # local import: optional dependency

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,   # so inner product == cosine similarity in FAISS
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype("float32")
