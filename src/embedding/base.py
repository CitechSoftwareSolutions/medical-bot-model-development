"""
Embedding backend interface.

Every embedder — real model or fallback — implements this so the pipeline
and vector store never need to know which one is active.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):
    name: str = "base"

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Length of the vectors this embedder produces."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a batch of texts.
        Returns a float32 array of shape (len(texts), self.dimension).
        """
