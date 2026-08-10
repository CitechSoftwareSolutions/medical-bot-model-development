"""
FAISS vector store for the Medical Guiding System.

Uses a flat inner-product index (IndexFlatIP). Vectors are expected to
already be L2-normalized by the embedder, so inner product == cosine
similarity. Flat index means exact (not approximate) search — the right
choice at this corpus size (tens to low thousands of chunks); revisit if
the corpus grows into the hundreds of thousands.

Chunk metadata is kept in a parallel JSON file, in the same order as
vectors are added to the FAISS index, so index position doubles as the
join key between the two.
"""

import json
import logging
import threading
from pathlib import Path

import faiss
import numpy as np

from src.extraction.schema import Chunk

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    def __init__(self, dimension: int, index_path: Path, metadata_path: Path):
        self.dimension = dimension
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self._lock = threading.Lock()

        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: list[dict] = []

    def __len__(self) -> int:
        return self.index.ntotal

    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        if len(vectors) != len(chunks):
            raise ValueError(f"vectors ({len(vectors)}) and chunks ({len(chunks)}) length mismatch")
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"vector dimension {vectors.shape[1]} does not match store dimension {self.dimension}"
            )

        with self._lock:
            self.index.add(np.ascontiguousarray(vectors, dtype="float32"))
            self.metadata.extend(chunk.to_dict() for chunk in chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        if len(self) == 0:
            return []

        query_vector = np.ascontiguousarray(query_vector, dtype="float32").reshape(1, -1)
        top_k = min(top_k, len(self))

        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            result = dict(self.metadata[idx])
            result["score"] = float(score)
            results.append(result)
        return results

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            faiss.write_index(self.index, str(self.index_path))
            self.metadata_path.write_text(json.dumps(self.metadata, indent=2))
        logger.info("Saved FAISS index (%d vectors) to %s", len(self), self.index_path)

    def load(self) -> bool:
        """Returns True if an existing index was found and loaded, False otherwise."""
        if not self.index_path.exists() or not self.metadata_path.exists():
            return False

        self.index = faiss.read_index(str(self.index_path))
        self.metadata = json.loads(self.metadata_path.read_text())

        if self.index.d != self.dimension:
            logger.warning(
                "Loaded index dimension (%d) != configured dimension (%d). "
                "This usually means the embedding backend changed since the "
                "index was built — rebuild it via POST /vector-store/rebuild.",
                self.index.d, self.dimension,
            )
        logger.info("Loaded FAISS index (%d vectors) from %s", len(self), self.index_path)
        return True

    def clear(self) -> None:
        with self._lock:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.metadata = []

    def doc_ids(self) -> list[str]:
        return sorted({m["doc_id"] for m in self.metadata})
