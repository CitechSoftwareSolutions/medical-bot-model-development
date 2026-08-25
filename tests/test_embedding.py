import numpy as np

from src.embedding import get_embedder
from src.embedding.hashing_embedder import HashingEmbedder


def test_hashing_embedder_forced_backend():
    embedder = get_embedder(force_backend="hashing")
    assert isinstance(embedder, HashingEmbedder)
    assert embedder.name == "hashing"


def test_hashing_embedder_output_shape():
    embedder = HashingEmbedder(dimension=384)
    vectors = embedder.embed(["hello world", "another sentence", "a third one"])
    assert vectors.shape == (3, 384)
    assert vectors.dtype == np.float32


def test_hashing_embedder_is_deterministic():
    embedder = HashingEmbedder(dimension=128)
    v1 = embedder.embed(["migraine treatment"])[0]
    v2 = embedder.embed(["migraine treatment"])[0]
    assert np.allclose(v1, v2)


def test_hashing_embedder_vectors_are_normalized():
    embedder = HashingEmbedder(dimension=128)
    vectors = embedder.embed(["some clinical text about NSAIDs and dosage"])
    norm = np.linalg.norm(vectors[0])
    assert abs(norm - 1.0) < 1e-5


def test_auto_backend_falls_back_gracefully_without_network():
    # In this test environment there's no access to huggingface.co, so
    # "auto" must fall back to hashing rather than raising.
    embedder = get_embedder(force_backend="auto")
    assert embedder.name in ("sentence-transformers", "hashing")
