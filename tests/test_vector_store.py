import numpy as np
import pytest

from src.vector_store.faiss_store import FAISSVectorStore
from src.extraction.schema import Chunk


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="test_doc",
        source_title="Test Doc",
        category="Clinical Guidelines",
        condition_name=chunk_id,
        text=text,
        key_features=[],
        red_flags=[],
        metadata={},
    )


def test_add_and_search_returns_closest_match(tmp_path):
    store = FAISSVectorStore(dimension=4, index_path=tmp_path / "i.faiss", metadata_path=tmp_path / "m.json")

    vectors = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype="float32")
    chunks = [make_chunk("a", "text a"), make_chunk("b", "text b"), make_chunk("c", "text c")]
    store.add(vectors, chunks)

    query = np.array([0.9, 0.1, 0.0, 0.0], dtype="float32")
    results = store.search(query, top_k=1)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "a"


def test_dimension_mismatch_raises(tmp_path):
    store = FAISSVectorStore(dimension=4, index_path=tmp_path / "i.faiss", metadata_path=tmp_path / "m.json")
    bad_vectors = np.zeros((1, 8), dtype="float32")
    with pytest.raises(ValueError):
        store.add(bad_vectors, [make_chunk("x", "x")])


def test_save_and_load_roundtrip(tmp_path):
    index_path = tmp_path / "i.faiss"
    metadata_path = tmp_path / "m.json"

    store = FAISSVectorStore(dimension=4, index_path=index_path, metadata_path=metadata_path)
    vectors = np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32")
    store.add(vectors, [make_chunk("a", "text a")])
    store.save()

    reloaded = FAISSVectorStore(dimension=4, index_path=index_path, metadata_path=metadata_path)
    loaded = reloaded.load()

    assert loaded is True
    assert len(reloaded) == 1
    assert reloaded.metadata[0]["chunk_id"] == "a"


def test_load_returns_false_when_no_index_exists(tmp_path):
    store = FAISSVectorStore(dimension=4, index_path=tmp_path / "missing.faiss", metadata_path=tmp_path / "missing.json")
    assert store.load() is False


def test_search_on_empty_store_returns_empty_list(tmp_path):
    store = FAISSVectorStore(dimension=4, index_path=tmp_path / "i.faiss", metadata_path=tmp_path / "m.json")
    results = store.search(np.zeros(4, dtype="float32"), top_k=5)
    assert results == []


def test_clear_resets_store(tmp_path):
    store = FAISSVectorStore(dimension=4, index_path=tmp_path / "i.faiss", metadata_path=tmp_path / "m.json")
    store.add(np.array([[1.0, 0.0, 0.0, 0.0]], dtype="float32"), [make_chunk("a", "text a")])
    assert len(store) == 1
    store.clear()
    assert len(store) == 0
    assert store.metadata == []
