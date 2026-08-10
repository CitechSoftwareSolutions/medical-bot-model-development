from src.document_loader.loader import DocumentLoader
from src.extraction.extractor import extract_all, extract_chunks, build_chunk_text


def test_extract_all_produces_one_chunk_per_condition():
    documents = DocumentLoader().load_all()
    total_conditions = sum(d.n_conditions for d in documents)
    chunks = extract_all(documents)
    assert len(chunks) == total_conditions


def test_chunk_id_is_stable_and_unique():
    documents = DocumentLoader().load_all()
    chunks = extract_all(documents)
    chunk_ids = [c.chunk_id for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids)), "chunk_ids must be unique"


def test_chunk_text_contains_management_plan_content():
    documents = DocumentLoader().load_all()
    headache = next(d for d in documents if d.doc_id == "clinical_guidelines_headache")
    chunks = extract_chunks(headache)

    migraine_chunk = next(c for c in chunks if c.condition_name == "Migraines")
    assert "Sumatriptan" in migraine_chunk.text
    assert "Trigger avoidance" in migraine_chunk.text
    assert migraine_chunk.doc_id == "clinical_guidelines_headache"
    assert migraine_chunk.red_flags  # doc-level red flags carried through


def test_chunk_roundtrips_through_dict():
    documents = DocumentLoader().load_all()
    chunks = extract_chunks(documents[0])
    original = chunks[0]

    from src.extraction.schema import Chunk
    restored = Chunk.from_dict(original.to_dict())

    assert restored.chunk_id == original.chunk_id
    assert restored.text == original.text
    assert restored.condition_name == original.condition_name
