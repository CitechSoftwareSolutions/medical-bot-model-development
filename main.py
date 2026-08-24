"""
Medical Guiding System - Core Model Pipeline Test

This script allows you to test the complete model development pipeline:
document loading -> extraction -> embedding -> FAISS indexing -> RAG Generation
"""

import sys
from src.pipeline import get_vector_store, rebuild_index
from src.embedding import get_embedder
from src.generation import get_generator

def main():
    print("=== Medical Guiding System: Model Development Pipeline ===")
    
    # 1. Initialize and/or rebuild the vector store from data/raw/
    print("\n1. Initializing vector store...")
    store = get_vector_store()
    
    # If the store is empty, or you want to force rebuild to catch new docs in data/raw:
    # Here we'll just rebuild it for testing purposes so it always catches new edits.
    print("Rebuilding index from documents in data/raw/...")
    result = rebuild_index(store)
    print(f"Loaded {result.documents_loaded} documents, adding {result.chunks_added} chunks to FAISS.")
    print(f"Total vectors in store: {result.total_vectors_in_store}")
    
    # Check if we have anything to test against
    if result.total_vectors_in_store == 0:
        print("\nNo documents found in data/raw/. Please add JSON guidelines there to test the model.")
        sys.exit(0)

    # 2. Interactive QA Loop
    print("\n=== RAG Pipeline Ready ===")
    embedder = get_embedder()
    generator = get_generator()
    
    print(f"Using Embedder: {embedder.name} (Dim: {embedder.dimension})")
    print(f"Using Generator: {generator.name}\n")
    
    while True:
        try:
            question = input("\nEnter a medical question (or 'q' to quit): ").strip()
            if not question or question.lower() in ['q', 'quit', 'exit']:
                break
                
            print("\n[Thinking...]")
            
            # Step A: Embed question
            query_vector = embedder.embed([question])[0]
            
            # Step B: Search vector store
            top_k = 3
            hits = store.search(query_vector, top_k=top_k)
            
            # Form context
            context = ""
            for idx, hit in enumerate(hits, start=1):
                condition = hit.get("condition_name", "Unknown")
                text = hit.get("text", "")
                context += f"[{idx}] {condition}: {text}\n\n"
            
            # Step C: Generate answer
            answer = generator.generate(question, context)
            
            print("\n=== ANSWER ===")
            print(answer)
            print("================\n")
            
            print("--- Retrieved Sources ---")
            for idx, hit in enumerate(hits, start=1):
                condition = hit.get('condition_name')
                score = hit.get('score', 0)
                print(f"[{idx}] {condition} (Relevance: {score:.4f})")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
