"""
Sprint 1 - Step 1: Document Loading demo.

Run:
    python main.py
"""

from src.document_loader.loader import DocumentLoader


def main():
    loader = DocumentLoader()
    documents = loader.load_all()

    print("\n=== Loaded Documents Summary ===")
    for doc in documents:
        print(f"- {doc.doc_id}")
        print(f"    title      : {doc.title}")
        print(f"    conditions : {doc.n_conditions}")
        print(f"    red flags  : {len(doc.global_red_flags)}")
    print(f"\nTotal documents loaded: {len(documents)}")


if __name__ == "__main__":
    main()
