"""
Validate a new guideline JSON file BEFORE moving it into data/raw/.

Usage:
    python validate_new_document.py path/to/new_guideline.json
"""

import sys
from pathlib import Path

from src.document_loader.loader import DocumentLoader, DocumentLoadError


def main():
    if len(sys.argv) != 2:
        print("Usage: python validate_new_document.py <path_to_json_file>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    loader = DocumentLoader()
    try:
        doc = loader.load_one(path)
    except DocumentLoadError as e:
        print(f"❌ INVALID — {e}")
        sys.exit(1)

    print(f"✅ VALID — {doc.title}")
    print(f"   conditions_registry entries: {doc.n_conditions}")
    print(f"   global_red_flags entries:    {len(doc.global_red_flags)}")
    print(f"\nSafe to copy into data/raw/{path.name}")


if __name__ == "__main__":
    main()
