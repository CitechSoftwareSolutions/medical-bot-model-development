"""
Document loading stage for the Medical Guiding System.

Responsibility of this module: read raw guideline files off disk and turn
each one into a validated Document object. It does NOT clean text, chunk
content, or generate embeddings — that happens in later pipeline stages.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from src.config import RAW_DATA_DIR, REQUIRED_TOP_LEVEL_KEYS, SUPPORTED_EXTENSIONS
from src.document_loader.schema import Document

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


class DocumentLoadError(Exception):
    """Raised when a single file cannot be turned into a valid Document."""


class DocumentLoader:
    """
    Loads clinical guideline documents from a directory of JSON files.

    Usage:
        loader = DocumentLoader()
        documents = loader.load_all()
    """

    def __init__(self, data_dir: Path = RAW_DATA_DIR):
        self.data_dir = Path(data_dir)

    def load_all(self) -> list[Document]:
        """
        Load every supported file in self.data_dir.
        Bad files are logged and skipped, not fatal to the whole batch.
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {self.data_dir}")

        files = sorted(
            p for p in self.data_dir.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not files:
            logger.warning("No supported files found in %s", self.data_dir)
            return []

        documents: list[Document] = []
        for path in files:
            try:
                documents.append(self.load_one(path))
                logger.info("Loaded OK: %s", path.name)
            except DocumentLoadError as e:
                logger.error("Skipping %s — %s", path.name, e)

        logger.info("Loaded %d/%d files successfully", len(documents), len(files))
        return documents

    def load_one(self, path: Path) -> Document:
        """
        Load and validate a single file from disk, returning a Document.
        Raises DocumentLoadError on anything that makes the file unusable.
        """
        path = Path(path)
        raw_text = self._read_text(path)
        data = self._parse_json(raw_text, context_name=path.name)
        return self._build_document(
            data, doc_id=path.stem, source_path=str(path.resolve()), context_name=path.name
        )

    def load_from_dict(self, data: dict, doc_id: str, source_path: str) -> Document:
        """
        Validate and wrap an already-parsed dict as a Document, without
        touching disk.

        Added for the upload API: it lets an in-memory uploaded file be
        validated before anything is written to data/raw/, so a bad upload
        never reaches the folder the batch loader scans. load_one() now
        calls the same shared validation path underneath — behavior for
        existing callers (load_all, load_one, validate_new_document.py) is
        unchanged.
        """
        return self._build_document(data, doc_id=doc_id, source_path=source_path, context_name=doc_id)

    # ---- internal helpers -------------------------------------------------

    def _build_document(self, data: dict, doc_id: str, source_path: str, context_name: str) -> Document:
        self._validate_schema(data, context_name)
        metadata = data.get("document_metadata", {})

        return Document(
            doc_id=doc_id,
            source_path=source_path,
            title=metadata.get("title"),
            category=metadata.get("category"),
            target_audience=metadata.get("target_audience"),
            workflow_steps=data.get("workflow_steps", []),
            conditions_registry=data.get("conditions_registry", []),
            global_red_flags=data.get("global_red_flags", []),
            clinical_best_practices=data.get("clinical_best_practices", []),
            disclaimer=data.get("disclaimer"),
            raw=data,
        )

    def _read_text(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except OSError as e:
            raise DocumentLoadError(f"Could not read file: {e}") from e

    def _parse_json(self, raw_text: str, context_name: str) -> dict:
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as e:
            raise DocumentLoadError(f"Invalid JSON at line {e.lineno}: {e.msg}") from e

    def _validate_schema(self, data: dict, context_name: str) -> None:
        if not isinstance(data, dict):
            raise DocumentLoadError("Top-level JSON is not an object")

        missing = REQUIRED_TOP_LEVEL_KEYS - data.keys()
        if missing:
            raise DocumentLoadError(f"Missing required key(s): {missing}")

        if not isinstance(data.get("conditions_registry"), list) or not data["conditions_registry"]:
            raise DocumentLoadError("conditions_registry must be a non-empty list")

        for i, condition in enumerate(data["conditions_registry"]):
            if "condition_name" not in condition:
                logger.warning(
                    "%s: condition at index %d has no condition_name", context_name, i
                )
