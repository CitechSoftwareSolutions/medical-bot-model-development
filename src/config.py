"""
Central configuration for the Medical Guiding System pipeline.
Keep every path/setting here so later stages (preprocessing, embedding,
vector store) import from one place instead of hardcoding strings.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# Project root = the folder that contains "src/"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where raw source documents live (input to the loader)
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Where we'll later cache processed/intermediate output (not used today)
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# File types the loader currently accepts.
# Kept as a set so adding PDFs/DOCX later is a one-line change.
SUPPORTED_EXTENSIONS = {".json"}

# Keys we require every guideline document to have.
# Anything else (like management_plan's internal shape) is allowed to vary
# between files, since we've already confirmed it does.
REQUIRED_TOP_LEVEL_KEYS = {"document_metadata", "conditions_registry"}

# --- Embedding settings -----------------------------------------------------

# "auto"               -> try sentence-transformers, fall back to hashing if
#                         the model can't be loaded (e.g. no internet access)
# "sentence-transformers" -> force it, raise if unavailable
# "hashing"            -> force the offline deterministic fallback (tests/CI)
EMBEDDING_BACKEND = "sentence-transformers"

# Any sentence-transformers model name works. all-MiniLM-L6-v2 is a solid
# general-purpose default. For better clinical-domain results, consider
# swapping to a biomedical model such as "pritamdeka/S-PubMedBert-MS-MARCO"
# once you've confirmed it in your own environment.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Vector dimension used by the hashing fallback, and the size the FAISS
# index is created with. 384 matches all-MiniLM-L6-v2's real output size,
# so switching backends later doesn't require touching this.
EMBEDDING_DIM = 384

# --- Generation settings ----------------------------------------------------

# "gemini"      -> Gemini API via GEMINI_API_KEY
# "huggingface" -> local Hugging Face generation backend
LLM_BACKEND = "gemini"

# LLM model used to turn retrieved guideline chunks into an answer.
# The model is loaded lazily on the first /ask request.
LLM_MODEL_NAME = "gemini-3.6-flash"
LLM_MAX_NEW_TOKENS = 512

# Gemini API key is environment-only by design.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Optional Hugging Face access token. Keep the credential in the environment,
# never in source control or configuration files.
HF_TOKEN = os.getenv("HF_TOKEN")

# --- Vector store settings --------------------------------------------------

VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "index.faiss"
FAISS_METADATA_PATH = VECTOR_STORE_DIR / "metadata.json"

# --- API / upload settings --------------------------------------------------

ALLOWED_UPLOAD_EXTENSIONS = {".json"}
MAX_UPLOAD_SIZE_MB = 10
