import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.resolve()

# Force load environment variables from .env
load_dotenv(BASE_DIR / ".env", override=True)

# OpenRouter Configuration
_openrouter_keys_str = os.getenv("OPENROUTER_API_KEYS", os.getenv("OPENROUTER_API_KEY", ""))
OPENROUTER_API_KEYS = [k.strip() for k in _openrouter_keys_str.split(",") if k.strip()]
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")

# Google AI Studio / Gemini OpenAI-compatible API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_BASE_URL = os.getenv(
    "GEMINI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/"
)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

# Voyage AI Configuration
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")
VOYAGE_EMBED_MODEL = os.getenv("VOYAGE_EMBED_MODEL", "voyage-4-lite")

# Paths Configuration
CORPUS_DIR = Path(os.getenv("CORPUS_DIR", BASE_DIR / "corpus")).resolve()
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", BASE_DIR / "data" / "chroma_db")).resolve()
BM25_PERSIST_PATH = Path(os.getenv("BM25_PERSIST_PATH", BASE_DIR / "data" / "bm25_index.pkl")).resolve()
CACHE_DIR = Path(os.getenv("CACHE_DIR", BASE_DIR / ".cache")).resolve()
LOGS_DIR = Path(os.getenv("LOGS_DIR", BASE_DIR / "logs")).resolve()
TRACES_DIR = LOGS_DIR / "traces"

# Ensure directories exist
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
BM25_PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
TRACES_DIR.mkdir(parents=True, exist_ok=True)

# Chunking Parameters
CHUNK_SIZE_TOKENS = 600
CHUNK_OVERLAP_TOKENS = 150
APPROX_CHARS_PER_TOKEN = 4  # ~2400 chars per chunk, ~600 chars overlap
