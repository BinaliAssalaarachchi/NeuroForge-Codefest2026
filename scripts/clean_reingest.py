import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.utils.cache import default_cache
from scripts.ingest_corpus import run_ingestion
import config

if __name__ == "__main__":
    print("Clearing old cache...")
    default_cache.clear()
    print("Re-running ingestion with live 1024-dim Voyage embeddings...")
    run_ingestion(config.CORPUS_DIR)
    print("Clean re-ingestion finished!")
