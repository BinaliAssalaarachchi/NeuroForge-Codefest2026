import sys
import time
import shutil
import argparse
import logging
from pathlib import Path

# Add root folder to python path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import config
from src.ingestion.parsers import DocumentParser
from src.ingestion.chunker import Chunker
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store
from src.utils.client_voyage import VoyageClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("IngestCorpus")

def run_ingestion(corpus_dir: Path):
    start_time = time.time()
    logger.info(f"=== Starting Ingestion Pipeline for Corpus at '{corpus_dir}' ===")

    if not corpus_dir.exists():
        logger.warning(f"Corpus directory '{corpus_dir}' does not exist. Creating empty directory...")
        corpus_dir.mkdir(parents=True, exist_ok=True)
        return

    # Clear old vector store folder entirely to prevent vector dimension mismatch
    if config.CHROMA_PERSIST_DIR.exists():
        logger.info(f"Clearing old vector database folder at '{config.CHROMA_PERSIST_DIR}'...")
        try:
            shutil.rmtree(config.CHROMA_PERSIST_DIR, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Could not remove old Chroma folder: {e}")

    config.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    # Find all candidate document files
    supported_extensions = {".pdf", ".docx", ".md", ".txt"}
    all_files = [
        f for f in corpus_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in supported_extensions and not f.name.startswith(".")
    ]

    logger.info(f"Found {len(all_files)} document files in corpus.")

    parsed_pages = []
    file_count = 0
    
    for f in all_files:
        pages = DocumentParser.parse_file(f)
        if pages:
            parsed_pages.extend(pages)
            file_count += 1

    logger.info(f"Parsing complete. Successfully parsed {file_count} files into {len(parsed_pages)} pages.")

    if not parsed_pages:
        logger.warning("No text could be parsed from documents. Exiting ingestion.")
        return

    # Chunking Stage
    chunker = Chunker()
    all_chunks = chunker.create_chunks(parsed_pages)
    logger.info(f"Chunking complete. Created {len(all_chunks)} overlapping chunks.")

    # Embedding Stage via Voyage AI
    logger.info("Generating embeddings via Voyage AI (voyage-4-lite)...")
    voyage_client = VoyageClient()
    chunk_texts = [c["text"] for c in all_chunks]
    
    embeddings = voyage_client.get_embeddings(chunk_texts, batch_size=64, use_cache=True)
    logger.info(f"Generated {len(embeddings)} embedding vectors.")

    # Vector Storage in ChromaDB
    logger.info(f"Indexing chunks into local ChromaDB store at '{config.CHROMA_PERSIST_DIR}'...")
    vector_store = VectorStore()
    vector_store.add_chunks(all_chunks, embeddings)

    # Sparse Keyword Storage in BM25
    logger.info(f"Indexing chunks into local BM25 index at '{config.BM25_PERSIST_PATH}'...")
    bm25_store = BM25Store()
    bm25_store.build_index(all_chunks)

    elapsed = time.time() - start_time
    logger.info(f"=== Ingestion Finished in {elapsed:.2f} seconds ===")
    logger.info(f"Summary:")
    logger.info(f"  - Document Files Parsed : {file_count}")
    logger.info(f"  - Total Pages Processed : {len(parsed_pages)}")
    logger.info(f"  - Total Chunks Indexed  : {len(all_chunks)}")
    logger.info(f"  - ChromaDB Vector Count : {vector_store.get_count()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest and Index Ashen Era Archive Corpus")
    parser.add_argument("--corpus_dir", type=str, default=str(config.CORPUS_DIR), help="Path to corpus directory")
    args = parser.parse_args()

    run_ingestion(Path(args.corpus_dir))
