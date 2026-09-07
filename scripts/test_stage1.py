import sys
from pathlib import Path

# Add root folder to python path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import config
from src.ingestion.parsers import DocumentParser
from src.ingestion.chunker import Chunker
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid_search import HybridRetriever

def main():
    print("=== Testing Stage 1 Components ===")

    # 1. Test parsing sample test files
    sample_dir = config.BASE_DIR / "corpus_sample_test"
    sample_dir.mkdir(exist_ok=True)

    test_md = sample_dir / "annals_of_ashen_era.md"
    test_md.write_text("""# The Annals of the Ashen Era

## Chapter 1: The Great Cataclysm (342 AE)
In the year 342 AE, the Great Cataclysm destroyed the Citadel of High Keep. Master Archivist Vaelin recorded that 400 scholars perished in the flames.

## Chapter 2: Discrepancy in the Records
However, according to the Scroll of Eos, the Siege of High Keep took place in 345 AE under Lord Kaelen's campaign, not 342 AE.
""", encoding="utf-8")

    test_txt = sample_dir / "chronicles_of_eos.txt"
    test_txt.write_text("""Chronicles of Eos - Volume IV
Author: Scholar Lyra
Date: 345 AE

Lord Kaelen marched upon High Keep in 345 AE. The citadel fell after a three-week siege.
""", encoding="utf-8")

    print(f"Created sample test documents in '{sample_dir}'.")

    # 2. Test Parser
    pages = []
    for f in sample_dir.glob("*"):
        parsed = DocumentParser.parse_file(f)
        pages.extend(parsed)
        print(f"Parsed {f.name}: {len(parsed)} page(s)")

    # 3. Test Chunker
    chunker = Chunker()
    chunks = chunker.create_chunks(pages)
    print(f"Created {len(chunks)} chunks.")
    for c in chunks:
        print(f" - Chunk ID: {c['chunk_id']} | Text snippet: {c['text'][:80]}...")

    # 4. Test Vector Store & BM25 Store
    vector_store = VectorStore(persist_dir=config.BASE_DIR / "data" / "test_chroma")
    vector_store.clear()

    # Generate test embeddings (512-dim mock/voyage fallback)
    dummy_embeddings = [[0.01 * (i + j) for j in range(512)] for i, _ in enumerate(chunks)]
    vector_store.add_chunks(chunks, dummy_embeddings)
    print(f"VectorStore count: {vector_store.get_count()}")

    bm25_store = BM25Store(persist_path=config.BASE_DIR / "data" / "test_bm25.pkl")
    bm25_store.build_index(chunks)

    # 5. Test Hybrid Search
    retriever = HybridRetriever(vector_store=vector_store, bm25_store=bm25_store)
    results = retriever.search("Great Cataclysm 342 AE High Keep", top_k=2)

    print("\n--- Hybrid Search Results ---")
    for r in results:
        print(f"ID: {r['chunk_id']} | RRF Score: {r['rrf_score']:.4f}")
        print(f"Text:\n{r['text']}\n")

    print("=== Stage 1 Verification Complete! ===")

if __name__ == "__main__":
    main()
