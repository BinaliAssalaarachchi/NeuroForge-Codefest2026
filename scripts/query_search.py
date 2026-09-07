import sys
import argparse
from pathlib import Path

# Ensure UTF-8 stdout encoding on Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add root folder to python path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import config
from src.retrieval.hybrid_search import HybridRetriever

def search_corpus(query: str, top_k: int = 5):
    print(f"\n[Search] Executing Hybrid RRF Search for: '{query}'")
    print("=" * 70)
    
    retriever = HybridRetriever()
    results = retriever.search(query, top_k=top_k)

    if not results:
        print("[Search] No matching chunks found.")
        return

    for i, res in enumerate(results, start=1):
        meta = res.get("metadata", {})
        chunk_id = res.get("chunk_id", "N/A")
        score = res.get("rrf_score", 0.0)
        file_name = meta.get("file_name", "N/A")
        page_num = meta.get("page_number", "N/A")
        text_snippet = res.get("text", "")

        print(f"\n--- Result #{i} [RRF Score: {score:.5f}] ---")
        print(f"  Chunk ID  : {chunk_id}")
        print(f"  Source    : {file_name} (Page {page_num})")
        print(f"  Content   :")
        print(text_snippet[:400] + ("..." if len(text_snippet) > 400 else ""))
        print("-" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query Ashen Era Archive via Hybrid RRF Search")
    parser.add_argument("query", type=str, nargs="?", default="Gauntlet of Sorrowfell forged year", help="Search query")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to retrieve")
    args = parser.parse_args()

    search_corpus(args.query, top_k=args.top_k)
