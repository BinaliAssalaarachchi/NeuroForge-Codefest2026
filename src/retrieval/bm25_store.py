import re
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import config

logger = logging.getLogger(__name__)

class BM25Store:
    """
    Local BM25 Sparse Keyword Index for exact entity, name, and date matching.
    """
    def __init__(self, persist_path: Optional[Path] = None):
        self.persist_path = Path(persist_path or config.BM25_PERSIST_PATH)
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[Dict[str, Any]] = []
        if self.persist_path.exists():
            self.load()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # Clean and tokenize text into lowercase alphanumeric tokens
        tokens = re.findall(r'\w+', text.lower())
        return tokens

    def build_index(self, chunks: List[Dict[str, Any]]):
        """
        Build BM25 index from document chunks and save to disk.
        """
        if not chunks:
            logger.warning("No chunks provided to build BM25 index.")
            return

        self.chunks = chunks
        corpus_tokens = [self._tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(corpus_tokens)
        self.save()
        logger.info(f"Built and persisted BM25 index with {len(chunks)} chunks at '{self.persist_path}'.")

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search BM25 index for query string and return top_k ranked chunks.
        """
        if not self.bm25 or not self.chunks:
            if self.persist_path.exists():
                self.load()
            else:
                return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        
        # Get top K indices
        top_k = min(top_k, len(self.chunks))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "metadata": {
                    "doc_id": chunk["doc_id"],
                    "file_name": chunk["file_name"],
                    "file_path": chunk["file_path"],
                    "file_type": chunk["file_type"],
                    "page_number": chunk["page_number"],
                    "chunk_index": chunk["chunk_index"],
                    "chunk_id": chunk["chunk_id"]
                },
                "score": score,
                "source": "bm25"
            })
        return results

    def save(self):
        with open(self.persist_path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.chunks}, f)

    def load(self):
        try:
            with open(self.persist_path, "rb") as f:
                data = pickle.load(f)
                self.bm25 = data.get("bm25")
                self.chunks = data.get("chunks", [])
        except Exception as e:
            logger.error(f"Failed to load BM25 index from '{self.persist_path}': {e}")
            self.bm25 = None
            self.chunks = []
