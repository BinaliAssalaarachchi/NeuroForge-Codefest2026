import logging
from typing import List, Dict, Any, Optional
from src.utils.client_voyage import VoyageClient
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store

logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    Hybrid Retriever using Reciprocal Rank Fusion (RRF) to combine
    dense vector search (ChromaDB + Voyage AI voyage-4-lite)
    and sparse keyword search (BM25Okapi).
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        bm25_store: Optional[BM25Store] = None,
        voyage_client: Optional[VoyageClient] = None,
        rrf_k: int = 60
    ):
        self.vector_store = vector_store or VectorStore()
        self.bm25_store = bm25_store or BM25Store()
        self.voyage_client = voyage_client or VoyageClient()
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 8, candidate_k: int = 20) -> List[Dict[str, Any]]:
        """
        Execute Hybrid Search using RRF over ChromaDB and BM25 store.
        """
        # 1. Get Dense Vector Results
        query_embedding = self.voyage_client.get_query_embedding(query)
        dense_results = self.vector_store.search(query_embedding, top_k=candidate_k)

        # 2. Get Sparse BM25 Results
        sparse_results = self.bm25_store.search(query, top_k=candidate_k)

        # 3. Apply Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}

        # Add ranks from dense search
        for rank, res in enumerate(dense_results, start=1):
            cid = res["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))
            if cid not in chunk_map:
                chunk_map[cid] = res

        # Add ranks from sparse BM25 search
        for rank, res in enumerate(sparse_results, start=1):
            cid = res["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))
            if cid not in chunk_map:
                chunk_map[cid] = res

        # Sort chunks by fused RRF score
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

        final_results = []
        for cid in sorted_cids:
            chunk_data = chunk_map[cid]
            chunk_data["rrf_score"] = rrf_scores[cid]
            final_results.append(chunk_data)

        return final_results
