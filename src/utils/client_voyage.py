import os
import requests
from typing import List, Optional
import config
from src.utils.retry import retry_with_backoff
from src.utils.cache import default_cache

class VoyageClient:
    """
    Resilient Voyage AI Embeddings Client supporting voyage-4-lite,
    exponential backoff retry, disk caching, and batched requests.
    """
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or config.VOYAGE_API_KEY
        self.model = model or config.VOYAGE_EMBED_MODEL
        self.endpoint = "https://api.voyageai.com/v1/embeddings"

    @retry_with_backoff(max_retries=5, initial_delay=1.0, backoff_factor=2.0)
    def _call_api(self, texts: List[str], input_type: Optional[str] = None) -> List[List[float]]:
        if not self.api_key or self.api_key == "pa-your-voyage-api-key-here":
            # For testing/mocking when key is not provided yet, fallback to dummy embeddings
            print("[Voyage Warning] VOYAGE_API_KEY is not configured in .env. Returning dummy fallback embeddings.")
            return [[0.0] * 512 for _ in texts]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "input": texts,
            "model": self.model
        }
        if input_type:
            payload["input_type"] = input_type

        response = requests.post(self.endpoint, json=payload, headers=headers, timeout=30)
        
        if response.status_code != 200:
            raise RuntimeError(f"Voyage API error ({response.status_code}): {response.text}")
        
        data = response.json()
        embeddings = [item["embedding"] for item in data["data"]]
        return embeddings

    def get_embeddings(self, texts: List[str], input_type: Optional[str] = None, batch_size: int = 64, use_cache: bool = True) -> List[List[float]]:
        """
        Get embeddings for a list of texts with disk caching and batching.
        """
        if not texts:
            return []

        results: List[Optional[List[float]]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        # Check cache first
        for i, text in enumerate(texts):
            if use_cache:
                cached_vec = default_cache.get(f"embed:{self.model}", text)
                if cached_vec is not None:
                    results[i] = cached_vec
                    continue
            uncached_indices.append(i)
            uncached_texts.append(text)

        # Batch call API for uncached items
        if uncached_texts:
            for start_idx in range(0, len(uncached_texts), batch_size):
                batch_texts = uncached_texts[start_idx : start_idx + batch_size]
                batch_indices = uncached_indices[start_idx : start_idx + batch_size]
                
                batch_vecs = self._call_api(batch_texts, input_type=input_type)
                
                for idx, text, vec in zip(batch_indices, batch_texts, batch_vecs):
                    results[idx] = vec
                    if use_cache:
                        default_cache.set(f"embed:{self.model}", text, vec)

        return [vec for vec in results if vec is not None]

    def get_query_embedding(self, query: str) -> List[float]:
        """
        Helper method to get embedding for a single query text.
        """
        vecs = self.get_embeddings([query], input_type="query")
        return vecs[0]
