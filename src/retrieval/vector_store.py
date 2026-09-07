import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import config

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Local ChromaDB Vector Store for managing document chunk embeddings and metadata.
    """
    COLLECTION_NAME = "ashen_era_archive"

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = str(persist_dir or config.CHROMA_PERSIST_DIR)
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]], batch_size: int = 200):
        """
        Add chunks and pre-computed embeddings to ChromaDB in batches.
        """
        if not chunks or not embeddings:
            return

        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match!")

        total = len(chunks)
        for i in range(0, total, batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]

            ids = [c["chunk_id"] for c in batch_chunks]
            documents = [c["text"] for c in batch_chunks]
            metadatas = [{
                "doc_id": c["doc_id"],
                "file_name": c["file_name"],
                "file_path": c["file_path"],
                "file_type": c["file_type"],
                "page_number": c["page_number"],
                "chunk_index": c["chunk_index"],
                "chunk_id": c["chunk_id"]
            } for c in batch_chunks]

            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=batch_embeddings,
                metadatas=metadatas
            )
        logger.info(f"Upserted {total} chunks into ChromaDB collection '{self.COLLECTION_NAME}'.")

    def search(self, query_embedding: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Query ChromaDB by embedding vector and return top_k results.
        """
        count = self.collection.count()
        if count == 0:
            return []

        actual_top_k = min(top_k, count)
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=actual_top_k,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            if "dimension" in str(e).lower():
                logger.warning(f"Vector dimension mismatch detected in ChromaDB ({e}). Please re-run ingestion with 'python scripts/ingest_corpus.py'.")
                return []
            raise e

        formatted_results = []
        if results and results.get("ids") and results["ids"][0]:
            ids = results["ids"][0]
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]

            for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists):
                similarity_score = 1.0 - dist  # Cosine distance to similarity conversion
                formatted_results.append({
                    "chunk_id": chunk_id,
                    "text": doc,
                    "metadata": meta,
                    "score": float(similarity_score),
                    "source": "vector"
                })

        return formatted_results

    def get_count(self) -> int:
        return self.collection.count()

    def clear(self):
        try:
            self.client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
