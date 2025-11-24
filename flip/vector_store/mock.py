"""Mock vector store for testing."""

from typing import List, Dict, Any, Optional
import time

from flip.vector_store.base import (
    BaseVectorStore,
    SearchResult,
    VectorStoreStats,
    HealthCheckResult,
    HealthStatus
)


class MockVectorStore(BaseVectorStore):
    """Mock vector store for testing purposes."""
    
    def __init__(self, collection_name: str = "test_collection", **kwargs):
        """Initialize mock vector store."""
        super().__init__(collection_name, **kwargs)
        self.vectors: Dict[str, Dict[str, Any]] = {}
        self._dimension = 0
    
    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """Add vectors to mock store."""
        if embeddings and len(embeddings) > 0:
            self._dimension = len(embeddings[0])
        
        for i, id in enumerate(ids):
            self.vectors[id] = {
                "embedding": embeddings[i],
                "text": texts[i],
                "metadata": metadatas[i] if metadatas else {}
            }
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search mock store."""
        # Simple cosine similarity
        import numpy as np
        
        results = []
        query_vec = np.array(query_embedding)
        
        for id, data in self.vectors.items():
            # Apply filter if provided
            if filter_dict:
                matches = all(
                    data["metadata"].get(k) == v
                    for k, v in filter_dict.items()
                )
                if not matches:
                    continue
            
            vec = np.array(data["embedding"])
            similarity = np.dot(query_vec, vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(vec)
            )
            
            results.append(SearchResult(
                id=id,
                text=data["text"],
                score=float(similarity),
                metadata=data["metadata"]
            ))
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def get_by_ids(self, ids: List[str]) -> List[SearchResult]:
        """Get vectors by IDs."""
        results = []
        for id in ids:
            if id in self.vectors:
                data = self.vectors[id]
                results.append(SearchResult(
                    id=id,
                    text=data["text"],
                    score=1.0,
                    metadata=data["metadata"]
                ))
        return results
    
    def delete(self, ids: List[str]):
        """Delete vectors."""
        for id in ids:
            if id in self.vectors:
                del self.vectors[id]
    
    def update(
        self,
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
        texts: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """Update vectors."""
        for i, id in enumerate(ids):
            if id in self.vectors:
                if embeddings:
                    self.vectors[id]["embedding"] = embeddings[i]
                if texts:
                    self.vectors[id]["text"] = texts[i]
                if metadatas:
                    self.vectors[id]["metadata"] = metadatas[i]
    
    def count(self) -> int:
        """Get count."""
        return len(self.vectors)
    
    def clear(self):
        """Clear all vectors."""
        self.vectors.clear()
    
    def health_check(self) -> HealthCheckResult:
        """Health check."""
        start = time.time()
        count = self.count()
        latency_ms = (time.time() - start) * 1000
        
        return HealthCheckResult(
            status=HealthStatus.HEALTHY,
            latency_ms=latency_ms,
            message=f"Mock store healthy with {count} vectors"
        )
    
    def get_stats(self) -> VectorStoreStats:
        """Get statistics."""
        return VectorStoreStats(
            total_vectors=self.count(),
            dimension=self._dimension,
            metadata={"provider": "mock"}
        )
    
    @property
    def provider_name(self) -> str:
        """Provider name."""
        return "mock"
