"""Base interface for vector stores with enhanced capabilities."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import time


class HealthStatus(Enum):
    """Health status for vector store."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class SearchResult:
    """Result from vector search."""
    
    id: str
    """Document/chunk ID."""
    
    text: str
    """Document/chunk text."""
    
    score: float
    """Similarity score."""
    
    metadata: Dict[str, Any]
    """Associated metadata."""


@dataclass
class VectorStoreStats:
    """Statistics for vector store."""
    total_vectors: int
    dimension: int
    index_size_bytes: Optional[int] = None
    memory_usage_bytes: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheckResult:
    """Result of health check."""
    status: HealthStatus
    latency_ms: float
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class BaseVectorStore(ABC):
    """Abstract base class for vector stores with enhanced capabilities."""
    
    def __init__(self, collection_name: str, **kwargs):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the collection/index
            **kwargs: Additional provider-specific arguments
        """
        self.collection_name = collection_name
        self.kwargs = kwargs
    
    @abstractmethod
    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Add vectors to the store.
        
        Args:
            ids: List of unique IDs
            embeddings: List of embedding vectors
            texts: List of text content
            metadatas: Optional list of metadata dictionaries
        """
        pass
    
    def batch_add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 100
    ):
        """
        Add vectors in batches for better performance.
        
        Default implementation splits into batches and calls add().
        Subclasses can override for optimized batch operations.
        
        Args:
            ids: List of unique IDs
            embeddings: List of embedding vectors
            texts: List of text content
            metadatas: Optional list of metadata dictionaries
            batch_size: Number of items per batch
        """
        total = len(ids)
        for i in range(0, total, batch_size):
            end_idx = min(i + batch_size, total)
            self.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                texts=texts[i:end_idx],
                metadatas=metadatas[i:end_idx] if metadatas else None
            )
    
    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of SearchResult objects
        """
        pass
    
    def filter_search(
        self,
        query_embedding: List[float],
        filters: Dict[str, Any],
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Search with metadata filtering.
        
        Default implementation uses search() with filter_dict.
        Subclasses can override for optimized filtering.
        
        Args:
            query_embedding: Query vector
            filters: Metadata filters (e.g., {"source": "doc1.txt", "page": 1})
            top_k: Number of results
            
        Returns:
            Filtered search results
        """
        return self.search(query_embedding, top_k, filter_dict=filters)
    
    def get_by_ids(
        self,
        ids: List[str]
    ) -> List[SearchResult]:
        """
        Retrieve vectors by their IDs.
        
        Default implementation returns empty list.
        Subclasses should override for actual retrieval.
        
        Args:
            ids: List of IDs to retrieve
            
        Returns:
            List of vectors with metadata
        """
        return []
    
    @abstractmethod
    def delete(self, ids: List[str]):
        """
        Delete vectors by IDs.
        
        Args:
            ids: List of IDs to delete
        """
        pass
    
    @abstractmethod
    def update(
        self,
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
        texts: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Update existing vectors.
        
        Args:
            ids: List of IDs to update
            embeddings: Optional new embeddings
            texts: Optional new texts
            metadatas: Optional new metadata
        """
        pass
    
    @abstractmethod
    def count(self) -> int:
        """
        Get the number of vectors in the store.
        
        Returns:
            Number of vectors
        """
        pass
    
    @abstractmethod
    def clear(self):
        """Clear all vectors from the store."""
        pass
    
    def health_check(self) -> HealthCheckResult:
        """
        Check the health of the vector store.
        
        Default implementation does a simple count operation.
        Subclasses can override for more comprehensive checks.
        
        Returns:
            Health check result with status and latency
        """
        try:
            start = time.time()
            count = self.count()
            latency_ms = (time.time() - start) * 1000
            
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                latency_ms=latency_ms,
                message=f"Store is healthy with {count} vectors",
                details={"vector_count": count}
            )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                latency_ms=0.0,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def get_stats(self) -> VectorStoreStats:
        """
        Get statistics about the vector store.
        
        Default implementation returns basic stats.
        Subclasses should override for detailed statistics.
        
        Returns:
            Vector store statistics
        """
        return VectorStoreStats(
            total_vectors=self.count(),
            dimension=0,  # Subclasses should provide actual dimension
            metadata={"provider": self.provider_name}
        )
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass
