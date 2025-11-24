"""Pinecone vector store implementation."""

from typing import List, Dict, Any, Optional
import time
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

from flip.vector_store.base import (
    BaseVectorStore,
    SearchResult,
    VectorStoreStats,
    HealthCheckResult,
    HealthStatus
)
from flip.core.exceptions import VectorStoreError


class PineconeVectorStore(BaseVectorStore):
    """Pinecone vector store implementation."""
    
    def __init__(
        self,
        collection_name: str,
        api_key: str,
        environment: Optional[str] = None,
        index_name: Optional[str] = None,
        namespace: str = "",
        dimension: int = 1536,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
        **kwargs
    ):
        """
        Initialize Pinecone vector store.
        
        Args:
            collection_name: Name of the collection (used as index name if index_name not provided)
            api_key: Pinecone API key
            environment: Pinecone environment (deprecated in newer versions)
            index_name: Specific index name (defaults to collection_name)
            namespace: Namespace for vectors (for multi-tenancy)
            dimension: Vector dimension
            metric: Distance metric (cosine, euclidean, dotproduct)
            cloud: Cloud provider (aws, gcp, azure)
            region: Cloud region
            **kwargs: Additional Pinecone settings
        """
        if not PINECONE_AVAILABLE:
            raise VectorStoreError(
                "Pinecone is not installed. Install with: pip install pinecone-client"
            )
        
        super().__init__(collection_name, **kwargs)
        
        self.api_key = api_key
        self.index_name = index_name or collection_name
        self.namespace = namespace
        self._dimension = dimension
        self.metric = metric
        
        # Initialize Pinecone client
        try:
            self.pc = Pinecone(api_key=api_key)
            
            # Create index if it doesn't exist
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric=metric,
                    spec=ServerlessSpec(
                        cloud=cloud,
                        region=region
                    )
                )
                # Wait for index to be ready
                time.sleep(1)
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize Pinecone: {str(e)}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """Add vectors to Pinecone with retry logic for rate limiting."""
        try:
            # Prepare vectors for Pinecone
            vectors = []
            for i, id in enumerate(ids):
                metadata = metadatas[i] if metadatas else {}
                # Add text to metadata
                metadata['text'] = texts[i]
                
                vectors.append({
                    'id': id,
                    'values': embeddings[i],
                    'metadata': metadata
                })
            
            # Upsert to Pinecone
            self.index.upsert(
                vectors=vectors,
                namespace=self.namespace
            )
            
        except Exception as e:
            raise VectorStoreError(f"Failed to add to Pinecone: {str(e)}")
    
    def batch_add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 100
    ):
        """Add vectors in batches with rate limiting."""
        total = len(ids)
        for i in range(0, total, batch_size):
            end_idx = min(i + batch_size, total)
            self.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                texts=texts[i:end_idx],
                metadatas=metadatas[i:end_idx] if metadatas else None
            )
            # Small delay to avoid rate limits
            if end_idx < total:
                time.sleep(0.1)
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search Pinecone for similar vectors."""
        try:
            # Query Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=self.namespace,
                filter=filter_dict,
                include_metadata=True
            )
            
            # Convert to SearchResult objects
            search_results = []
            for match in results.matches:
                metadata = match.metadata or {}
                text = metadata.pop('text', '')
                
                search_results.append(SearchResult(
                    id=match.id,
                    text=text,
                    score=match.score,
                    metadata=metadata
                ))
            
            return search_results
            
        except Exception as e:
            raise VectorStoreError(f"Failed to search Pinecone: {str(e)}")
    
    def filter_search(
        self,
        query_embedding: List[float],
        filters: Dict[str, Any],
        top_k: int = 5
    ) -> List[SearchResult]:
        """Search with metadata filtering."""
        return self.search(query_embedding, top_k, filter_dict=filters)
    
    def get_by_ids(
        self,
        ids: List[str]
    ) -> List[SearchResult]:
        """Retrieve vectors by their IDs."""
        try:
            results = self.index.fetch(
                ids=ids,
                namespace=self.namespace
            )
            
            search_results = []
            for id, vector_data in results.vectors.items():
                metadata = vector_data.metadata or {}
                text = metadata.pop('text', '')
                
                search_results.append(SearchResult(
                    id=id,
                    text=text,
                    score=1.0,
                    metadata=metadata
                ))
            
            return search_results
            
        except Exception as e:
            raise VectorStoreError(f"Failed to get by IDs from Pinecone: {str(e)}")
    
    def delete(self, ids: List[str]):
        """Delete vectors from Pinecone."""
        try:
            self.index.delete(
                ids=ids,
                namespace=self.namespace
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to delete from Pinecone: {str(e)}")
    
    def update(
        self,
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
        texts: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """Update vectors in Pinecone (uses upsert)."""
        if not embeddings:
            raise VectorStoreError("Pinecone update requires embeddings")
        
        # Pinecone doesn't have a separate update, use upsert
        self.add(ids, embeddings, texts or [""] * len(ids), metadatas)
    
    def count(self) -> int:
        """Get count of vectors in Pinecone."""
        try:
            stats = self.index.describe_index_stats()
            if self.namespace:
                return stats.namespaces.get(self.namespace, {}).get('vector_count', 0)
            return stats.total_vector_count
        except Exception as e:
            raise VectorStoreError(f"Failed to count Pinecone vectors: {str(e)}")
    
    def clear(self):
        """Clear all vectors from Pinecone namespace."""
        try:
            self.index.delete(
                delete_all=True,
                namespace=self.namespace
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to clear Pinecone: {str(e)}")
    
    def health_check(self) -> HealthCheckResult:
        """Check Pinecone health."""
        try:
            start = time.time()
            stats = self.index.describe_index_stats()
            latency_ms = (time.time() - start) * 1000
            
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                latency_ms=latency_ms,
                message=f"Pinecone index '{self.index_name}' is healthy",
                details={
                    "total_vectors": stats.total_vector_count,
                    "dimension": stats.dimension,
                    "index_fullness": stats.index_fullness
                }
            )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                latency_ms=0.0,
                message=f"Pinecone health check failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def get_stats(self) -> VectorStoreStats:
        """Get Pinecone statistics."""
        try:
            stats = self.index.describe_index_stats()
            
            total_vectors = stats.total_vector_count
            if self.namespace:
                total_vectors = stats.namespaces.get(self.namespace, {}).get('vector_count', 0)
            
            return VectorStoreStats(
                total_vectors=total_vectors,
                dimension=stats.dimension,
                metadata={
                    "provider": self.provider_name,
                    "index_name": self.index_name,
                    "namespace": self.namespace,
                    "metric": self.metric,
                    "index_fullness": stats.index_fullness
                }
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to get Pinecone stats: {str(e)}")
    
    @property
    def dimension(self) -> int:
        """Get vector dimension."""
        return self._dimension
    
    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "pinecone"
