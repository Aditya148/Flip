"""Elasticsearch vector store implementation."""

from typing import List, Dict, Any, Optional
import time

try:
    from elasticsearch import Elasticsearch
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False

from flip.vector_store.base import (
    BaseVectorStore,
    SearchResult,
    VectorStoreStats,
    HealthCheckResult,
    HealthStatus
)
from flip.core.exceptions import VectorStoreError


class ElasticsearchVectorStore(BaseVectorStore):
    """Elasticsearch vector store implementation."""
    
    def __init__(
        self,
        collection_name: str,
        url: str = "http://localhost:9200",
        api_key: Optional[str] = None,
        dimension: int = 1536,
        **kwargs
    ):
        """
        Initialize Elasticsearch vector store.
        
        Args:
            collection_name: Name of the index
            url: Elasticsearch URL
            api_key: API key for authentication
            dimension: Vector dimension
            **kwargs: Additional Elasticsearch settings
        """
        if not ELASTICSEARCH_AVAILABLE:
            raise VectorStoreError(
                "Elasticsearch is not installed. Install with: pip install elasticsearch"
            )
        
        super().__init__(collection_name, **kwargs)
        
        self._dimension = dimension
        self.index_name = collection_name.lower()
        
        # Connect to Elasticsearch
        try:
            if api_key:
                self.client = Elasticsearch(
                    url,
                    api_key=api_key
                )
            else:
                self.client = Elasticsearch(url)
            
            # Create index if not exists
            self._create_index_if_needed()
            
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize Elasticsearch: {str(e)}")
    
    def _create_index_if_needed(self):
        """Create Elasticsearch index if it doesn't exist."""
        if not self.client.indices.exists(index=self.index_name):
            mappings = {
                "properties": {
                    "embedding": {
                        "type": "dense_vector",
                        "dims": self._dimension,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "text": {"type": "text"},
                    "metadata": {"type": "object", "enabled": True}
                }
            }
            
            self.client.indices.create(
                index=self.index_name,
                mappings=mappings
            )
    
    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """Add vectors to Elasticsearch."""
        try:
            operations = []
            for i, id in enumerate(ids):
                metadata = metadatas[i] if metadatas else {}
                
                operations.append({"index": {"_index": self.index_name, "_id": id}})
                operations.append({
                    "embedding": embeddings[i],
                    "text": texts[i],
                    "metadata": metadata
                })
            
            if operations:
                self.client.bulk(operations=operations, refresh=True)
            
        except Exception as e:
            raise VectorStoreError(f"Failed to add to Elasticsearch: {str(e)}")
    
    def batch_add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 100
    ):
        """Add vectors in batches."""
        total = len(ids)
        for i in range(0, total, batch_size):
            end_idx = min(i + batch_size, total)
            self.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                texts=texts[i:end_idx],
                metadatas=metadatas[i:end_idx] if metadatas else None
            )
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search Elasticsearch for similar vectors."""
        try:
            # Build query
            knn = {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": top_k,
                "num_candidates": top_k * 10
            }
            
            # Add filter if provided
            if filter_dict:
                knn["filter"] = [
                    {"term": {f"metadata.{k}": v}}
                    for k, v in filter_dict.items()
                ]
            
            response = self.client.search(
                index=self.index_name,
                knn=knn,
                size=top_k
            )
            
            # Convert to SearchResult objects
            search_results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                
                search_results.append(SearchResult(
                    id=hit["_id"],
                    text=source.get("text", ""),
                    score=hit["_score"],
                    metadata=source.get("metadata", {})
                ))
            
            return search_results
            
        except Exception as e:
            raise VectorStoreError(f"Failed to search Elasticsearch: {str(e)}")
    
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
            response = self.client.mget(
                index=self.index_name,
                ids=ids
            )
            
            search_results = []
            for doc in response["docs"]:
                if doc.get("found"):
                    source = doc["_source"]
                    search_results.append(SearchResult(
                        id=doc["_id"],
                        text=source.get("text", ""),
                        score=1.0,
                        metadata=source.get("metadata", {})
                    ))
            
            return search_results
            
        except Exception as e:
            raise VectorStoreError(f"Failed to get by IDs from Elasticsearch: {str(e)}")
    
    def delete(self, ids: List[str]):
        """Delete vectors from Elasticsearch."""
        try:
            operations = []
            for id in ids:
                operations.append({"delete": {"_index": self.index_name, "_id": id}})
            
            if operations:
                self.client.bulk(operations=operations, refresh=True)
        except Exception as e:
            raise VectorStoreError(f"Failed to delete from Elasticsearch: {str(e)}")
    
    def update(
        self,
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
        texts: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """Update vectors in Elasticsearch."""
        if not embeddings:
            raise VectorStoreError("Elasticsearch update requires embeddings")
        
        # Elasticsearch uses upsert
        self.add(ids, embeddings, texts or [""] * len(ids), metadatas)
    
    def count(self) -> int:
        """Get count of vectors in Elasticsearch."""
        try:
            response = self.client.count(index=self.index_name)
            return response["count"]
        except Exception as e:
            raise VectorStoreError(f"Failed to count Elasticsearch vectors: {str(e)}")
    
    def clear(self):
        """Clear all vectors from Elasticsearch index."""
        try:
            self.client.delete_by_query(
                index=self.index_name,
                body={"query": {"match_all": {}}}
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to clear Elasticsearch: {str(e)}")
    
    def health_check(self) -> HealthCheckResult:
        """Check Elasticsearch health."""
        try:
            start = time.time()
            health = self.client.cluster.health()
            latency_ms = (time.time() - start) * 1000
            
            status_map = {
                "green": HealthStatus.HEALTHY,
                "yellow": HealthStatus.DEGRADED,
                "red": HealthStatus.UNHEALTHY
            }
            
            return HealthCheckResult(
                status=status_map.get(health["status"], HealthStatus.UNKNOWN),
                latency_ms=latency_ms,
                message=f"Elasticsearch cluster status: {health['status']}",
                details=health
            )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                latency_ms=0.0,
                message=f"Elasticsearch health check failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def get_stats(self) -> VectorStoreStats:
        """Get Elasticsearch statistics."""
        try:
            count = self.count()
            
            return VectorStoreStats(
                total_vectors=count,
                dimension=self._dimension,
                metadata={
                    "provider": self.provider_name,
                    "index_name": self.index_name
                }
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to get Elasticsearch stats: {str(e)}")
    
    @property
    def dimension(self) -> int:
        """Get vector dimension."""
        return self._dimension
    
    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "elasticsearch"
