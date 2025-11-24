"""Qdrant vector store implementation."""

from typing import List, Dict, Any, Optional
import time
from uuid import uuid4

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
        SearchParams
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

from flip.vector_store.base import (
    BaseVectorStore,
    SearchResult,
    VectorStoreStats,
    HealthCheckResult,
    HealthStatus
)
from flip.core.exceptions import VectorStoreError


class QdrantVectorStore(BaseVectorStore):
    """Qdrant vector store implementation."""
    
    def __init__(
        self,
        collection_name: str,
        url: Optional[str] = None,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        dimension: int = 1536,
        distance: str = "cosine",
        prefer_grpc: bool = False,
        **kwargs
    ):
        """
        Initialize Qdrant vector store.
        
        Args:
            collection_name: Name of the collection
            url: Full URL to Qdrant instance (e.g., "http://localhost:6333")
            host: Qdrant host (used if url not provided)
            port: Qdrant port (used if url not provided)
            api_key: API key for Qdrant Cloud
            dimension: Vector dimension
            distance: Distance metric (cosine, euclidean, dot)
            prefer_grpc: Use gRPC instead of HTTP
            **kwargs: Additional Qdrant settings
        """
        if not QDRANT_AVAILABLE:
            raise VectorStoreError(
                "Qdrant is not installed. Install with: pip install qdrant-client"
            )
        
        super().__init__(collection_name, **kwargs)
        
        self._dimension = dimension
        self.distance_metric = distance
        
        # Map distance metric
        distance_map = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dot": Distance.DOT,
        }
        self.distance = distance_map.get(distance.lower(), Distance.COSINE)
        
        # Initialize Qdrant client
        try:
            if url:
                self.client = QdrantClient(
                    url=url,
                    api_key=api_key,
                    prefer_grpc=prefer_grpc,
                    **kwargs
                )
            else:
                self.client = QdrantClient(
                    host=host,
                    port=port,
                    api_key=api_key,
                    prefer_grpc=prefer_grpc,
                    **kwargs
                )
            
            # Create collection if it doesn't exist
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=dimension,
                        distance=self.distance
                    )
                )
            
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize Qdrant: {str(e)}")
    
    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """Add vectors to Qdrant."""
        try:
            points = []
            for i, id in enumerate(ids):
                payload = metadatas[i].copy() if metadatas else {}
                payload['text'] = texts[i]
                
                points.append(PointStruct(
                    id=id,
                    vector=embeddings[i],
                    payload=payload
                ))
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
        except Exception as e:
            raise VectorStoreError(f"Failed to add to Qdrant: {str(e)}")
    
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
        """Search Qdrant for similar vectors."""
        try:
            # Build filter if provided
            query_filter = None
            if filter_dict:
                query_filter = self._build_filter(filter_dict)
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=query_filter
            )
            
            # Convert to SearchResult objects
            search_results = []
            for result in results:
                payload = result.payload or {}
                text = payload.pop('text', '')
                
                search_results.append(SearchResult(
                    id=str(result.id),
                    text=text,
                    score=result.score,
                    metadata=payload
                ))
            
            return search_results
            
        except Exception as e:
            raise VectorStoreError(f"Failed to search Qdrant: {str(e)}")
    
    def filter_search(
        self,
        query_embedding: List[float],
        filters: Dict[str, Any],
        top_k: int = 5
    ) -> List[SearchResult]:
        """Search with metadata filtering."""
        return self.search(query_embedding, top_k, filter_dict=filters)
    
    def _build_filter(self, filter_dict: Dict[str, Any]) -> Filter:
        """Build Qdrant filter from dictionary."""
        conditions = []
        for key, value in filter_dict.items():
            conditions.append(
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                )
            )
        return Filter(must=conditions)
    
    def get_by_ids(
        self,
        ids: List[str]
    ) -> List[SearchResult]:
        """Retrieve vectors by their IDs."""
        try:
            results = self.client.retrieve(
                collection_name=self.collection_name,
                ids=ids,
                with_payload=True,
                with_vectors=False
            )
            
            search_results = []
            for result in results:
                payload = result.payload or {}
                text = payload.pop('text', '')
                
                search_results.append(SearchResult(
                    id=str(result.id),
                    text=text,
                    score=1.0,
                    metadata=payload
                ))
            
            return search_results
            
        except Exception as e:
            raise VectorStoreError(f"Failed to get by IDs from Qdrant: {str(e)}")
    
    def delete(self, ids: List[str]):
        """Delete vectors from Qdrant."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=ids
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to delete from Qdrant: {str(e)}")
    
    def update(
        self,
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
        texts: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """Update vectors in Qdrant (uses upsert)."""
        if not embeddings:
            raise VectorStoreError("Qdrant update requires embeddings")
        
        # Qdrant uses upsert for updates
        self.add(ids, embeddings, texts or [""] * len(ids), metadatas)
    
    def count(self) -> int:
        """Get count of vectors in Qdrant."""
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count
        except Exception as e:
            raise VectorStoreError(f"Failed to count Qdrant vectors: {str(e)}")
    
    def clear(self):
        """Clear all vectors from Qdrant collection."""
        try:
            # Delete and recreate collection
            self.client.delete_collection(self.collection_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self._dimension,
                    distance=self.distance
                )
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to clear Qdrant: {str(e)}")
    
    def health_check(self) -> HealthCheckResult:
        """Check Qdrant health."""
        try:
            start = time.time()
            info = self.client.get_collection(self.collection_name)
            latency_ms = (time.time() - start) * 1000
            
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                latency_ms=latency_ms,
                message=f"Qdrant collection '{self.collection_name}' is healthy",
                details={
                    "points_count": info.points_count,
                    "vectors_count": info.vectors_count,
                    "status": info.status.value
                }
            )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                latency_ms=0.0,
                message=f"Qdrant health check failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def get_stats(self) -> VectorStoreStats:
        """Get Qdrant statistics."""
        try:
            info = self.client.get_collection(self.collection_name)
            
            return VectorStoreStats(
                total_vectors=info.points_count,
                dimension=self._dimension,
                metadata={
                    "provider": self.provider_name,
                    "collection_name": self.collection_name,
                    "distance_metric": self.distance_metric,
                    "vectors_count": info.vectors_count,
                    "indexed_vectors_count": info.indexed_vectors_count,
                    "status": info.status.value
                }
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to get Qdrant stats: {str(e)}")
    
    def create_snapshot(self) -> str:
        """Create a snapshot of the collection."""
        try:
            snapshot = self.client.create_snapshot(
                collection_name=self.collection_name
            )
            return snapshot.name
        except Exception as e:
            raise VectorStoreError(f"Failed to create Qdrant snapshot: {str(e)}")
    
    @property
    def dimension(self) -> int:
        """Get vector dimension."""
        return self._dimension
    
    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "qdrant"
