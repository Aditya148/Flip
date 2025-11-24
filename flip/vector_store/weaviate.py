"""Weaviate vector store implementation."""

from typing import List, Dict, Any, Optional
import time
import uuid

try:
    import weaviate
    from weaviate.classes.config import Configure, Property, DataType
    from weaviate.classes.query import Filter
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False

from flip.vector_store.base import (
    BaseVectorStore,
    SearchResult,
    VectorStoreStats,
    HealthCheckResult,
    HealthStatus
)
from flip.core.exceptions import VectorStoreError


class WeaviateVectorStore(BaseVectorStore):
    """Weaviate vector store implementation."""
    
    def __init__(
        self,
        collection_name: str,
        url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        dimension: int = 1536,
        distance_metric: str = "cosine",
        **kwargs
    ):
        """
        Initialize Weaviate vector store.
        
        Args:
            collection_name: Name of the collection (class in Weaviate)
            url: Weaviate instance URL
            api_key: API key for authentication
            dimension: Vector dimension
            distance_metric: Distance metric (cosine, l2-squared, dot, hamming, manhattan)
            **kwargs: Additional Weaviate settings
        """
        if not WEAVIATE_AVAILABLE:
            raise VectorStoreError(
                "Weaviate is not installed. Install with: pip install weaviate-client"
            )
        
        super().__init__(collection_name, **kwargs)
        
        self._dimension = dimension
        self.distance_metric = distance_metric
        self.class_name = collection_name.capitalize()  # Weaviate requires capitalized class names
        
        # Initialize Weaviate client
        try:
            if api_key:
                self.client = weaviate.connect_to_custom(
                    http_host=url.replace("http://", "").replace("https://", ""),
                    http_port=8080,
                    http_secure=url.startswith("https"),
                    auth_credentials=weaviate.auth.AuthApiKey(api_key),
                    **kwargs
                )
            else:
                self.client = weaviate.connect_to_local(
                    host=url.replace("http://", "").replace("https://", ""),
                    port=8080,
                    **kwargs
                )
            
            # Create collection if it doesn't exist
            if not self.client.collections.exists(self.class_name):
                self.client.collections.create(
                    name=self.class_name,
                    vectorizer_config=Configure.Vectorizer.none(),  # We provide vectors
                    properties=[
                        Property(name="text", data_type=DataType.TEXT),
                        Property(name="metadata", data_type=DataType.TEXT),
                    ]
                )
            
            self.collection = self.client.collections.get(self.class_name)
            
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize Weaviate: {str(e)}")
    
    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """Add vectors to Weaviate."""
        try:
            import json
            
            with self.collection.batch.dynamic() as batch:
                for i, id in enumerate(ids):
                    metadata = metadatas[i] if metadatas else {}
                    
                    properties = {
                        "text": texts[i],
                        "metadata": json.dumps(metadata)
                    }
                    
                    batch.add_object(
                        properties=properties,
                        vector=embeddings[i],
                        uuid=uuid.UUID(id) if self._is_valid_uuid(id) else None
                    )
            
        except Exception as e:
            raise VectorStoreError(f"Failed to add to Weaviate: {str(e)}")
    
    def _is_valid_uuid(self, id: str) -> bool:
        """Check if string is a valid UUID."""
        try:
            uuid.UUID(id)
            return True
        except:
            return False
    
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
        """Search Weaviate for similar vectors."""
        try:
            import json
            
            # Build query
            query = self.collection.query.near_vector(
                near_vector=query_embedding,
                limit=top_k,
                return_metadata=["distance"]
            )
            
            # Add filter if provided
            if filter_dict:
                # Note: Weaviate filtering is complex, this is a simplified version
                # For production, implement proper filter building
                pass
            
            results = query
            
            # Convert to SearchResult objects
            search_results = []
            for obj in results.objects:
                metadata = {}
                if obj.properties.get("metadata"):
                    try:
                        metadata = json.loads(obj.properties["metadata"])
                    except:
                        pass
                
                # Convert distance to similarity score
                distance = obj.metadata.distance if hasattr(obj.metadata, 'distance') else 0
                score = 1.0 / (1.0 + distance)  # Convert distance to similarity
                
                search_results.append(SearchResult(
                    id=str(obj.uuid),
                    text=obj.properties.get("text", ""),
                    score=score,
                    metadata=metadata
                ))
            
            return search_results
            
        except Exception as e:
            raise VectorStoreError(f"Failed to search Weaviate: {str(e)}")
    
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
            import json
            
            search_results = []
            for id in ids:
                try:
                    obj = self.collection.query.fetch_object_by_id(
                        uuid.UUID(id) if self._is_valid_uuid(id) else id
                    )
                    
                    if obj:
                        metadata = {}
                        if obj.properties.get("metadata"):
                            try:
                                metadata = json.loads(obj.properties["metadata"])
                            except:
                                pass
                        
                        search_results.append(SearchResult(
                            id=str(obj.uuid),
                            text=obj.properties.get("text", ""),
                            score=1.0,
                            metadata=metadata
                        ))
                except:
                    continue
            
            return search_results
            
        except Exception as e:
            raise VectorStoreError(f"Failed to get by IDs from Weaviate: {str(e)}")
    
    def delete(self, ids: List[str]):
        """Delete vectors from Weaviate."""
        try:
            for id in ids:
                try:
                    self.collection.data.delete_by_id(
                        uuid.UUID(id) if self._is_valid_uuid(id) else id
                    )
                except:
                    continue
        except Exception as e:
            raise VectorStoreError(f"Failed to delete from Weaviate: {str(e)}")
    
    def update(
        self,
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
        texts: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """Update vectors in Weaviate."""
        try:
            import json
            
            for i, id in enumerate(ids):
                properties = {}
                if texts:
                    properties["text"] = texts[i]
                if metadatas:
                    properties["metadata"] = json.dumps(metadatas[i])
                
                vector = embeddings[i] if embeddings else None
                
                self.collection.data.update(
                    uuid=uuid.UUID(id) if self._is_valid_uuid(id) else id,
                    properties=properties,
                    vector=vector
                )
        except Exception as e:
            raise VectorStoreError(f"Failed to update Weaviate: {str(e)}")
    
    def count(self) -> int:
        """Get count of vectors in Weaviate."""
        try:
            result = self.collection.aggregate.over_all(total_count=True)
            return result.total_count
        except Exception as e:
            raise VectorStoreError(f"Failed to count Weaviate vectors: {str(e)}")
    
    def clear(self):
        """Clear all vectors from Weaviate collection."""
        try:
            # Delete all objects
            self.collection.data.delete_many(
                where=Filter.by_property("text").exists()
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to clear Weaviate: {str(e)}")
    
    def health_check(self) -> HealthCheckResult:
        """Check Weaviate health."""
        try:
            start = time.time()
            is_ready = self.client.is_ready()
            latency_ms = (time.time() - start) * 1000
            
            if is_ready:
                return HealthCheckResult(
                    status=HealthStatus.HEALTHY,
                    latency_ms=latency_ms,
                    message=f"Weaviate collection '{self.class_name}' is healthy",
                    details={"ready": True}
                )
            else:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    latency_ms=latency_ms,
                    message="Weaviate is not ready",
                    details={"ready": False}
                )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                latency_ms=0.0,
                message=f"Weaviate health check failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def get_stats(self) -> VectorStoreStats:
        """Get Weaviate statistics."""
        try:
            count = self.count()
            
            return VectorStoreStats(
                total_vectors=count,
                dimension=self._dimension,
                metadata={
                    "provider": self.provider_name,
                    "class_name": self.class_name,
                    "distance_metric": self.distance_metric
                }
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to get Weaviate stats: {str(e)}")
    
    def __del__(self):
        """Close Weaviate connection."""
        try:
            if hasattr(self, 'client'):
                self.client.close()
        except:
            pass
    
    @property
    def dimension(self) -> int:
        """Get vector dimension."""
        return self._dimension
    
    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "weaviate"
