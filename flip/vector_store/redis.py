"""Redis vector store implementation with RediSearch."""

from typing import List, Dict, Any, Optional
import time
import json

try:
    import redis
    from redis.commands.search.field import VectorField, TextField, TagField
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
    from redis.commands.search.query import Query
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from flip.vector_store.base import (
    BaseVectorStore,
    SearchResult,
    VectorStoreStats,
    HealthCheckResult,
    HealthStatus
)
from flip.core.exceptions import VectorStoreError


class RedisVectorStore(BaseVectorStore):
    """Redis vector store implementation with RediSearch."""
    
    def __init__(
        self,
        collection_name: str,
        host: str = "localhost",
        port: int = 6379,
        password: Optional[str] = None,
        db: int = 0,
        dimension: int = 1536,
        distance_metric: str = "COSINE",
        **kwargs
    ):
        """
        Initialize Redis vector store.
        
        Args:
            collection_name: Name of the index
            host: Redis host
            port: Redis port
            password: Redis password
            db: Redis database number
            dimension: Vector dimension
            distance_metric: Distance metric (COSINE, L2, IP)
            **kwargs: Additional Redis settings
        """
        if not REDIS_AVAILABLE:
            raise VectorStoreError(
                "Redis is not installed. Install with: pip install redis"
            )
        
        super().__init__(collection_name, **kwargs)
        
        self._dimension = dimension
        self.distance_metric = distance_metric
        self.index_name = f"idx:{collection_name}"
        self.prefix = f"{collection_name}:"
        
        # Connect to Redis
        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                password=password,
                db=db,
                decode_responses=False
            )
            
            # Create index if not exists
            self._create_index_if_needed()
            
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize Redis: {str(e)}")
    
    def _create_index_if_needed(self):
        """Create RediSearch index if it doesn't exist."""
        try:
            self.client.ft(self.index_name).info()
        except:
            # Index doesn't exist, create it
            schema = (
                VectorField(
                    "embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self._dimension,
                        "DISTANCE_METRIC": self.distance_metric
                    }
                ),
                TextField("text"),
                TextField("metadata")
            )
            
            definition = IndexDefinition(
                prefix=[self.prefix],
                index_type=IndexType.HASH
            )
            
            self.client.ft(self.index_name).create_index(
                fields=schema,
                definition=definition
            )
    
    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """Add vectors to Redis."""
        try:
            import numpy as np
            
            pipe = self.client.pipeline()
            for i, id in enumerate(ids):
                key = f"{self.prefix}{id}"
                metadata = metadatas[i] if metadatas else {}
                
                # Convert embedding to bytes
                embedding_bytes = np.array(embeddings[i], dtype=np.float32).tobytes()
                
                pipe.hset(
                    key,
                    mapping={
                        "embedding": embedding_bytes,
                        "text": texts[i],
                        "metadata": json.dumps(metadata)
                    }
                )
            
            pipe.execute()
            
        except Exception as e:
            raise VectorStoreError(f"Failed to add to Redis: {str(e)}")
    
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
        """Search Redis for similar vectors."""
        try:
            import numpy as np
            
            # Convert query to bytes
            query_bytes = np.array(query_embedding, dtype=np.float32).tobytes()
            
            # Build query
            query = (
                Query(f"*=>[KNN {top_k} @embedding $vec AS score]")
                .return_fields("text", "metadata", "score")
                .sort_by("score")
                .dialect(2)
            )
            
            results = self.client.ft(self.index_name).search(
                query,
                query_params={"vec": query_bytes}
            )
            
            # Convert to SearchResult objects
            search_results = []
            for doc in results.docs:
                text = doc.text.decode() if isinstance(doc.text, bytes) else doc.text
                metadata_str = doc.metadata.decode() if isinstance(doc.metadata, bytes) else doc.metadata
                metadata = json.loads(metadata_str) if metadata_str else {}
                
                # Apply filter if provided
                if filter_dict:
                    if not all(metadata.get(k) == v for k, v in filter_dict.items()):
                        continue
                
                # Extract ID from doc.id
                id = doc.id.replace(self.prefix, "")
                
                search_results.append(SearchResult(
                    id=id,
                    text=text,
                    score=1.0 - float(doc.score),  # Convert distance to similarity
                    metadata=metadata
                ))
            
            return search_results
            
        except Exception as e:
            raise VectorStoreError(f"Failed to search Redis: {str(e)}")
    
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
            search_results = []
            for id in ids:
                key = f"{self.prefix}{id}"
                data = self.client.hgetall(key)
                
                if data:
                    text = data[b'text'].decode() if b'text' in data else ""
                    metadata_str = data[b'metadata'].decode() if b'metadata' in data else "{}"
                    metadata = json.loads(metadata_str)
                    
                    search_results.append(SearchResult(
                        id=id,
                        text=text,
                        score=1.0,
                        metadata=metadata
                    ))
            
            return search_results
            
        except Exception as e:
            raise VectorStoreError(f"Failed to get by IDs from Redis: {str(e)}")
    
    def delete(self, ids: List[str]):
        """Delete vectors from Redis."""
        try:
            keys = [f"{self.prefix}{id}" for id in ids]
            self.client.delete(*keys)
        except Exception as e:
            raise VectorStoreError(f"Failed to delete from Redis: {str(e)}")
    
    def update(
        self,
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
        texts: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """Update vectors in Redis."""
        if not embeddings:
            raise VectorStoreError("Redis update requires embeddings")
        
        # Redis uses upsert
        self.add(ids, embeddings, texts or [""] * len(ids), metadatas)
    
    def count(self) -> int:
        """Get count of vectors in Redis."""
        try:
            info = self.client.ft(self.index_name).info()
            return int(info["num_docs"])
        except Exception as e:
            raise VectorStoreError(f"Failed to count Redis vectors: {str(e)}")
    
    def clear(self):
        """Clear all vectors from Redis index."""
        try:
            # Delete all keys with prefix
            keys = self.client.keys(f"{self.prefix}*")
            if keys:
                self.client.delete(*keys)
        except Exception as e:
            raise VectorStoreError(f"Failed to clear Redis: {str(e)}")
    
    def health_check(self) -> HealthCheckResult:
        """Check Redis health."""
        try:
            start = time.time()
            self.client.ping()
            latency_ms = (time.time() - start) * 1000
            
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                latency_ms=latency_ms,
                message=f"Redis index '{self.index_name}' is healthy",
                details={"connected": True}
            )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                latency_ms=0.0,
                message=f"Redis health check failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def get_stats(self) -> VectorStoreStats:
        """Get Redis statistics."""
        try:
            count = self.count()
            
            return VectorStoreStats(
                total_vectors=count,
                dimension=self._dimension,
                metadata={
                    "provider": self.provider_name,
                    "index_name": self.index_name,
                    "distance_metric": self.distance_metric
                }
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to get Redis stats: {str(e)}")
    
    @property
    def dimension(self) -> int:
        """Get vector dimension."""
        return self._dimension
    
    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "redis"
