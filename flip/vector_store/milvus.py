"""Milvus vector store implementation."""

from typing import List, Dict, Any, Optional
import time

try:
    from pymilvus import (
        connections,
        Collection,
        CollectionSchema,
        FieldSchema,
        DataType,
        utility
    )
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

from flip.vector_store.base import (
    BaseVectorStore,
    SearchResult,
    VectorStoreStats,
    HealthCheckResult,
    HealthStatus
)
from flip.core.exceptions import VectorStoreError


class MilvusVectorStore(BaseVectorStore):
    """Milvus vector store implementation."""
    
    def __init__(
        self,
        collection_name: str,
        host: str = "localhost",
        port: int = 19530,
        user: str = "",
        password: str = "",
        dimension: int = 1536,
        index_type: str = "IVF_FLAT",
        metric_type: str = "L2",
        consistency_level: str = "Strong",
        **kwargs
    ):
        """
        Initialize Milvus vector store.
        
        Args:
            collection_name: Name of the collection
            host: Milvus host
            port: Milvus port
            user: Username for authentication
            password: Password for authentication
            dimension: Vector dimension
            index_type: Index type (IVF_FLAT, IVF_SQ8, IVF_PQ, HNSW, FLAT)
            metric_type: Distance metric (L2, IP, COSINE)
            consistency_level: Consistency level (Strong, Session, Bounded, Eventually)
            **kwargs: Additional Milvus settings
        """
        if not MILVUS_AVAILABLE:
            raise VectorStoreError(
                "Milvus is not installed. Install with: pip install pymilvus"
            )
        
        super().__init__(collection_name, **kwargs)
        
        self._dimension = dimension
        self.index_type = index_type
        self.metric_type = metric_type
        self.consistency_level = consistency_level
        
        # Connect to Milvus
        try:
            connections.connect(
                alias="default",
                host=host,
                port=port,
                user=user,
                password=password
            )
            
            # Create collection if it doesn't exist
            if not utility.has_collection(collection_name):
                self._create_collection()
            
            self.collection = Collection(collection_name)
            
            # Create index if not exists
            if not self.collection.has_index():
                self._create_index()
            
            # Load collection into memory
            self.collection.load()
            
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize Milvus: {str(e)}")
    
    def _create_collection(self):
        """Create Milvus collection with schema."""
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self._dimension),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description=f"Collection for {self.collection_name}"
        )
        
        Collection(
            name=self.collection_name,
            schema=schema,
            consistency_level=self.consistency_level
        )
    
    def _create_index(self):
        """Create index on vector field."""
        index_params = {
            "metric_type": self.metric_type,
            "index_type": self.index_type,
            "params": self._get_index_params()
        }
        
        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
    
    def _get_index_params(self) -> Dict[str, Any]:
        """Get index parameters based on index type."""
        if self.index_type == "IVF_FLAT":
            return {"nlist": 1024}
        elif self.index_type == "IVF_SQ8":
            return {"nlist": 1024}
        elif self.index_type == "IVF_PQ":
            return {"nlist": 1024, "m": 8}
        elif self.index_type == "HNSW":
            return {"M": 16, "efConstruction": 200}
        elif self.index_type == "FLAT":
            return {}
        else:
            return {"nlist": 1024}
    
    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """Add vectors to Milvus."""
        try:
            import json
            
            # Prepare data
            metadata_strs = []
            for i in range(len(ids)):
                metadata = metadatas[i] if metadatas else {}
                metadata_strs.append(json.dumps(metadata))
            
            data = [
                ids,
                embeddings,
                texts,
                metadata_strs
            ]
            
            self.collection.insert(data)
            self.collection.flush()
            
        except Exception as e:
            raise VectorStoreError(f"Failed to add to Milvus: {str(e)}")
    
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
        """Search Milvus for similar vectors."""
        try:
            search_params = {
                "metric_type": self.metric_type,
                "params": self._get_search_params()
            }
            
            # Build filter expression if provided
            expr = None
            if filter_dict:
                expr = self._build_filter_expr(filter_dict)
            
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=["text", "metadata"]
            )
            
            # Convert to SearchResult objects
            search_results = []
            if results and len(results) > 0:
                import json
                for hit in results[0]:
                    metadata = {}
                    if hit.entity.get("metadata"):
                        try:
                            metadata = json.loads(hit.entity.get("metadata"))
                        except:
                            pass
                    
                    # Convert distance to similarity score
                    score = 1.0 / (1.0 + hit.distance) if self.metric_type == "L2" else hit.distance
                    
                    search_results.append(SearchResult(
                        id=str(hit.id),
                        text=hit.entity.get("text", ""),
                        score=score,
                        metadata=metadata
                    ))
            
            return search_results
            
        except Exception as e:
            raise VectorStoreError(f"Failed to search Milvus: {str(e)}")
    
    def _get_search_params(self) -> Dict[str, Any]:
        """Get search parameters based on index type."""
        if self.index_type == "IVF_FLAT" or self.index_type == "IVF_SQ8":
            return {"nprobe": 10}
        elif self.index_type == "IVF_PQ":
            return {"nprobe": 10}
        elif self.index_type == "HNSW":
            return {"ef": 64}
        else:
            return {}
    
    def _build_filter_expr(self, filter_dict: Dict[str, Any]) -> str:
        """Build Milvus filter expression."""
        # Simple implementation - can be enhanced
        conditions = []
        for key, value in filter_dict.items():
            if isinstance(value, str):
                conditions.append(f'{key} == "{value}"')
            else:
                conditions.append(f'{key} == {value}')
        return " and ".join(conditions) if conditions else None
    
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
            
            expr = f'id in {ids}'
            results = self.collection.query(
                expr=expr,
                output_fields=["id", "text", "metadata"]
            )
            
            search_results = []
            for result in results:
                metadata = {}
                if result.get("metadata"):
                    try:
                        metadata = json.loads(result["metadata"])
                    except:
                        pass
                
                search_results.append(SearchResult(
                    id=result["id"],
                    text=result.get("text", ""),
                    score=1.0,
                    metadata=metadata
                ))
            
            return search_results
            
        except Exception as e:
            raise VectorStoreError(f"Failed to get by IDs from Milvus: {str(e)}")
    
    def delete(self, ids: List[str]):
        """Delete vectors from Milvus."""
        try:
            expr = f'id in {ids}'
            self.collection.delete(expr)
            self.collection.flush()
        except Exception as e:
            raise VectorStoreError(f"Failed to delete from Milvus: {str(e)}")
    
    def update(
        self,
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
        texts: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """Update vectors in Milvus (delete and re-insert)."""
        if not embeddings:
            raise VectorStoreError("Milvus update requires embeddings")
        
        # Milvus doesn't support direct update, so delete and re-insert
        self.delete(ids)
        self.add(ids, embeddings, texts or [""] * len(ids), metadatas)
    
    def count(self) -> int:
        """Get count of vectors in Milvus."""
        try:
            return self.collection.num_entities
        except Exception as e:
            raise VectorStoreError(f"Failed to count Milvus vectors: {str(e)}")
    
    def clear(self):
        """Clear all vectors from Milvus collection."""
        try:
            # Drop and recreate collection
            self.collection.drop()
            self._create_collection()
            self.collection = Collection(self.collection_name)
            self._create_index()
            self.collection.load()
        except Exception as e:
            raise VectorStoreError(f"Failed to clear Milvus: {str(e)}")
    
    def health_check(self) -> HealthCheckResult:
        """Check Milvus health."""
        try:
            start = time.time()
            # Check if collection is loaded
            is_loaded = utility.load_state(self.collection_name)
            latency_ms = (time.time() - start) * 1000
            
            if is_loaded:
                return HealthCheckResult(
                    status=HealthStatus.HEALTHY,
                    latency_ms=latency_ms,
                    message=f"Milvus collection '{self.collection_name}' is healthy",
                    details={"loaded": True}
                )
            else:
                return HealthCheckResult(
                    status=HealthStatus.DEGRADED,
                    latency_ms=latency_ms,
                    message="Collection exists but not loaded",
                    details={"loaded": False}
                )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                latency_ms=0.0,
                message=f"Milvus health check failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def get_stats(self) -> VectorStoreStats:
        """Get Milvus statistics."""
        try:
            count = self.count()
            
            return VectorStoreStats(
                total_vectors=count,
                dimension=self._dimension,
                metadata={
                    "provider": self.provider_name,
                    "collection_name": self.collection_name,
                    "index_type": self.index_type,
                    "metric_type": self.metric_type,
                    "consistency_level": self.consistency_level
                }
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to get Milvus stats: {str(e)}")
    
    def __del__(self):
        """Disconnect from Milvus."""
        try:
            connections.disconnect("default")
        except:
            pass
    
    @property
    def dimension(self) -> int:
        """Get vector dimension."""
        return self._dimension
    
    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "milvus"
