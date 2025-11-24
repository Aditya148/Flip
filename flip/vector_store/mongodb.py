"""MongoDB vector store implementation (for metadata storage)."""

from typing import List, Dict, Any, Optional
import time

try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

from flip.vector_store.base import (
    BaseVectorStore,
    SearchResult,
    VectorStoreStats,
    HealthCheckResult,
    HealthStatus
)
from flip.core.exceptions import VectorStoreError


class MongoDBVectorStore(BaseVectorStore):
    """MongoDB vector store implementation (primarily for metadata storage)."""
    
    def __init__(
        self,
        collection_name: str,
        uri: str = "mongodb://localhost:27017/",
        database: str = "flip_db",
        dimension: int = 1536,
        **kwargs
    ):
        """
        Initialize MongoDB vector store.
        
        Args:
            collection_name: Name of the collection
            uri: MongoDB connection URI
            database: Database name
            dimension: Vector dimension
            **kwargs: Additional MongoDB settings
        """
        if not MONGODB_AVAILABLE:
            raise VectorStoreError(
                "MongoDB is not installed. Install with: pip install pymongo"
            )
        
        super().__init__(collection_name, **kwargs)
        
        self._dimension = dimension
        
        # Connect to MongoDB
        try:
            self.client = MongoClient(uri)
            self.db = self.client[database]
            self.collection = self.db[collection_name]
            
            # Create index on id field
            self.collection.create_index("id", unique=True)
            
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize MongoDB: {str(e)}")
    
    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """Add vectors to MongoDB."""
        try:
            documents = []
            for i, id in enumerate(ids):
                metadata = metadatas[i] if metadatas else {}
                
                doc = {
                    "id": id,
                    "embedding": embeddings[i],
                    "text": texts[i],
                    "metadata": metadata
                }
                documents.append(doc)
            
            # Use replace_one with upsert for each document
            for doc in documents:
                self.collection.replace_one(
                    {"id": doc["id"]},
                    doc,
                    upsert=True
                )
            
        except Exception as e:
            raise VectorStoreError(f"Failed to add to MongoDB: {str(e)}")
    
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
        """
        Search MongoDB for similar vectors.
        Note: MongoDB doesn't have native vector similarity search,
        so this performs a simple cosine similarity calculation.
        """
        try:
            import numpy as np
            
            # Get all documents (or filtered)
            query = {}
            if filter_dict:
                for key, value in filter_dict.items():
                    query[f"metadata.{key}"] = value
            
            docs = list(self.collection.find(query))
            
            # Calculate cosine similarity
            query_vec = np.array(query_embedding)
            results = []
            
            for doc in docs:
                doc_vec = np.array(doc["embedding"])
                similarity = np.dot(query_vec, doc_vec) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
                )
                
                results.append({
                    "id": doc["id"],
                    "text": doc["text"],
                    "metadata": doc.get("metadata", {}),
                    "score": float(similarity)
                })
            
            # Sort by similarity and take top_k
            results.sort(key=lambda x: x["score"], reverse=True)
            results = results[:top_k]
            
            # Convert to SearchResult objects
            search_results = [
                SearchResult(
                    id=r["id"],
                    text=r["text"],
                    score=r["score"],
                    metadata=r["metadata"]
                )
                for r in results
            ]
            
            return search_results
            
        except Exception as e:
            raise VectorStoreError(f"Failed to search MongoDB: {str(e)}")
    
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
            docs = self.collection.find({"id": {"$in": ids}})
            
            search_results = []
            for doc in docs:
                search_results.append(SearchResult(
                    id=doc["id"],
                    text=doc["text"],
                    score=1.0,
                    metadata=doc.get("metadata", {})
                ))
            
            return search_results
            
        except Exception as e:
            raise VectorStoreError(f"Failed to get by IDs from MongoDB: {str(e)}")
    
    def delete(self, ids: List[str]):
        """Delete vectors from MongoDB."""
        try:
            self.collection.delete_many({"id": {"$in": ids}})
        except Exception as e:
            raise VectorStoreError(f"Failed to delete from MongoDB: {str(e)}")
    
    def update(
        self,
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
        texts: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """Update vectors in MongoDB."""
        if not embeddings:
            raise VectorStoreError("MongoDB update requires embeddings")
        
        # MongoDB uses upsert
        self.add(ids, embeddings, texts or [""] * len(ids), metadatas)
    
    def count(self) -> int:
        """Get count of vectors in MongoDB."""
        try:
            return self.collection.count_documents({})
        except Exception as e:
            raise VectorStoreError(f"Failed to count MongoDB vectors: {str(e)}")
    
    def clear(self):
        """Clear all vectors from MongoDB collection."""
        try:
            self.collection.delete_many({})
        except Exception as e:
            raise VectorStoreError(f"Failed to clear MongoDB: {str(e)}")
    
    def health_check(self) -> HealthCheckResult:
        """Check MongoDB health."""
        try:
            start = time.time()
            self.client.admin.command('ping')
            latency_ms = (time.time() - start) * 1000
            
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                latency_ms=latency_ms,
                message=f"MongoDB collection '{self.collection_name}' is healthy",
                details={"connected": True}
            )
        except ConnectionFailure as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                latency_ms=0.0,
                message=f"MongoDB health check failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def get_stats(self) -> VectorStoreStats:
        """Get MongoDB statistics."""
        try:
            count = self.count()
            
            return VectorStoreStats(
                total_vectors=count,
                dimension=self._dimension,
                metadata={
                    "provider": self.provider_name,
                    "collection_name": self.collection_name,
                    "database": self.db.name
                }
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to get MongoDB stats: {str(e)}")
    
    def __del__(self):
        """Close MongoDB connection."""
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
        return "mongodb"
