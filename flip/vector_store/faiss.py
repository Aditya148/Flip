"""FAISS vector store implementation."""

from typing import List, Dict, Any, Optional
import time
import pickle
import os
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from flip.vector_store.base import (
    BaseVectorStore,
    SearchResult,
    VectorStoreStats,
    HealthCheckResult,
    HealthStatus
)
from flip.core.exceptions import VectorStoreError


class FAISSVectorStore(BaseVectorStore):
    """FAISS vector store implementation."""
    
    def __init__(
        self,
        collection_name: str,
        dimension: int = 1536,
        index_type: str = "Flat",
        metric: str = "L2",
        persist_directory: Optional[str] = None,
        use_gpu: bool = False,
        **kwargs
    ):
        """
        Initialize FAISS vector store.
        
        Args:
            collection_name: Name of the collection
            dimension: Vector dimension
            index_type: Index type (Flat, IVF, HNSW)
            metric: Distance metric (L2, IP for inner product)
            persist_directory: Directory to persist index
            use_gpu: Whether to use GPU (if available)
            **kwargs: Additional FAISS settings
        """
        if not FAISS_AVAILABLE:
            raise VectorStoreError(
                "FAISS is not installed. Install with: pip install faiss-cpu or faiss-gpu"
            )
        
        super().__init__(collection_name, **kwargs)
        
        self._dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.persist_directory = persist_directory
        self.use_gpu = use_gpu and self._gpu_available()
        
        # Storage for metadata and texts
        self.id_to_index = {}  # Map IDs to FAISS indices
        self.index_to_id = {}  # Map FAISS indices to IDs
        self.texts = {}  # Store texts
        self.metadatas = {}  # Store metadata
        self.next_index = 0
        
        # Create FAISS index
        self.index = self._create_index()
        
        # Load existing data if available
        if persist_directory:
            self._load()
    
    def _gpu_available(self) -> bool:
        """Check if GPU is available for FAISS."""
        try:
            return faiss.get_num_gpus() > 0
        except:
            return False
    
    def _create_index(self):
        """Create FAISS index based on type."""
        if self.metric == "L2":
            metric_type = faiss.METRIC_L2
        elif self.metric == "IP":
            metric_type = faiss.METRIC_INNER_PRODUCT
        else:
            metric_type = faiss.METRIC_L2
        
        if self.index_type == "Flat":
            index = faiss.IndexFlatL2(self._dimension) if metric_type == faiss.METRIC_L2 else faiss.IndexFlatIP(self._dimension)
        elif self.index_type == "IVF":
            quantizer = faiss.IndexFlatL2(self._dimension)
            index = faiss.IndexIVFFlat(quantizer, self._dimension, 100, metric_type)
            # Need to train IVF index
            index.nprobe = 10
        elif self.index_type == "HNSW":
            index = faiss.IndexHNSWFlat(self._dimension, 32, metric_type)
        else:
            index = faiss.IndexFlatL2(self._dimension)
        
        # Move to GPU if requested and available
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            except:
                pass
        
        return index
    
    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """Add vectors to FAISS."""
        try:
            vectors = np.array(embeddings, dtype=np.float32)
            
            # Train index if needed (for IVF)
            if self.index_type == "IVF" and not self.index.is_trained:
                self.index.train(vectors)
            
            # Add vectors
            start_idx = self.next_index
            self.index.add(vectors)
            
            # Store metadata
            for i, id in enumerate(ids):
                idx = start_idx + i
                self.id_to_index[id] = idx
                self.index_to_id[idx] = id
                self.texts[id] = texts[i]
                if metadatas:
                    self.metadatas[id] = metadatas[i]
            
            self.next_index += len(ids)
            
            # Persist if directory is set
            if self.persist_directory:
                self._save()
            
        except Exception as e:
            raise VectorStoreError(f"Failed to add to FAISS: {str(e)}")
    
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
        """Search FAISS for similar vectors."""
        try:
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # Search
            distances, indices = self.index.search(query_vector, top_k)
            
            # Convert to SearchResult objects
            search_results = []
            for i, idx in enumerate(indices[0]):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue
                
                id = self.index_to_id.get(idx)
                if not id:
                    continue
                
                # Apply filter if provided
                if filter_dict:
                    metadata = self.metadatas.get(id, {})
                    if not self._matches_filter(metadata, filter_dict):
                        continue
                
                # Convert distance to similarity score
                distance = distances[0][i]
                score = 1.0 / (1.0 + distance) if self.metric == "L2" else distance
                
                search_results.append(SearchResult(
                    id=id,
                    text=self.texts.get(id, ""),
                    score=float(score),
                    metadata=self.metadatas.get(id, {})
                ))
            
            return search_results
            
        except Exception as e:
            raise VectorStoreError(f"Failed to search FAISS: {str(e)}")
    
    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if metadata matches filter."""
        for key, value in filter_dict.items():
            if metadata.get(key) != value:
                return False
        return True
    
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
                if id in self.texts:
                    search_results.append(SearchResult(
                        id=id,
                        text=self.texts[id],
                        score=1.0,
                        metadata=self.metadatas.get(id, {})
                    ))
            
            return search_results
            
        except Exception as e:
            raise VectorStoreError(f"Failed to get by IDs from FAISS: {str(e)}")
    
    def delete(self, ids: List[str]):
        """Delete vectors from FAISS."""
        try:
            # FAISS doesn't support deletion, so we mark as deleted
            for id in ids:
                if id in self.id_to_index:
                    idx = self.id_to_index[id]
                    del self.id_to_index[id]
                    del self.index_to_id[idx]
                    del self.texts[id]
                    if id in self.metadatas:
                        del self.metadatas[id]
            
            # Persist if directory is set
            if self.persist_directory:
                self._save()
            
        except Exception as e:
            raise VectorStoreError(f"Failed to delete from FAISS: {str(e)}")
    
    def update(
        self,
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
        texts: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """Update vectors in FAISS (delete and re-add)."""
        if not embeddings:
            raise VectorStoreError("FAISS update requires embeddings")
        
        # FAISS doesn't support direct update, delete and re-add
        self.delete(ids)
        self.add(ids, embeddings, texts or [""] * len(ids), metadatas)
    
    def count(self) -> int:
        """Get count of vectors in FAISS."""
        try:
            return len(self.id_to_index)
        except Exception as e:
            raise VectorStoreError(f"Failed to count FAISS vectors: {str(e)}")
    
    def clear(self):
        """Clear all vectors from FAISS."""
        try:
            self.index = self._create_index()
            self.id_to_index = {}
            self.index_to_id = {}
            self.texts = {}
            self.metadatas = {}
            self.next_index = 0
            
            # Clear persisted data
            if self.persist_directory:
                self._save()
            
        except Exception as e:
            raise VectorStoreError(f"Failed to clear FAISS: {str(e)}")
    
    def health_check(self) -> HealthCheckResult:
        """Check FAISS health."""
        try:
            start = time.time()
            count = self.count()
            latency_ms = (time.time() - start) * 1000
            
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                latency_ms=latency_ms,
                message=f"FAISS index '{self.collection_name}' is healthy",
                details={
                    "total_vectors": count,
                    "is_trained": self.index.is_trained if hasattr(self.index, 'is_trained') else True
                }
            )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                latency_ms=0.0,
                message=f"FAISS health check failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def get_stats(self) -> VectorStoreStats:
        """Get FAISS statistics."""
        try:
            count = self.count()
            
            return VectorStoreStats(
                total_vectors=count,
                dimension=self._dimension,
                metadata={
                    "provider": self.provider_name,
                    "collection_name": self.collection_name,
                    "index_type": self.index_type,
                    "metric": self.metric,
                    "use_gpu": self.use_gpu,
                    "persisted": self.persist_directory is not None
                }
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to get FAISS stats: {str(e)}")
    
    def _save(self):
        """Save FAISS index and metadata to disk."""
        if not self.persist_directory:
            return
        
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Save FAISS index
            index_path = os.path.join(self.persist_directory, f"{self.collection_name}.index")
            
            # Move to CPU before saving if on GPU
            if self.use_gpu:
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, index_path)
            else:
                faiss.write_index(self.index, index_path)
            
            # Save metadata
            metadata_path = os.path.join(self.persist_directory, f"{self.collection_name}.pkl")
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'id_to_index': self.id_to_index,
                    'index_to_id': self.index_to_id,
                    'texts': self.texts,
                    'metadatas': self.metadatas,
                    'next_index': self.next_index
                }, f)
            
        except Exception as e:
            raise VectorStoreError(f"Failed to save FAISS index: {str(e)}")
    
    def _load(self):
        """Load FAISS index and metadata from disk."""
        if not self.persist_directory:
            return
        
        try:
            index_path = os.path.join(self.persist_directory, f"{self.collection_name}.index")
            metadata_path = os.path.join(self.persist_directory, f"{self.collection_name}.pkl")
            
            if os.path.exists(index_path) and os.path.exists(metadata_path):
                # Load FAISS index
                self.index = faiss.read_index(index_path)
                
                # Move to GPU if requested
                if self.use_gpu:
                    try:
                        res = faiss.StandardGpuResources()
                        self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                    except:
                        pass
                
                # Load metadata
                with open(metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.id_to_index = data['id_to_index']
                    self.index_to_id = data['index_to_id']
                    self.texts = data['texts']
                    self.metadatas = data['metadatas']
                    self.next_index = data['next_index']
            
        except Exception as e:
            # If loading fails, start fresh
            pass
    
    @property
    def dimension(self) -> int:
        """Get vector dimension."""
        return self._dimension
    
    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "faiss"
