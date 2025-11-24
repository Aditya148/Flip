"""ChromaDB vector store implementation."""

from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings

from flip.vector_store.base import (
    BaseVectorStore, 
    SearchResult, 
    VectorStoreStats,
    HealthCheckResult,
    HealthStatus
)
from flip.core.exceptions import VectorStoreError


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB vector store implementation."""
    
    def __init__(
        self,
        collection_name: str,
        persist_directory: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize ChromaDB vector store.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist data (None for in-memory)
            **kwargs: Additional ChromaDB settings
        """
        super().__init__(collection_name, **kwargs)
        
        # Initialize ChromaDB client
        if persist_directory:
            persist_path = Path(persist_directory)
            persist_path.mkdir(parents=True, exist_ok=True)
            
            self.client = chromadb.PersistentClient(
                path=str(persist_path),
                settings=Settings(**kwargs)
            )
        else:
            self.client = chromadb.Client(settings=Settings(**kwargs))
        
        # Get or create collection
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to create ChromaDB collection: {str(e)}")
    
    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """Add vectors to ChromaDB."""
        try:
            # ChromaDB requires documents (texts) to be provided
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to add to ChromaDB: {str(e)}")
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search ChromaDB for similar vectors."""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_dict
            )
            
            # Convert to SearchResult objects
            search_results = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    search_results.append(SearchResult(
                        id=results['ids'][0][i],
                        text=results['documents'][0][i],
                        score=1.0 - results['distances'][0][i],  # Convert distance to similarity
                        metadata=results['metadatas'][0][i] if results['metadatas'] else {}
                    ))
            
            return search_results
            
        except Exception as e:
            raise VectorStoreError(f"Failed to search ChromaDB: {str(e)}")
    
    def get_by_ids(
        self,
        ids: List[str]
    ) -> List[SearchResult]:
        """Retrieve vectors by their IDs from ChromaDB."""
        try:
            results = self.collection.get(
                ids=ids,
                include=["documents", "metadatas", "embeddings"]
            )
            
            search_results = []
            if results['ids']:
                for i in range(len(results['ids'])):
                    search_results.append(SearchResult(
                        id=results['ids'][i],
                        text=results['documents'][i] if results['documents'] else "",
                        score=1.0,  # No score for direct retrieval
                        metadata=results['metadatas'][i] if results['metadatas'] else {}
                    ))
            
            return search_results
            
        except Exception as e:
            raise VectorStoreError(f"Failed to get by IDs from ChromaDB: {str(e)}")
    
    def delete(self, ids: List[str]):
        """Delete vectors from ChromaDB."""
        try:
            self.collection.delete(ids=ids)
        except Exception as e:
            raise VectorStoreError(f"Failed to delete from ChromaDB: {str(e)}")
    
    def update(
        self,
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
        texts: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """Update vectors in ChromaDB."""
        try:
            self.collection.update(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to update ChromaDB: {str(e)}")
    
    def count(self) -> int:
        """Get count of vectors in ChromaDB."""
        try:
            return self.collection.count()
        except Exception as e:
            raise VectorStoreError(f"Failed to count ChromaDB vectors: {str(e)}")
    
    def clear(self):
        """Clear all vectors from ChromaDB collection."""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to clear ChromaDB: {str(e)}")
    
    def get_stats(self) -> VectorStoreStats:
        """Get statistics about ChromaDB collection."""
        try:
            count = self.count()
            
            # Try to get a sample to determine dimension
            dimension = 0
            if count > 0:
                sample = self.collection.get(limit=1, include=["embeddings"])
                if sample['embeddings'] and len(sample['embeddings']) > 0:
                    dimension = len(sample['embeddings'][0])
            
            return VectorStoreStats(
                total_vectors=count,
                dimension=dimension,
                metadata={
                    "provider": self.provider_name,
                    "collection_name": self.collection_name,
                    "distance_metric": "cosine"
                }
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to get ChromaDB stats: {str(e)}")
    
    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "chroma"
