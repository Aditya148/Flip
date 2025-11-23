"""Base interface for vector stores."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


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


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""
    
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
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass
