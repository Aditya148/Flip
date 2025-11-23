"""Main retrieval module."""

from typing import List, Dict, Any, Optional
from flip.vector_store.base import BaseVectorStore, SearchResult
from flip.embedding.base import BaseEmbedder


class Retriever:
    """Main retrieval component for vector search."""
    
    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedder: BaseEmbedder,
        top_k: int = 5
    ):
        """
        Initialize retriever.
        
        Args:
            vector_store: Vector store instance
            embedder: Embedder instance
            top_k: Number of results to retrieve
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query text
            top_k: Number of results (uses default if None)
            filter_dict: Optional metadata filter
            
        Returns:
            List of SearchResult objects
        """
        k = top_k or self.top_k
        
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)
        
        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=k,
            filter_dict=filter_dict
        )
        
        return results
