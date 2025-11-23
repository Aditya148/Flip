"""Re-ranking module for improving retrieval accuracy."""

from typing import List, Optional
from sentence_transformers import CrossEncoder

from flip.vector_store.base import SearchResult
from flip.core.exceptions import RetrievalError


class Reranker:
    """Re-rank search results using cross-encoder models."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize reranker.
        
        Args:
            model_name: Cross-encoder model name
        """
        self.model_name = model_name
        try:
            self.model = CrossEncoder(model_name)
        except Exception as e:
            raise RetrievalError(f"Failed to load reranker model: {str(e)}")
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Re-rank search results.
        
        Args:
            query: Query text
            results: List of search results to rerank
            top_k: Number of top results to return (None = all)
            
        Returns:
            Re-ranked list of SearchResult objects
        """
        if not results:
            return []
        
        # Prepare query-document pairs
        pairs = [[query, result.text] for result in results]
        
        # Get relevance scores from cross-encoder
        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            raise RetrievalError(f"Re-ranking failed: {str(e)}")
        
        # Update scores and sort
        for i, result in enumerate(results):
            result.score = float(scores[i])
        
        # Sort by new scores
        reranked = sorted(results, key=lambda x: x.score, reverse=True)
        
        # Return top-k if specified
        if top_k is not None:
            return reranked[:top_k]
        
        return reranked
