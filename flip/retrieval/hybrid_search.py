"""Hybrid search combining vector and keyword search."""

from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import numpy as np

from flip.vector_store.base import BaseVectorStore, SearchResult
from flip.embedding.base import BaseEmbedder


class HybridSearchRetriever:
    """Hybrid retriever combining dense (vector) and sparse (BM25) search."""
    
    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedder: BaseEmbedder,
        top_k: int = 5,
        alpha: float = 0.5
    ):
        """
        Initialize hybrid search retriever.
        
        Args:
            vector_store: Vector store instance
            embedder: Embedder instance
            top_k: Number of results to retrieve
            alpha: Weight for vector search (1-alpha for BM25). 0.5 = equal weight
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k
        self.alpha = alpha
        
        # BM25 index (will be built on first retrieval)
        self.bm25 = None
        self.documents = []
        self.doc_ids = []
        self._index_built = False
    
    def _build_bm25_index(self, documents: List[SearchResult]):
        """Build BM25 index from documents."""
        if self._index_built:
            return
        
        # Tokenize documents
        tokenized_docs = [doc.text.lower().split() for doc in documents]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_docs)
        self.documents = documents
        self.doc_ids = [doc.id for doc in documents]
        self._index_built = True
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        alpha: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Retrieve using hybrid search (vector + BM25).
        
        Args:
            query: Query text
            top_k: Number of results (uses default if None)
            filter_dict: Optional metadata filter
            alpha: Weight for vector search (overrides default if provided)
            
        Returns:
            List of SearchResult objects ranked by hybrid score
        """
        k = top_k or self.top_k
        weight = alpha if alpha is not None else self.alpha
        
        # 1. Vector search
        query_embedding = self.embedder.embed_text(query)
        vector_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=k * 2,  # Get more for fusion
            filter_dict=filter_dict
        )
        
        if not vector_results:
            return []
        
        # 2. Build BM25 index if needed
        if not self._index_built:
            # Get all documents for BM25 indexing
            all_results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=1000,  # Get many documents for BM25
                filter_dict=filter_dict
            )
            self._build_bm25_index(all_results)
        
        # 3. BM25 search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # 4. Reciprocal Rank Fusion (RRF)
        # Create score dictionaries
        vector_scores = {result.id: result.score for result in vector_results}
        
        # Normalize BM25 scores
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        bm25_dict = {
            self.doc_ids[i]: bm25_scores[i] / max_bm25
            for i in range(len(self.doc_ids))
        }
        
        # Combine scores
        all_ids = set(vector_scores.keys()) | set(bm25_dict.keys())
        hybrid_scores = {}
        
        for doc_id in all_ids:
            vector_score = vector_scores.get(doc_id, 0.0)
            bm25_score = bm25_dict.get(doc_id, 0.0)
            
            # Weighted combination
            hybrid_scores[doc_id] = (weight * vector_score) + ((1 - weight) * bm25_score)
        
        # 5. Sort by hybrid score and get top-k
        sorted_ids = sorted(hybrid_scores.keys(), key=lambda x: hybrid_scores[x], reverse=True)
        top_ids = sorted_ids[:k]
        
        # 6. Create results with hybrid scores
        results = []
        id_to_result = {r.id: r for r in vector_results}
        id_to_doc = {d.id: d for d in self.documents}
        
        for doc_id in top_ids:
            if doc_id in id_to_result:
                result = id_to_result[doc_id]
            elif doc_id in id_to_doc:
                result = id_to_doc[doc_id]
            else:
                continue
            
            # Update score to hybrid score
            result.score = hybrid_scores[doc_id]
            results.append(result)
        
        return results
    
    def reset_index(self):
        """Reset BM25 index (call when documents change)."""
        self.bm25 = None
        self.documents = []
        self.doc_ids = []
        self._index_built = False
