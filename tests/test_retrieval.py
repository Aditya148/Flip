"""Unit tests for retrieval components."""

import pytest
import numpy as np
from flip.retrieval.hybrid import HybridRetriever
from flip.retrieval.reranker import CrossEncoderReranker


class TestHybridRetriever:
    """Test hybrid retrieval (vector + BM25)."""
    
    def test_initialization(self):
        """Test hybrid retriever initialization."""
        retriever = HybridRetriever()
        
        assert retriever is not None
    
    def test_bm25_search(self):
        """Test BM25 keyword search."""
        retriever = HybridRetriever()
        
        documents = [
            "The quick brown fox jumps over the lazy dog",
            "A fast brown fox leaps over a sleepy dog",
            "The weather is sunny today"
        ]
        
        retriever.index_documents(documents)
        
        results = retriever.bm25_search("quick fox", top_k=2)
        
        assert len(results) <= 2
        assert results[0]["text"] == documents[0]  # Should match best
    
    def test_reciprocal_rank_fusion(self):
        """Test RRF combination of results."""
        retriever = HybridRetriever()
        
        vector_results = [
            {"id": "1", "score": 0.9},
            {"id": "2", "score": 0.7},
            {"id": "3", "score": 0.5}
        ]
        
        bm25_results = [
            {"id": "2", "score": 10.0},
            {"id": "1", "score": 8.0},
            {"id": "4", "score": 6.0}
        ]
        
        fused = retriever.reciprocal_rank_fusion(
            vector_results,
            bm25_results,
            k=60
        )
        
        assert len(fused) > 0
        # ID "2" appears in both, should rank high
        assert any(r["id"] == "2" for r in fused[:2])


class TestCrossEncoderReranker:
    """Test cross-encoder re-ranking."""
    
    def test_initialization(self):
        """Test reranker initialization."""
        reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        
        assert reranker is not None
    
    def test_rerank(self):
        """Test re-ranking results."""
        reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        
        query = "What is artificial intelligence?"
        
        results = [
            {"text": "AI is machine intelligence", "score": 0.5},
            {"text": "The weather is nice", "score": 0.8},
            {"text": "Artificial intelligence simulates human cognition", "score": 0.6}
        ]
        
        reranked = reranker.rerank(query, results, top_k=3)
        
        assert len(reranked) == 3
        # Most relevant should be ranked higher
        assert "intelligence" in reranked[0]["text"].lower()
    
    def test_score_pairs(self):
        """Test scoring query-document pairs."""
        reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        
        query = "machine learning"
        documents = [
            "Machine learning is a subset of AI",
            "The sky is blue",
            "Deep learning uses neural networks"
        ]
        
        scores = reranker.score_pairs(query, documents)
        
        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)
        # First doc should score higher than second
        assert scores[0] > scores[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
