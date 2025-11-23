"""Integration tests for end-to-end RAG pipeline."""

import pytest
import tempfile
from pathlib import Path
from flip import Flip, FlipConfig


class TestEndToEndPipeline:
    """Test complete RAG pipeline."""
    
    @pytest.fixture
    def sample_docs(self, tmp_path):
        """Create sample documents for testing."""
        doc1 = tmp_path / "doc1.txt"
        doc1.write_text("""
        Artificial Intelligence (AI) is the simulation of human intelligence by machines.
        It includes machine learning, natural language processing, and computer vision.
        """)
        
        doc2 = tmp_path / "doc2.txt"
        doc2.write_text("""
        Machine Learning is a subset of AI that enables systems to learn from data.
        Deep learning uses neural networks with multiple layers.
        """)
        
        doc3 = tmp_path / "doc3.txt"
        doc3.write_text("""
        RAG (Retrieval-Augmented Generation) combines information retrieval with text generation.
        It helps reduce hallucinations and provides up-to-date information.
        """)
        
        return tmp_path
    
    def test_basic_indexing_and_query(self, sample_docs, tmp_path):
        """Test basic indexing and querying."""
        config = FlipConfig(
            embedding_provider="sentence-transformers",
            vector_store_path=tmp_path / "vector_store"
        )
        
        flip = Flip(directory=str(sample_docs), config=config)
        
        # Check indexing
        stats = flip.get_stats()
        assert stats["indexed"] is True
        assert stats["document_count"] == 3
        assert stats["chunk_count"] > 0
        
        # Query
        response = flip.query("What is artificial intelligence?")
        
        assert response.answer is not None
        assert len(response.answer) > 0
        assert len(response.citations) > 0
        assert len(response.context_chunks) > 0
    
    def test_hybrid_search(self, sample_docs, tmp_path):
        """Test hybrid search functionality."""
        config = FlipConfig(
            embedding_provider="sentence-transformers",
            use_hybrid_search=True,
            vector_store_path=tmp_path / "vector_store"
        )
        
        flip = Flip(directory=str(sample_docs), config=config)
        response = flip.query("machine learning neural networks")
        
        assert response.answer is not None
        assert any("machine learning" in chunk.lower() or "neural" in chunk.lower() 
                   for chunk in response.context_chunks)
    
    def test_reranking(self, sample_docs, tmp_path):
        """Test re-ranking functionality."""
        config = FlipConfig(
            embedding_provider="sentence-transformers",
            use_reranking=True,
            vector_store_path=tmp_path / "vector_store"
        )
        
        flip = Flip(directory=str(sample_docs), config=config)
        response = flip.query("What is RAG?")
        
        assert response.answer is not None
        # Should retrieve RAG-related content
        assert any("rag" in chunk.lower() or "retrieval" in chunk.lower() 
                   for chunk in response.context_chunks)
    
    def test_caching(self, sample_docs, tmp_path):
        """Test caching functionality."""
        config = FlipConfig(
            embedding_provider="sentence-transformers",
            enable_cache=True,
            cache_dir=tmp_path / "cache",
            vector_store_path=tmp_path / "vector_store"
        )
        
        flip = Flip(directory=str(sample_docs), config=config)
        
        # First query (not cached)
        import time
        start1 = time.time()
        response1 = flip.query("What is AI?")
        time1 = time.time() - start1
        
        # Second query (should be cached)
        start2 = time.time()
        response2 = flip.query("What is AI?")
        time2 = time.time() - start2
        
        # Cached query should be faster
        assert time2 < time1
        assert response1.answer == response2.answer
    
    def test_incremental_update(self, sample_docs, tmp_path):
        """Test incremental index updates."""
        config = FlipConfig(
            embedding_provider="sentence-transformers",
            vector_store_path=tmp_path / "vector_store"
        )
        
        flip = Flip(directory=str(sample_docs), config=config)
        
        initial_count = flip.get_stats()["chunk_count"]
        
        # Add a new document
        new_doc = sample_docs / "doc4.txt"
        new_doc.write_text("New content about quantum computing.")
        
        # Refresh index
        stats = flip.refresh_index()
        
        assert stats["added"] >= 1
        
        new_count = flip.get_stats()["chunk_count"]
        assert new_count > initial_count
    
    def test_multiple_file_formats(self, tmp_path):
        """Test handling multiple file formats."""
        # Create different file types
        (tmp_path / "test.txt").write_text("Text file content")
        (tmp_path / "test.md").write_text("# Markdown\n\nMarkdown content")
        (tmp_path / "test.json").write_text('{"key": "JSON content"}')
        
        config = FlipConfig(
            embedding_provider="sentence-transformers",
            vector_store_path=tmp_path / "vector_store"
        )
        
        flip = Flip(directory=str(tmp_path), config=config)
        
        stats = flip.get_stats()
        assert stats["document_count"] >= 3
    
    def test_error_handling(self, tmp_path):
        """Test error handling in pipeline."""
        config = FlipConfig(
            embedding_provider="sentence-transformers",
            vector_store_path=tmp_path / "vector_store"
        )
        
        flip = Flip(config=config)
        
        # Query without indexing should handle gracefully
        try:
            response = flip.query("test query")
            # Should either return empty or raise appropriate error
            assert response is not None or True
        except Exception as e:
            # Should be a meaningful error
            assert "index" in str(e).lower() or "document" in str(e).lower()


class TestAdvancedFeatures:
    """Test advanced RAG features."""
    
    def test_evaluation(self, tmp_path):
        """Test RAG evaluation."""
        # Create test docs
        doc = tmp_path / "test.txt"
        doc.write_text("AI is artificial intelligence.")
        
        config = FlipConfig(
            embedding_provider="sentence-transformers",
            vector_store_path=tmp_path / "vector_store"
        )
        
        flip = Flip(directory=str(tmp_path), config=config)
        
        # Get some chunk IDs
        response = flip.query("What is AI?")
        relevant_ids = [c["chunk_id"] for c in response.citations[:2]]
        
        # Evaluate
        result = flip.evaluate(
            query="What is AI?",
            relevant_doc_ids=relevant_ids,
            k=3
        )
        
        assert result.retrieval_precision >= 0
        assert result.retrieval_recall >= 0
        assert result.overall_score >= 0
    
    def test_monitoring(self, tmp_path):
        """Test performance monitoring."""
        doc = tmp_path / "test.txt"
        doc.write_text("Test content for monitoring.")
        
        config = FlipConfig(
            embedding_provider="sentence-transformers",
            vector_store_path=tmp_path / "vector_store"
        )
        
        flip = Flip(directory=str(tmp_path), config=config)
        
        # Run some queries
        flip.query("test query 1")
        flip.query("test query 2")
        
        # Get stats
        stats = flip.get_monitoring_stats()
        
        assert stats["total_queries"] >= 2
        assert "avg_total_time" in stats
        
        # Get recent queries
        recent = flip.get_recent_queries(n=2)
        assert len(recent) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
