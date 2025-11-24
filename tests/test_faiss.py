"""Unit tests for FAISS vector store."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import tempfile
import os

from flip.vector_store.faiss import FAISSVectorStore, FAISS_AVAILABLE
from flip.vector_store.base import SearchResult, HealthStatus
from flip.core.exceptions import VectorStoreError


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not installed")
class TestFAISSVectorStore:
    """Test FAISS vector store."""
    
    def test_initialization(self):
        """Test FAISS initialization."""
        store = FAISSVectorStore(
            collection_name="test",
            dimension=384
        )
        
        assert store.collection_name == "test"
        assert store._dimension == 384
        assert store.index is not None
    
    def test_add_vectors(self):
        """Test adding vectors."""
        store = FAISSVectorStore(collection_name="test", dimension=2)
        
        ids = ["id1", "id2"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        texts = ["text1", "text2"]
        metadatas = [{"key": "val1"}, {"key": "val2"}]
        
        store.add(ids, embeddings, texts, metadatas)
        
        assert store.count() == 2
        assert "id1" in store.texts
        assert "id2" in store.texts
    
    def test_search(self):
        """Test vector search."""
        store = FAISSVectorStore(collection_name="test", dimension=2)
        
        # Add vectors
        ids = ["id1", "id2", "id3"]
        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        texts = ["text1", "text2", "text3"]
        
        store.add(ids, embeddings, texts)
        
        # Search
        results = store.search([0.1, 0.2], top_k=2)
        
        assert len(results) <= 2
        assert results[0].id == "id1"  # Closest match
    
    def test_filter_search(self):
        """Test search with filters."""
        store = FAISSVectorStore(collection_name="test", dimension=2)
        
        # Add vectors with metadata
        ids = ["id1", "id2", "id3"]
        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        texts = ["text1", "text2", "text3"]
        metadatas = [
            {"category": "A"},
            {"category": "B"},
            {"category": "A"}
        ]
        
        store.add(ids, embeddings, texts, metadatas)
        
        # Search with filter
        results = store.filter_search(
            [0.1, 0.2],
            filters={"category": "A"},
            top_k=5
        )
        
        assert len(results) == 2
        for result in results:
            assert result.metadata["category"] == "A"
    
    def test_get_by_ids(self):
        """Test retrieval by IDs."""
        store = FAISSVectorStore(collection_name="test", dimension=2)
        
        ids = ["id1", "id2"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        texts = ["text1", "text2"]
        
        store.add(ids, embeddings, texts)
        
        results = store.get_by_ids(["id1"])
        
        assert len(results) == 1
        assert results[0].id == "id1"
        assert results[0].text == "text1"
    
    def test_delete(self):
        """Test vector deletion."""
        store = FAISSVectorStore(collection_name="test", dimension=2)
        
        ids = ["id1", "id2"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        texts = ["text1", "text2"]
        
        store.add(ids, embeddings, texts)
        assert store.count() == 2
        
        store.delete(["id1"])
        assert store.count() == 1
        assert "id1" not in store.texts
    
    def test_count(self):
        """Test vector count."""
        store = FAISSVectorStore(collection_name="test", dimension=2)
        
        assert store.count() == 0
        
        store.add(["id1"], [[0.1, 0.2]], ["text1"])
        assert store.count() == 1
        
        store.add(["id2"], [[0.3, 0.4]], ["text2"])
        assert store.count() == 2
    
    def test_clear(self):
        """Test clearing all vectors."""
        store = FAISSVectorStore(collection_name="test", dimension=2)
        
        store.add(["id1", "id2"], [[0.1, 0.2], [0.3, 0.4]], ["text1", "text2"])
        assert store.count() == 2
        
        store.clear()
        assert store.count() == 0
    
    def test_health_check(self):
        """Test health check."""
        store = FAISSVectorStore(collection_name="test", dimension=2)
        
        result = store.health_check()
        
        assert result.status == HealthStatus.HEALTHY
        assert result.latency_ms >= 0
    
    def test_get_stats(self):
        """Test getting statistics."""
        store = FAISSVectorStore(collection_name="test", dimension=512)
        
        store.add(["id1"], [[0.1] * 512], ["text1"])
        
        stats = store.get_stats()
        
        assert stats.total_vectors == 1
        assert stats.dimension == 512
        assert stats.metadata["provider"] == "faiss"
    
    def test_persistence(self):
        """Test saving and loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and populate store
            store1 = FAISSVectorStore(
                collection_name="test",
                dimension=2,
                persist_directory=tmpdir
            )
            
            ids = ["id1", "id2"]
            embeddings = [[0.1, 0.2], [0.3, 0.4]]
            texts = ["text1", "text2"]
            
            store1.add(ids, embeddings, texts)
            assert store1.count() == 2
            
            # Create new store with same directory
            store2 = FAISSVectorStore(
                collection_name="test",
                dimension=2,
                persist_directory=tmpdir
            )
            
            # Should load existing data
            assert store2.count() == 2
            assert "id1" in store2.texts
    
    def test_index_types(self):
        """Test different index types."""
        # Flat index
        store_flat = FAISSVectorStore(
            collection_name="test_flat",
            dimension=2,
            index_type="Flat"
        )
        assert store_flat.index_type == "Flat"
        
        # HNSW index
        store_hnsw = FAISSVectorStore(
            collection_name="test_hnsw",
            dimension=2,
            index_type="HNSW"
        )
        assert store_hnsw.index_type == "HNSW"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
