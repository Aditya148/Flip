"""Unit tests for Pinecone vector store."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from flip.vector_store.pinecone import PineconeVectorStore, PINECONE_AVAILABLE
from flip.vector_store.base import SearchResult, HealthStatus
from flip.core.exceptions import VectorStoreError


@pytest.mark.skipif(not PINECONE_AVAILABLE, reason="Pinecone not installed")
class TestPineconeVectorStore:
    """Test Pinecone vector store."""
    
    @patch('flip.vector_store.pinecone.Pinecone')
    def test_initialization(self, mock_pinecone):
        """Test Pinecone initialization."""
        mock_pc = Mock()
        mock_pinecone.return_value = mock_pc
        mock_pc.list_indexes.return_value = []
        mock_index = Mock()
        mock_pc.Index.return_value = mock_index
        
        store = PineconeVectorStore(
            collection_name="test",
            api_key="test-key",
            dimension=384
        )
        
        assert store.index_name == "test"
        assert store._dimension == 384
        mock_pc.create_index.assert_called_once()
    
    @patch('flip.vector_store.pinecone.Pinecone')
    def test_add_vectors(self, mock_pinecone):
        """Test adding vectors."""
        mock_pc = Mock()
        mock_pinecone.return_value = mock_pc
        mock_pc.list_indexes.return_value = [Mock(name="test")]
        mock_index = Mock()
        mock_pc.Index.return_value = mock_index
        
        store = PineconeVectorStore(
            collection_name="test",
            api_key="test-key"
        )
        
        ids = ["id1", "id2"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        texts = ["text1", "text2"]
        metadatas = [{"key": "val1"}, {"key": "val2"}]
        
        store.add(ids, embeddings, texts, metadatas)
        
        mock_index.upsert.assert_called_once()
        call_args = mock_index.upsert.call_args
        vectors = call_args.kwargs['vectors']
        assert len(vectors) == 2
        assert vectors[0]['id'] == 'id1'
        assert vectors[0]['metadata']['text'] == 'text1'
    
    @patch('flip.vector_store.pinecone.Pinecone')
    def test_search(self, mock_pinecone):
        """Test vector search."""
        mock_pc = Mock()
        mock_pinecone.return_value = mock_pc
        mock_pc.list_indexes.return_value = [Mock(name="test")]
        mock_index = Mock()
        mock_pc.Index.return_value = mock_index
        
        # Mock search results
        mock_match1 = Mock()
        mock_match1.id = "id1"
        mock_match1.score = 0.95
        mock_match1.metadata = {"text": "result1", "key": "val1"}
        
        mock_match2 = Mock()
        mock_match2.id = "id2"
        mock_match2.score = 0.85
        mock_match2.metadata = {"text": "result2", "key": "val2"}
        
        mock_results = Mock()
        mock_results.matches = [mock_match1, mock_match2]
        mock_index.query.return_value = mock_results
        
        store = PineconeVectorStore(
            collection_name="test",
            api_key="test-key"
        )
        
        results = store.search([0.1, 0.2], top_k=2)
        
        assert len(results) == 2
        assert results[0].id == "id1"
        assert results[0].score == 0.95
        assert results[0].text == "result1"
    
    @patch('flip.vector_store.pinecone.Pinecone')
    def test_filter_search(self, mock_pinecone):
        """Test search with filters."""
        mock_pc = Mock()
        mock_pinecone.return_value = mock_pc
        mock_pc.list_indexes.return_value = [Mock(name="test")]
        mock_index = Mock()
        mock_pc.Index.return_value = mock_index
        
        mock_results = Mock()
        mock_results.matches = []
        mock_index.query.return_value = mock_results
        
        store = PineconeVectorStore(
            collection_name="test",
            api_key="test-key"
        )
        
        filters = {"category": "tech"}
        store.filter_search([0.1, 0.2], filters, top_k=5)
        
        mock_index.query.assert_called_once()
        call_args = mock_index.query.call_args
        assert call_args.kwargs['filter'] == filters
    
    @patch('flip.vector_store.pinecone.Pinecone')
    def test_get_by_ids(self, mock_pinecone):
        """Test retrieval by IDs."""
        mock_pc = Mock()
        mock_pinecone.return_value = mock_pc
        mock_pc.list_indexes.return_value = [Mock(name="test")]
        mock_index = Mock()
        mock_pc.Index.return_value = mock_index
        
        # Mock fetch results
        mock_vector1 = Mock()
        mock_vector1.metadata = {"text": "content1", "key": "val1"}
        
        mock_fetch_result = Mock()
        mock_fetch_result.vectors = {"id1": mock_vector1}
        mock_index.fetch.return_value = mock_fetch_result
        
        store = PineconeVectorStore(
            collection_name="test",
            api_key="test-key"
        )
        
        results = store.get_by_ids(["id1"])
        
        assert len(results) == 1
        assert results[0].id == "id1"
        assert results[0].text == "content1"
    
    @patch('flip.vector_store.pinecone.Pinecone')
    def test_delete(self, mock_pinecone):
        """Test vector deletion."""
        mock_pc = Mock()
        mock_pinecone.return_value = mock_pc
        mock_pc.list_indexes.return_value = [Mock(name="test")]
        mock_index = Mock()
        mock_pc.Index.return_value = mock_index
        
        store = PineconeVectorStore(
            collection_name="test",
            api_key="test-key"
        )
        
        store.delete(["id1", "id2"])
        
        mock_index.delete.assert_called_once_with(
            ids=["id1", "id2"],
            namespace=""
        )
    
    @patch('flip.vector_store.pinecone.Pinecone')
    def test_count(self, mock_pinecone):
        """Test vector count."""
        mock_pc = Mock()
        mock_pinecone.return_value = mock_pc
        mock_pc.list_indexes.return_value = [Mock(name="test")]
        mock_index = Mock()
        mock_pc.Index.return_value = mock_index
        
        mock_stats = Mock()
        mock_stats.total_vector_count = 100
        mock_index.describe_index_stats.return_value = mock_stats
        
        store = PineconeVectorStore(
            collection_name="test",
            api_key="test-key"
        )
        
        count = store.count()
        
        assert count == 100
    
    @patch('flip.vector_store.pinecone.Pinecone')
    def test_clear(self, mock_pinecone):
        """Test clearing all vectors."""
        mock_pc = Mock()
        mock_pinecone.return_value = mock_pc
        mock_pc.list_indexes.return_value = [Mock(name="test")]
        mock_index = Mock()
        mock_pc.Index.return_value = mock_index
        
        store = PineconeVectorStore(
            collection_name="test",
            api_key="test-key"
        )
        
        store.clear()
        
        mock_index.delete.assert_called_once_with(
            delete_all=True,
            namespace=""
        )
    
    @patch('flip.vector_store.pinecone.Pinecone')
    def test_health_check(self, mock_pinecone):
        """Test health check."""
        mock_pc = Mock()
        mock_pinecone.return_value = mock_pc
        mock_pc.list_indexes.return_value = [Mock(name="test")]
        mock_index = Mock()
        mock_pc.Index.return_value = mock_index
        
        mock_stats = Mock()
        mock_stats.total_vector_count = 50
        mock_stats.dimension = 384
        mock_stats.index_fullness = 0.25
        mock_index.describe_index_stats.return_value = mock_stats
        
        store = PineconeVectorStore(
            collection_name="test",
            api_key="test-key"
        )
        
        result = store.health_check()
        
        assert result.status == HealthStatus.HEALTHY
        assert result.latency_ms >= 0
        assert result.details["total_vectors"] == 50
    
    @patch('flip.vector_store.pinecone.Pinecone')
    def test_get_stats(self, mock_pinecone):
        """Test getting statistics."""
        mock_pc = Mock()
        mock_pinecone.return_value = mock_pc
        mock_pc.list_indexes.return_value = [Mock(name="test")]
        mock_index = Mock()
        mock_pc.Index.return_value = mock_index
        
        mock_stats = Mock()
        mock_stats.total_vector_count = 75
        mock_stats.dimension = 512
        mock_stats.index_fullness = 0.5
        mock_stats.namespaces = {}
        mock_index.describe_index_stats.return_value = mock_stats
        
        store = PineconeVectorStore(
            collection_name="test",
            api_key="test-key",
            dimension=512
        )
        
        stats = store.get_stats()
        
        assert stats.total_vectors == 75
        assert stats.dimension == 512
        assert stats.metadata["provider"] == "pinecone"
    
    @patch('flip.vector_store.pinecone.Pinecone')
    def test_namespace_support(self, mock_pinecone):
        """Test namespace support."""
        mock_pc = Mock()
        mock_pinecone.return_value = mock_pc
        mock_pc.list_indexes.return_value = [Mock(name="test")]
        mock_index = Mock()
        mock_pc.Index.return_value = mock_index
        
        store = PineconeVectorStore(
            collection_name="test",
            api_key="test-key",
            namespace="tenant1"
        )
        
        assert store.namespace == "tenant1"
        
        # Test that namespace is used in operations
        store.add(["id1"], [[0.1, 0.2]], ["text1"])
        call_args = mock_index.upsert.call_args
        assert call_args.kwargs['namespace'] == "tenant1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
