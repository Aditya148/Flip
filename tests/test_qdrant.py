"""Unit tests for Qdrant vector store."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from flip.vector_store.qdrant import QdrantVectorStore, QDRANT_AVAILABLE
from flip.vector_store.base import SearchResult, HealthStatus
from flip.core.exceptions import VectorStoreError


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="Qdrant not installed")
class TestQdrantVectorStore:
    """Test Qdrant vector store."""
    
    @patch('flip.vector_store.qdrant.QdrantClient')
    def test_initialization(self, mock_client_class):
        """Test Qdrant initialization."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_collections = Mock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        
        store = QdrantVectorStore(
            collection_name="test",
            host="localhost",
            port=6333,
            dimension=384
        )
        
        assert store.collection_name == "test"
        assert store._dimension == 384
        mock_client.create_collection.assert_called_once()
    
    @patch('flip.vector_store.qdrant.QdrantClient')
    def test_add_vectors(self, mock_client_class):
        """Test adding vectors."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_collections = Mock()
        mock_collection = Mock()
        mock_collection.name = "test"
        mock_collections.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections
        
        store = QdrantVectorStore(
            collection_name="test",
            dimension=2
        )
        
        ids = ["id1", "id2"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        texts = ["text1", "text2"]
        metadatas = [{"key": "val1"}, {"key": "val2"}]
        
        store.add(ids, embeddings, texts, metadatas)
        
        mock_client.upsert.assert_called_once()
        call_args = mock_client.upsert.call_args
        points = call_args.kwargs['points']
        assert len(points) == 2
    
    @patch('flip.vector_store.qdrant.QdrantClient')
    def test_search(self, mock_client_class):
        """Test vector search."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_collections = Mock()
        mock_collection = Mock()
        mock_collection.name = "test"
        mock_collections.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections
        
        # Mock search results
        mock_result1 = Mock()
        mock_result1.id = "id1"
        mock_result1.score = 0.95
        mock_result1.payload = {"text": "result1", "key": "val1"}
        
        mock_result2 = Mock()
        mock_result2.id = "id2"
        mock_result2.score = 0.85
        mock_result2.payload = {"text": "result2", "key": "val2"}
        
        mock_client.search.return_value = [mock_result1, mock_result2]
        
        store = QdrantVectorStore(collection_name="test")
        
        results = store.search([0.1, 0.2], top_k=2)
        
        assert len(results) == 2
        assert results[0].id == "id1"
        assert results[0].score == 0.95
        assert results[0].text == "result1"
    
    @patch('flip.vector_store.qdrant.QdrantClient')
    def test_filter_search(self, mock_client_class):
        """Test search with filters."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_collections = Mock()
        mock_collection = Mock()
        mock_collection.name = "test"
        mock_collections.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections
        
        mock_client.search.return_value = []
        
        store = QdrantVectorStore(collection_name="test")
        
        filters = {"category": "tech"}
        store.filter_search([0.1, 0.2], filters, top_k=5)
        
        mock_client.search.assert_called_once()
        call_args = mock_client.search.call_args
        assert call_args.kwargs['query_filter'] is not None
    
    @patch('flip.vector_store.qdrant.QdrantClient')
    def test_get_by_ids(self, mock_client_class):
        """Test retrieval by IDs."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_collections = Mock()
        mock_collection = Mock()
        mock_collection.name = "test"
        mock_collections.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections
        
        # Mock retrieve results
        mock_result = Mock()
        mock_result.id = "id1"
        mock_result.payload = {"text": "content1", "key": "val1"}
        
        mock_client.retrieve.return_value = [mock_result]
        
        store = QdrantVectorStore(collection_name="test")
        
        results = store.get_by_ids(["id1"])
        
        assert len(results) == 1
        assert results[0].id == "id1"
        assert results[0].text == "content1"
    
    @patch('flip.vector_store.qdrant.QdrantClient')
    def test_delete(self, mock_client_class):
        """Test vector deletion."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_collections = Mock()
        mock_collection = Mock()
        mock_collection.name = "test"
        mock_collections.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections
        
        store = QdrantVectorStore(collection_name="test")
        
        store.delete(["id1", "id2"])
        
        mock_client.delete.assert_called_once()
    
    @patch('flip.vector_store.qdrant.QdrantClient')
    def test_count(self, mock_client_class):
        """Test vector count."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_collections = Mock()
        mock_collection = Mock()
        mock_collection.name = "test"
        mock_collections.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections
        
        mock_info = Mock()
        mock_info.points_count = 100
        mock_client.get_collection.return_value = mock_info
        
        store = QdrantVectorStore(collection_name="test")
        
        count = store.count()
        
        assert count == 100
    
    @patch('flip.vector_store.qdrant.QdrantClient')
    def test_clear(self, mock_client_class):
        """Test clearing all vectors."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_collections = Mock()
        mock_collection = Mock()
        mock_collection.name = "test"
        mock_collections.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections
        
        store = QdrantVectorStore(collection_name="test", dimension=384)
        
        store.clear()
        
        mock_client.delete_collection.assert_called_once()
        assert mock_client.create_collection.call_count == 2  # Once in init, once in clear
    
    @patch('flip.vector_store.qdrant.QdrantClient')
    def test_health_check(self, mock_client_class):
        """Test health check."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_collections = Mock()
        mock_collection = Mock()
        mock_collection.name = "test"
        mock_collections.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections
        
        mock_info = Mock()
        mock_info.points_count = 50
        mock_info.vectors_count = 50
        mock_info.status.value = "green"
        mock_client.get_collection.return_value = mock_info
        
        store = QdrantVectorStore(collection_name="test")
        
        result = store.health_check()
        
        assert result.status == HealthStatus.HEALTHY
        assert result.latency_ms >= 0
        assert result.details["points_count"] == 50
    
    @patch('flip.vector_store.qdrant.QdrantClient')
    def test_get_stats(self, mock_client_class):
        """Test getting statistics."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_collections = Mock()
        mock_collection = Mock()
        mock_collection.name = "test"
        mock_collections.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections
        
        mock_info = Mock()
        mock_info.points_count = 75
        mock_info.vectors_count = 75
        mock_info.indexed_vectors_count = 75
        mock_info.status.value = "green"
        mock_client.get_collection.return_value = mock_info
        
        store = QdrantVectorStore(collection_name="test", dimension=512)
        
        stats = store.get_stats()
        
        assert stats.total_vectors == 75
        assert stats.dimension == 512
        assert stats.metadata["provider"] == "qdrant"
    
    @patch('flip.vector_store.qdrant.QdrantClient')
    def test_create_snapshot(self, mock_client_class):
        """Test snapshot creation."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_collections = Mock()
        mock_collection = Mock()
        mock_collection.name = "test"
        mock_collections.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections
        
        mock_snapshot = Mock()
        mock_snapshot.name = "snapshot_123"
        mock_client.create_snapshot.return_value = mock_snapshot
        
        store = QdrantVectorStore(collection_name="test")
        
        snapshot_name = store.create_snapshot()
        
        assert snapshot_name == "snapshot_123"
        mock_client.create_snapshot.assert_called_once()
    
    @patch('flip.vector_store.qdrant.QdrantClient')
    def test_url_connection(self, mock_client_class):
        """Test connection with URL."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_collections = Mock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        
        store = QdrantVectorStore(
            collection_name="test",
            url="http://localhost:6333",
            api_key="test-key"
        )
        
        # Verify client was initialized with URL
        mock_client_class.assert_called()
        call_kwargs = mock_client_class.call_args.kwargs
        assert call_kwargs.get('url') == "http://localhost:6333"
        assert call_kwargs.get('api_key') == "test-key"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
