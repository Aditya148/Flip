"""Unit tests for Weaviate vector store."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import uuid

from flip.vector_store.weaviate import WeaviateVectorStore, WEAVIATE_AVAILABLE
from flip.vector_store.base import SearchResult, HealthStatus
from flip.core.exceptions import VectorStoreError


@pytest.mark.skipif(not WEAVIATE_AVAILABLE, reason="Weaviate not installed")
class TestWeaviateVectorStore:
    """Test Weaviate vector store."""
    
    @patch('flip.vector_store.weaviate.weaviate')
    def test_initialization(self, mock_weaviate):
        """Test Weaviate initialization."""
        mock_client = Mock()
        mock_weaviate.connect_to_local.return_value = mock_client
        
        mock_collections = Mock()
        mock_collections.exists.return_value = False
        mock_client.collections = mock_collections
        
        mock_collection = Mock()
        mock_collections.get.return_value = mock_collection
        
        store = WeaviateVectorStore(
            collection_name="test",
            url="http://localhost:8080",
            dimension=384
        )
        
        assert store.class_name == "Test"  # Capitalized
        assert store._dimension == 384
    
    @patch('flip.vector_store.weaviate.weaviate')
    def test_add_vectors(self, mock_weaviate):
        """Test adding vectors."""
        mock_client = Mock()
        mock_weaviate.connect_to_local.return_value = mock_client
        
        mock_collections = Mock()
        mock_collections.exists.return_value = True
        mock_client.collections = mock_collections
        
        mock_collection = Mock()
        mock_batch = Mock()
        mock_batch.__enter__ = Mock(return_value=mock_batch)
        mock_batch.__exit__ = Mock(return_value=False)
        mock_collection.batch.dynamic.return_value = mock_batch
        mock_collections.get.return_value = mock_collection
        
        store = WeaviateVectorStore(collection_name="test", dimension=2)
        
        ids = [str(uuid.uuid4()), str(uuid.uuid4())]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        texts = ["text1", "text2"]
        metadatas = [{"key": "val1"}, {"key": "val2"}]
        
        store.add(ids, embeddings, texts, metadatas)
        
        assert mock_batch.add_object.call_count == 2
    
    @patch('flip.vector_store.weaviate.weaviate')
    def test_search(self, mock_weaviate):
        """Test vector search."""
        mock_client = Mock()
        mock_weaviate.connect_to_local.return_value = mock_client
        
        mock_collections = Mock()
        mock_collections.exists.return_value = True
        mock_client.collections = mock_collections
        
        # Mock search results
        mock_obj1 = Mock()
        mock_obj1.uuid = uuid.uuid4()
        mock_obj1.properties = {"text": "result1", "metadata": '{"key": "val1"}'}
        mock_metadata1 = Mock()
        mock_metadata1.distance = 0.1
        mock_obj1.metadata = mock_metadata1
        
        mock_obj2 = Mock()
        mock_obj2.uuid = uuid.uuid4()
        mock_obj2.properties = {"text": "result2", "metadata": '{"key": "val2"}'}
        mock_metadata2 = Mock()
        mock_metadata2.distance = 0.2
        mock_obj2.metadata = mock_metadata2
        
        mock_results = Mock()
        mock_results.objects = [mock_obj1, mock_obj2]
        
        mock_collection = Mock()
        mock_query = Mock()
        mock_query.near_vector.return_value = mock_results
        mock_collection.query = mock_query
        mock_collections.get.return_value = mock_collection
        
        store = WeaviateVectorStore(collection_name="test")
        
        results = store.search([0.1, 0.2], top_k=2)
        
        assert len(results) == 2
        assert results[0].text == "result1"
    
    @patch('flip.vector_store.weaviate.weaviate')
    def test_get_by_ids(self, mock_weaviate):
        """Test retrieval by IDs."""
        mock_client = Mock()
        mock_weaviate.connect_to_local.return_value = mock_client
        
        mock_collections = Mock()
        mock_collections.exists.return_value = True
        mock_client.collections = mock_collections
        
        # Mock fetch result
        test_uuid = uuid.uuid4()
        mock_obj = Mock()
        mock_obj.uuid = test_uuid
        mock_obj.properties = {"text": "content1", "metadata": '{"key": "val1"}'}
        
        mock_collection = Mock()
        mock_query = Mock()
        mock_query.fetch_object_by_id.return_value = mock_obj
        mock_collection.query = mock_query
        mock_collections.get.return_value = mock_collection
        
        store = WeaviateVectorStore(collection_name="test")
        
        results = store.get_by_ids([str(test_uuid)])
        
        assert len(results) == 1
        assert results[0].text == "content1"
    
    @patch('flip.vector_store.weaviate.weaviate')
    def test_delete(self, mock_weaviate):
        """Test vector deletion."""
        mock_client = Mock()
        mock_weaviate.connect_to_local.return_value = mock_client
        
        mock_collections = Mock()
        mock_collections.exists.return_value = True
        mock_client.collections = mock_collections
        
        mock_collection = Mock()
        mock_data = Mock()
        mock_collection.data = mock_data
        mock_collections.get.return_value = mock_collection
        
        store = WeaviateVectorStore(collection_name="test")
        
        test_id = str(uuid.uuid4())
        store.delete([test_id])
        
        mock_data.delete_by_id.assert_called()
    
    @patch('flip.vector_store.weaviate.weaviate')
    def test_count(self, mock_weaviate):
        """Test vector count."""
        mock_client = Mock()
        mock_weaviate.connect_to_local.return_value = mock_client
        
        mock_collections = Mock()
        mock_collections.exists.return_value = True
        mock_client.collections = mock_collections
        
        mock_result = Mock()
        mock_result.total_count = 100
        
        mock_collection = Mock()
        mock_aggregate = Mock()
        mock_aggregate.over_all.return_value = mock_result
        mock_collection.aggregate = mock_aggregate
        mock_collections.get.return_value = mock_collection
        
        store = WeaviateVectorStore(collection_name="test")
        
        count = store.count()
        
        assert count == 100
    
    @patch('flip.vector_store.weaviate.weaviate')
    def test_health_check(self, mock_weaviate):
        """Test health check."""
        mock_client = Mock()
        mock_client.is_ready.return_value = True
        mock_weaviate.connect_to_local.return_value = mock_client
        
        mock_collections = Mock()
        mock_collections.exists.return_value = True
        mock_client.collections = mock_collections
        
        mock_collection = Mock()
        mock_collections.get.return_value = mock_collection
        
        store = WeaviateVectorStore(collection_name="test")
        
        result = store.health_check()
        
        assert result.status == HealthStatus.HEALTHY
        assert result.latency_ms >= 0
    
    @patch('flip.vector_store.weaviate.weaviate')
    def test_get_stats(self, mock_weaviate):
        """Test getting statistics."""
        mock_client = Mock()
        mock_weaviate.connect_to_local.return_value = mock_client
        
        mock_collections = Mock()
        mock_collections.exists.return_value = True
        mock_client.collections = mock_collections
        
        mock_result = Mock()
        mock_result.total_count = 75
        
        mock_collection = Mock()
        mock_aggregate = Mock()
        mock_aggregate.over_all.return_value = mock_result
        mock_collection.aggregate = mock_aggregate
        mock_collections.get.return_value = mock_collection
        
        store = WeaviateVectorStore(collection_name="test", dimension=512)
        
        stats = store.get_stats()
        
        assert stats.total_vectors == 75
        assert stats.dimension == 512
        assert stats.metadata["provider"] == "weaviate"
    
    @patch('flip.vector_store.weaviate.weaviate')
    def test_api_key_authentication(self, mock_weaviate):
        """Test authentication with API key."""
        mock_client = Mock()
        mock_weaviate.connect_to_custom.return_value = mock_client
        
        mock_collections = Mock()
        mock_collections.exists.return_value = False
        mock_client.collections = mock_collections
        
        mock_collection = Mock()
        mock_collections.get.return_value = mock_collection
        
        store = WeaviateVectorStore(
            collection_name="test",
            url="https://my-cluster.weaviate.network",
            api_key="test-key"
        )
        
        # Verify connect_to_custom was called
        mock_weaviate.connect_to_custom.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
