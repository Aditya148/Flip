"""Unit tests for Milvus vector store."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from flip.vector_store.milvus import MilvusVectorStore, MILVUS_AVAILABLE
from flip.vector_store.base import SearchResult, HealthStatus
from flip.core.exceptions import VectorStoreError


@pytest.mark.skipif(not MILVUS_AVAILABLE, reason="Milvus not installed")
class TestMilvusVectorStore:
    """Test Milvus vector store."""
    
    @patch('flip.vector_store.milvus.connections')
    @patch('flip.vector_store.milvus.utility')
    @patch('flip.vector_store.milvus.Collection')
    def test_initialization(self, mock_collection_class, mock_utility, mock_connections):
        """Test Milvus initialization."""
        mock_utility.has_collection.return_value = False
        mock_collection = Mock()
        mock_collection.has_index.return_value = False
        mock_collection_class.return_value = mock_collection
        
        store = MilvusVectorStore(
            collection_name="test",
            host="localhost",
            port=19530,
            dimension=384
        )
        
        assert store.collection_name == "test"
        assert store._dimension == 384
        mock_connections.connect.assert_called_once()
    
    @patch('flip.vector_store.milvus.connections')
    @patch('flip.vector_store.milvus.utility')
    @patch('flip.vector_store.milvus.Collection')
    def test_add_vectors(self, mock_collection_class, mock_utility, mock_connections):
        """Test adding vectors."""
        mock_utility.has_collection.return_value = True
        mock_collection = Mock()
        mock_collection.has_index.return_value = True
        mock_collection_class.return_value = mock_collection
        
        store = MilvusVectorStore(collection_name="test", dimension=2)
        
        ids = ["id1", "id2"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        texts = ["text1", "text2"]
        metadatas = [{"key": "val1"}, {"key": "val2"}]
        
        store.add(ids, embeddings, texts, metadatas)
        
        mock_collection.insert.assert_called_once()
        mock_collection.flush.assert_called()
    
    @patch('flip.vector_store.milvus.connections')
    @patch('flip.vector_store.milvus.utility')
    @patch('flip.vector_store.milvus.Collection')
    def test_search(self, mock_collection_class, mock_utility, mock_connections):
        """Test vector search."""
        mock_utility.has_collection.return_value = True
        
        # Mock search results
        mock_hit1 = Mock()
        mock_hit1.id = "id1"
        mock_hit1.distance = 0.1
        mock_hit1.entity = {"text": "result1", "metadata": '{"key": "val1"}'}
        
        mock_hit2 = Mock()
        mock_hit2.id = "id2"
        mock_hit2.distance = 0.2
        mock_hit2.entity = {"text": "result2", "metadata": '{"key": "val2"}'}
        
        mock_collection = Mock()
        mock_collection.has_index.return_value = True
        mock_collection.search.return_value = [[mock_hit1, mock_hit2]]
        mock_collection_class.return_value = mock_collection
        
        store = MilvusVectorStore(collection_name="test")
        
        results = store.search([0.1, 0.2], top_k=2)
        
        assert len(results) == 2
        assert results[0].text == "result1"
    
    @patch('flip.vector_store.milvus.connections')
    @patch('flip.vector_store.milvus.utility')
    @patch('flip.vector_store.milvus.Collection')
    def test_get_by_ids(self, mock_collection_class, mock_utility, mock_connections):
        """Test retrieval by IDs."""
        mock_utility.has_collection.return_value = True
        
        mock_collection = Mock()
        mock_collection.has_index.return_value = True
        mock_collection.query.return_value = [
            {"id": "id1", "text": "content1", "metadata": '{"key": "val1"}'}
        ]
        mock_collection_class.return_value = mock_collection
        
        store = MilvusVectorStore(collection_name="test")
        
        results = store.get_by_ids(["id1"])
        
        assert len(results) == 1
        assert results[0].text == "content1"
    
    @patch('flip.vector_store.milvus.connections')
    @patch('flip.vector_store.milvus.utility')
    @patch('flip.vector_store.milvus.Collection')
    def test_delete(self, mock_collection_class, mock_utility, mock_connections):
        """Test vector deletion."""
        mock_utility.has_collection.return_value = True
        
        mock_collection = Mock()
        mock_collection.has_index.return_value = True
        mock_collection_class.return_value = mock_collection
        
        store = MilvusVectorStore(collection_name="test")
        
        store.delete(["id1", "id2"])
        
        mock_collection.delete.assert_called_once()
        mock_collection.flush.assert_called()
    
    @patch('flip.vector_store.milvus.connections')
    @patch('flip.vector_store.milvus.utility')
    @patch('flip.vector_store.milvus.Collection')
    def test_count(self, mock_collection_class, mock_utility, mock_connections):
        """Test vector count."""
        mock_utility.has_collection.return_value = True
        
        mock_collection = Mock()
        mock_collection.has_index.return_value = True
        mock_collection.num_entities = 100
        mock_collection_class.return_value = mock_collection
        
        store = MilvusVectorStore(collection_name="test")
        
        count = store.count()
        
        assert count == 100
    
    @patch('flip.vector_store.milvus.connections')
    @patch('flip.vector_store.milvus.utility')
    @patch('flip.vector_store.milvus.Collection')
    def test_health_check(self, mock_collection_class, mock_utility, mock_connections):
        """Test health check."""
        mock_utility.has_collection.return_value = True
        mock_utility.load_state.return_value = True
        
        mock_collection = Mock()
        mock_collection.has_index.return_value = True
        mock_collection_class.return_value = mock_collection
        
        store = MilvusVectorStore(collection_name="test")
        
        result = store.health_check()
        
        assert result.status == HealthStatus.HEALTHY
        assert result.latency_ms >= 0
    
    @patch('flip.vector_store.milvus.connections')
    @patch('flip.vector_store.milvus.utility')
    @patch('flip.vector_store.milvus.Collection')
    def test_get_stats(self, mock_collection_class, mock_utility, mock_connections):
        """Test getting statistics."""
        mock_utility.has_collection.return_value = True
        
        mock_collection = Mock()
        mock_collection.has_index.return_value = True
        mock_collection.num_entities = 75
        mock_collection_class.return_value = mock_collection
        
        store = MilvusVectorStore(collection_name="test", dimension=512)
        
        stats = store.get_stats()
        
        assert stats.total_vectors == 75
        assert stats.dimension == 512
        assert stats.metadata["provider"] == "milvus"
    
    @patch('flip.vector_store.milvus.connections')
    @patch('flip.vector_store.milvus.utility')
    @patch('flip.vector_store.milvus.Collection')
    def test_index_types(self, mock_collection_class, mock_utility, mock_connections):
        """Test different index types."""
        mock_utility.has_collection.return_value = False
        
        mock_collection = Mock()
        mock_collection.has_index.return_value = False
        mock_collection_class.return_value = mock_collection
        
        # Test HNSW
        store_hnsw = MilvusVectorStore(
            collection_name="test_hnsw",
            index_type="HNSW"
        )
        assert store_hnsw.index_type == "HNSW"
        
        # Test IVF_FLAT
        store_ivf = MilvusVectorStore(
            collection_name="test_ivf",
            index_type="IVF_FLAT"
        )
        assert store_ivf.index_type == "IVF_FLAT"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
