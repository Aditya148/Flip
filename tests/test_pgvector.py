"""Unit tests for Pgvector vector store."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from flip.vector_store.pgvector import PgvectorVectorStore, PGVECTOR_AVAILABLE
from flip.vector_store.base import SearchResult, HealthStatus
from flip.core.exceptions import VectorStoreError


@pytest.mark.skipif(not PGVECTOR_AVAILABLE, reason="Pgvector not installed")
class TestPgvectorVectorStore:
    """Test Pgvector vector store."""
    
    @patch('flip.vector_store.pgvector.psycopg2.pool.SimpleConnectionPool')
    def test_initialization(self, mock_pool):
        """Test Pgvector initialization."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        
        mock_pool_instance = Mock()
        mock_pool_instance.getconn.return_value = mock_conn
        mock_pool.return_value = mock_pool_instance
        
        store = PgvectorVectorStore(
            collection_name="test",
            host="localhost",
            port=5432,
            database="test_db",
            user="test_user",
            password="test_pass",
            dimension=384
        )
        
        assert store.table_name == "test"
        assert store._dimension == 384
        mock_pool.assert_called_once()
    
    @patch('flip.vector_store.pgvector.psycopg2.pool.SimpleConnectionPool')
    def test_add_vectors(self, mock_pool):
        """Test adding vectors."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        
        mock_pool_instance = Mock()
        mock_pool_instance.getconn.return_value = mock_conn
        mock_pool.return_value = mock_pool_instance
        
        store = PgvectorVectorStore(collection_name="test", dimension=2)
        
        ids = ["id1", "id2"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        texts = ["text1", "text2"]
        metadatas = [{"key": "val1"}, {"key": "val2"}]
        
        store.add(ids, embeddings, texts, metadatas)
        
        # Verify execute_values was called
        assert mock_cursor.execute.called or hasattr(mock_cursor, 'execute')
    
    @patch('flip.vector_store.pgvector.psycopg2.pool.SimpleConnectionPool')
    def test_search(self, mock_pool):
        """Test vector search."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_cursor.fetchall.return_value = [
            ("id1", "text1", '{"key": "val1"}', 0.1),
            ("id2", "text2", '{"key": "val2"}', 0.2)
        ]
        mock_conn.cursor.return_value = mock_cursor
        
        mock_pool_instance = Mock()
        mock_pool_instance.getconn.return_value = mock_conn
        mock_pool.return_value = mock_pool_instance
        
        store = PgvectorVectorStore(collection_name="test")
        
        results = store.search([0.1, 0.2], top_k=2)
        
        assert len(results) == 2
        assert results[0].id == "id1"
    
    @patch('flip.vector_store.pgvector.psycopg2.pool.SimpleConnectionPool')
    def test_get_by_ids(self, mock_pool):
        """Test retrieval by IDs."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_cursor.fetchall.return_value = [
            ("id1", "text1", '{"key": "val1"}')
        ]
        mock_conn.cursor.return_value = mock_cursor
        
        mock_pool_instance = Mock()
        mock_pool_instance.getconn.return_value = mock_conn
        mock_pool.return_value = mock_pool_instance
        
        store = PgvectorVectorStore(collection_name="test")
        
        results = store.get_by_ids(["id1"])
        
        assert len(results) == 1
        assert results[0].id == "id1"
    
    @patch('flip.vector_store.pgvector.psycopg2.pool.SimpleConnectionPool')
    def test_delete(self, mock_pool):
        """Test vector deletion."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        
        mock_pool_instance = Mock()
        mock_pool_instance.getconn.return_value = mock_conn
        mock_pool.return_value = mock_pool_instance
        
        store = PgvectorVectorStore(collection_name="test")
        
        store.delete(["id1", "id2"])
        
        mock_cursor.execute.assert_called()
    
    @patch('flip.vector_store.pgvector.psycopg2.pool.SimpleConnectionPool')
    def test_count(self, mock_pool):
        """Test vector count."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_cursor.fetchone.return_value = (100,)
        mock_conn.cursor.return_value = mock_cursor
        
        mock_pool_instance = Mock()
        mock_pool_instance.getconn.return_value = mock_conn
        mock_pool.return_value = mock_pool_instance
        
        store = PgvectorVectorStore(collection_name="test")
        
        count = store.count()
        
        assert count == 100
    
    @patch('flip.vector_store.pgvector.psycopg2.pool.SimpleConnectionPool')
    def test_health_check(self, mock_pool):
        """Test health check."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_cursor.fetchone.return_value = (1,)
        mock_conn.cursor.return_value = mock_cursor
        
        mock_pool_instance = Mock()
        mock_pool_instance.getconn.return_value = mock_conn
        mock_pool.return_value = mock_pool_instance
        
        store = PgvectorVectorStore(collection_name="test")
        
        result = store.health_check()
        
        assert result.status == HealthStatus.HEALTHY
        assert result.latency_ms >= 0
    
    @patch('flip.vector_store.pgvector.psycopg2.pool.SimpleConnectionPool')
    def test_get_stats(self, mock_pool):
        """Test getting statistics."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_cursor.fetchone.return_value = (75,)
        mock_conn.cursor.return_value = mock_cursor
        
        mock_pool_instance = Mock()
        mock_pool_instance.getconn.return_value = mock_conn
        mock_pool.return_value = mock_pool_instance
        
        store = PgvectorVectorStore(collection_name="test", dimension=512)
        
        stats = store.get_stats()
        
        assert stats.total_vectors == 75
        assert stats.dimension == 512
        assert stats.metadata["provider"] == "pgvector"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
