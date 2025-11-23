"""Unit tests for vector store components."""

import pytest
import tempfile
from pathlib import Path
from flip.vector_store.chroma import ChromaVectorStore


class TestChromaVectorStore:
    """Test ChromaDB vector store."""
    
    def test_initialization(self, tmp_path):
        """Test vector store initialization."""
        store = ChromaVectorStore(
            collection_name="test_collection",
            persist_directory=str(tmp_path)
        )
        
        assert store.collection_name == "test_collection"
        assert store.count() == 0
    
    def test_add_embeddings(self, tmp_path):
        """Test adding embeddings."""
        store = ChromaVectorStore(
            collection_name="test_add",
            persist_directory=str(tmp_path)
        )
        
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        texts = ["Text 1", "Text 2"]
        metadatas = [{"source": "test1"}, {"source": "test2"}]
        ids = ["id1", "id2"]
        
        store.add(
            embeddings=embeddings,
            texts=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        assert store.count() == 2
    
    def test_search(self, tmp_path):
        """Test similarity search."""
        store = ChromaVectorStore(
            collection_name="test_search",
            persist_directory=str(tmp_path)
        )
        
        # Add some embeddings
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
        texts = ["Text A", "Text B", "Text C"]
        metadatas = [{"id": i} for i in range(3)]
        ids = [f"id{i}" for i in range(3)]
        
        store.add(embeddings, texts, metadatas, ids)
        
        # Search for similar to first embedding
        results = store.search(
            query_embedding=[1.0, 0.0, 0.0],
            top_k=2
        )
        
        assert len(results) == 2
        assert results[0]["text"] == "Text A"  # Should be most similar
    
    def test_delete(self, tmp_path):
        """Test deleting embeddings."""
        store = ChromaVectorStore(
            collection_name="test_delete",
            persist_directory=str(tmp_path)
        )
        
        # Add embeddings
        store.add(
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            texts=["Text 1", "Text 2"],
            metadatas=[{}, {}],
            ids=["id1", "id2"]
        )
        
        assert store.count() == 2
        
        # Delete one
        store.delete(ids=["id1"])
        
        assert store.count() == 1
    
    def test_clear(self, tmp_path):
        """Test clearing all embeddings."""
        store = ChromaVectorStore(
            collection_name="test_clear",
            persist_directory=str(tmp_path)
        )
        
        # Add embeddings
        store.add(
            embeddings=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            texts=["A", "B", "C"],
            metadatas=[{}, {}, {}],
            ids=["1", "2", "3"]
        )
        
        assert store.count() == 3
        
        store.clear()
        
        assert store.count() == 0
    
    def test_persistence(self, tmp_path):
        """Test that data persists across instances."""
        collection_name = "test_persist"
        
        # Create and populate store
        store1 = ChromaVectorStore(
            collection_name=collection_name,
            persist_directory=str(tmp_path)
        )
        
        store1.add(
            embeddings=[[0.1, 0.2]],
            texts=["Persistent text"],
            metadatas=[{"key": "value"}],
            ids=["persist_id"]
        )
        
        count1 = store1.count()
        
        # Create new instance with same collection
        store2 = ChromaVectorStore(
            collection_name=collection_name,
            persist_directory=str(tmp_path)
        )
        
        count2 = store2.count()
        
        assert count1 == count2 == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
