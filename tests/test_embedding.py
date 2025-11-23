"""Unit tests for embedding components."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from flip.embedding.sentence_transformers import SentenceTransformerEmbedder


class TestSentenceTransformerEmbedder:
    """Test Sentence Transformer embedder."""
    
    def test_initialization(self):
        """Test embedder initialization."""
        embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
        
        assert embedder.model_name == "all-MiniLM-L6-v2"
        assert embedder.dimension > 0
    
    def test_embed_single_text(self):
        """Test embedding a single text."""
        embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
        
        text = "This is a test sentence."
        embedding = embedder.embed(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) == embedder.dimension
        assert all(isinstance(x, float) for x in embedding)
    
    def test_embed_batch(self):
        """Test batch embedding."""
        embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
        
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        embeddings = embedder.embed_batch(texts)
        
        assert len(embeddings) == 3
        assert all(len(emb) == embedder.dimension for emb in embeddings)
    
    def test_embedding_similarity(self):
        """Test that similar texts have similar embeddings."""
        embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
        
        text1 = "The cat sits on the mat."
        text2 = "A cat is sitting on a mat."
        text3 = "The weather is sunny today."
        
        emb1 = np.array(embedder.embed(text1))
        emb2 = np.array(embedder.embed(text2))
        emb3 = np.array(embedder.embed(text3))
        
        # Cosine similarity
        sim_12 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        sim_13 = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))
        
        # Similar sentences should be more similar
        assert sim_12 > sim_13
    
    def test_empty_text(self):
        """Test handling of empty text."""
        embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
        
        embedding = embedder.embed("")
        
        assert isinstance(embedding, list)
        assert len(embedding) == embedder.dimension


class TestEmbeddingFactory:
    """Test embedding factory."""
    
    def test_create_sentence_transformer(self):
        """Test creating sentence transformer embedder."""
        from flip.embedding.factory import EmbedderFactory
        
        embedder = EmbedderFactory.create(
            provider="sentence-transformers",
            model="all-MiniLM-L6-v2"
        )
        
        assert embedder is not None
        assert embedder.provider_name == "sentence-transformers"
    
    def test_unsupported_provider(self):
        """Test error handling for unsupported provider."""
        from flip.embedding.factory import EmbedderFactory
        from flip.core.exceptions import ConfigurationError
        
        with pytest.raises(ConfigurationError):
            EmbedderFactory.create(
                provider="unsupported_provider",
                model="some-model"
            )
    
    def test_get_supported_providers(self):
        """Test getting list of supported providers."""
        from flip.embedding.factory import EmbedderFactory
        
        providers = EmbedderFactory.get_supported_providers()
        
        assert "sentence-transformers" in providers
        assert "openai" in providers
        assert "azure-openai" in providers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
