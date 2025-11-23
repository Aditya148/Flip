"""Sentence Transformers embeddings implementation."""

from typing import List
from sentence_transformers import SentenceTransformer

from flip.embedding.base import BaseEmbedder
from flip.core.exceptions import EmbeddingError


class SentenceTransformerEmbedder(BaseEmbedder):
    """Sentence Transformers (local) embeddings implementation."""
    
    def __init__(self, model: str, **kwargs):
        """Initialize Sentence Transformer embedder."""
        super().__init__(model, **kwargs)
        
        try:
            self.model_instance = SentenceTransformer(model)
        except Exception as e:
            raise EmbeddingError(f"Failed to load Sentence Transformer model: {str(e)}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            embedding = self.model_instance.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            raise EmbeddingError(f"Sentence Transformer embedding failed: {str(e)}")
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            embeddings = self.model_instance.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            raise EmbeddingError(f"Sentence Transformer batch embedding failed: {str(e)}")
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self.model_instance.get_sentence_embedding_dimension()
    
    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "sentence-transformers"
