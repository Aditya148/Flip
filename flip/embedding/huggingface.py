"""HuggingFace embeddings implementation."""

import os
from typing import List, Optional
from huggingface_hub import InferenceClient

from flip.embedding.base import BaseEmbedder
from flip.core.exceptions import EmbeddingError, APIKeyMissingError


class HuggingFaceEmbedder(BaseEmbedder):
    """HuggingFace embeddings implementation."""
    
    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        """Initialize HuggingFace embedder."""
        super().__init__(model, **kwargs)
        
        # Get API key from parameter or environment
        api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        if not api_key:
            raise APIKeyMissingError(
                "HuggingFace API key not found. Set HUGGINGFACE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = InferenceClient(token=api_key)
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            embedding = self.client.feature_extraction(text, model=self.model)
            # Handle different return formats
            if isinstance(embedding, list) and len(embedding) > 0:
                if isinstance(embedding[0], list):
                    # Pooling: take mean of all token embeddings
                    import numpy as np
                    return np.mean(embedding, axis=0).tolist()
                return embedding
            return embedding
        except Exception as e:
            raise EmbeddingError(f"HuggingFace embedding failed: {str(e)}")
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            embeddings = []
            for text in texts:
                embeddings.append(self.embed_text(text))
            return embeddings
        except Exception as e:
            raise EmbeddingError(f"HuggingFace batch embedding failed: {str(e)}")
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        # Common dimension for sentence transformers
        return 768
    
    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "huggingface"
