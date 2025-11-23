"""OpenAI embeddings implementation."""

import os
from typing import List, Optional
from openai import OpenAI

from flip.embedding.base import BaseEmbedder
from flip.core.exceptions import EmbeddingError, APIKeyMissingError


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embeddings implementation."""
    
    # Model dimensions
    _dimensions = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        """Initialize OpenAI embedder."""
        super().__init__(model, **kwargs)
        
        # Get API key from parameter or environment
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise APIKeyMissingError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = OpenAI(api_key=api_key)
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            raise EmbeddingError(f"OpenAI embedding failed: {str(e)}")
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            raise EmbeddingError(f"OpenAI batch embedding failed: {str(e)}")
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimensions.get(self.model, 1536)
    
    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "openai"
