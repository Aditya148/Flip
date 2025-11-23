"""Google embeddings implementation."""

import os
from typing import List, Optional
import google.generativeai as genai

from flip.embedding.base import BaseEmbedder
from flip.core.exceptions import EmbeddingError, APIKeyMissingError


class GoogleEmbedder(BaseEmbedder):
    """Google embeddings implementation."""
    
    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        """Initialize Google embedder."""
        super().__init__(model, **kwargs)
        
        # Get API key from parameter or environment
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise APIKeyMissingError(
                "Google API key not found. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        genai.configure(api_key=api_key)
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            raise EmbeddingError(f"Google embedding failed: {str(e)}")
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            embeddings = []
            # Google API doesn't support batch, so we process one by one
            for text in texts:
                result = genai.embed_content(
                    model=self.model,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            return embeddings
        except Exception as e:
            raise EmbeddingError(f"Google batch embedding failed: {str(e)}")
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        # Google's embedding-001 has 768 dimensions
        return 768
    
    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "google"
