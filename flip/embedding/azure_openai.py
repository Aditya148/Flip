"""Azure OpenAI embedder."""

import os
from typing import List, Optional
from openai import AzureOpenAI
import numpy as np

from flip.embedding.base import BaseEmbedder
from flip.core.exceptions import EmbeddingError


class AzureOpenAIEmbedder(BaseEmbedder):
    """Azure OpenAI embedder."""
    
    def __init__(
        self,
        model: str = "text-embedding-ada-002",
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: str = "2024-02-15-preview",
        deployment_name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Azure OpenAI embedder.
        
        Args:
            model: Model name (e.g., "text-embedding-ada-002", "text-embedding-3-small")
            api_key: Azure OpenAI API key
            azure_endpoint: Azure OpenAI endpoint URL
            api_version: API version
            deployment_name: Deployment name (defaults to model name)
            **kwargs: Additional arguments
        """
        super().__init__(model, **kwargs)
        
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = api_version
        self.deployment_name = deployment_name or model
        
        if not self.api_key:
            raise EmbeddingError("Azure OpenAI API key not provided")
        
        if not self.azure_endpoint:
            raise EmbeddingError("Azure OpenAI endpoint not provided")
        
        # Initialize client
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint
        )
        
        # Cache dimension
        self._dimension = None
    
    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return "azure-openai"
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None:
            # Get dimension by embedding a test string
            test_embedding = self.embed("test")
            self._dimension = len(test_embedding)
        return self._dimension
    
    def embed(self, text: str) -> List[float]:
        """
        Embed a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            response = self.client.embeddings.create(
                model=self.deployment_name,
                input=text
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            raise EmbeddingError(f"Azure OpenAI embedding failed: {str(e)}")
    
    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Embed multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for API calls
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        try:
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                response = self.client.embeddings.create(
                    model=self.deployment_name,
                    input=batch
                )
                
                # Extract embeddings in order
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
            
            return embeddings
            
        except Exception as e:
            raise EmbeddingError(f"Azure OpenAI batch embedding failed: {str(e)}")
