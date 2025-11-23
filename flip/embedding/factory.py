"""Factory for creating embedder instances dynamically."""

from typing import Optional
from flip.embedding.base import BaseEmbedder
from flip.embedding.openai import OpenAIEmbedder
from flip.embedding.azure_openai import AzureOpenAIEmbedder
from flip.embedding.google import GoogleEmbedder
from flip.embedding.sentence_transformers import SentenceTransformerEmbedder
from flip.embedding.huggingface import HuggingFaceEmbedder
from flip.core.exceptions import ConfigurationError


class EmbedderFactory:
    """Factory for creating embedder instances based on provider."""
    
    _providers = {
        "openai": OpenAIEmbedder,
        "azure-openai": AzureOpenAIEmbedder,
        "google": GoogleEmbedder,
        "sentence-transformers": SentenceTransformerEmbedder,
        "huggingface": HuggingFaceEmbedder,
    }
    
    @classmethod
    def create(
        cls,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        **kwargs
    ) -> BaseEmbedder:
        """
        Create an embedder instance for the specified provider.
        
        Args:
            provider: Provider name (openai, google, sentence-transformers, huggingface)
            model: Model name to use
            api_key: API key for the provider (if required)
            **kwargs: Additional provider-specific arguments
            
        Returns:
            BaseEmbedder instance
            
        Raises:
            ConfigurationError: If provider is not supported
        """
        provider = provider.lower()
        
        if provider not in cls._providers:
            raise ConfigurationError(
                f"Unsupported embedding provider: {provider}. "
                f"Supported providers: {', '.join(cls._providers.keys())}"
            )
        
        embedder_class = cls._providers[provider]
        
        # sentence-transformers doesn't need API key
        if provider == "sentence-transformers":
            return embedder_class(model=model, **kwargs)
        else:
            return embedder_class(model=model, api_key=api_key, **kwargs)
    
    @classmethod
    def get_supported_providers(cls) -> list[str]:
        """Get list of supported providers."""
        return list(cls._providers.keys())
    
    @classmethod
    def register_provider(cls, name: str, embedder_class: type[BaseEmbedder]):
        """
        Register a custom embedder provider.
        
        Args:
            name: Provider name
            embedder_class: Embedder class that inherits from BaseEmbedder
        """
        if not issubclass(embedder_class, BaseEmbedder):
            raise ConfigurationError(
                f"Embedder class must inherit from BaseEmbedder"
            )
        cls._providers[name.lower()] = embedder_class
