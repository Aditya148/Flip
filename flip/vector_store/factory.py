"""Factory for creating vector store instances."""

from typing import Optional
from pathlib import Path

from flip.vector_store.base import BaseVectorStore
from flip.vector_store.chroma import ChromaVectorStore
from flip.core.exceptions import ConfigurationError


class VectorStoreFactory:
    """Factory for creating vector store instances."""
    
    _providers = {
        "chroma": ChromaVectorStore,
        # Additional providers can be added here
        # "pinecone": PineconeVectorStore,
        # "qdrant": QdrantVectorStore,
        # "weaviate": WeaviateVectorStore,
        # "faiss": FAISSVectorStore,
    }
    
    @classmethod
    def create(
        cls,
        provider: str,
        collection_name: str,
        persist_directory: Optional[str] = None,
        **kwargs
    ) -> BaseVectorStore:
        """
        Create a vector store instance.
        
        Args:
            provider: Provider name (chroma, pinecone, qdrant, weaviate, faiss)
            collection_name: Name of the collection/index
            persist_directory: Directory to persist data (for local stores)
            **kwargs: Additional provider-specific arguments
            
        Returns:
            BaseVectorStore instance
            
        Raises:
            ConfigurationError: If provider is not supported
        """
        provider = provider.lower()
        
        if provider not in cls._providers:
            raise ConfigurationError(
                f"Unsupported vector store provider: {provider}. "
                f"Supported providers: {', '.join(cls._providers.keys())}"
            )
        
        store_class = cls._providers[provider]
        
        # Create instance with appropriate parameters
        if provider == "chroma":
            return store_class(
                collection_name=collection_name,
                persist_directory=persist_directory,
                **kwargs
            )
        else:
            return store_class(
                collection_name=collection_name,
                **kwargs
            )
    
    @classmethod
    def get_supported_providers(cls) -> list[str]:
        """Get list of supported providers."""
        return list(cls._providers.keys())
    
    @classmethod
    def register_provider(cls, name: str, store_class: type[BaseVectorStore]):
        """
        Register a custom vector store provider.
        
        Args:
            name: Provider name
            store_class: Vector store class that inherits from BaseVectorStore
        """
        if not issubclass(store_class, BaseVectorStore):
            raise ConfigurationError(
                "Vector store class must inherit from BaseVectorStore"
            )
        cls._providers[name.lower()] = store_class
