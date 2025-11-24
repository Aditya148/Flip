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
    }
    
    @classmethod
    def _lazy_load_providers(cls):
        """Lazy load optional providers."""
        if "pinecone" not in cls._providers:
            try:
                from flip.vector_store.pinecone import PineconeVectorStore
                cls._providers["pinecone"] = PineconeVectorStore
            except ImportError:
                pass
        
        if "qdrant" not in cls._providers:
            try:
                from flip.vector_store.qdrant import QdrantVectorStore
                cls._providers["qdrant"] = QdrantVectorStore
            except ImportError:
                pass
        
        if "weaviate" not in cls._providers:
            try:
                from flip.vector_store.weaviate import WeaviateVectorStore
                cls._providers["weaviate"] = WeaviateVectorStore
            except ImportError:
                pass
        
        if "milvus" not in cls._providers:
            try:
                from flip.vector_store.milvus import MilvusVectorStore
                cls._providers["milvus"] = MilvusVectorStore
            except ImportError:
                pass
        
        if "faiss" not in cls._providers:
            try:
                from flip.vector_store.faiss import FAISSVectorStore
                cls._providers["faiss"] = FAISSVectorStore
            except ImportError:
                pass
        
        if "pgvector" not in cls._providers:
            try:
                from flip.vector_store.pgvector import PgvectorVectorStore
                cls._providers["pgvector"] = PgvectorVectorStore
            except ImportError:
                pass
        
        if "redis" not in cls._providers:
            try:
                from flip.vector_store.redis import RedisVectorStore
                cls._providers["redis"] = RedisVectorStore
            except ImportError:
                pass
        
        if "elasticsearch" not in cls._providers:
            try:
                from flip.vector_store.elasticsearch import ElasticsearchVectorStore
                cls._providers["elasticsearch"] = ElasticsearchVectorStore
            except ImportError:
                pass
        
        if "mongodb" not in cls._providers:
            try:
                from flip.vector_store.mongodb import MongoDBVectorStore
                cls._providers["mongodb"] = MongoDBVectorStore
            except ImportError:
                pass
    
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
            provider: Provider name (chroma, pinecone, qdrant, weaviate, etc.)
            collection_name: Name of the collection/index
            persist_directory: Directory to persist data (for local stores)
            **kwargs: Additional provider-specific arguments
            
        Returns:
            BaseVectorStore instance
            
        Raises:
            ConfigurationError: If provider is not supported
        """
        provider = provider.lower()
        
        # Lazy load optional providers
        cls._lazy_load_providers()
        
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
        elif provider == "pinecone":
            return store_class(
                collection_name=collection_name,
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
        cls._lazy_load_providers()
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
