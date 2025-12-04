"""Factory for creating image embedders."""

from typing import Optional
from flip.embedding.image.base import BaseImageEmbedder


class ImageEmbedderFactory:
    """Factory for creating image embedders."""
    
    _embedders = {}
    
    @classmethod
    def _lazy_load_embedders(cls):
        """Lazy load image embedders."""
        if "openai-clip" not in cls._embedders:
            try:
                from flip.embedding.image.openai_clip import OpenAICLIPEmbedder
                cls._embedders["openai-clip"] = OpenAICLIPEmbedder
            except ImportError:
                pass
        
        if "huggingface-clip" not in cls._embedders:
            try:
                from flip.embedding.image.huggingface_clip import HuggingFaceCLIPEmbedder
                cls._embedders["huggingface-clip"] = HuggingFaceCLIPEmbedder
            except ImportError:
                pass
        
        if "sentence-transformers-clip" not in cls._embedders:
            try:
                from flip.embedding.image.sentence_transformers_clip import SentenceTransformersCLIPEmbedder
                cls._embedders["sentence-transformers-clip"] = SentenceTransformersCLIPEmbedder
            except ImportError:
                pass
    
    @classmethod
    def create(
        cls,
        provider: str = "sentence-transformers-clip",
        model_name: str = None,
        **kwargs
    ) -> Optional[BaseImageEmbedder]:
        """
        Create an image embedder.
        
        Args:
            provider: Provider name ('openai-clip', 'huggingface-clip', 'sentence-transformers-clip')
            model_name: Model name (provider-specific)
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Image embedder instance or None if provider not available
        """
        cls._lazy_load_embedders()
        
        if provider not in cls._embedders:
            available = list(cls._embedders.keys())
            raise ValueError(
                f"Image embedder '{provider}' not available. "
                f"Available providers: {available}"
            )
        
        embedder_class = cls._embedders[provider]
        
        # Create instance with model name if provided
        if model_name:
            return embedder_class(model_name=model_name, **kwargs)
        else:
            return embedder_class(**kwargs)
    
    @classmethod
    def list_providers(cls) -> list:
        """
        List available image embedding providers.
        
        Returns:
            List of provider names
        """
        cls._lazy_load_embedders()
        return list(cls._embedders.keys())
