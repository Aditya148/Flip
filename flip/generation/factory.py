"""Factory for creating LLM instances dynamically."""

from typing import Optional
from flip.generation.base import BaseLLM
from flip.generation.openai import OpenAILLM
from flip.generation.azure_openai import AzureOpenAILLM
from flip.generation.anthropic import AnthropicLLM
from flip.generation.gemini import GeminiLLM
from flip.generation.huggingface import HuggingFaceLLM
from flip.generation.ollama import OllamaLLM
from flip.core.exceptions import ConfigurationError


class LLMFactory:
    """Factory for creating LLM instances based on provider."""
    
    _providers = {
        "openai": OpenAILLM,
        "azure-openai": AzureOpenAILLM,
        "anthropic": AnthropicLLM,
        "google": GeminiLLM,
        "huggingface": HuggingFaceLLM,
        "meta": HuggingFaceLLM,  # Meta models via HuggingFace
        "ollama": OllamaLLM,
    }
    
    @classmethod
    def create(
        cls,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        **kwargs
    ) -> BaseLLM:
        """
        Create an LLM instance for the specified provider.
        
        Args:
            provider: Provider name (openai, anthropic, google, huggingface, meta, ollama)
            model: Model name to use
            api_key: API key for the provider (if required)
            **kwargs: Additional provider-specific arguments
            
        Returns:
            BaseLLM instance
            
        Raises:
            ConfigurationError: If provider is not supported
        """
        provider = provider.lower()
        
        if provider not in cls._providers:
            raise ConfigurationError(
                f"Unsupported LLM provider: {provider}. "
                f"Supported providers: {', '.join(cls._providers.keys())}"
            )
        
        llm_class = cls._providers[provider]
        return llm_class(model=model, api_key=api_key, **kwargs)
    
    @classmethod
    def get_supported_providers(cls) -> list[str]:
        """Get list of supported providers."""
        return list(cls._providers.keys())
    
    @classmethod
    def register_provider(cls, name: str, llm_class: type[BaseLLM]):
        """
        Register a custom LLM provider.
        
        Args:
            name: Provider name
            llm_class: LLM class that inherits from BaseLLM
        """
        if not issubclass(llm_class, BaseLLM):
            raise ConfigurationError(
                f"LLM class must inherit from BaseLLM"
            )
        cls._providers[name.lower()] = llm_class
