"""Base interface for LLM providers."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Response from LLM generation."""
    
    answer: str
    """The generated answer."""
    
    model: str
    """Model used for generation."""
    
    tokens_used: int
    """Total tokens used."""
    
    finish_reason: str
    """Reason for completion (e.g., 'stop', 'length')."""
    
    metadata: Dict[str, Any]
    """Additional metadata from the provider."""


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        """
        Initialize the LLM provider.
        
        Args:
            model: Model name to use
            api_key: API key for the provider (if required)
            **kwargs: Additional provider-specific arguments
        """
        self.model = model
        self.api_key = api_key
        self.kwargs = kwargs
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        context: List[str],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a response using the LLM.
        
        Args:
            prompt: The user's query/prompt
            context: List of retrieved context chunks
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            LLMResponse object with the generated answer
        """
        pass
    
    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        context: List[str],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ):
        """
        Generate a streaming response.
        
        Args:
            prompt: The user's query/prompt
            context: List of retrieved context chunks
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters
            
        Yields:
            Chunks of the generated response
        """
        pass
    
    def format_prompt(self, prompt: str, context: List[str]) -> str:
        """
        Format the prompt with context.
        
        Args:
            prompt: User's query
            context: Retrieved context chunks
            
        Returns:
            Formatted prompt string
        """
        context_str = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(context)])
        
        return f"""You are a helpful assistant that answers questions based on the provided context.
Use the context below to answer the question. If the answer cannot be found in the context, say so.
Always cite the context numbers (e.g., [1], [2]) when using information from them.

Context:
{context_str}

Question: {prompt}

Answer:"""
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in the text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        pass
