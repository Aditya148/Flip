"""HuggingFace LLM implementation."""

import os
from typing import List, Optional
from huggingface_hub import InferenceClient

from flip.generation.base import BaseLLM, LLMResponse
from flip.core.exceptions import GenerationError, APIKeyMissingError


class HuggingFaceLLM(BaseLLM):
    """HuggingFace Inference API implementation."""
    
    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        """Initialize HuggingFace LLM."""
        super().__init__(model, api_key, **kwargs)
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        if not self.api_key:
            raise APIKeyMissingError(
                "HuggingFace API key not found. Set HUGGINGFACE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = InferenceClient(token=self.api_key)
    
    def generate(
        self,
        prompt: str,
        context: List[str],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> LLMResponse:
        """Generate response using HuggingFace."""
        try:
            formatted_prompt = self.format_prompt(prompt, context)
            
            response = self.client.text_generation(
                formatted_prompt,
                model=self.model,
                max_new_tokens=max_tokens,
                temperature=temperature,
                return_full_text=False,
                **kwargs
            )
            
            return LLMResponse(
                answer=response,
                model=self.model,
                tokens_used=self.count_tokens(formatted_prompt + response),
                finish_reason="stop",
                metadata={}
            )
            
        except Exception as e:
            raise GenerationError(f"HuggingFace generation failed: {str(e)}")
    
    def generate_stream(
        self,
        prompt: str,
        context: List[str],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ):
        """Generate streaming response using HuggingFace."""
        try:
            formatted_prompt = self.format_prompt(prompt, context)
            
            stream = self.client.text_generation(
                formatted_prompt,
                model=self.model,
                max_new_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                **kwargs
            )
            
            for chunk in stream:
                yield chunk
                    
        except Exception as e:
            raise GenerationError(f"HuggingFace streaming failed: {str(e)}")
    
    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "huggingface"
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens (approximation).
        For exact counts, would need to load the specific tokenizer.
        """
        return len(text) // 4
