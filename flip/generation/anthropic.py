"""Anthropic Claude LLM implementation."""

import os
from typing import List, Optional
from anthropic import Anthropic

from flip.generation.base import BaseLLM, LLMResponse
from flip.core.exceptions import GenerationError, APIKeyMissingError


class AnthropicLLM(BaseLLM):
    """Anthropic Claude implementation."""
    
    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        """Initialize Anthropic LLM."""
        super().__init__(model, api_key, **kwargs)
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise APIKeyMissingError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = Anthropic(api_key=self.api_key)
    
    def generate(
        self,
        prompt: str,
        context: List[str],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Claude."""
        try:
            formatted_prompt = self.format_prompt(prompt, context)
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": formatted_prompt}
                ],
                **kwargs
            )
            
            return LLMResponse(
                answer=response.content[0].text,
                model=response.model,
                tokens_used=response.usage.input_tokens + response.usage.output_tokens,
                finish_reason=response.stop_reason,
                metadata={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }
            )
            
        except Exception as e:
            raise GenerationError(f"Anthropic generation failed: {str(e)}")
    
    def generate_stream(
        self,
        prompt: str,
        context: List[str],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ):
        """Generate streaming response using Claude."""
        try:
            formatted_prompt = self.format_prompt(prompt, context)
            
            with self.client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": formatted_prompt}
                ],
                **kwargs
            ) as stream:
                for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            raise GenerationError(f"Anthropic streaming failed: {str(e)}")
    
    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "anthropic"
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens using Anthropic's token counting.
        Note: This is an approximation. For exact counts, use Anthropic's API.
        """
        # Rough approximation: ~4 characters per token
        return len(text) // 4
