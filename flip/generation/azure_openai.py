"""Azure OpenAI LLM provider."""

import os
from typing import List, Optional, Dict, Any
from openai import AzureOpenAI

from flip.generation.base import BaseLLM, LLMResponse
from flip.core.exceptions import GenerationError


class AzureOpenAILLM(BaseLLM):
    """Azure OpenAI LLM provider."""
    
    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: str = "2024-02-15-preview",
        deployment_name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Azure OpenAI LLM.
        
        Args:
            model: Model name (e.g., "gpt-4", "gpt-35-turbo")
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
            raise GenerationError("Azure OpenAI API key not provided")
        
        if not self.azure_endpoint:
            raise GenerationError("Azure OpenAI endpoint not provided")
        
        # Initialize client
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint
        )
    
    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return "azure-openai"
    
    def generate(
        self,
        prompt: str,
        context: List[str],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Azure OpenAI."""
        try:
            formatted_prompt = self.format_prompt(prompt, context)
            
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            return LLMResponse(
                answer=response.choices[0].message.content,
                model=self.deployment_name,
                tokens_used=response.usage.total_tokens,
                finish_reason=response.choices[0].finish_reason,
                metadata={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                }
            )
            
        except Exception as e:
            raise GenerationError(f"Azure OpenAI generation failed: {str(e)}")
    
    def stream(
        self,
        prompt: str,
        context: List[str],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ):
        """Stream response using Azure OpenAI."""
        try:
            formatted_prompt = self.format_prompt(prompt, context)
            
            stream = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            raise GenerationError(f"Azure OpenAI streaming failed: {str(e)}")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        try:
            import tiktoken
            
            # Use appropriate encoding for the model
            if "gpt-4" in self.model:
                encoding = tiktoken.encoding_for_model("gpt-4")
            elif "gpt-35" in self.model or "gpt-3.5" in self.model:
                encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                encoding = tiktoken.get_encoding("cl100k_base")
            
            return len(encoding.encode(text))
        except:
            # Fallback to rough estimation
            return len(text) // 4
