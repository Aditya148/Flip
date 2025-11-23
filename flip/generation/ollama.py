"""Ollama LLM implementation for local models."""

import os
from typing import List, Optional
import requests

from flip.generation.base import BaseLLM, LLMResponse
from flip.core.exceptions import GenerationError


class OllamaLLM(BaseLLM):
    """Ollama local LLM implementation."""
    
    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        """Initialize Ollama LLM."""
        super().__init__(model, api_key, **kwargs)
        
        # Ollama runs locally, get base URL from env or use default
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    def generate(
        self,
        prompt: str,
        context: List[str],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Ollama."""
        try:
            formatted_prompt = self.format_prompt(prompt, context)
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": formatted_prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        **kwargs
                    }
                }
            )
            response.raise_for_status()
            data = response.json()
            
            return LLMResponse(
                answer=data["response"],
                model=self.model,
                tokens_used=data.get("eval_count", 0) + data.get("prompt_eval_count", 0),
                finish_reason=data.get("done_reason", "stop"),
                metadata={
                    "eval_count": data.get("eval_count", 0),
                    "prompt_eval_count": data.get("prompt_eval_count", 0),
                    "total_duration": data.get("total_duration", 0),
                }
            )
            
        except Exception as e:
            raise GenerationError(f"Ollama generation failed: {str(e)}")
    
    def generate_stream(
        self,
        prompt: str,
        context: List[str],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ):
        """Generate streaming response using Ollama."""
        try:
            formatted_prompt = self.format_prompt(prompt, context)
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": formatted_prompt,
                    "stream": True,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        **kwargs
                    }
                },
                stream=True
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    import json
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                    
        except Exception as e:
            raise GenerationError(f"Ollama streaming failed: {str(e)}")
    
    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "ollama"
    
    def count_tokens(self, text: str) -> int:
        """Count tokens (approximation)."""
        return len(text) // 4
