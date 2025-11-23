"""Google Gemini LLM implementation."""

import os
from typing import List, Optional
import google.generativeai as genai

from flip.generation.base import BaseLLM, LLMResponse
from flip.core.exceptions import GenerationError, APIKeyMissingError


class GeminiLLM(BaseLLM):
    """Google Gemini implementation."""
    
    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        """Initialize Gemini LLM."""
        super().__init__(model, api_key, **kwargs)
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise APIKeyMissingError(
                "Google API key not found. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(model)
    
    def generate(
        self,
        prompt: str,
        context: List[str],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Gemini."""
        try:
            formatted_prompt = self.format_prompt(prompt, context)
            
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                **kwargs
            )
            
            response = self.client.generate_content(
                formatted_prompt,
                generation_config=generation_config
            )
            
            # Check if response was blocked or has no content
            if not response.candidates:
                raise GenerationError(
                    "Gemini response was blocked. No candidates returned. "
                    "This may be due to safety filters or content policy."
                )
            
            candidate = response.candidates[0]
            finish_reason = candidate.finish_reason
            
            # Handle different finish reasons
            # 0 = FINISH_REASON_UNSPECIFIED
            # 1 = STOP (normal completion)
            # 2 = MAX_TOKENS (hit token limit)
            # 3 = SAFETY (blocked by safety filters)
            # 4 = RECITATION (blocked due to recitation)
            # 5 = OTHER
            
            if finish_reason == 3:  # SAFETY
                safety_info = []
                if candidate.safety_ratings:
                    safety_info = [
                        f"{rating.category.name}: {rating.probability.name}"
                        for rating in candidate.safety_ratings
                    ]
                raise GenerationError(
                    f"Gemini response blocked by safety filters. "
                    f"Safety ratings: {', '.join(safety_info) if safety_info else 'N/A'}. "
                    f"Try rephrasing your query or adjusting safety settings."
                )
            
            if finish_reason == 4:  # RECITATION
                raise GenerationError(
                    "Gemini response blocked due to recitation detection. "
                    "The model detected potential copyrighted content. "
                    "Try rephrasing your query."
                )
            
            # Try to get text, handle if not available
            try:
                answer_text = response.text
            except ValueError as ve:
                # If finish_reason is 2 (MAX_TOKENS), we might have partial content
                if finish_reason == 2:
                    # Try to extract partial content from parts
                    if candidate.content and candidate.content.parts:
                        answer_text = "".join([part.text for part in candidate.content.parts if hasattr(part, 'text')])
                        if not answer_text:
                            raise GenerationError(
                                "Gemini hit max tokens limit and returned no usable content. "
                                "Try increasing max_tokens or shortening your context."
                            )
                    else:
                        raise GenerationError(
                            "Gemini hit max tokens limit and returned no content. "
                            "Try increasing max_tokens or shortening your context."
                        )
                else:
                    raise GenerationError(
                        f"Gemini response has no valid text content. "
                        f"Finish reason: {finish_reason}. Original error: {str(ve)}"
                    )
            
            # Extract token counts if available
            tokens_used = 0
            if hasattr(response, 'usage_metadata'):
                tokens_used = (
                    response.usage_metadata.prompt_token_count +
                    response.usage_metadata.candidates_token_count
                )
            
            return LLMResponse(
                answer=answer_text,
                model=self.model,
                tokens_used=tokens_used,
                finish_reason=str(finish_reason),
                metadata={
                    "safety_ratings": [
                        {
                            "category": rating.category.name,
                            "probability": rating.probability.name
                        }
                        for rating in candidate.safety_ratings
                    ] if candidate.safety_ratings else []
                }
            )
            
        except GenerationError:
            # Re-raise our custom errors
            raise
        except Exception as e:
            raise GenerationError(f"Gemini generation failed: {str(e)}")
    
    def generate_stream(
        self,
        prompt: str,
        context: List[str],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ):
        """Generate streaming response using Gemini."""
        try:
            formatted_prompt = self.format_prompt(prompt, context)
            
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                **kwargs
            )
            
            response = self.client.generate_content(
                formatted_prompt,
                generation_config=generation_config,
                stream=True
            )
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            raise GenerationError(f"Gemini streaming failed: {str(e)}")
    
    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "google"
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using Gemini's token counting."""
        try:
            result = self.client.count_tokens(text)
            return result.total_tokens
        except:
            # Fallback approximation
            return len(text) // 4
