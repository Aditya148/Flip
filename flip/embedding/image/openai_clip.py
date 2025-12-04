"""OpenAI CLIP image embedder."""

from typing import List, Union
import base64
import io

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from flip.embedding.image.base import BaseImageEmbedder


class OpenAICLIPEmbedder(BaseImageEmbedder):
    """OpenAI CLIP image embedder using OpenAI API."""
    
    def __init__(
        self,
        model_name: str = "clip-vit-base-patch32",
        api_key: str = None
    ):
        """
        Initialize OpenAI CLIP embedder.
        
        Args:
            model_name: CLIP model to use
            api_key: OpenAI API key
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI is required for OpenAI CLIP embeddings. "
                "Install with: pip install openai"
            )
        
        super().__init__(model_name)
        
        # Set API key
        if api_key:
            openai.api_key = api_key
        
        self._dimension = 512  # CLIP embedding dimension
    
    def embed_image(self, image: Union[Image.Image, str]) -> List[float]:
        """
        Generate embedding for a single image.
        
        Args:
            image: PIL Image object or path to image file
            
        Returns:
            Embedding vector
        """
        # Load and preprocess image
        img = self._load_image(image)
        img = self._preprocess_image(img)
        
        # Convert to base64
        img_base64 = self._image_to_base64(img)
        
        # Note: OpenAI doesn't have a direct CLIP embedding API yet
        # This is a placeholder for when they add it
        # For now, we'll use a workaround or alternative approach
        
        # Placeholder implementation
        # In production, you'd use the actual OpenAI CLIP API when available
        raise NotImplementedError(
            "OpenAI CLIP embeddings are not yet available via API. "
            "Use HuggingFaceCLIPEmbedder or SentenceTransformersCLIPEmbedder instead."
        )
    
    def embed_images_batch(self, images: List[Union[Image.Image, str]]) -> List[List[float]]:
        """
        Generate embeddings for multiple images.
        
        Args:
            images: List of PIL Image objects or paths
            
        Returns:
            List of embedding vectors
        """
        return [self.embed_image(img) for img in images]
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')
