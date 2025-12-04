"""Base class for image embedders."""

from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class BaseImageEmbedder(ABC):
    """Abstract base class for image embedding providers."""
    
    def __init__(self, model_name: str = "default"):
        """
        Initialize image embedder.
        
        Args:
            model_name: Name of the model to use
        """
        if not PIL_AVAILABLE:
            raise ImportError("PIL/Pillow is required for image embedding. Install with: pip install Pillow")
        
        self.model_name = model_name
    
    @abstractmethod
    def embed_image(self, image: Union[Image.Image, str]) -> List[float]:
        """
        Generate embedding for a single image.
        
        Args:
            image: PIL Image object or path to image file
            
        Returns:
            Embedding vector as list of floats
        """
        pass
    
    @abstractmethod
    def embed_images_batch(self, images: List[Union[Image.Image, str]]) -> List[List[float]]:
        """
        Generate embeddings for multiple images.
        
        Args:
            images: List of PIL Image objects or paths to image files
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the embeddings."""
        pass
    
    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return self.__class__.__name__.replace("ImageEmbedder", "").lower()
    
    def _load_image(self, image: Union[Image.Image, str]) -> Image.Image:
        """
        Load image from path or return PIL Image.
        
        Args:
            image: PIL Image or path to image
            
        Returns:
            PIL Image object
        """
        if isinstance(image, str):
            return Image.open(image)
        return image
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image (convert to RGB, resize if needed).
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed PIL Image
        """
        # Convert to RGB if needed
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        
        return image
    
    def _normalize_embedding(self, embedding: np.ndarray) -> List[float]:
        """
        Normalize embedding to unit length.
        
        Args:
            embedding: Numpy array
            
        Returns:
            Normalized embedding as list
        """
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.tolist()
