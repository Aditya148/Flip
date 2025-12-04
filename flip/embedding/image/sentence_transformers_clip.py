"""Sentence Transformers CLIP image embedder."""

from typing import List, Union
import numpy as np

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from flip.embedding.image.base import BaseImageEmbedder


class SentenceTransformersCLIPEmbedder(BaseImageEmbedder):
    """Sentence Transformers CLIP image embedder."""
    
    # Popular CLIP models available in Sentence Transformers
    MODELS = {
        "clip-vit-b-32": "clip-ViT-B-32",
        "clip-vit-b-16": "clip-ViT-B-16",
        "clip-vit-l-14": "clip-ViT-L-14",
    }
    
    def __init__(
        self,
        model_name: str = "clip-vit-b-32",
        device: str = None
    ):
        """
        Initialize Sentence Transformers CLIP embedder.
        
        Args:
            model_name: CLIP model to use
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Sentence Transformers is required for CLIP embeddings. "
                "Install with: pip install sentence-transformers"
            )
        
        super().__init__(model_name)
        
        # Get model path
        model_path = self.MODELS.get(model_name, model_name)
        
        # Load model
        print(f"Loading Sentence Transformers CLIP model: {model_path}...")
        self.model = SentenceTransformer(model_path, device=device)
        
        # Get dimension
        self._dimension = self.model.get_sentence_embedding_dimension()
    
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
        
        # Generate embedding
        embedding = self.model.encode(img, convert_to_numpy=True)
        
        return self._normalize_embedding(embedding)
    
    def embed_images_batch(self, images: List[Union[Image.Image, str]]) -> List[List[float]]:
        """
        Generate embeddings for multiple images.
        
        Args:
            images: List of PIL Image objects or paths
            
        Returns:
            List of embedding vectors
        """
        # Load and preprocess all images
        pil_images = []
        for img in images:
            img = self._load_image(img)
            img = self._preprocess_image(img)
            pil_images.append(img)
        
        # Generate embeddings in batch
        embeddings = self.model.encode(
            pil_images,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        return [self._normalize_embedding(emb) for emb in embeddings]
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension
