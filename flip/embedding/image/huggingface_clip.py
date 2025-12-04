"""HuggingFace CLIP image embedder."""

from typing import List, Union
import numpy as np

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from flip.embedding.image.base import BaseImageEmbedder


class HuggingFaceCLIPEmbedder(BaseImageEmbedder):
    """HuggingFace CLIP image embedder."""
    
    # Popular CLIP models
    MODELS = {
        "clip-vit-base-patch32": "openai/clip-vit-base-patch32",
        "clip-vit-base-patch16": "openai/clip-vit-base-patch16",
        "clip-vit-large-patch14": "openai/clip-vit-large-patch14",
    }
    
    def __init__(
        self,
        model_name: str = "clip-vit-base-patch32",
        device: str = None
    ):
        """
        Initialize HuggingFace CLIP embedder.
        
        Args:
            model_name: CLIP model to use
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers is required for HuggingFace CLIP embeddings. "
                "Install with: pip install transformers torch"
            )
        
        super().__init__(model_name)
        
        # Get model path
        model_path = self.MODELS.get(model_name, model_name)
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load model and processor
        print(f"Loading CLIP model: {model_path} on {self.device}...")
        self.model = CLIPModel.from_pretrained(model_path).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        
        # Set to eval mode
        self.model.eval()
        
        # Get dimension
        self._dimension = self.model.config.projection_dim
    
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
        
        # Process image
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embedding
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        
        # Convert to numpy and normalize
        embedding = image_features.cpu().numpy()[0]
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
        
        # Process batch
        inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        
        # Convert to list of embeddings
        embeddings = image_features.cpu().numpy()
        return [self._normalize_embedding(emb) for emb in embeddings]
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension
