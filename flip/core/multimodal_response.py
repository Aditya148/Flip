"""Multimodal RAG response with image results."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@dataclass
class ImageResult:
    """Result containing an image."""
    
    id: str
    image: Any  # PIL.Image.Image
    score: float
    metadata: Dict[str, Any]
    caption: Optional[str] = None
    context: Optional[str] = None
    
    def __post_init__(self):
        """Extract caption and context from metadata."""
        if self.caption is None:
            self.caption = self.metadata.get("caption") or self.metadata.get("alt_text")
        
        if self.context is None:
            self.context = self.metadata.get("context")
    
    def save(self, path: str):
        """Save image to file."""
        if PIL_AVAILABLE and self.image:
            self.image.save(path)
    
    def show(self):
        """Display image."""
        if PIL_AVAILABLE and self.image:
            self.image.show()


@dataclass
class MultimodalRAGResponse:
    """Response from multimodal RAG query."""
    
    answer: str
    text_results: List[Dict[str, Any]]
    image_results: List[ImageResult]
    citations: List[str]
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Initialize empty lists if None."""
        if self.text_results is None:
            self.text_results = []
        if self.image_results is None:
            self.image_results = []
        if self.citations is None:
            self.citations = []
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def has_images(self) -> bool:
        """Check if response contains images."""
        return len(self.image_results) > 0
    
    @property
    def total_results(self) -> int:
        """Get total number of results."""
        return len(self.text_results) + len(self.image_results)
    
    def save_images(self, directory: str):
        """
        Save all images to directory.
        
        Args:
            directory: Directory to save images
        """
        from pathlib import Path
        
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        for i, img_result in enumerate(self.image_results):
            img_path = dir_path / f"image_{i}.png"
            img_result.save(str(img_path))
    
    def display_images(self):
        """Display all images."""
        for img_result in self.image_results:
            print(f"\nImage: {img_result.id}")
            if img_result.caption:
                print(f"Caption: {img_result.caption}")
            print(f"Score: {img_result.score:.3f}")
            img_result.show()
