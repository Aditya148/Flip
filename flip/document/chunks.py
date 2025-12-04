"""Enhanced document chunk with image support."""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document."""
    
    text: str
    metadata: Dict[str, Any]
    chunk_id: Optional[str] = None
    chunk_type: Literal["text", "image"] = "text"
    
    def __post_init__(self):
        """Ensure metadata has required fields."""
        if "source" not in self.metadata:
            self.metadata["source"] = "unknown"


@dataclass
class ImageChunk:
    """Represents an image chunk from a document."""
    
    image: Any  # PIL.Image.Image
    embedding: Optional[list] = None
    metadata: Dict[str, Any] = None
    chunk_id: Optional[str] = None
    chunk_type: Literal["text", "image"] = "image"
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}
        
        # Ensure chunk_type is set
        self.metadata["chunk_type"] = "image"
        
        if "source" not in self.metadata:
            self.metadata["source"] = "unknown"
    
    def to_storage_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for storage (without PIL Image).
        
        Returns:
            Dictionary with metadata and embedding
        """
        return {
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "text": self.metadata.get("caption", "") or self.metadata.get("alt_text", "") or ""
        }
