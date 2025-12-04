"""Image extraction from various document formats."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path
import io

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@dataclass
class ImageMetadata:
    """Metadata for extracted images."""
    
    source_file: str
    page_number: Optional[int] = None
    image_index: int = 0
    width: Optional[int] = None
    height: Optional[int] = None
    format: Optional[str] = None
    caption: Optional[str] = None
    alt_text: Optional[str] = None
    position: Optional[Dict[str, float]] = None  # x, y coordinates
    context: Optional[str] = None  # Surrounding text
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_file": self.source_file,
            "page_number": self.page_number,
            "image_index": self.image_index,
            "width": self.width,
            "height": self.height,
            "format": self.format,
            "caption": self.caption,
            "alt_text": self.alt_text,
            "position": self.position,
            "context": self.context
        }


@dataclass
class ExtractedImage:
    """Container for extracted image with metadata."""
    
    image: Any  # PIL.Image.Image
    metadata: ImageMetadata
    
    def save(self, path: str):
        """Save image to file."""
        if not PIL_AVAILABLE:
            raise ImportError("PIL/Pillow is required for image operations")
        self.image.save(path)
    
    def resize(self, max_size: int = 1024) -> 'ExtractedImage':
        """Resize image maintaining aspect ratio."""
        if not PIL_AVAILABLE:
            raise ImportError("PIL/Pillow is required for image operations")
        
        width, height = self.image.size
        if width <= max_size and height <= max_size:
            return self
        
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        resized_image = self.image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return ExtractedImage(
            image=resized_image,
            metadata=ImageMetadata(
                source_file=self.metadata.source_file,
                page_number=self.metadata.page_number,
                image_index=self.metadata.image_index,
                width=new_width,
                height=new_height,
                format=self.metadata.format,
                caption=self.metadata.caption,
                alt_text=self.metadata.alt_text,
                position=self.metadata.position,
                context=self.metadata.context
            )
        )
    
    def to_bytes(self, format: str = "PNG") -> bytes:
        """Convert image to bytes."""
        if not PIL_AVAILABLE:
            raise ImportError("PIL/Pillow is required for image operations")
        
        buffer = io.BytesIO()
        self.image.save(buffer, format=format)
        return buffer.getvalue()


class BaseImageExtractor(ABC):
    """Base class for image extractors."""
    
    def __init__(self, max_image_size: int = 1024, min_image_size: int = 100):
        """
        Initialize image extractor.
        
        Args:
            max_image_size: Maximum dimension for extracted images
            min_image_size: Minimum dimension to consider (filter out icons)
        """
        if not PIL_AVAILABLE:
            raise ImportError("PIL/Pillow is required for image extraction. Install with: pip install Pillow")
        
        self.max_image_size = max_image_size
        self.min_image_size = min_image_size
    
    @abstractmethod
    def extract_images(self, file_path: str) -> List[ExtractedImage]:
        """
        Extract images from a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of extracted images with metadata
        """
        pass
    
    def _should_include_image(self, width: int, height: int) -> bool:
        """Check if image meets size requirements."""
        return width >= self.min_image_size and height >= self.min_image_size
    
    def _resize_if_needed(self, image: Image.Image) -> Image.Image:
        """Resize image if it exceeds max size."""
        width, height = image.size
        
        if width <= self.max_image_size and height <= self.max_image_size:
            return image
        
        if width > height:
            new_width = self.max_image_size
            new_height = int(height * (self.max_image_size / width))
        else:
            new_height = self.max_image_size
            new_width = int(width * (self.max_image_size / height))
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


class StandaloneImageLoader(BaseImageExtractor):
    """Loader for standalone image files."""
    
    SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff'}
    
    def extract_images(self, file_path: str) -> List[ExtractedImage]:
        """
        Load a standalone image file.
        
        Args:
            file_path: Path to image file
            
        Returns:
            List containing single extracted image
        """
        path = Path(file_path)
        
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            return []
        
        try:
            image = Image.open(file_path)
            
            # Convert to RGB if needed
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            
            width, height = image.size
            
            if not self._should_include_image(width, height):
                return []
            
            # Resize if needed
            image = self._resize_if_needed(image)
            
            metadata = ImageMetadata(
                source_file=str(path),
                page_number=None,
                image_index=0,
                width=image.size[0],
                height=image.size[1],
                format=image.format or path.suffix[1:].upper(),
                caption=path.stem,  # Use filename as caption
                alt_text=None,
                position=None,
                context=None
            )
            
            return [ExtractedImage(image=image, metadata=metadata)]
            
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            return []
    
    @classmethod
    def is_supported(cls, file_path: str) -> bool:
        """Check if file is a supported image format."""
        return Path(file_path).suffix.lower() in cls.SUPPORTED_FORMATS
