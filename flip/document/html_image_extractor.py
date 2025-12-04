"""HTML image extraction using BeautifulSoup."""

from typing import List
from pathlib import Path
import io
import requests

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from flip.document.image_extractor import (
    BaseImageExtractor,
    ExtractedImage,
    ImageMetadata
)


class HTMLImageExtractor(BaseImageExtractor):
    """Extract images from HTML files."""
    
    def __init__(
        self,
        max_image_size: int = 1024,
        min_image_size: int = 100,
        download_external: bool = False,
        timeout: int = 10
    ):
        """
        Initialize HTML image extractor.
        
        Args:
            max_image_size: Maximum dimension for extracted images
            min_image_size: Minimum dimension to consider
            download_external: Whether to download external images
            timeout: Timeout for downloading external images
        """
        if not BS4_AVAILABLE:
            raise ImportError(
                "BeautifulSoup4 is required for HTML image extraction. "
                "Install with: pip install beautifulsoup4"
            )
        
        super().__init__(max_image_size, min_image_size)
        self.download_external = download_external
        self.timeout = timeout
    
    def extract_images(self, file_path: str) -> List[ExtractedImage]:
        """
        Extract images from HTML file.
        
        Args:
            file_path: Path to HTML file
            
        Returns:
            List of extracted images with metadata
        """
        path = Path(file_path)
        
        if path.suffix.lower() not in ['.html', '.htm']:
            return []
        
        extracted_images = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find all img tags
            img_tags = soup.find_all('img')
            
            for img_index, img_tag in enumerate(img_tags):
                try:
                    src = img_tag.get('src')
                    if not src:
                        continue
                    
                    # Handle data URLs
                    if src.startswith('data:image'):
                        image = self._load_data_url(src)
                    # Handle local files
                    elif not src.startswith(('http://', 'https://')):
                        image_path = path.parent / src
                        if image_path.exists():
                            image = Image.open(image_path)
                        else:
                            continue
                    # Handle external URLs
                    elif self.download_external:
                        image = self._download_image(src)
                    else:
                        continue
                    
                    if image is None:
                        continue
                    
                    # Convert to RGB if needed
                    if image.mode not in ('RGB', 'L'):
                        image = image.convert('RGB')
                    
                    width, height = image.size
                    
                    # Filter small images
                    if not self._should_include_image(width, height):
                        continue
                    
                    # Resize if needed
                    image = self._resize_if_needed(image)
                    
                    # Extract metadata
                    alt_text = img_tag.get('alt', '')
                    caption = self._extract_caption(img_tag)
                    context = self._extract_context(img_tag)
                    
                    metadata = ImageMetadata(
                        source_file=str(path),
                        page_number=None,
                        image_index=img_index,
                        width=image.size[0],
                        height=image.size[1],
                        format=image.format or 'UNKNOWN',
                        caption=caption,
                        alt_text=alt_text if alt_text else None,
                        position=None,
                        context=context
                    )
                    
                    extracted_images.append(
                        ExtractedImage(image=image, metadata=metadata)
                    )
                    
                except Exception as e:
                    print(f"Error extracting image {img_index}: {e}")
                    continue
            
        except Exception as e:
            print(f"Error processing HTML {file_path}: {e}")
        
        return extracted_images
    
    def _load_data_url(self, data_url: str):
        """Load image from data URL."""
        try:
            import base64
            
            # Parse data URL
            header, data = data_url.split(',', 1)
            image_data = base64.b64decode(data)
            
            return Image.open(io.BytesIO(image_data))
        except Exception:
            return None
    
    def _download_image(self, url: str):
        """Download image from URL."""
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            return Image.open(io.BytesIO(response.content))
        except Exception:
            return None
    
    def _extract_caption(self, img_tag) -> str:
        """Extract caption from figure or nearby elements."""
        try:
            # Check if image is in a figure with figcaption
            figure = img_tag.find_parent('figure')
            if figure:
                figcaption = figure.find('figcaption')
                if figcaption:
                    return figcaption.get_text(strip=True)
            
            # Check for title attribute
            title = img_tag.get('title')
            if title:
                return title
            
            return None
        except Exception:
            return None
    
    def _extract_context(self, img_tag, context_chars: int = 500) -> str:
        """Extract surrounding text context."""
        try:
            context_parts = []
            
            # Get parent element
            parent = img_tag.find_parent(['p', 'div', 'section', 'article'])
            if parent:
                text = parent.get_text(strip=True)
                if text and len(text) > 0:
                    context_parts.append(text)
            
            context = " ".join(context_parts)
            if len(context) > context_chars:
                context = context[:context_chars] + "..."
            
            return context if context else None
        except Exception:
            return None
    
    @classmethod
    def is_supported(cls, file_path: str) -> bool:
        """Check if file is HTML."""
        return Path(file_path).suffix.lower() in ['.html', '.htm']
