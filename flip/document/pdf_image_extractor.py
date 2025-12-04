"""PDF image extraction using PyMuPDF."""

from typing import List
from pathlib import Path
import io

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

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


class PDFImageExtractor(BaseImageExtractor):
    """Extract images from PDF files using PyMuPDF."""
    
    def __init__(self, max_image_size: int = 1024, min_image_size: int = 100, extract_context: bool = True):
        """
        Initialize PDF image extractor.
        
        Args:
            max_image_size: Maximum dimension for extracted images
            min_image_size: Minimum dimension to consider
            extract_context: Whether to extract surrounding text as context
        """
        if not PYMUPDF_AVAILABLE:
            raise ImportError(
                "PyMuPDF is required for PDF image extraction. "
                "Install with: pip install PyMuPDF"
            )
        
        super().__init__(max_image_size, min_image_size)
        self.extract_context = extract_context
    
    def extract_images(self, file_path: str) -> List[ExtractedImage]:
        """
        Extract images from PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of extracted images with metadata
        """
        path = Path(file_path)
        
        if path.suffix.lower() != '.pdf':
            return []
        
        extracted_images = []
        
        try:
            doc = fitz.open(file_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get images from page
                image_list = page.get_images(full=True)
                
                for img_index, img_info in enumerate(image_list):
                    try:
                        # Extract image
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        
                        if not base_image:
                            continue
                        
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # Convert to PIL Image
                        image = Image.open(io.BytesIO(image_bytes))
                        
                        # Convert to RGB if needed
                        if image.mode not in ('RGB', 'L'):
                            image = image.convert('RGB')
                        
                        width, height = image.size
                        
                        # Filter small images (likely icons/logos)
                        if not self._should_include_image(width, height):
                            continue
                        
                        # Resize if needed
                        image = self._resize_if_needed(image)
                        
                        # Extract context (surrounding text)
                        context = None
                        if self.extract_context:
                            context = self._extract_context(page, img_info)
                        
                        # Get image position
                        position = self._get_image_position(page, img_info)
                        
                        metadata = ImageMetadata(
                            source_file=str(path),
                            page_number=page_num + 1,  # 1-indexed
                            image_index=img_index,
                            width=image.size[0],
                            height=image.size[1],
                            format=image_ext.upper(),
                            caption=None,  # PDFs don't typically have captions
                            alt_text=None,
                            position=position,
                            context=context
                        )
                        
                        extracted_images.append(
                            ExtractedImage(image=image, metadata=metadata)
                        )
                        
                    except Exception as e:
                        print(f"Error extracting image {img_index} from page {page_num + 1}: {e}")
                        continue
            
            doc.close()
            
        except Exception as e:
            print(f"Error processing PDF {file_path}: {e}")
        
        return extracted_images
    
    def _extract_context(self, page, img_info, context_chars: int = 500) -> str:
        """
        Extract text context around the image.
        
        Args:
            page: PyMuPDF page object
            img_info: Image information tuple
            context_chars: Number of characters to extract
            
        Returns:
            Context text
        """
        try:
            # Get all text from the page
            text = page.get_text()
            
            # For now, return a snippet of page text
            # In a more sophisticated implementation, we'd extract text
            # specifically around the image position
            if len(text) > context_chars:
                return text[:context_chars] + "..."
            return text
            
        except Exception:
            return None
    
    def _get_image_position(self, page, img_info) -> dict:
        """
        Get image position on page.
        
        Args:
            page: PyMuPDF page object
            img_info: Image information tuple
            
        Returns:
            Dictionary with position information
        """
        try:
            # Get image rectangles
            xref = img_info[0]
            image_rects = page.get_image_rects(xref)
            
            if image_rects:
                rect = image_rects[0]
                return {
                    "x0": rect.x0,
                    "y0": rect.y0,
                    "x1": rect.x1,
                    "y1": rect.y1
                }
        except Exception:
            pass
        
        return None
    
    @classmethod
    def is_supported(cls, file_path: str) -> bool:
        """Check if file is a PDF."""
        return Path(file_path).suffix.lower() == '.pdf'
