"""Factory for creating appropriate image extractors."""

from typing import Optional, List
from pathlib import Path

from flip.document.image_extractor import BaseImageExtractor, ExtractedImage


class ImageExtractorFactory:
    """Factory for creating image extractors based on file type."""
    
    _extractors = {}
    
    @classmethod
    def _lazy_load_extractors(cls):
        """Lazy load image extractors."""
        if "standalone" not in cls._extractors:
            from flip.document.image_extractor import StandaloneImageLoader
            cls._extractors["standalone"] = StandaloneImageLoader
        
        if "pdf" not in cls._extractors:
            try:
                from flip.document.pdf_image_extractor import PDFImageExtractor
                cls._extractors["pdf"] = PDFImageExtractor
            except ImportError:
                pass
        
        if "docx" not in cls._extractors:
            try:
                from flip.document.docx_image_extractor import DOCXImageExtractor
                cls._extractors["docx"] = DOCXImageExtractor
            except ImportError:
                pass
        
        if "html" not in cls._extractors:
            try:
                from flip.document.html_image_extractor import HTMLImageExtractor
                cls._extractors["html"] = HTMLImageExtractor
            except ImportError:
                pass
    
    @classmethod
    def create(
        cls,
        file_path: str,
        max_image_size: int = 1024,
        min_image_size: int = 100,
        **kwargs
    ) -> Optional[BaseImageExtractor]:
        """
        Create appropriate image extractor for the file.
        
        Args:
            file_path: Path to the file
            max_image_size: Maximum dimension for extracted images
            min_image_size: Minimum dimension to consider
            **kwargs: Additional extractor-specific arguments
            
        Returns:
            Image extractor instance or None if not supported
        """
        cls._lazy_load_extractors()
        
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        # Determine extractor type
        if suffix == '.pdf' and 'pdf' in cls._extractors:
            return cls._extractors['pdf'](max_image_size, min_image_size, **kwargs)
        
        elif suffix in ['.docx', '.doc'] and 'docx' in cls._extractors:
            return cls._extractors['docx'](max_image_size, min_image_size, **kwargs)
        
        elif suffix in ['.html', '.htm'] and 'html' in cls._extractors:
            return cls._extractors['html'](max_image_size, min_image_size, **kwargs)
        
        elif suffix in {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff'} and 'standalone' in cls._extractors:
            return cls._extractors['standalone'](max_image_size, min_image_size)
        
        return None
    
    @classmethod
    def extract_images(
        cls,
        file_path: str,
        max_image_size: int = 1024,
        min_image_size: int = 100,
        **kwargs
    ) -> List[ExtractedImage]:
        """
        Extract images from a file using the appropriate extractor.
        
        Args:
            file_path: Path to the file
            max_image_size: Maximum dimension for extracted images
            min_image_size: Minimum dimension to consider
            **kwargs: Additional extractor-specific arguments
            
        Returns:
            List of extracted images
        """
        extractor = cls.create(file_path, max_image_size, min_image_size, **kwargs)
        
        if extractor is None:
            return []
        
        return extractor.extract_images(file_path)
    
    @classmethod
    def is_supported(cls, file_path: str) -> bool:
        """
        Check if file type is supported for image extraction.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if supported, False otherwise
        """
        cls._lazy_load_extractors()
        
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        supported_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff'}
        
        if 'pdf' in cls._extractors:
            supported_extensions.add('.pdf')
        
        if 'docx' in cls._extractors:
            supported_extensions.update(['.docx', '.doc'])
        
        if 'html' in cls._extractors:
            supported_extensions.update(['.html', '.htm'])
        
        return suffix in supported_extensions
