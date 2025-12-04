"""Document loader with image extraction support."""

from typing import List, Optional
from pathlib import Path

from flip.document.chunks import DocumentChunk, ImageChunk
from flip.document.image_factory import ImageExtractorFactory
from flip.core.config import FlipConfig


class DocumentLoaderWithImages:
    """Enhanced document loader that can extract images."""
    
    def __init__(self, config: FlipConfig):
        """
        Initialize document loader.
        
        Args:
            config: Flip configuration
        """
        self.config = config
        self.enable_images = config.enable_image_extraction
    
    def load_document(self, file_path: str) -> tuple[List[DocumentChunk], List[ImageChunk]]:
        """
        Load document and extract both text and images.
        
        Args:
            file_path: Path to document file
            
        Returns:
            Tuple of (text_chunks, image_chunks)
        """
        text_chunks = []
        image_chunks = []
        
        # Load text chunks (existing functionality)
        # This would call the existing document loading logic
        text_chunks = self._load_text_chunks(file_path)
        
        # Extract images if enabled
        if self.enable_images and ImageExtractorFactory.is_supported(file_path):
            image_chunks = self._extract_image_chunks(file_path)
        
        return text_chunks, image_chunks
    
    def _load_text_chunks(self, file_path: str) -> List[DocumentChunk]:
        """
        Load text chunks from document.
        
        This is a placeholder - in production, this would call
        the existing document loading and chunking logic.
        
        Args:
            file_path: Path to document
            
        Returns:
            List of text chunks
        """
        # Placeholder - integrate with existing loader
        return []
    
    def _extract_image_chunks(self, file_path: str) -> List[ImageChunk]:
        """
        Extract image chunks from document.
        
        Args:
            file_path: Path to document
            
        Returns:
            List of image chunks
        """
        image_chunks = []
        
        try:
            # Extract images
            extracted_images = ImageExtractorFactory.extract_images(
                file_path,
                max_image_size=self.config.max_image_size,
                min_image_size=self.config.min_image_size,
                extract_context=self.config.extract_image_context
            )
            
            # Convert to ImageChunk objects
            for i, extracted_img in enumerate(extracted_images):
                chunk_id = f"{Path(file_path).stem}_image_{i}"
                
                # Prepare metadata
                metadata = extracted_img.metadata.to_dict()
                metadata["chunk_type"] = "image"
                
                image_chunk = ImageChunk(
                    image=extracted_img.image,
                    embedding=None,  # Will be filled later
                    metadata=metadata,
                    chunk_id=chunk_id
                )
                
                image_chunks.append(image_chunk)
        
        except Exception as e:
            print(f"Error extracting images from {file_path}: {e}")
        
        return image_chunks


def integrate_images_into_existing_loader(existing_loader, config: FlipConfig):
    """
    Helper function to add image extraction to existing document loader.
    
    Args:
        existing_loader: Existing document loader instance
        config: Flip configuration
        
    Returns:
        Enhanced loader with image support
    """
    # This would wrap or extend the existing loader
    # to add image extraction capabilities
    pass
