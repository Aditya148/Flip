"""Multimodal pipeline for processing text and images."""

from typing import List, Optional, Dict, Any
from pathlib import Path

from flip.document.chunks import DocumentChunk, ImageChunk
from flip.document.loader_with_images import DocumentLoaderWithImages
from flip.embedding.image.factory import ImageEmbedderFactory
from flip.core.config import FlipConfig


class MultimodalPipeline:
    """Pipeline for processing both text and images."""
    
    def __init__(self, config: FlipConfig):
        """
        Initialize multimodal pipeline.
        
        Args:
            config: Flip configuration
        """
        self.config = config
        self.loader = DocumentLoaderWithImages(config)
        
        # Initialize image embedder if enabled
        self.image_embedder = None
        if config.enable_image_extraction:
            try:
                self.image_embedder = ImageEmbedderFactory.create(
                    provider=config.image_embedding_provider,
                    model_name=config.image_embedding_model
                )
                print(f"Initialized image embedder: {config.image_embedding_provider}")
            except Exception as e:
                print(f"Warning: Could not initialize image embedder: {e}")
                print("Image extraction will be disabled.")
                self.config.enable_image_extraction = False
    
    def process_document(self, file_path: str) -> tuple[List[DocumentChunk], List[ImageChunk]]:
        """
        Process document to extract text and image chunks.
        
        Args:
            file_path: Path to document
            
        Returns:
            Tuple of (text_chunks, image_chunks)
        """
        # Load document
        text_chunks, image_chunks = self.loader.load_document(file_path)
        
        # Generate embeddings for images
        if image_chunks and self.image_embedder:
            image_chunks = self._embed_images(image_chunks)
        
        return text_chunks, image_chunks
    
    def _embed_images(self, image_chunks: List[ImageChunk]) -> List[ImageChunk]:
        """
        Generate embeddings for image chunks.
        
        Args:
            image_chunks: List of image chunks
            
        Returns:
            Image chunks with embeddings
        """
        if not self.image_embedder:
            return image_chunks
        
        try:
            # Extract PIL images
            images = [chunk.image for chunk in image_chunks]
            
            # Generate embeddings in batch
            embeddings = self.image_embedder.embed_images_batch(images)
            
            # Assign embeddings to chunks
            for chunk, embedding in zip(image_chunks, embeddings):
                chunk.embedding = embedding
            
            print(f"Generated embeddings for {len(image_chunks)} images")
            
        except Exception as e:
            print(f"Error generating image embeddings: {e}")
        
        return image_chunks
    
    def process_directory(self, directory: str) -> tuple[List[DocumentChunk], List[ImageChunk]]:
        """
        Process all documents in a directory.
        
        Args:
            directory: Path to directory
            
        Returns:
            Tuple of (all_text_chunks, all_image_chunks)
        """
        all_text_chunks = []
        all_image_chunks = []
        
        dir_path = Path(directory)
        
        # Process all supported files
        for file_path in dir_path.rglob("*"):
            if file_path.is_file():
                try:
                    text_chunks, image_chunks = self.process_document(str(file_path))
                    all_text_chunks.extend(text_chunks)
                    all_image_chunks.extend(image_chunks)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        return all_text_chunks, all_image_chunks
