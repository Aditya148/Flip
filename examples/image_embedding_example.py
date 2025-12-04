"""
Example demonstrating image embedding with Flip SDK.

This example shows how to extract images from documents and generate embeddings.
"""

from flip.document.image_factory import ImageExtractorFactory
from flip.embedding.image.factory import ImageEmbedderFactory
from pathlib import Path


def example_extract_images_from_pdf():
    """Extract images from a PDF file."""
    print("=" * 60)
    print("Example 1: Extract Images from PDF")
    print("=" * 60)
    
    # Extract images from PDF
    pdf_path = "./sample.pdf"
    
    if not Path(pdf_path).exists():
        print(f"PDF file not found: {pdf_path}")
        print("Skipping this example...")
        return
    
    images = ImageExtractorFactory.extract_images(
        pdf_path,
        max_image_size=1024,
        min_image_size=100
    )
    
    print(f"\nExtracted {len(images)} images from PDF")
    
    for i, img in enumerate(images):
        print(f"\nImage {i + 1}:")
        print(f"  Page: {img.metadata.page_number}")
        print(f"  Size: {img.metadata.width}x{img.metadata.height}")
        print(f"  Format: {img.metadata.format}")
        if img.metadata.context:
            print(f"  Context: {img.metadata.context[:100]}...")


def example_embed_images_huggingface():
    """Generate embeddings using HuggingFace CLIP."""
    print("\n" + "=" * 60)
    print("Example 2: Image Embeddings with HuggingFace CLIP")
    print("=" * 60)
    
    try:
        # Create embedder
        embedder = ImageEmbedderFactory.create(
            provider="huggingface-clip",
            model_name="clip-vit-base-patch32"
        )
        
        print(f"\nEmbedder: {embedder.provider_name}")
        print(f"Model: {embedder.model_name}")
        print(f"Dimension: {embedder.dimension}")
        
        # Load sample image
        image_path = "./sample_image.jpg"
        
        if not Path(image_path).exists():
            print(f"\nImage file not found: {image_path}")
            print("Skipping embedding generation...")
            return
        
        # Generate embedding
        print(f"\nGenerating embedding for {image_path}...")
        embedding = embedder.embed_image(image_path)
        
        print(f"Embedding shape: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure transformers and torch are installed:")
        print("  pip install transformers torch")


def example_embed_images_sentence_transformers():
    """Generate embeddings using Sentence Transformers CLIP."""
    print("\n" + "=" * 60)
    print("Example 3: Image Embeddings with Sentence Transformers")
    print("=" * 60)
    
    try:
        # Create embedder
        embedder = ImageEmbedderFactory.create(
            provider="sentence-transformers-clip",
            model_name="clip-vit-b-32"
        )
        
        print(f"\nEmbedder: {embedder.provider_name}")
        print(f"Model: {embedder.model_name}")
        print(f"Dimension: {embedder.dimension}")
        
        # Load sample image
        image_path = "./sample_image.jpg"
        
        if not Path(image_path).exists():
            print(f"\nImage file not found: {image_path}")
            print("Skipping embedding generation...")
            return
        
        # Generate embedding
        print(f"\nGenerating embedding for {image_path}...")
        embedding = embedder.embed_image(image_path)
        
        print(f"Embedding shape: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure sentence-transformers is installed:")
        print("  pip install sentence-transformers")


def example_batch_embedding():
    """Generate embeddings for multiple images in batch."""
    print("\n" + "=" * 60)
    print("Example 4: Batch Image Embedding")
    print("=" * 60)
    
    try:
        # Create embedder
        embedder = ImageEmbedderFactory.create(
            provider="sentence-transformers-clip"
        )
        
        # Extract images from PDF
        pdf_path = "./sample.pdf"
        
        if not Path(pdf_path).exists():
            print(f"PDF file not found: {pdf_path}")
            print("Skipping this example...")
            return
        
        images = ImageExtractorFactory.extract_images(pdf_path)
        
        if not images:
            print("No images found in PDF")
            return
        
        print(f"\nGenerating embeddings for {len(images)} images...")
        
        # Get PIL images
        pil_images = [img.image for img in images]
        
        # Generate embeddings in batch
        embeddings = embedder.embed_images_batch(pil_images)
        
        print(f"Generated {len(embeddings)} embeddings")
        print(f"Embedding dimension: {len(embeddings[0])}")
        
        # Show similarity between first two images
        if len(embeddings) >= 2:
            import numpy as np
            
            emb1 = np.array(embeddings[0])
            emb2 = np.array(embeddings[1])
            
            similarity = np.dot(emb1, emb2)
            print(f"\nSimilarity between image 1 and 2: {similarity:.3f}")
        
    except Exception as e:
        print(f"Error: {e}")


def example_list_providers():
    """List available image embedding providers."""
    print("\n" + "=" * 60)
    print("Example 5: List Available Providers")
    print("=" * 60)
    
    providers = ImageEmbedderFactory.list_providers()
    
    print(f"\nAvailable image embedding providers:")
    for provider in providers:
        print(f"  - {provider}")
    
    if not providers:
        print("  No providers available. Install dependencies:")
        print("    pip install transformers torch")
        print("    pip install sentence-transformers")


def main():
    """Run all examples."""
    print("\n🖼️  Flip SDK - Image Embedding Examples\n")
    
    try:
        example_extract_images_from_pdf()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_embed_images_huggingface()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_embed_images_sentence_transformers()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_batch_embedding()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_list_providers()
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n✅ Image embedding examples complete!")


if __name__ == "__main__":
    main()
