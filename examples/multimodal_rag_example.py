"""
Complete multimodal RAG example with Flip SDK.

This example demonstrates end-to-end multimodal RAG with image extraction and search.
"""

from flip import Flip, FlipConfig
from pathlib import Path


def example_basic_multimodal_rag():
    """Basic multimodal RAG with image extraction."""
    print("=" * 60)
    print("Example 1: Basic Multimodal RAG")
    print("=" * 60)
    
    # Configure Flip with image support
    config = FlipConfig(
        # Enable image processing
        enable_image_extraction=True,
        image_embedding_provider="sentence-transformers-clip",
        
        # Use local models for demo
        embedding_provider="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        
        # Vector store
        vector_store="chroma"
    )
    
    # Initialize Flip
    flip = Flip(directory="./docs_with_images", config=config)
    
    # Query for text
    response = flip.query("What is machine learning?")
    print(f"\nAnswer: {response.answer}")
    
    # Check if response has images
    if hasattr(response, 'image_results') and response.image_results:
        print(f"\nFound {len(response.image_results)} related images")
        for img in response.image_results:
            print(f"  - {img.caption or 'Untitled'} (score: {img.score:.3f})")


def example_image_search():
    """Search for images using text query."""
    print("\n" + "=" * 60)
    print("Example 2: Text-to-Image Search")
    print("=" * 60)
    
    config = FlipConfig(
        enable_image_extraction=True,
        enable_multimodal_search=True,
        image_embedding_provider="sentence-transformers-clip",
        embedding_provider="sentence-transformers"
    )
    
    flip = Flip(directory="./docs_with_images", config=config)
    
    # Search for images
    query = "Show me diagrams about neural networks"
    response = flip.query(query)
    
    if hasattr(response, 'image_results'):
        print(f"\nFound {len(response.image_results)} images for: '{query}'")
        
        # Save images
        if response.image_results:
            response.save_images("./search_results")
            print("Images saved to ./search_results/")


def example_pdf_with_images():
    """Extract and process images from PDF."""
    print("\n" + "=" * 60)
    print("Example 3: PDF with Images")
    print("=" * 60)
    
    from flip.document.image_factory import ImageExtractorFactory
    from flip.embedding.image.factory import ImageEmbedderFactory
    
    pdf_path = "./research_paper.pdf"
    
    if not Path(pdf_path).exists():
        print(f"PDF not found: {pdf_path}")
        return
    
    # Extract images
    print(f"\nExtracting images from {pdf_path}...")
    images = ImageExtractorFactory.extract_images(pdf_path)
    
    print(f"Extracted {len(images)} images")
    
    # Generate embeddings
    if images:
        embedder = ImageEmbedderFactory.create("sentence-transformers-clip")
        
        pil_images = [img.image for img in images]
        embeddings = embedder.embed_images_batch(pil_images)
        
        print(f"Generated {len(embeddings)} embeddings")
        print(f"Embedding dimension: {len(embeddings[0])}")
        
        # Show image info
        for i, img in enumerate(images):
            print(f"\nImage {i + 1}:")
            print(f"  Page: {img.metadata.page_number}")
            print(f"  Size: {img.metadata.width}x{img.metadata.height}")
            if img.metadata.context:
                print(f"  Context: {img.metadata.context[:100]}...")


def example_multimodal_weights():
    """Adjust weights for text vs image results."""
    print("\n" + "=" * 60)
    print("Example 4: Adjusting Multimodal Weights")
    print("=" * 60)
    
    # Favor text results
    config_text = FlipConfig(
        enable_image_extraction=True,
        enable_multimodal_search=True,
        text_weight=0.8,
        image_weight=0.2
    )
    
    # Favor image results
    config_images = FlipConfig(
        enable_image_extraction=True,
        enable_multimodal_search=True,
        text_weight=0.2,
        image_weight=0.8
    )
    
    print("\nConfiguration 1: Text-focused (80% text, 20% images)")
    print("Configuration 2: Image-focused (20% text, 80% images)")
    print("\nUse different configs based on your use case!")


def example_image_quality_settings():
    """Configure image extraction quality."""
    print("\n" + "=" * 60)
    print("Example 5: Image Quality Settings")
    print("=" * 60)
    
    # High quality, larger images
    config_hq = FlipConfig(
        enable_image_extraction=True,
        max_image_size=2048,  # Larger images
        min_image_size=200,   # Filter out small images
        image_quality=95      # High JPEG quality
    )
    
    # Fast processing, smaller images
    config_fast = FlipConfig(
        enable_image_extraction=True,
        max_image_size=512,   # Smaller images
        min_image_size=50,    # Include more images
        image_quality=75      # Lower quality, faster
    )
    
    print("\nHigh Quality Config:")
    print(f"  Max size: {config_hq.max_image_size}px")
    print(f"  Min size: {config_hq.min_image_size}px")
    print(f"  Quality: {config_hq.image_quality}%")
    
    print("\nFast Processing Config:")
    print(f"  Max size: {config_fast.max_image_size}px")
    print(f"  Min size: {config_fast.min_image_size}px")
    print(f"  Quality: {config_fast.image_quality}%")


def main():
    """Run all multimodal examples."""
    print("\n🖼️  Flip SDK - Multimodal RAG Examples\n")
    
    try:
        example_basic_multimodal_rag()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_image_search()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_pdf_with_images()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_multimodal_weights()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_image_quality_settings()
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n✅ Multimodal RAG examples complete!")
    print("\n📝 Note: To use multimodal features, install:")
    print("  pip install sentence-transformers")
    print("  pip install PyMuPDF python-docx beautifulsoup4")


if __name__ == "__main__":
    main()
