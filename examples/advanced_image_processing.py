"""
Advanced image processing examples.

This example demonstrates caching, preprocessing, and parallel processing.
"""

from flip.embedding.image.cache import ImageEmbeddingCache
from flip.embedding.image.preprocessing import ImagePreprocessor, ImageAugmentor
from flip.embedding.image.parallel import ParallelImageProcessor, ProgressTracker
from flip.embedding.image.factory import ImageEmbedderFactory
from flip.document.image_factory import ImageExtractorFactory
from pathlib import Path


def example_image_caching():
    """Demonstrate image embedding caching."""
    print("=" * 60)
    print("Example 1: Image Embedding Cache")
    print("=" * 60)
    
    # Initialize cache
    cache = ImageEmbeddingCache(cache_dir="./.image_cache", max_size_mb=100)
    
    # Initialize embedder
    embedder = ImageEmbedderFactory.create("sentence-transformers-clip")
    
    # Load sample image
    image_path = "./sample_image.jpg"
    
    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        return
    
    from PIL import Image
    image = Image.open(image_path)
    
    # First embedding (not cached)
    import time
    
    print("\nFirst embedding (no cache)...")
    start = time.time()
    embedding1 = embedder.embed_image(image)
    time1 = time.time() - start
    
    # Cache the embedding
    cache.put(image, embedder.model_name, embedding1)
    
    # Second embedding (from cache)
    print("Second embedding (from cache)...")
    start = time.time()
    embedding2 = cache.get(image, embedder.model_name)
    time2 = time.time() - start
    
    print(f"\nTime without cache: {time1:.3f}s")
    print(f"Time with cache: {time2:.3f}s")
    print(f"Speedup: {time1/time2:.1f}x")
    
    # Show cache stats
    stats = cache.get_stats()
    print(f"\nCache stats:")
    print(f"  Entries: {stats['total_entries']}")
    print(f"  Size: {stats['total_size_mb']:.2f} MB")
    print(f"  Usage: {stats['usage_percent']:.1f}%")


def example_image_preprocessing():
    """Demonstrate image preprocessing."""
    print("\n" + "=" * 60)
    print("Example 2: Image Preprocessing")
    print("=" * 60)
    
    from PIL import Image
    
    # Load image
    image_path = "./sample_image.jpg"
    
    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        return
    
    image = Image.open(image_path)
    print(f"\nOriginal size: {image.size}")
    
    # Create preprocessor
    preprocessor = ImagePreprocessor(
        target_size=(512, 512),
        normalize=True,
        enhance_contrast=True,
        denoise=True
    )
    
    # Preprocess image
    processed = preprocessor.preprocess(image)
    print(f"Processed size: {processed.size}")
    
    # Save processed image
    processed.save("./processed_image.jpg")
    print("Saved processed image to ./processed_image.jpg")


def example_image_augmentation():
    """Demonstrate image augmentation."""
    print("\n" + "=" * 60)
    print("Example 3: Image Augmentation")
    print("=" * 60)
    
    from PIL import Image
    
    image_path = "./sample_image.jpg"
    
    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        return
    
    image = Image.open(image_path)
    
    # Apply augmentations
    augmentor = ImageAugmentor()
    
    rotated = augmentor.rotate(image, 45)
    flipped = augmentor.flip_horizontal(image)
    brightened = augmentor.adjust_brightness(image, 1.5)
    
    print("\nGenerated augmented images:")
    print("  - Rotated 45 degrees")
    print("  - Flipped horizontally")
    print("  - Brightened 1.5x")
    
    # Save augmented images
    rotated.save("./augmented_rotated.jpg")
    flipped.save("./augmented_flipped.jpg")
    brightened.save("./augmented_bright.jpg")
    print("\nSaved augmented images")


def example_parallel_processing():
    """Demonstrate parallel image processing."""
    print("\n" + "=" * 60)
    print("Example 4: Parallel Processing")
    print("=" * 60)
    
    # Get list of PDF files
    pdf_files = list(Path(".").glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in current directory")
        return
    
    print(f"\nFound {len(pdf_files)} PDF files")
    
    # Create parallel processor
    processor = ParallelImageProcessor(max_workers=4)
    
    # Extract images in parallel
    print("\nExtracting images in parallel...")
    
    def extract_func(file_path):
        return ImageExtractorFactory.extract_images(str(file_path))
    
    results = processor.extract_images_parallel(
        [str(f) for f in pdf_files],
        extract_func,
        show_progress=True
    )
    
    total_images = sum(len(imgs) for imgs in results if imgs)
    print(f"\nExtracted {total_images} total images from {len(pdf_files)} files")


def example_progress_tracking():
    """Demonstrate progress tracking."""
    print("\n" + "=" * 60)
    print("Example 5: Progress Tracking")
    print("=" * 60)
    
    import time
    
    # Simulate processing
    items = list(range(100))
    
    with ProgressTracker(len(items), "Processing items") as tracker:
        for item in items:
            # Simulate work
            time.sleep(0.01)
            tracker.update(1)
    
    print("\nProgress tracking complete!")


def example_complete_pipeline():
    """Complete pipeline with all optimizations."""
    print("\n" + "=" * 60)
    print("Example 6: Complete Optimized Pipeline")
    print("=" * 60)
    
    # Initialize components
    cache = ImageEmbeddingCache()
    preprocessor = ImagePreprocessor(target_size=(512, 512))
    embedder = ImageEmbedderFactory.create("sentence-transformers-clip")
    
    # Extract images
    pdf_path = "./sample.pdf"
    
    if not Path(pdf_path).exists():
        print(f"PDF not found: {pdf_path}")
        return
    
    print(f"\nProcessing {pdf_path}...")
    
    # Extract
    images = ImageExtractorFactory.extract_images(pdf_path)
    print(f"Extracted {len(images)} images")
    
    if not images:
        return
    
    # Preprocess
    print("Preprocessing images...")
    pil_images = [preprocessor.preprocess(img.image) for img in images]
    
    # Generate embeddings with caching
    print("Generating embeddings (with cache)...")
    embeddings = []
    
    for img in pil_images:
        # Check cache first
        cached = cache.get(img, embedder.model_name)
        
        if cached:
            embeddings.append(cached)
        else:
            # Generate and cache
            emb = embedder.embed_image(img)
            cache.put(img, embedder.model_name, emb)
            embeddings.append(emb)
    
    print(f"Generated {len(embeddings)} embeddings")
    
    # Show cache stats
    stats = cache.get_stats()
    print(f"\nCache efficiency: {stats['total_entries']} cached embeddings")


def main():
    """Run all advanced examples."""
    print("\n⚡ Flip SDK - Advanced Image Processing Examples\n")
    
    try:
        example_image_caching()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_image_preprocessing()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_image_augmentation()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_parallel_processing()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_progress_tracking()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_complete_pipeline()
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n✅ Advanced image processing examples complete!")


if __name__ == "__main__":
    main()
