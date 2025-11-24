"""
Example using FAISS vector store with Flip SDK.

This example demonstrates how to use FAISS as the vector store backend.
FAISS is a local, in-memory vector store that's excellent for fast similarity search.
"""

from flip import Flip, FlipConfig
import os


def example_local_faiss():
    """Basic FAISS usage."""
    print("=" * 60)
    print("Example 1: Local FAISS Usage")
    print("=" * 60)
    
    config = FlipConfig(
        vector_store="faiss",
        
        # FAISS configuration
        faiss_index_type="Flat",  # Exact search
        
        # Use sentence transformers for local embeddings
        embedding_provider="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2"
    )
    
    flip = Flip(directory="./sample_docs", config=config)
    
    # Query
    response = flip.query("What is artificial intelligence?")
    print(f"\nAnswer: {response.answer}")
    print(f"Sources: {len(response.citations)}")


def example_faiss_with_persistence():
    """FAISS with persistence."""
    print("\n" + "=" * 60)
    print("Example 2: FAISS with Persistence")
    print("=" * 60)
    
    from flip.vector_store.faiss import FAISSVectorStore
    from flip.embedding.sentence_transformers import SentenceTransformerEmbedder
    
    embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
    
    # Create store with persistence
    store = FAISSVectorStore(
        collection_name="flip_persistent",
        dimension=384,
        persist_directory="./faiss_data"
    )
    
    # Add data
    texts = ["Document 1", "Document 2", "Document 3"]
    embeddings = embedder.embed_batch(texts)
    ids = ["1", "2", "3"]
    
    store.add(ids, embeddings, texts)
    print(f"\nAdded {store.count()} vectors")
    print(f"Data persisted to: ./faiss_data")
    
    # Data will be automatically loaded next time


def example_faiss_index_types():
    """Different FAISS index types."""
    print("\n" + "=" * 60)
    print("Example 3: Index Types")
    print("=" * 60)
    
    from flip.vector_store.faiss import FAISSVectorStore
    
    # Flat - Exact search (most accurate, slower for large datasets)
    store_flat = FAISSVectorStore(
        collection_name="flip_flat",
        dimension=384,
        index_type="Flat"
    )
    
    # IVF - Inverted file index (faster, approximate)
    store_ivf = FAISSVectorStore(
        collection_name="flip_ivf",
        dimension=384,
        index_type="IVF"
    )
    
    # HNSW - Hierarchical Navigable Small World (best balance)
    store_hnsw = FAISSVectorStore(
        collection_name="flip_hnsw",
        dimension=384,
        index_type="HNSW"
    )
    
    print("\nCreated indices:")
    print(f"- Flat (exact): {store_flat.index_type}")
    print(f"- IVF (fast): {store_ivf.index_type}")
    print(f"- HNSW (balanced): {store_hnsw.index_type}")


def example_faiss_health_monitoring():
    """Monitor FAISS health and stats."""
    print("\n" + "=" * 60)
    print("Example 4: Health Monitoring")
    print("=" * 60)
    
    from flip.vector_store.faiss import FAISSVectorStore
    
    store = FAISSVectorStore(
        collection_name="flip_health",
        dimension=384
    )
    
    # Health check
    health = store.health_check()
    print(f"\nHealth Status: {health.status.value}")
    print(f"Latency: {health.latency_ms:.2f}ms")
    print(f"Message: {health.message}")
    
    # Get statistics
    stats = store.get_stats()
    print(f"\nTotal Vectors: {stats.total_vectors}")
    print(f"Dimension: {stats.dimension}")
    print(f"Index Type: {stats.metadata.get('index_type', 'N/A')}")
    print(f"GPU Enabled: {stats.metadata.get('use_gpu', False)}")


def example_faiss_batch_operations():
    """Batch operations with FAISS."""
    print("\n" + "=" * 60)
    print("Example 5: Batch Operations")
    print("=" * 60)
    
    from flip.vector_store.faiss import FAISSVectorStore
    from flip.embedding.sentence_transformers import SentenceTransformerEmbedder
    
    # Initialize
    embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
    store = FAISSVectorStore(
        collection_name="flip_batch",
        dimension=384
    )
    
    # Prepare data
    texts = [f"Document {i} about AI and machine learning." for i in range(1000)]
    embeddings = embedder.embed_batch(texts)
    ids = [str(i) for i in range(1000)]
    
    # Batch add
    import time
    start = time.time()
    store.batch_add(
        ids=ids,
        embeddings=embeddings,
        texts=texts,
        batch_size=100
    )
    duration = time.time() - start
    
    print(f"\nAdded {len(texts)} vectors in {duration:.2f}s")
    print(f"Rate: {len(texts)/duration:.1f} vectors/sec")
    print(f"FAISS is extremely fast for local operations!")


def example_faiss_filtering():
    """Search with metadata filtering."""
    print("\n" + "=" * 60)
    print("Example 6: Filtered Search")
    print("=" * 60)
    
    from flip.vector_store.faiss import FAISSVectorStore
    from flip.embedding.sentence_transformers import SentenceTransformerEmbedder
    
    embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
    store = FAISSVectorStore(
        collection_name="flip_filter",
        dimension=384
    )
    
    # Add data with metadata
    texts = ["AI document", "ML document", "DL document", "NLP document"]
    embeddings = embedder.embed_batch(texts)
    ids = ["1", "2", "3", "4"]
    metadatas = [
        {"category": "AI", "year": 2024},
        {"category": "ML", "year": 2024},
        {"category": "DL", "year": 2023},
        {"category": "NLP", "year": 2024}
    ]
    
    store.add(ids, embeddings, texts, metadatas)
    
    # Search with filter
    query_embedding = embedder.embed("artificial intelligence")
    results = store.filter_search(
        query_embedding,
        filters={"year": 2024},
        top_k=10
    )
    
    print(f"\nFiltered results (year=2024): {len(results)}")
    for result in results:
        print(f"  - {result.text} (score: {result.score:.3f})")


def example_faiss_gpu():
    """FAISS with GPU support."""
    print("\n" + "=" * 60)
    print("Example 7: GPU Support")
    print("=" * 60)
    
    from flip.vector_store.faiss import FAISSVectorStore
    
    # Try to use GPU if available
    store = FAISSVectorStore(
        collection_name="flip_gpu",
        dimension=384,
        use_gpu=True  # Will use GPU if available
    )
    
    stats = store.get_stats()
    gpu_enabled = stats.metadata.get('use_gpu', False)
    
    print(f"\nGPU Enabled: {gpu_enabled}")
    if gpu_enabled:
        print("Using GPU acceleration for faster search!")
    else:
        print("GPU not available, using CPU (still very fast!)")


def main():
    """Run all FAISS examples."""
    print("\n🚀 Flip SDK - FAISS Examples\n")
    
    try:
        example_local_faiss()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_faiss_with_persistence()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_faiss_index_types()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_faiss_health_monitoring()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_faiss_batch_operations()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_faiss_filtering()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_faiss_gpu()
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n✅ FAISS examples complete!")


if __name__ == "__main__":
    main()
