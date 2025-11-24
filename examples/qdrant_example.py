"""
Example using Qdrant vector store with Flip SDK.

This example demonstrates how to use Qdrant as the vector store backend.
Qdrant supports both local and cloud deployments.
"""

from flip import Flip, FlipConfig
import os

# Set your Qdrant credentials (for cloud)
os.environ["QDRANT_URL"] = "https://your-cluster.qdrant.io"
os.environ["QDRANT_API_KEY"] = "your-qdrant-api-key"


def example_local_qdrant():
    """Basic Qdrant usage with local instance."""
    print("=" * 60)
    print("Example 1: Local Qdrant Usage")
    print("=" * 60)
    
    config = FlipConfig(
        vector_store="qdrant",
        
        # Local Qdrant configuration
        qdrant_host="localhost",
        qdrant_port=6333,
        
        # Use sentence transformers for local embeddings
        embedding_provider="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2"
    )
    
    flip = Flip(directory="./sample_docs", config=config)
    
    # Query
    response = flip.query("What is artificial intelligence?")
    print(f"\nAnswer: {response.answer}")
    print(f"Sources: {len(response.citations)}")


def example_cloud_qdrant():
    """Qdrant Cloud usage."""
    print("\n" + "=" * 60)
    print("Example 2: Qdrant Cloud")
    print("=" * 60)
    
    config = FlipConfig(
        vector_store="qdrant",
        
        # Cloud Qdrant configuration
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        
        embedding_provider="sentence-transformers"
    )
    
    flip = Flip(directory="./sample_docs", config=config)
    
    stats = flip.get_stats()
    print(f"\nVectors: {stats['chunk_count']}")
    print(f"Collection: {stats.get('collection_name', 'N/A')}")


def example_qdrant_with_filtering():
    """Qdrant with payload filtering."""
    print("\n" + "=" * 60)
    print("Example 3: Payload Filtering")
    print("=" * 60)
    
    config = FlipConfig(
        vector_store="qdrant",
        qdrant_host="localhost",
        qdrant_port=6333,
        embedding_provider="sentence-transformers"
    )
    
    flip = Flip(directory="./sample_docs", config=config)
    
    # Query with metadata filter
    response = flip.query(
        "What is machine learning?",
        # filter={"category": "AI"}  # Example filter
    )
    
    print(f"\nFiltered results: {len(response.context_chunks)}")


def example_qdrant_snapshots():
    """Create and manage Qdrant snapshots."""
    print("\n" + "=" * 60)
    print("Example 4: Snapshots")
    print("=" * 60)
    
    from flip.vector_store.qdrant import QdrantVectorStore
    
    store = QdrantVectorStore(
        collection_name="flip-backup",
        host="localhost",
        port=6333,
        dimension=384
    )
    
    # Add some data
    from flip.embedding.sentence_transformers import SentenceTransformerEmbedder
    
    embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
    texts = ["Document 1", "Document 2", "Document 3"]
    embeddings = embedder.embed_batch(texts)
    ids = ["doc1", "doc2", "doc3"]
    
    store.add(ids, embeddings, texts)
    
    # Create snapshot
    snapshot_name = store.create_snapshot()
    print(f"\nSnapshot created: {snapshot_name}")


def example_qdrant_health_monitoring():
    """Monitor Qdrant health and stats."""
    print("\n" + "=" * 60)
    print("Example 5: Health Monitoring")
    print("=" * 60)
    
    from flip.vector_store.qdrant import QdrantVectorStore
    
    store = QdrantVectorStore(
        collection_name="flip-health",
        host="localhost",
        port=6333,
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
    print(f"Indexed Vectors: {stats.metadata.get('indexed_vectors_count', 'N/A')}")
    print(f"Status: {stats.metadata.get('status', 'N/A')}")


def example_qdrant_batch_operations():
    """Batch operations with Qdrant."""
    print("\n" + "=" * 60)
    print("Example 6: Batch Operations")
    print("=" * 60)
    
    from flip.vector_store.qdrant import QdrantVectorStore
    from flip.embedding.sentence_transformers import SentenceTransformerEmbedder
    
    # Initialize
    embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
    store = QdrantVectorStore(
        collection_name="flip-batch",
        host="localhost",
        port=6333,
        dimension=384
    )
    
    # Prepare data
    texts = [f"Document {i} about AI and machine learning." for i in range(100)]
    embeddings = embedder.embed_batch(texts)
    ids = [f"doc_{i}" for i in range(100)]
    
    # Batch add
    import time
    start = time.time()
    store.batch_add(
        ids=ids,
        embeddings=embeddings,
        texts=texts,
        batch_size=50
    )
    duration = time.time() - start
    
    print(f"\nAdded {len(texts)} vectors in {duration:.2f}s")
    print(f"Rate: {len(texts)/duration:.1f} vectors/sec")


def example_qdrant_distance_metrics():
    """Different distance metrics."""
    print("\n" + "=" * 60)
    print("Example 7: Distance Metrics")
    print("=" * 60)
    
    from flip.vector_store.qdrant import QdrantVectorStore
    
    # Cosine similarity (default)
    store_cosine = QdrantVectorStore(
        collection_name="flip-cosine",
        host="localhost",
        port=6333,
        dimension=384,
        distance="cosine"
    )
    
    # Euclidean distance
    store_euclidean = QdrantVectorStore(
        collection_name="flip-euclidean",
        host="localhost",
        port=6333,
        dimension=384,
        distance="euclidean"
    )
    
    # Dot product
    store_dot = QdrantVectorStore(
        collection_name="flip-dot",
        host="localhost",
        port=6333,
        dimension=384,
        distance="dot"
    )
    
    print("\nCreated collections with different distance metrics:")
    print(f"- Cosine: {store_cosine.distance_metric}")
    print(f"- Euclidean: {store_euclidean.distance_metric}")
    print(f"- Dot Product: {store_dot.distance_metric}")


def main():
    """Run all Qdrant examples."""
    print("\n🚀 Flip SDK - Qdrant Examples\n")
    
    try:
        example_local_qdrant()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_cloud_qdrant()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_qdrant_with_filtering()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_qdrant_snapshots()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_qdrant_health_monitoring()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_qdrant_batch_operations()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_qdrant_distance_metrics()
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n✅ Qdrant examples complete!")


if __name__ == "__main__":
    main()
