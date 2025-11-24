"""
Example using Milvus vector store with Flip SDK.

This example demonstrates how to use Milvus as the vector store backend.
"""

from flip import Flip, FlipConfig
import os


def example_local_milvus():
    """Basic Milvus usage with local instance."""
    print("=" * 60)
    print("Example 1: Local Milvus Usage")
    print("=" * 60)
    
    config = FlipConfig(
        vector_store="milvus",
        
        # Local Milvus configuration
        milvus_host="localhost",
        milvus_port=19530,
        
        # Use sentence transformers for local embeddings
        embedding_provider="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2"
    )
    
    flip = Flip(directory="./sample_docs", config=config)
    
    # Query
    response = flip.query("What is artificial intelligence?")
    print(f"\nAnswer: {response.answer}")
    print(f"Sources: {len(response.citations)}")


def example_milvus_index_types():
    """Different index types in Milvus."""
    print("\n" + "=" * 60)
    print("Example 2: Index Types")
    print("=" * 60)
    
    from flip.vector_store.milvus import MilvusVectorStore
    
    # IVF_FLAT - Good balance of speed and accuracy
    store_ivf = MilvusVectorStore(
        collection_name="flip_ivf",
        host="localhost",
        port=19530,
        dimension=384,
        index_type="IVF_FLAT"
    )
    
    # HNSW - Best for high-dimensional vectors
    store_hnsw = MilvusVectorStore(
        collection_name="flip_hnsw",
        host="localhost",
        port=19530,
        dimension=384,
        index_type="HNSW"
    )
    
    # FLAT - Exact search (slower but most accurate)
    store_flat = MilvusVectorStore(
        collection_name="flip_flat",
        host="localhost",
        port=19530,
        dimension=384,
        index_type="FLAT"
    )
    
    print("\nCreated collections with different index types:")
    print(f"- IVF_FLAT: {store_ivf.index_type}")
    print(f"- HNSW: {store_hnsw.index_type}")
    print(f"- FLAT: {store_flat.index_type}")


def example_milvus_consistency_levels():
    """Different consistency levels."""
    print("\n" + "=" * 60)
    print("Example 3: Consistency Levels")
    print("=" * 60)
    
    from flip.vector_store.milvus import MilvusVectorStore
    
    # Strong consistency (default)
    store_strong = MilvusVectorStore(
        collection_name="flip_strong",
        host="localhost",
        port=19530,
        dimension=384,
        consistency_level="Strong"
    )
    
    # Eventually consistent (faster)
    store_eventual = MilvusVectorStore(
        collection_name="flip_eventual",
        host="localhost",
        port=19530,
        dimension=384,
        consistency_level="Eventually"
    )
    
    print("\nConsistency levels:")
    print(f"- Strong: {store_strong.consistency_level}")
    print(f"- Eventually: {store_eventual.consistency_level}")


def example_milvus_health_monitoring():
    """Monitor Milvus health and stats."""
    print("\n" + "=" * 60)
    print("Example 4: Health Monitoring")
    print("=" * 60)
    
    from flip.vector_store.milvus import MilvusVectorStore
    
    store = MilvusVectorStore(
        collection_name="flip_health",
        host="localhost",
        port=19530,
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
    print(f"Metric Type: {stats.metadata.get('metric_type', 'N/A')}")


def example_milvus_batch_operations():
    """Batch operations with Milvus."""
    print("\n" + "=" * 60)
    print("Example 5: Batch Operations")
    print("=" * 60)
    
    from flip.vector_store.milvus import MilvusVectorStore
    from flip.embedding.sentence_transformers import SentenceTransformerEmbedder
    
    # Initialize
    embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
    store = MilvusVectorStore(
        collection_name="flip_batch",
        host="localhost",
        port=19530,
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


def example_milvus_filtering():
    """Search with filtering."""
    print("\n" + "=" * 60)
    print("Example 6: Filtered Search")
    print("=" * 60)
    
    from flip.vector_store.milvus import MilvusVectorStore
    from flip.embedding.sentence_transformers import SentenceTransformerEmbedder
    
    embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
    store = MilvusVectorStore(
        collection_name="flip_filter",
        host="localhost",
        port=19530,
        dimension=384
    )
    
    # Add data with metadata
    texts = ["AI document", "ML document", "DL document"]
    embeddings = embedder.embed_batch(texts)
    ids = ["1", "2", "3"]
    metadatas = [
        {"category": "AI"},
        {"category": "ML"},
        {"category": "DL"}
    ]
    
    store.add(ids, embeddings, texts, metadatas)
    
    # Search with filter
    query_embedding = embedder.embed("artificial intelligence")
    results = store.filter_search(
        query_embedding,
        filters={"category": "AI"},
        top_k=5
    )
    
    print(f"\nFiltered results: {len(results)}")


def example_milvus_distance_metrics():
    """Different distance metrics."""
    print("\n" + "=" * 60)
    print("Example 7: Distance Metrics")
    print("=" * 60)
    
    from flip.vector_store.milvus import MilvusVectorStore
    
    # L2 distance (Euclidean)
    store_l2 = MilvusVectorStore(
        collection_name="flip_l2",
        host="localhost",
        port=19530,
        dimension=384,
        metric_type="L2"
    )
    
    # Inner Product
    store_ip = MilvusVectorStore(
        collection_name="flip_ip",
        host="localhost",
        port=19530,
        dimension=384,
        metric_type="IP"
    )
    
    # Cosine similarity
    store_cosine = MilvusVectorStore(
        collection_name="flip_cosine",
        host="localhost",
        port=19530,
        dimension=384,
        metric_type="COSINE"
    )
    
    print("\nDistance metrics:")
    print(f"- L2: {store_l2.metric_type}")
    print(f"- IP: {store_ip.metric_type}")
    print(f"- COSINE: {store_cosine.metric_type}")


def main():
    """Run all Milvus examples."""
    print("\n🚀 Flip SDK - Milvus Examples\n")
    
    try:
        example_local_milvus()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_milvus_index_types()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_milvus_consistency_levels()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_milvus_health_monitoring()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_milvus_batch_operations()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_milvus_filtering()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_milvus_distance_metrics()
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n✅ Milvus examples complete!")


if __name__ == "__main__":
    main()
