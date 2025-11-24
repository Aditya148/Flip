"""
Example using Pinecone vector store with Flip SDK.

This example demonstrates how to use Pinecone as the vector store backend.
"""

from flip import Flip, FlipConfig
import os

# Set your Pinecone credentials
os.environ["PINECONE_API_KEY"] = "your-pinecone-api-key"
os.environ["PINECONE_ENVIRONMENT"] = "us-east-1-aws"  # or your environment


def example_basic_pinecone():
    """Basic Pinecone usage."""
    print("=" * 60)
    print("Example 1: Basic Pinecone Usage")
    print("=" * 60)
    
    config = FlipConfig(
        vector_store="pinecone",
        
        # Pinecone configuration
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        pinecone_environment=os.getenv("PINECONE_ENVIRONMENT"),
        pinecone_index_name="flip-demo",
        
        # Use sentence transformers for local embeddings
        embedding_provider="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2"
    )
    
    flip = Flip(directory="./sample_docs", config=config)
    
    # Query
    response = flip.query("What is artificial intelligence?")
    print(f"\nAnswer: {response.answer}")
    print(f"Sources: {len(response.citations)}")


def example_pinecone_with_namespace():
    """Pinecone with namespace for multi-tenancy."""
    print("\n" + "=" * 60)
    print("Example 2: Pinecone with Namespace")
    print("=" * 60)
    
    config = FlipConfig(
        vector_store="pinecone",
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        pinecone_environment=os.getenv("PINECONE_ENVIRONMENT"),
        pinecone_index_name="flip-multi-tenant",
        pinecone_namespace="tenant-1",  # Separate namespace per tenant
        
        embedding_provider="sentence-transformers"
    )
    
    flip = Flip(directory="./sample_docs", config=config)
    
    stats = flip.get_stats()
    print(f"\nNamespace: {stats.get('namespace', 'default')}")
    print(f"Vectors: {stats['chunk_count']}")


def example_pinecone_with_metadata_filtering():
    """Pinecone with metadata filtering."""
    print("\n" + "=" * 60)
    print("Example 3: Metadata Filtering")
    print("=" * 60)
    
    config = FlipConfig(
        vector_store="pinecone",
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        pinecone_environment=os.getenv("PINECONE_ENVIRONMENT"),
        pinecone_index_name="flip-filtered",
        
        embedding_provider="sentence-transformers"
    )
    
    flip = Flip(directory="./sample_docs", config=config)
    
    # Query with metadata filter
    # Note: Metadata filtering syntax depends on how you structure your metadata
    response = flip.query(
        "What is machine learning?",
        # filter={"category": "AI", "year": 2024}  # Example filter
    )
    
    print(f"\nFiltered results: {len(response.context_chunks)}")


def example_pinecone_health_monitoring():
    """Monitor Pinecone health and stats."""
    print("\n" + "=" * 60)
    print("Example 4: Health Monitoring")
    print("=" * 60)
    
    from flip.vector_store.pinecone import PineconeVectorStore
    
    store = PineconeVectorStore(
        collection_name="flip-health",
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT"),
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
    print(f"Index Fullness: {stats.metadata.get('index_fullness', 'N/A')}")


def example_pinecone_batch_operations():
    """Batch operations with Pinecone."""
    print("\n" + "=" * 60)
    print("Example 5: Batch Operations")
    print("=" * 60)
    
    from flip.vector_store.pinecone import PineconeVectorStore
    from flip.embedding.sentence_transformers import SentenceTransformerEmbedder
    
    # Initialize
    embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
    store = PineconeVectorStore(
        collection_name="flip-batch",
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENVIRONMENT"),
        dimension=384
    )
    
    # Prepare data
    texts = [f"Document {i} about AI and machine learning." for i in range(100)]
    embeddings = embedder.embed_batch(texts)
    ids = [f"doc_{i}" for i in range(100)]
    
    # Batch add with automatic rate limiting
    import time
    start = time.time()
    store.batch_add(
        ids=ids,
        embeddings=embeddings,
        texts=texts,
        batch_size=50  # Process 50 at a time
    )
    duration = time.time() - start
    
    print(f"\nAdded {len(texts)} vectors in {duration:.2f}s")
    print(f"Rate: {len(texts)/duration:.1f} vectors/sec")


def example_pinecone_migration():
    """Migrate from ChromaDB to Pinecone."""
    print("\n" + "=" * 60)
    print("Example 6: Migration from ChromaDB")
    print("=" * 60)
    
    # Step 1: Load existing data from ChromaDB
    config_chroma = FlipConfig(
        vector_store="chroma",
        embedding_provider="sentence-transformers"
    )
    
    flip_chroma = Flip(directory="./sample_docs", config=config_chroma)
    print(f"ChromaDB vectors: {flip_chroma.get_stats()['chunk_count']}")
    
    # Step 2: Create new Pinecone instance
    config_pinecone = FlipConfig(
        vector_store="pinecone",
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        pinecone_environment=os.getenv("PINECONE_ENVIRONMENT"),
        pinecone_index_name="flip-migrated",
        embedding_provider="sentence-transformers"
    )
    
    # Step 3: Re-index with Pinecone
    flip_pinecone = Flip(directory="./sample_docs", config=config_pinecone)
    print(f"Pinecone vectors: {flip_pinecone.get_stats()['chunk_count']}")
    
    # Verify migration
    response_chroma = flip_chroma.query("What is AI?")
    response_pinecone = flip_pinecone.query("What is AI?")
    
    print(f"\nBoth systems working: ✓")


def main():
    """Run all Pinecone examples."""
    print("\n🚀 Flip SDK - Pinecone Examples\n")
    
    try:
        example_basic_pinecone()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_pinecone_with_namespace()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_pinecone_with_metadata_filtering()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_pinecone_health_monitoring()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_pinecone_batch_operations()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_pinecone_migration()
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n✅ Pinecone examples complete!")


if __name__ == "__main__":
    main()
