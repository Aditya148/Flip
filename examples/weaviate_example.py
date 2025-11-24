"""
Example using Weaviate vector store with Flip SDK.

This example demonstrates how to use Weaviate as the vector store backend.
"""

from flip import Flip, FlipConfig
import os

# Set your Weaviate credentials
os.environ["WEAVIATE_URL"] = "http://localhost:8080"
os.environ["WEAVIATE_API_KEY"] = "your-weaviate-api-key"  # For cloud


def example_local_weaviate():
    """Basic Weaviate usage with local instance."""
    print("=" * 60)
    print("Example 1: Local Weaviate Usage")
    print("=" * 60)
    
    config = FlipConfig(
        vector_store="weaviate",
        
        # Local Weaviate configuration
        weaviate_url="http://localhost:8080",
        
        # Use sentence transformers for local embeddings
        embedding_provider="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2"
    )
    
    flip = Flip(directory="./sample_docs", config=config)
    
    # Query
    response = flip.query("What is artificial intelligence?")
    print(f"\nAnswer: {response.answer}")
    print(f"Sources: {len(response.citations)}")


def example_cloud_weaviate():
    """Weaviate Cloud usage."""
    print("\n" + "=" * 60)
    print("Example 2: Weaviate Cloud")
    print("=" * 60)
    
    config = FlipConfig(
        vector_store="weaviate",
        
        # Cloud Weaviate configuration
        weaviate_url=os.getenv("WEAVIATE_URL"),
        weaviate_api_key=os.getenv("WEAVIATE_API_KEY"),
        
        embedding_provider="sentence-transformers"
    )
    
    flip = Flip(directory="./sample_docs", config=config)
    
    stats = flip.get_stats()
    print(f"\nVectors: {stats['chunk_count']}")


def example_weaviate_graphql():
    """Weaviate with GraphQL queries."""
    print("\n" + "=" * 60)
    print("Example 3: GraphQL Queries")
    print("=" * 60)
    
    from flip.vector_store.weaviate import WeaviateVectorStore
    
    store = WeaviateVectorStore(
        collection_name="flip_docs",
        url="http://localhost:8080",
        dimension=384
    )
    
    # Weaviate supports powerful GraphQL queries
    # This is handled internally by the search method
    from flip.embedding.sentence_transformers import SentenceTransformerEmbedder
    
    embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
    query_embedding = embedder.embed("What is machine learning?")
    
    results = store.search(query_embedding, top_k=3)
    
    print(f"\nFound {len(results)} results")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.text[:100]}... (score: {result.score:.3f})")


def example_weaviate_schema():
    """Weaviate schema management."""
    print("\n" + "=" * 60)
    print("Example 4: Schema Management")
    print("=" * 60)
    
    from flip.vector_store.weaviate import WeaviateVectorStore
    
    # Weaviate automatically creates schema
    store = WeaviateVectorStore(
        collection_name="custom_schema",
        url="http://localhost:8080",
        dimension=768  # Custom dimension
    )
    
    print(f"\nCollection created: {store.class_name}")
    print(f"Dimension: {store.dimension}")


def example_weaviate_health_monitoring():
    """Monitor Weaviate health and stats."""
    print("\n" + "=" * 60)
    print("Example 5: Health Monitoring")
    print("=" * 60)
    
    from flip.vector_store.weaviate import WeaviateVectorStore
    
    store = WeaviateVectorStore(
        collection_name="flip_health",
        url="http://localhost:8080",
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
    print(f"Class Name: {stats.metadata.get('class_name', 'N/A')}")


def example_weaviate_batch_operations():
    """Batch operations with Weaviate."""
    print("\n" + "=" * 60)
    print("Example 6: Batch Operations")
    print("=" * 60)
    
    from flip.vector_store.weaviate import WeaviateVectorStore
    from flip.embedding.sentence_transformers import SentenceTransformerEmbedder
    import uuid
    
    # Initialize
    embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
    store = WeaviateVectorStore(
        collection_name="flip_batch",
        url="http://localhost:8080",
        dimension=384
    )
    
    # Prepare data
    texts = [f"Document {i} about AI and machine learning." for i in range(100)]
    embeddings = embedder.embed_batch(texts)
    ids = [str(uuid.uuid4()) for _ in range(100)]
    
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


def example_weaviate_update():
    """Update vectors in Weaviate."""
    print("\n" + "=" * 60)
    print("Example 7: Update Operations")
    print("=" * 60)
    
    from flip.vector_store.weaviate import WeaviateVectorStore
    from flip.embedding.sentence_transformers import SentenceTransformerEmbedder
    import uuid
    
    embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
    store = WeaviateVectorStore(
        collection_name="flip_update",
        url="http://localhost:8080",
        dimension=384
    )
    
    # Add initial data
    test_id = str(uuid.uuid4())
    embedding = embedder.embed("Original text")
    store.add([test_id], [embedding], ["Original text"])
    
    print(f"Added document: {test_id}")
    
    # Update the document
    new_embedding = embedder.embed("Updated text")
    store.update(
        ids=[test_id],
        embeddings=[new_embedding],
        texts=["Updated text"],
        metadatas=[{"updated": True}]
    )
    
    print(f"Updated document: {test_id}")
    
    # Retrieve to verify
    results = store.get_by_ids([test_id])
    if results:
        print(f"Retrieved: {results[0].text}")


def main():
    """Run all Weaviate examples."""
    print("\n🚀 Flip SDK - Weaviate Examples\n")
    
    try:
        example_local_weaviate()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_cloud_weaviate()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_weaviate_graphql()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_weaviate_schema()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_weaviate_health_monitoring()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_weaviate_batch_operations()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_weaviate_update()
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n✅ Weaviate examples complete!")


if __name__ == "__main__":
    main()
