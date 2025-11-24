"""
Example using Pgvector (PostgreSQL) vector store with Flip SDK.

This example demonstrates how to use PostgreSQL with the pgvector extension
as the vector store backend.
"""

from flip import Flip, FlipConfig
import os


def example_local_pgvector():
    """Basic Pgvector usage."""
    print("=" * 60)
    print("Example 1: Local Pgvector Usage")
    print("=" * 60)
    
    config = FlipConfig(
        vector_store="pgvector",
        
        # PostgreSQL configuration
        pgvector_host="localhost",
        pgvector_port=5432,
        pgvector_database="flip_db",
        pgvector_user="postgres",
        pgvector_password="password",
        
        # Use sentence transformers for local embeddings
        embedding_provider="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2"
    )
    
    flip = Flip(directory="./sample_docs", config=config)
    
    # Query
    response = flip.query("What is artificial intelligence?")
    print(f"\nAnswer: {response.answer}")
    print(f"Sources: {len(response.citations)}")


def example_pgvector_index_types():
    """Different index types in Pgvector."""
    print("\n" + "=" * 60)
    print("Example 2: Index Types")
    print("=" * 60)
    
    from flip.vector_store.pgvector import PgvectorVectorStore
    
    # IVF Flat - Good for large datasets
    store_ivf = PgvectorVectorStore(
        collection_name="flip_ivf",
        host="localhost",
        port=5432,
        database="flip_db",
        user="postgres",
        password="password",
        dimension=384,
        index_type="ivfflat"
    )
    
    # HNSW - Best for high recall
    store_hnsw = PgvectorVectorStore(
        collection_name="flip_hnsw",
        host="localhost",
        port=5432,
        database="flip_db",
        user="postgres",
        password="password",
        dimension=384,
        index_type="hnsw"
    )
    
    print("\nCreated tables with different index types:")
    print(f"- IVF Flat: {store_ivf.index_type}")
    print(f"- HNSW: {store_hnsw.index_type}")


def example_pgvector_distance_metrics():
    """Different distance metrics."""
    print("\n" + "=" * 60)
    print("Example 3: Distance Metrics")
    print("=" * 60)
    
    from flip.vector_store.pgvector import PgvectorVectorStore
    
    # Cosine similarity
    store_cosine = PgvectorVectorStore(
        collection_name="flip_cosine",
        host="localhost",
        database="flip_db",
        user="postgres",
        password="password",
        dimension=384,
        distance_metric="cosine"
    )
    
    # L2 distance
    store_l2 = PgvectorVectorStore(
        collection_name="flip_l2",
        host="localhost",
        database="flip_db",
        user="postgres",
        password="password",
        dimension=384,
        distance_metric="l2"
    )
    
    # Inner product
    store_ip = PgvectorVectorStore(
        collection_name="flip_ip",
        host="localhost",
        database="flip_db",
        user="postgres",
        password="password",
        dimension=384,
        distance_metric="inner_product"
    )
    
    print("\nDistance metrics:")
    print(f"- Cosine: {store_cosine.distance_metric}")
    print(f"- L2: {store_l2.distance_metric}")
    print(f"- Inner Product: {store_ip.distance_metric}")


def example_pgvector_health_monitoring():
    """Monitor Pgvector health and stats."""
    print("\n" + "=" * 60)
    print("Example 4: Health Monitoring")
    print("=" * 60)
    
    from flip.vector_store.pgvector import PgvectorVectorStore
    
    store = PgvectorVectorStore(
        collection_name="flip_health",
        host="localhost",
        database="flip_db",
        user="postgres",
        password="password",
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
    print(f"Table Name: {stats.metadata.get('table_name', 'N/A')}")


def example_pgvector_batch_operations():
    """Batch operations with Pgvector."""
    print("\n" + "=" * 60)
    print("Example 5: Batch Operations")
    print("=" * 60)
    
    from flip.vector_store.pgvector import PgvectorVectorStore
    from flip.embedding.sentence_transformers import SentenceTransformerEmbedder
    
    # Initialize
    embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
    store = PgvectorVectorStore(
        collection_name="flip_batch",
        host="localhost",
        database="flip_db",
        user="postgres",
        password="password",
        dimension=384
    )
    
    # Prepare data
    texts = [f"Document {i} about AI and machine learning." for i in range(100)]
    embeddings = embedder.embed_batch(texts)
    ids = [str(i) for i in range(100)]
    
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


def example_pgvector_filtering():
    """Search with metadata filtering."""
    print("\n" + "=" * 60)
    print("Example 6: Filtered Search")
    print("=" * 60)
    
    from flip.vector_store.pgvector import PgvectorVectorStore
    from flip.embedding.sentence_transformers import SentenceTransformerEmbedder
    
    embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
    store = PgvectorVectorStore(
        collection_name="flip_filter",
        host="localhost",
        database="flip_db",
        user="postgres",
        password="password",
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


def example_pgvector_transactions():
    """PostgreSQL transactions with Pgvector."""
    print("\n" + "=" * 60)
    print("Example 7: Transactions")
    print("=" * 60)
    
    from flip.vector_store.pgvector import PgvectorVectorStore
    from flip.embedding.sentence_transformers import SentenceTransformerEmbedder
    
    embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
    store = PgvectorVectorStore(
        collection_name="flip_transactions",
        host="localhost",
        database="flip_db",
        user="postgres",
        password="password",
        dimension=384
    )
    
    print("\nPgvector uses PostgreSQL transactions automatically")
    print("All operations are ACID compliant!")
    
    # Add operation is transactional
    texts = ["Doc 1", "Doc 2"]
    embeddings = embedder.embed_batch(texts)
    ids = ["1", "2"]
    
    store.add(ids, embeddings, texts)
    print(f"Added {len(texts)} documents (transactional)")


def main():
    """Run all Pgvector examples."""
    print("\n🚀 Flip SDK - Pgvector Examples\n")
    
    try:
        example_local_pgvector()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_pgvector_index_types()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_pgvector_distance_metrics()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_pgvector_health_monitoring()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_pgvector_batch_operations()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_pgvector_filtering()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_pgvector_transactions()
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n✅ Pgvector examples complete!")


if __name__ == "__main__":
    main()
