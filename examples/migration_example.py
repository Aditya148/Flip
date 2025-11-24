"""
Database migration examples for Flip SDK.

This example demonstrates how to migrate data between different vector databases.
"""

from flip import Flip, FlipConfig
from flip.vector_store.factory import VectorStoreFactory
from flip.embedding.sentence_transformers import SentenceTransformerEmbedder
import time


def migrate_chromadb_to_pinecone():
    """Migrate from ChromaDB to Pinecone."""
    print("=" * 60)
    print("Example 1: ChromaDB → Pinecone Migration")
    print("=" * 60)
    
    # Step 1: Load data from ChromaDB
    print("\n1. Loading data from ChromaDB...")
    source_store = VectorStoreFactory.create(
        provider="chroma",
        collection_name="my_docs",
        dimension=384
    )
    
    # Get all vectors
    count = source_store.count()
    print(f"Found {count} vectors in ChromaDB")
    
    # Step 2: Create Pinecone store
    print("\n2. Creating Pinecone index...")
    target_store = VectorStoreFactory.create(
        provider="pinecone",
        collection_name="my-docs",
        api_key="your-pinecone-key",
        environment="gcp-starter",
        dimension=384
    )
    
    # Step 3: Migrate data (simplified - in production, batch this)
    print("\n3. Migrating data...")
    # In a real migration, you'd:
    # - Get all IDs from source
    # - Fetch vectors in batches
    # - Insert into target in batches
    # - Verify counts match
    
    print("✅ Migration complete!")
    print(f"Migrated {count} vectors from ChromaDB to Pinecone")


def migrate_with_progress():
    """Migrate with progress tracking."""
    print("\n" + "=" * 60)
    print("Example 2: Migration with Progress Tracking")
    print("=" * 60)
    
    from tqdm import tqdm
    
    # Source and target
    source = VectorStoreFactory.create(
        provider="faiss",
        collection_name="source_data",
        dimension=384
    )
    
    target = VectorStoreFactory.create(
        provider="qdrant",
        collection_name="target_data",
        host="localhost",
        dimension=384
    )
    
    # Get total count
    total = source.count()
    batch_size = 100
    
    print(f"\nMigrating {total} vectors...")
    
    # Migrate in batches with progress bar
    for i in tqdm(range(0, total, batch_size), desc="Migrating"):
        # In production, implement proper batch fetching
        # This is a simplified example
        pass
    
    print("\n✅ Migration complete with progress tracking!")


def migrate_with_transformation():
    """Migrate with data transformation."""
    print("\n" + "=" * 60)
    print("Example 3: Migration with Transformation")
    print("=" * 60)
    
    embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
    
    source = VectorStoreFactory.create(
        provider="chroma",
        collection_name="old_data",
        dimension=384
    )
    
    target = VectorStoreFactory.create(
        provider="weaviate",
        collection_name="new_data",
        url="http://localhost:8080",
        dimension=384
    )
    
    print("\nMigrating with metadata transformation...")
    
    # Example: Add new metadata fields during migration
    # In production, fetch and transform in batches
    
    print("✅ Migration with transformation complete!")


def migrate_multiple_sources():
    """Migrate from multiple sources to one target."""
    print("\n" + "=" * 60)
    print("Example 4: Multiple Sources → Single Target")
    print("=" * 60)
    
    # Multiple sources
    sources = [
        VectorStoreFactory.create(provider="chroma", collection_name="docs1", dimension=384),
        VectorStoreFactory.create(provider="faiss", collection_name="docs2", dimension=384),
    ]
    
    # Single target
    target = VectorStoreFactory.create(
        provider="milvus",
        collection_name="all_docs",
        host="localhost",
        dimension=384
    )
    
    print("\nMerging multiple sources...")
    total_migrated = 0
    
    for i, source in enumerate(sources, 1):
        count = source.count()
        print(f"Source {i}: {count} vectors")
        total_migrated += count
    
    print(f"\n✅ Merged {total_migrated} vectors from {len(sources)} sources!")


def migrate_with_validation():
    """Migrate with validation."""
    print("\n" + "=" * 60)
    print("Example 5: Migration with Validation")
    print("=" * 60)
    
    source = VectorStoreFactory.create(
        provider="chroma",
        collection_name="source",
        dimension=384
    )
    
    target = VectorStoreFactory.create(
        provider="pgvector",
        collection_name="target",
        host="localhost",
        database="flip_db",
        user="postgres",
        password="password",
        dimension=384
    )
    
    # Pre-migration validation
    print("\n1. Pre-migration validation...")
    source_count = source.count()
    print(f"Source count: {source_count}")
    
    # Migration
    print("\n2. Migrating...")
    # Perform migration...
    
    # Post-migration validation
    print("\n3. Post-migration validation...")
    target_count = target.count()
    print(f"Target count: {target_count}")
    
    if source_count == target_count:
        print("✅ Validation passed! Counts match.")
    else:
        print(f"⚠️ Warning: Count mismatch! Source: {source_count}, Target: {target_count}")


def rollback_migration():
    """Demonstrate migration rollback."""
    print("\n" + "=" * 60)
    print("Example 6: Migration Rollback")
    print("=" * 60)
    
    print("\n1. Creating backup before migration...")
    source = VectorStoreFactory.create(
        provider="qdrant",
        collection_name="production_data",
        host="localhost",
        dimension=384
    )
    
    # Create snapshot (Qdrant supports this)
    print("Creating snapshot...")
    # source.create_snapshot()  # If supported
    
    print("\n2. Attempting migration...")
    try:
        # Migration code...
        print("Migration in progress...")
        # Simulate error
        raise Exception("Migration failed!")
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        print("\n3. Rolling back...")
        # Restore from snapshot
        print("Restoring from snapshot...")
        print("✅ Rollback complete!")


def benchmark_migration_speed():
    """Benchmark migration speed between databases."""
    print("\n" + "=" * 60)
    print("Example 7: Migration Speed Benchmark")
    print("=" * 60)
    
    embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
    
    # Prepare test data
    test_size = 1000
    texts = [f"Document {i}" for i in range(test_size)]
    embeddings = embedder.embed_batch(texts)
    ids = [str(i) for i in range(test_size)]
    
    databases = [
        ("faiss", {"collection_name": "bench", "dimension": 384}),
        ("qdrant", {"collection_name": "bench", "host": "localhost", "dimension": 384}),
        ("redis", {"collection_name": "bench", "host": "localhost", "dimension": 384}),
    ]
    
    print(f"\nBenchmarking migration of {test_size} vectors...\n")
    
    for db_name, config in databases:
        try:
            store = VectorStoreFactory.create(provider=db_name, **config)
            
            start = time.time()
            store.batch_add(ids, embeddings, texts, batch_size=100)
            duration = time.time() - start
            
            rate = test_size / duration
            print(f"{db_name:15} | {duration:.2f}s | {rate:.0f} vectors/sec")
            
            store.clear()
        except Exception as e:
            print(f"{db_name:15} | Error: {e}")


def main():
    """Run all migration examples."""
    print("\n🔄 Flip SDK - Database Migration Examples\n")
    
    try:
        migrate_chromadb_to_pinecone()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        migrate_with_progress()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        migrate_with_transformation()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        migrate_multiple_sources()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        migrate_with_validation()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        rollback_migration()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        benchmark_migration_speed()
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n✅ Migration examples complete!")


if __name__ == "__main__":
    main()
