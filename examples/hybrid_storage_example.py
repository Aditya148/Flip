"""
Hybrid storage examples for Flip SDK.

This example demonstrates how to use multiple databases together
for different purposes (e.g., cache + persistent storage).
"""

from flip import Flip, FlipConfig
from flip.vector_store.factory import VectorStoreFactory
from flip.embedding.sentence_transformers import SentenceTransformerEmbedder
import time


def redis_cache_with_pgvector_storage():
    """Use Redis as cache and Pgvector as persistent storage."""
    print("=" * 60)
    print("Example 1: Redis Cache + Pgvector Storage")
    print("=" * 60)
    
    embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
    
    # Redis for fast cache
    cache = VectorStoreFactory.create(
        provider="redis",
        collection_name="cache",
        host="localhost",
        dimension=384
    )
    
    # Pgvector for persistent storage
    storage = VectorStoreFactory.create(
        provider="pgvector",
        collection_name="persistent",
        host="localhost",
        database="flip_db",
        user="postgres",
        password="password",
        dimension=384
    )
    
    # Add data to both
    texts = ["Document 1", "Document 2", "Document 3"]
    embeddings = embedder.embed_batch(texts)
    ids = ["1", "2", "3"]
    
    print("\nAdding to persistent storage...")
    storage.add(ids, embeddings, texts)
    
    print("Caching frequently accessed data...")
    cache.add(ids[:2], embeddings[:2], texts[:2])  # Cache first 2
    
    # Search: Check cache first, then storage
    query = embedder.embed("Document")
    
    print("\nSearching cache first...")
    start = time.time()
    cache_results = cache.search(query, top_k=2)
    cache_time = (time.time() - start) * 1000
    
    print(f"Cache search: {cache_time:.2f}ms")
    
    if not cache_results:
        print("Cache miss, searching storage...")
        start = time.time()
        storage_results = storage.search(query, top_k=2)
        storage_time = (time.time() - start) * 1000
        print(f"Storage search: {storage_time:.2f}ms")
    
    print(f"\n✅ Hybrid cache + storage pattern complete!")


def faiss_local_with_pinecone_cloud():
    """Use FAISS for local dev and Pinecone for production."""
    print("\n" + "=" * 60)
    print("Example 2: FAISS (Dev) + Pinecone (Prod)")
    print("=" * 60)
    
    import os
    
    # Determine environment
    is_production = os.getenv("ENVIRONMENT") == "production"
    
    if is_production:
        print("\nProduction mode: Using Pinecone...")
        config = FlipConfig(
            vector_store="pinecone",
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
            pinecone_environment="gcp-starter"
        )
    else:
        print("\nDevelopment mode: Using FAISS...")
        config = FlipConfig(
            vector_store="faiss",
            faiss_index_type="Flat",
            persist_directory="./dev_faiss"
        )
    
    flip = Flip(directory="./docs", config=config)
    
    response = flip.query("What is AI?")
    print(f"\nAnswer: {response.answer[:100]}...")
    print(f"Using: {config.vector_store}")


def elasticsearch_fulltext_with_qdrant_vectors():
    """Use Elasticsearch for full-text and Qdrant for vectors."""
    print("\n" + "=" * 60)
    print("Example 3: Elasticsearch (Text) + Qdrant (Vectors)")
    print("=" * 60)
    
    embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
    
    # Elasticsearch for full-text search
    text_search = VectorStoreFactory.create(
        provider="elasticsearch",
        collection_name="text_index",
        url="http://localhost:9200",
        dimension=384
    )
    
    # Qdrant for vector search
    vector_search = VectorStoreFactory.create(
        provider="qdrant",
        collection_name="vector_index",
        host="localhost",
        dimension=384
    )
    
    # Add same data to both
    texts = [
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks",
        "Natural language processing handles text"
    ]
    embeddings = embedder.embed_batch(texts)
    ids = ["1", "2", "3"]
    
    print("\nIndexing in both systems...")
    text_search.add(ids, embeddings, texts)
    vector_search.add(ids, embeddings, texts)
    
    # Keyword search in Elasticsearch
    print("\nKeyword search (Elasticsearch):")
    # In production, use ES's text search capabilities
    
    # Vector search in Qdrant
    print("Vector search (Qdrant):")
    query = embedder.embed("neural networks")
    results = vector_search.search(query, top_k=2)
    for r in results:
        print(f"  - {r.text} (score: {r.score:.3f})")
    
    print("\n✅ Hybrid text + vector search complete!")


def mongodb_metadata_with_milvus_vectors():
    """Use MongoDB for metadata and Milvus for vectors."""
    print("\n" + "=" * 60)
    print("Example 4: MongoDB (Metadata) + Milvus (Vectors)")
    print("=" * 60)
    
    embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
    
    # MongoDB for rich metadata
    metadata_store = VectorStoreFactory.create(
        provider="mongodb",
        collection_name="metadata",
        uri="mongodb://localhost:27017/",
        database="flip_db",
        dimension=384
    )
    
    # Milvus for fast vector search
    vector_store = VectorStoreFactory.create(
        provider="milvus",
        collection_name="vectors",
        host="localhost",
        dimension=384
    )
    
    # Add data with rich metadata
    texts = ["Doc 1", "Doc 2"]
    embeddings = embedder.embed_batch(texts)
    ids = ["1", "2"]
    metadatas = [
        {
            "title": "Introduction to AI",
            "author": "John Doe",
            "date": "2024-01-01",
            "tags": ["AI", "ML"],
            "version": 1
        },
        {
            "title": "Deep Learning Basics",
            "author": "Jane Smith",
            "date": "2024-01-02",
            "tags": ["DL", "NN"],
            "version": 1
        }
    ]
    
    print("\nStoring metadata in MongoDB...")
    metadata_store.add(ids, embeddings, texts, metadatas)
    
    print("Storing vectors in Milvus...")
    vector_store.add(ids, embeddings, texts, metadatas)
    
    # Search vectors in Milvus
    query = embedder.embed("deep learning")
    results = vector_store.search(query, top_k=1)
    
    # Get full metadata from MongoDB
    if results:
        doc_id = results[0].id
        full_docs = metadata_store.get_by_ids([doc_id])
        print(f"\nFound: {full_docs[0].text}")
        print(f"Metadata: {full_docs[0].metadata}")
    
    print("\n✅ Hybrid metadata + vector storage complete!")


def multi_region_deployment():
    """Use different databases for different regions."""
    print("\n" + "=" * 60)
    print("Example 5: Multi-Region Deployment")
    print("=" * 60)
    
    regions = {
        "us-east": {
            "provider": "pinecone",
            "config": {
                "collection_name": "us-east-index",
                "api_key": "us-key",
                "environment": "us-east1-gcp"
            }
        },
        "eu-west": {
            "provider": "qdrant",
            "config": {
                "collection_name": "eu-west-index",
                "url": "https://eu-cluster.qdrant.io",
                "api_key": "eu-key"
            }
        },
        "ap-south": {
            "provider": "weaviate",
            "config": {
                "collection_name": "ap-south-index",
                "url": "https://ap-cluster.weaviate.io",
                "api_key": "ap-key"
            }
        }
    }
    
    print("\nMulti-region configuration:")
    for region, config in regions.items():
        print(f"  {region}: {config['provider']}")
    
    # Route requests based on user location
    user_region = "us-east"
    print(f"\nRouting user to: {user_region}")
    
    print("✅ Multi-region deployment configured!")


def hot_cold_storage():
    """Use Redis for hot data and Pgvector for cold data."""
    print("\n" + "=" * 60)
    print("Example 6: Hot/Cold Storage Pattern")
    print("=" * 60)
    
    embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
    
    # Redis for frequently accessed (hot) data
    hot_storage = VectorStoreFactory.create(
        provider="redis",
        collection_name="hot_data",
        host="localhost",
        dimension=384
    )
    
    # Pgvector for rarely accessed (cold) data
    cold_storage = VectorStoreFactory.create(
        provider="pgvector",
        collection_name="cold_data",
        host="localhost",
        database="flip_db",
        user="postgres",
        password="password",
        dimension=384
    )
    
    # Simulate access patterns
    recent_docs = ["Recent doc 1", "Recent doc 2"]
    old_docs = ["Old doc 1", "Old doc 2", "Old doc 3"]
    
    recent_embeddings = embedder.embed_batch(recent_docs)
    old_embeddings = embedder.embed_batch(old_docs)
    
    print("\nStoring recent data in hot storage (Redis)...")
    hot_storage.add(
        ["r1", "r2"],
        recent_embeddings,
        recent_docs
    )
    
    print("Storing old data in cold storage (Pgvector)...")
    cold_storage.add(
        ["o1", "o2", "o3"],
        old_embeddings,
        old_docs
    )
    
    # Search hot first, then cold
    query = embedder.embed("document")
    
    print("\nSearching hot storage first...")
    results = hot_storage.search(query, top_k=5)
    
    if len(results) < 5:
        print("Searching cold storage for more results...")
        cold_results = cold_storage.search(query, top_k=5 - len(results))
        results.extend(cold_results)
    
    print(f"Total results: {len(results)}")
    print("✅ Hot/cold storage pattern complete!")


def main():
    """Run all hybrid storage examples."""
    print("\n🔀 Flip SDK - Hybrid Storage Examples\n")
    
    try:
        redis_cache_with_pgvector_storage()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        faiss_local_with_pinecone_cloud()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        elasticsearch_fulltext_with_qdrant_vectors()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        mongodb_metadata_with_milvus_vectors()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        multi_region_deployment()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        hot_cold_storage()
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n✅ Hybrid storage examples complete!")


if __name__ == "__main__":
    main()
