"""
Advanced features example for Flip SDK.

This example demonstrates the advanced RAG features:
1. Hybrid search (vector + keyword)
2. Re-ranking for better accuracy
3. Query enhancement
4. Caching
"""

from flip import Flip, FlipConfig


def example_hybrid_search():
    """Example using hybrid search."""
    print("=" * 60)
    print("Example 1: Hybrid Search (Vector + Keyword)")
    print("=" * 60)
    
    config = FlipConfig(
        llm_provider="openai",
        use_hybrid_search=True,  # Enable hybrid search
        use_reranking=False,     # Disable reranking for comparison
    )
    
    flip = Flip(directory="./sample_docs", config=config)
    
    response = flip.query("What are the key components of RAG?")
    print(f"\nAnswer: {response.answer}")
    print(f"\nMetadata:")
    print(f"  Hybrid search: {response.metadata.get('hybrid_search_used', False)}")
    print(f"  Top result score: {response.citations[0]['score']:.3f}")


def example_reranking():
    """Example using re-ranking."""
    print("\n" + "=" * 60)
    print("Example 2: Re-ranking for Better Accuracy")
    print("=" * 60)
    
    config = FlipConfig(
        llm_provider="openai",
        use_hybrid_search=True,
        use_reranking=True,      # Enable re-ranking
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    
    flip = Flip(directory="./sample_docs", config=config)
    
    response = flip.query("Explain how RAG reduces hallucinations")
    print(f"\nAnswer: {response.answer}")
    print(f"\nMetadata:")
    print(f"  Re-ranking used: {response.metadata.get('reranking_used', False)}")
    print(f"  Citations: {len(response.citations)}")
    
    # Show citation scores
    print("\n  Citation scores (after re-ranking):")
    for i, citation in enumerate(response.citations[:3], 1):
        print(f"    [{i}] Score: {citation['score']:.3f} - {citation['source']}")


def example_caching():
    """Example demonstrating caching."""
    print("\n" + "=" * 60)
    print("Example 3: Caching for Performance")
    print("=" * 60)
    
    import time
    
    config = FlipConfig(
        llm_provider="openai",
        enable_cache=True,       # Enable caching
        cache_dir="./flip_cache"
    )
    
    flip = Flip(directory="./sample_docs", config=config)
    
    query = "What is artificial intelligence?"
    
    # First query (no cache)
    print("\nFirst query (no cache)...")
    start = time.time()
    response1 = flip.query(query)
    time1 = time.time() - start
    print(f"  Time: {time1:.2f}s")
    print(f"  Answer: {response1.answer[:100]}...")
    
    # Second query (cached)
    print("\nSecond query (cached)...")
    start = time.time()
    response2 = flip.query(query)
    time2 = time.time() - start
    print(f"  Time: {time2:.2f}s")
    print(f"  Speedup: {time1/time2:.1f}x faster!")


def example_all_features():
    """Example with all advanced features enabled."""
    print("\n" + "=" * 60)
    print("Example 4: All Advanced Features Combined")
    print("=" * 60)
    
    config = FlipConfig(
        llm_provider="openai",
        llm_model="gpt-4-turbo-preview",
        
        # Advanced retrieval
        use_hybrid_search=True,
        use_reranking=True,
        retrieval_top_k=5,
        
        # Caching
        enable_cache=True,
        
        # Chunking
        chunking_strategy="semantic",
        chunk_size=512,
    )
    
    flip = Flip(directory="./sample_docs", config=config)
    
    # Get stats
    stats = flip.get_stats()
    print("\n📊 Configuration:")
    print(f"  LLM: {stats['llm_provider']} ({stats['llm_model']})")
    print(f"  Hybrid Search: {'✅' if stats['hybrid_search'] else '❌'}")
    print(f"  Re-ranking: {'✅' if stats['reranking'] else '❌'}")
    print(f"  Caching: {'✅' if stats['caching'] else '❌'}")
    
    # Query
    response = flip.query("What are the benefits of using RAG?")
    
    print(f"\n💡 Answer: {response.answer}")
    print(f"\n📚 Sources: {len(response.citations)} documents")
    print(f"🔢 Tokens used: {response.metadata['tokens_used']}")
    
    # Show metadata
    print("\n🔍 Retrieval Details:")
    print(f"  Hybrid search: {response.metadata.get('hybrid_search_used', False)}")
    print(f"  Re-ranking: {response.metadata.get('reranking_used', False)}")


def example_chunking_comparison():
    """Compare different chunking strategies with advanced features."""
    print("\n" + "=" * 60)
    print("Example 5: Chunking Strategies with Advanced Features")
    print("=" * 60)
    
    strategies = ["token", "sentence", "semantic"]
    
    for strategy in strategies:
        print(f"\n📝 Testing {strategy} chunking...")
        
        config = FlipConfig(
            llm_provider="openai",
            chunking_strategy=strategy,
            use_hybrid_search=True,
            use_reranking=True,
        )
        
        flip = Flip(directory="./sample_docs", config=config)
        stats = flip.get_stats()
        
        print(f"  Chunks created: {stats['chunk_count']}")
        
        response = flip.query("What is RAG?")
        print(f"  Answer length: {len(response.answer)} chars")
        print(f"  Citations: {len(response.citations)}")


def main():
    """Run all advanced examples."""
    print("\n🚀 Flip SDK - Advanced Features Examples\n")
    
    try:
        example_hybrid_search()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_reranking()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_caching()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_all_features()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_chunking_comparison()
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n✅ Advanced examples complete!")


if __name__ == "__main__":
    main()
