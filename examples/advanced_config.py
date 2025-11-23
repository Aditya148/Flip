"""
Advanced configuration example for Flip SDK.

This example demonstrates:
1. Custom configuration
2. Multiple LLM providers
3. Different chunking strategies
4. Custom settings
"""

from flip import Flip, FlipConfig
import os


def example_custom_config():
    """Example with custom configuration."""
    print("=" * 60)
    print("Example 1: Custom Configuration")
    print("=" * 60)
    
    config = FlipConfig(
        # LLM settings
        llm_provider="openai",
        llm_model="gpt-4-turbo-preview",
        llm_temperature=0.7,
        llm_max_tokens=1024,
        
        # Embedding settings
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        
        # Chunking settings
        chunking_strategy="semantic",
        chunk_size=512,
        chunk_overlap=50,
        
        # Retrieval settings
        retrieval_top_k=5,
        use_hybrid_search=True,
        use_reranking=True,
        
        # Performance settings
        batch_size=32,
        show_progress=True,
        enable_cache=True,
    )
    
    flip = Flip(directory="./sample_docs", config=config)
    response = flip.query("What is this about?")
    
    print(f"Answer: {response.answer}")
    print(f"Model: {response.metadata['model']}")
    print(f"Tokens: {response.metadata['tokens_used']}")


def example_multiple_providers():
    """Example comparing different LLM providers."""
    print("\n" + "=" * 60)
    print("Example 2: Multiple LLM Providers")
    print("=" * 60)
    
    providers = [
        ("openai", "gpt-4-turbo-preview"),
        ("anthropic", "claude-3-sonnet-20240229"),
        ("google", "gemini-pro"),
    ]
    
    question = "Summarize the main points in one sentence."
    
    for provider, model in providers:
        print(f"\n🤖 Testing {provider} ({model})...")
        
        try:
            flip = Flip(
                directory="./sample_docs",
                llm_provider=provider,
                llm_model=model
            )
            
            response = flip.query(question)
            print(f"Answer: {response.answer}")
            print(f"Tokens: {response.metadata['tokens_used']}")
            
        except Exception as e:
            print(f"Error: {str(e)}")


def example_chunking_strategies():
    """Example comparing different chunking strategies."""
    print("\n" + "=" * 60)
    print("Example 3: Chunking Strategies")
    print("=" * 60)
    
    strategies = ["token", "sentence", "semantic", "recursive"]
    
    for strategy in strategies:
        print(f"\n📝 Testing {strategy} chunking...")
        
        config = FlipConfig(
            chunking_strategy=strategy,
            chunk_size=512,
            chunk_overlap=50
        )
        
        flip = Flip(directory="./sample_docs", config=config)
        stats = flip.get_stats()
        
        print(f"  Chunks created: {stats['chunk_count']}")
        print(f"  Avg chunks per doc: {stats['chunk_count'] / max(stats['document_count'], 1):.1f}")


def example_incremental_indexing():
    """Example of adding documents incrementally."""
    print("\n" + "=" * 60)
    print("Example 4: Incremental Indexing")
    print("=" * 60)
    
    # Start with empty index
    flip = Flip()
    
    print("Initial stats:", flip.get_stats())
    
    # Add documents one by one
    documents = [
        "./sample_docs/doc1.txt",
        "./sample_docs/doc2.pdf",
        "./sample_docs/doc3.md",
    ]
    
    for doc in documents:
        if os.path.exists(doc):
            print(f"\nAdding {doc}...")
            flip.add_documents([doc])
            stats = flip.get_stats()
            print(f"  Total chunks: {stats['chunk_count']}")
    
    # Query
    response = flip.query("What do these documents contain?")
    print(f"\nAnswer: {response.answer}")


def example_local_embeddings():
    """Example using local embeddings (no API key required)."""
    print("\n" + "=" * 60)
    print("Example 5: Local Embeddings")
    print("=" * 60)
    
    config = FlipConfig(
        # Use local embeddings (no API key needed)
        embedding_provider="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        
        # Use Ollama for local LLM (no API key needed)
        llm_provider="ollama",
        llm_model="llama2",
    )
    
    print("Using fully local setup (no API keys required)!")
    print("  Embeddings: sentence-transformers")
    print("  LLM: Ollama")
    
    flip = Flip(directory="./sample_docs", config=config)
    response = flip.query("What is this about?")
    
    print(f"\nAnswer: {response.answer}")


def main():
    """Run all examples."""
    print("\n🚀 Flip SDK - Advanced Examples\n")
    
    # Run examples
    try:
        example_custom_config()
    except Exception as e:
        print(f"Error in example 1: {e}")
    
    try:
        example_multiple_providers()
    except Exception as e:
        print(f"Error in example 2: {e}")
    
    try:
        example_chunking_strategies()
    except Exception as e:
        print(f"Error in example 3: {e}")
    
    try:
        example_incremental_indexing()
    except Exception as e:
        print(f"Error in example 4: {e}")
    
    try:
        example_local_embeddings()
    except Exception as e:
        print(f"Error in example 5: {e}")
    
    print("\n✅ Examples complete!")


if __name__ == "__main__":
    main()
