"""
Quick start example using HuggingFace google/embeddinggemma-300m for embeddings.

This example demonstrates how to use Google's EmbeddingGemma model
from HuggingFace for generating embeddings in the Flip SDK.
"""

from flip import Flip, FlipConfig
import os

# Set your API keys
os.environ["HUGGINGFACE_API_KEY"] = "your-huggingface-api-key-here"
os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"  # For LLM


def main():
    """Use google/embeddinggemma-300m for embeddings."""
    
    print("🚀 Flip SDK with google/embeddinggemma-300m")
    print("=" * 60)
    
    # Configure Flip to use HuggingFace embeddings
    config = FlipConfig(
        # Embedding configuration - Use HuggingFace with EmbeddingGemma
        embedding_provider="huggingface",
        embedding_model="google/embeddinggemma-300m",
        
        # LLM configuration - You can use any provider
        llm_provider="openai",  # or "anthropic", "google", etc.
        llm_model="gpt-4-turbo-preview",
        
        # Optional: Advanced features
        use_hybrid_search=True,
        use_reranking=True,
        enable_cache=True,
        
        # Chunking
        chunking_strategy="semantic",
        chunk_size=512,
    )
    
    # Initialize Flip with your documents
    print("\n📂 Indexing documents...")
    flip = Flip(directory="./sample_docs", config=config)
    
    # Get stats to verify configuration
    stats = flip.get_stats()
    print(f"\n📊 Configuration:")
    print(f"  Embedding Provider: {stats['embedding_provider']}")
    print(f"  Embedding Model: {stats['embedding_model']}")
    print(f"  LLM Provider: {stats['llm_provider']}")
    print(f"  LLM Model: {stats['llm_model']}")
    print(f"  Documents: {stats['document_count']}")
    print(f"  Chunks: {stats['chunk_count']}")
    
    # Query your documents
    print("\n🔍 Querying documents...")
    questions = [
        "What is artificial intelligence?",
        "Explain the benefits of RAG",
        "How does machine learning work?"
    ]
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        response = flip.query(question)
        
        print(f"💡 Answer: {response.answer}")
        print(f"📚 Sources: {len(response.citations)} documents")
        print(f"🔢 Tokens: {response.metadata['tokens_used']}")


if __name__ == "__main__":
    main()
