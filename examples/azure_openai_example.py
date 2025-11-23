"""
Example using Azure OpenAI with Flip SDK.

This example demonstrates how to use Azure OpenAI for both
LLM generation and embeddings.
"""

from flip import Flip, FlipConfig
import os

# Set your Azure OpenAI credentials
os.environ["AZURE_OPENAI_API_KEY"] = "your-azure-openai-api-key"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-resource.openai.azure.com/"


def example_azure_openai_llm():
    """Example using Azure OpenAI for LLM."""
    print("=" * 60)
    print("Example 1: Azure OpenAI LLM")
    print("=" * 60)
    
    config = FlipConfig(
        # Use Azure OpenAI for LLM
        llm_provider="azure-openai",
        llm_model="gpt-4",  # Your deployment name
        
        # Use regular OpenAI for embeddings (or azure-openai)
        embedding_provider="openai",
        
        # Azure OpenAI configuration
        azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_openai_api_version="2024-02-15-preview",
    )
    
    flip = Flip(directory="./sample_docs", config=config)
    
    response = flip.query("What is artificial intelligence?")
    print(f"\nAnswer: {response.answer}")
    print(f"Model: {response.metadata['model']}")
    print(f"Tokens: {response.metadata['tokens_used']}")


def example_azure_openai_embeddings():
    """Example using Azure OpenAI for embeddings."""
    print("\n" + "=" * 60)
    print("Example 2: Azure OpenAI Embeddings")
    print("=" * 60)
    
    config = FlipConfig(
        # Use OpenAI for LLM
        llm_provider="openai",
        
        # Use Azure OpenAI for embeddings
        embedding_provider="azure-openai",
        embedding_model="text-embedding-ada-002",  # Your deployment name
        
        # Azure OpenAI configuration
        azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )
    
    flip = Flip(directory="./sample_docs", config=config)
    
    stats = flip.get_stats()
    print(f"\nEmbedding Provider: {stats['embedding_provider']}")
    print(f"Embedding Model: {stats['embedding_model']}")
    
    response = flip.query("Explain RAG")
    print(f"\nAnswer: {response.answer[:200]}...")


def example_azure_openai_both():
    """Example using Azure OpenAI for both LLM and embeddings."""
    print("\n" + "=" * 60)
    print("Example 3: Azure OpenAI for LLM + Embeddings")
    print("=" * 60)
    
    config = FlipConfig(
        # Use Azure OpenAI for both
        llm_provider="azure-openai",
        llm_model="gpt-4",  # Your LLM deployment name
        
        embedding_provider="azure-openai",
        embedding_model="text-embedding-ada-002",  # Your embedding deployment name
        
        # Azure OpenAI configuration
        azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_openai_api_version="2024-02-15-preview",
        
        # Advanced features
        use_hybrid_search=True,
        use_reranking=True,
        enable_cache=True,
    )
    
    flip = Flip(directory="./sample_docs", config=config)
    
    stats = flip.get_stats()
    print(f"\n📊 Configuration:")
    print(f"  LLM: {stats['llm_provider']} ({stats['llm_model']})")
    print(f"  Embedding: {stats['embedding_provider']} ({stats['embedding_model']})")
    print(f"  Hybrid Search: {'✅' if stats['hybrid_search'] else '❌'}")
    print(f"  Re-ranking: {'✅' if stats['reranking'] else '❌'}")
    
    response = flip.query("What are the benefits of using RAG?")
    print(f"\n💡 Answer: {response.answer}")
    print(f"\n📚 Sources: {len(response.citations)} documents")


def example_azure_openai_with_deployment_names():
    """Example with explicit deployment names."""
    print("\n" + "=" * 60)
    print("Example 4: Explicit Deployment Names")
    print("=" * 60)
    
    config = FlipConfig(
        llm_provider="azure-openai",
        embedding_provider="azure-openai",
    )
    
    # Initialize with deployment names
    from flip.generation.factory import LLMFactory
    from flip.embedding.factory import EmbedderFactory
    
    # Create LLM with specific deployment
    llm = LLMFactory.create(
        provider="azure-openai",
        model="gpt-4",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        deployment_name="my-gpt4-deployment"  # Your actual deployment name
    )
    
    # Create embedder with specific deployment
    embedder = EmbedderFactory.create(
        provider="azure-openai",
        model="text-embedding-ada-002",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        deployment_name="my-embedding-deployment"  # Your actual deployment name
    )
    
    print("✅ Created Azure OpenAI LLM and Embedder with custom deployments")


def main():
    """Run all Azure OpenAI examples."""
    print("\n🚀 Flip SDK - Azure OpenAI Examples\n")
    
    try:
        example_azure_openai_llm()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_azure_openai_embeddings()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_azure_openai_both()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_azure_openai_with_deployment_names()
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n✅ Azure OpenAI examples complete!")


if __name__ == "__main__":
    main()
