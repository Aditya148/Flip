"""
Basic usage example for Flip SDK.

This example demonstrates the simplest way to use Flip:
1. Initialize with a directory
2. Query your documents
"""

from flip import Flip


def main():
    # Initialize Flip with a directory
    # This will automatically:
    # - Load all supported documents
    # - Chunk them appropriately
    # - Generate embeddings
    # - Store in vector database
    
    print("Initializing Flip...")
    flip = Flip(directory="./sample_docs")
    
    # Get stats
    stats = flip.get_stats()
    print(f"\n📊 Indexing Stats:")
    print(f"  Documents: {stats['document_count']}")
    print(f"  Chunks: {stats['chunk_count']}")
    print(f"  LLM: {stats['llm_provider']} ({stats['llm_model']})")
    print(f"  Embeddings: {stats['embedding_provider']} ({stats['embedding_model']})")
    
    # Query the documents
    print("\n🔍 Querying documents...")
    
    questions = [
        "What is the main topic of these documents?",
        "Can you summarize the key points?",
        "What are the most important takeaways?"
    ]
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        response = flip.query(question)
        
        print(f"💡 Answer: {response.answer}")
        print(f"📚 Sources: {len(response.citations)} documents used")
        print(f"🔢 Tokens: {response.metadata['tokens_used']}")
        
        # Show citations
        if response.citations:
            print("\n📖 Citations:")
            for i, citation in enumerate(response.citations[:3], 1):
                print(f"  [{i}] {citation['source']}")
                print(f"      Score: {citation['score']:.3f}")
                print(f"      Preview: {citation['text_preview'][:100]}...")


if __name__ == "__main__":
    main()
