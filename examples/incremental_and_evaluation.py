"""
Example demonstrating incremental updates and evaluation features.

This example shows:
1. Incremental document updates
2. RAG evaluation metrics
3. Performance monitoring
"""

from flip import Flip, FlipConfig
from flip.evaluation.metrics import RAGMetrics
import time


def example_incremental_updates():
    """Demonstrate incremental updates."""
    print("=" * 60)
    print("Example 1: Incremental Updates")
    print("=" * 60)
    
    config = FlipConfig(
        llm_provider="openai",
        use_hybrid_search=True,
        enable_cache=True,
    )
    
    # Initial indexing
    print("\n📚 Initial indexing...")
    flip = Flip(directory="./sample_docs", config=config)
    
    stats = flip.get_stats()
    print(f"  Documents: {stats['document_count']}")
    print(f"  Chunks: {stats['chunk_count']}")
    
    # Simulate adding a new document
    print("\n➕ Adding new documents...")
    print("  (Add some new files to ./sample_docs)")
    input("  Press Enter when ready...")
    
    # Refresh index (incremental update)
    update_stats = flip.refresh_index()
    
    print(f"\n📊 Update Summary:")
    print(f"  Added: {update_stats['added']}")
    print(f"  Updated: {update_stats['updated']}")
    print(f"  Deleted: {update_stats['deleted']}")
    
    # New stats
    new_stats = flip.get_stats()
    print(f"\n📈 New Stats:")
    print(f"  Total chunks: {new_stats['chunk_count']}")


def example_evaluation():
    """Demonstrate RAG evaluation."""
    print("\n" + "=" * 60)
    print("Example 2: RAG Evaluation")
    print("=" * 60)
    
    flip = Flip(directory="./sample_docs")
    
    # Define a test query with ground truth
    query = "What is artificial intelligence?"
    
    # You would normally have ground truth relevant doc IDs
    # For demo, we'll use the first few retrieved
    response = flip.query(query)
    relevant_ids = [c["chunk_id"] for c in response.citations[:3]]
    
    print(f"\n🔍 Evaluating query: '{query}'")
    
    # Evaluate
    result = flip.evaluate(query, relevant_ids, k=5)
    
    print(f"\n📊 Evaluation Results:")
    print(f"  Retrieval Precision@5: {result.retrieval_precision:.3f}")
    print(f"  Retrieval Recall@5: {result.retrieval_recall:.3f}")
    print(f"  Retrieval F1@5: {result.retrieval_f1:.3f}")
    print(f"  MRR: {result.mrr:.3f}")
    print(f"  NDCG@5: {result.ndcg:.3f}")
    print(f"  Answer Relevance: {result.answer_relevance:.3f}")
    print(f"  Faithfulness: {result.faithfulness:.3f}")
    print(f"  Overall Score: {result.overall_score:.3f}")


def example_monitoring():
    """Demonstrate performance monitoring."""
    print("\n" + "=" * 60)
    print("Example 3: Performance Monitoring")
    print("=" * 60)
    
    flip = Flip(directory="./sample_docs")
    
    # Run several queries
    queries = [
        "What is AI?",
        "Explain machine learning",
        "What are the benefits of RAG?",
        "How does deep learning work?",
    ]
    
    print("\n🔍 Running queries...")
    for query in queries:
        response = flip.query(query)
        print(f"  ✓ {query}")
    
    # Get monitoring stats
    stats = flip.get_monitoring_stats()
    
    print(f"\n📊 Monitoring Statistics:")
    print(f"  Total Queries: {stats['total_queries']}")
    print(f"  Total Tokens: {stats['total_tokens']}")
    print(f"  Avg Tokens/Query: {stats['avg_tokens_per_query']:.1f}")
    print(f"  Avg Retrieval Time: {stats['avg_retrieval_time']:.3f}s")
    print(f"  Avg Generation Time: {stats['avg_generation_time']:.3f}s")
    print(f"  Avg Total Time: {stats['avg_total_time']:.3f}s")
    
    # Get recent queries
    print(f"\n📝 Recent Queries:")
    recent = flip.get_recent_queries(n=3)
    for i, q in enumerate(recent, 1):
        print(f"  [{i}] {q['query'][:50]}...")
        print(f"      Tokens: {q['tokens_used']}, Time: {q['total_time']:.2f}s")


def example_batch_evaluation():
    """Demonstrate batch evaluation."""
    print("\n" + "=" * 60)
    print("Example 4: Batch Evaluation")
    print("=" * 60)
    
    flip = Flip(directory="./sample_docs")
    
    # Test queries with ground truth
    test_cases = [
        {
            "query": "What is artificial intelligence?",
            "relevant_ids": ["chunk_1", "chunk_2"]  # Replace with actual IDs
        },
        {
            "query": "Explain RAG",
            "relevant_ids": ["chunk_3", "chunk_4"]  # Replace with actual IDs
        },
    ]
    
    print("\n🧪 Running batch evaluation...")
    
    results = []
    for i, test in enumerate(test_cases, 1):
        print(f"\n  Test {i}: {test['query']}")
        result = flip.evaluate(test['query'], test['relevant_ids'])
        results.append(result)
        print(f"    Overall Score: {result.overall_score:.3f}")
    
    # Average scores
    avg_precision = sum(r.retrieval_precision for r in results) / len(results)
    avg_recall = sum(r.retrieval_recall for r in results) / len(results)
    avg_f1 = sum(r.retrieval_f1 for r in results) / len(results)
    avg_overall = sum(r.overall_score for r in results) / len(results)
    
    print(f"\n📊 Average Scores:")
    print(f"  Precision: {avg_precision:.3f}")
    print(f"  Recall: {avg_recall:.3f}")
    print(f"  F1: {avg_f1:.3f}")
    print(f"  Overall: {avg_overall:.3f}")


def main():
    """Run all examples."""
    print("\n🚀 Flip SDK - Incremental Updates & Evaluation Examples\n")
    
    try:
        example_incremental_updates()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_evaluation()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_monitoring()
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        example_batch_evaluation()
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n✅ Examples complete!")


if __name__ == "__main__":
    main()
