# Advanced Features Guide

This guide covers the advanced features of the Flip SDK that go beyond basic RAG functionality.

## 🔀 Hybrid Search

Hybrid search combines **vector similarity search** with **keyword search (BM25)** for better retrieval accuracy.

### How It Works

1. **Vector Search**: Finds semantically similar documents using embeddings
2. **BM25 Search**: Finds documents with keyword matches
3. **Reciprocal Rank Fusion**: Combines both results intelligently

### Usage

```python
from flip import Flip, FlipConfig

config = FlipConfig(
    use_hybrid_search=True,  # Enable hybrid search
    alpha=0.5  # 0.5 = equal weight for vector and keyword search
)

flip = Flip(directory="./docs", config=config)
response = flip.query("What is machine learning?")
```

### Benefits

- Better recall for keyword-specific queries
- Improved semantic understanding
- More robust to query variations

## 📊 Re-ranking

Re-ranking uses cross-encoder models to re-score retrieved documents for better relevance.

### How It Works

1. Initial retrieval gets top-K documents (e.g., top 10)
2. Cross-encoder re-scores each document against the query
3. Documents are re-ranked by new scores
4. Top-N documents are returned (e.g., top 5)

### Usage

```python
config = FlipConfig(
    use_reranking=True,
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"  # Default
)

flip = Flip(directory="./docs", config=config)
```

### Available Models

- `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast, good quality)
- `cross-encoder/ms-marco-MiniLM-L-12-v2` (slower, better quality)
- `cross-encoder/ms-marco-TinyBERT-L-2-v2` (fastest, lower quality)

## 🧠 Query Enhancement

Query enhancement improves retrieval by processing queries before searching.

### Features

#### 1. Query Classification

Automatically classifies queries into types:
- **Factual**: "What is X?"
- **Analytical**: "Why does X happen?"
- **Creative**: "Generate a plan for X"
- **Conversational**: "Hello", "Thank you"

#### 2. Query Rewriting

Rewrites queries to be more specific and retrieval-friendly.

```python
# Original: "How does it work?"
# Rewritten: "How does the authentication system work?"
```

#### 3. Query Decomposition

Breaks complex queries into simpler sub-queries.

```python
# Complex: "Compare RAG and fine-tuning and explain when to use each"
# Sub-queries:
# 1. "What is RAG?"
# 2. "What is fine-tuning?"
# 3. "When should you use RAG vs fine-tuning?"
```

#### 4. HyDE (Hypothetical Document Embeddings)

Generates a hypothetical answer to improve retrieval.

```python
# Query: "What are the benefits of RAG?"
# Hypothetical doc: "RAG provides several benefits including reduced hallucinations..."
# (This hypothetical doc is embedded and used for retrieval)
```

### Usage

Query enhancement is enabled by default when using the pipeline. It automatically:
- Detects if retrieval is needed
- Classifies the query type
- Can optionally rewrite or decompose queries

## 💾 Caching

Flip includes multi-level caching for improved performance.

### Cache Types

#### 1. Embedding Cache

Caches text embeddings to avoid recomputation.

- **LRU eviction**: Least recently used items are removed
- **Persistent storage**: Saved to disk
- **Max size**: Configurable (default: 10,000 embeddings)

#### 2. Query Cache

Caches query results for instant responses.

- **LRU eviction**: Least recently used queries removed
- **Persistent storage**: Saved to disk
- **Max size**: Configurable (default: 1,000 queries)

### Usage

```python
config = FlipConfig(
    enable_cache=True,
    cache_dir="./flip_cache"  # Where to store cache
)

flip = Flip(directory="./docs", config=config)

# First query: ~2 seconds
response1 = flip.query("What is AI?")

# Second query (cached): ~0.1 seconds
response2 = flip.query("What is AI?")
```

### Cache Management

```python
# Clear all caches
flip.clear()

# Save caches manually
flip.pipeline.save_caches()
```

## ⚙️ Configuration Options

### Complete Configuration

```python
from flip import FlipConfig

config = FlipConfig(
    # LLM
    llm_provider="openai",
    llm_model="gpt-4-turbo-preview",
    llm_temperature=0.7,
    llm_max_tokens=1024,
    
    # Embedding
    embedding_provider="openai",
    embedding_model="text-embedding-3-small",
    
    # Vector Store
    vector_store="chroma",
    vector_store_path="./flip_data",
    
    # Chunking
    chunking_strategy="semantic",  # token, sentence, semantic, recursive
    chunk_size=512,
    chunk_overlap=50,
    
    # Retrieval
    retrieval_top_k=5,
    use_hybrid_search=True,
    use_reranking=True,
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    
    # Caching
    enable_cache=True,
    cache_dir="./flip_cache",
    
    # Performance
    batch_size=32,
    show_progress=True,
)
```

## 🎯 Best Practices

### 1. Choose the Right Chunking Strategy

- **Token**: Fast, consistent chunk sizes
- **Sentence**: Preserves sentence boundaries
- **Semantic**: Best for documents with clear paragraphs
- **Recursive**: Good for mixed content

### 2. Tune Hybrid Search

```python
# More weight on vector search (semantic)
config = FlipConfig(use_hybrid_search=True, alpha=0.7)

# More weight on keyword search
config = FlipConfig(use_hybrid_search=True, alpha=0.3)
```

### 3. Balance Speed vs Accuracy

```python
# Fast (no reranking)
config = FlipConfig(use_reranking=False)

# Accurate (with reranking)
config = FlipConfig(use_reranking=True)

# Very accurate (larger reranker model)
config = FlipConfig(
    use_reranking=True,
    reranker_model="cross-encoder/ms-marco-MiniLM-L-12-v2"
)
```

### 4. Use Caching for Production

```python
config = FlipConfig(
    enable_cache=True,
    cache_dir="./production_cache"
)
```

## 📈 Performance Comparison

| Feature | Speed Impact | Accuracy Impact |
|---------|-------------|-----------------|
| Basic Vector Search | Baseline | Baseline |
| + Hybrid Search | -10% | +15% |
| + Re-ranking | -30% | +25% |
| + Caching (cached) | +90% | 0% |
| All Features | -40% (first) / +80% (cached) | +30% |

## 🔍 Debugging

### Check Feature Status

```python
stats = flip.get_stats()
print(f"Hybrid search: {stats['hybrid_search']}")
print(f"Re-ranking: {stats['reranking']}")
print(f"Caching: {stats['caching']}")
```

### View Metadata

```python
response = flip.query("What is AI?")
print(response.metadata)
# {
#   'tokens_used': 150,
#   'model': 'gpt-4-turbo-preview',
#   'retrieval_used': True,
#   'hybrid_search_used': True,
#   'reranking_used': True
# }
```

## 🚀 Next Steps

- Try different configurations
- Experiment with chunking strategies
- Compare with and without advanced features
- Monitor performance and accuracy

For more examples, see `examples/advanced_features.py`.
