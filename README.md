# Flip RAG SDK 🚀

**Fully Automated Retrieval-Augmented Generation**

Flip is the simplest yet most powerful RAG SDK. Initialize with a directory and start querying - that's it!

## ✨ Features

- 🎯 **Ultra-Simple API**: Just `Flip(directory="./docs")` to get started
- 🔄 **Fully Automated**: Handles everything from document loading to response generation
- 🤖 **Multiple LLM Providers**: OpenAI, Azure OpenAI, Anthropic, Google, HuggingFace, Meta, Ollama
- 📊 **Smart Chunking**: Token, sentence, semantic, and recursive strategies
- 🔍 **Hybrid Search**: Vector + keyword search for better accuracy
- 💾 **9 Vector Databases**: Pinecone, Qdrant, Weaviate, Milvus, FAISS, Pgvector, Redis, Elasticsearch, MongoDB
- 📝 **Rich File Support**: PDF, DOCX, TXT, MD, JSON, CSV, HTML, code files
- 🎨 **Extensible**: Easy to add custom providers and strategies

## 🚀 Advanced Features

Flip includes production-ready advanced features:

- **🔀 Hybrid Search**: Combines vector similarity + BM25 keyword search with reciprocal rank fusion
- **📊 Re-ranking**: Cross-encoder models for improved retrieval accuracy
- **🧠 Query Enhancement**: Automatic query classification, rewriting, decomposition, and HyDE
- **💾 Intelligent Caching**: LRU caching for embeddings and queries with disk persistence
- **⚡ Pipeline Orchestration**: Seamless integration of all components
- **🔄 Incremental Updates**: Efficient document change tracking
- **📈 Performance Monitoring**: Query logging and analytics
- **📊 Built-in Evaluation**: Precision, recall, F1, MRR, NDCG metrics

See [ADVANCED_FEATURES.md](ADVANCED_FEATURES.md) for detailed documentation.

## 🚀 Quick Start

### Installation

```bash
pip install flip-rag
```

### Basic Usage

```python
from flip import Flip

# Initialize with a directory - that's it!
flip = Flip(directory="./docs")

# Query your documents
response = flip.query("What is the main topic?")

print(response.answer)
print(response.citations)
```

### Choose Your LLM Provider

```python
# Use OpenAI (default)
flip = Flip(directory="./docs", llm_provider="openai")

# Use Azure OpenAI
flip = Flip(directory="./docs", llm_provider="azure-openai")

# Use Anthropic Claude
flip = Flip(directory="./docs", llm_provider="anthropic")

# Use Google Gemini
flip = Flip(directory="./docs", llm_provider="google")

# Use HuggingFace
flip = Flip(directory="./docs", llm_provider="huggingface")

# Use local Ollama
flip = Flip(directory="./docs", llm_provider="ollama")
```

### Advanced Configuration

```python
from flip import Flip, FlipConfig

# Custom configuration
config = FlipConfig(
    llm_provider="openai",
    llm_model="gpt-4-turbo-preview",
    embedding_provider="openai",
    chunking_strategy="semantic",
    chunk_size=512,
    retrieval_top_k=5,
    use_hybrid_search=True,
)

flip = Flip(directory="./docs", config=config)
```

### Azure OpenAI Configuration

```python
from flip import Flip, FlipConfig

config = FlipConfig(
    llm_provider="azure-openai",
    llm_model="gpt-4",  # Your deployment name
    
    embedding_provider="azure-openai",
    embedding_model="text-embedding-ada-002",  # Your deployment name
    
    azure_openai_api_key="your-key",
    azure_openai_endpoint="https://your-resource.openai.azure.com/",
)

flip = Flip(directory="./docs", config=config)
response = flip.query("What is AI?")
```

## 📚 Supported LLM Providers

| Provider | Models | API Key Required |
|----------|--------|------------------|
| **OpenAI** | GPT-4, GPT-3.5 | Yes (`OPENAI_API_KEY`) |
| **Azure OpenAI** | GPT-4, GPT-3.5 | Yes (`AZURE_OPENAI_API_KEY`) |
| **Anthropic** | Claude 3 (Opus, Sonnet, Haiku) | Yes (`ANTHROPIC_API_KEY`) |
| **Google** | Gemini Pro | Yes (`GOOGLE_API_KEY`) |
| **HuggingFace** | Llama 2, Mistral, etc. | Yes (`HUGGINGFACE_API_KEY`) |
| **Meta** | Llama models via HuggingFace | Yes (`HUGGINGFACE_API_KEY`) |
| **Ollama** | Any local model | No (runs locally) |

## 🔗 Supported Embedding Providers

| Provider | Models | API Key Required |
|----------|--------|------------------|
| **OpenAI** | text-embedding-3-small/large | Yes |
| **Azure OpenAI** | text-embedding-ada-002 | Yes |
| **Google** | models/embedding-001 | Yes |
| **HuggingFace** | Various models | Yes |
| **Sentence Transformers** | all-MiniLM-L6-v2, etc. | No (local) |

## 📄 Supported File Formats

- **Documents**: PDF, DOCX, TXT, MD
- **Data**: JSON, CSV
- **Web**: HTML, HTM
- **Code**: Python, JavaScript, TypeScript, Java, C++, Go, Rust, Ruby

## 🔧 API Reference

### Flip Class

```python
flip = Flip(
    directory: Optional[str] = None,
    config: Optional[FlipConfig] = None,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    **kwargs
)
```

#### Methods

- **`query(question: str) -> FlipResponse`**: Query your documents
- **`index_directory(directory: str)`**: Index a directory
- **`add_documents(file_paths: List[str])`**: Add specific files
- **`refresh_index()`**: Incrementally update index with changed documents
- **`evaluate(query, relevant_ids, k=5)`**: Evaluate RAG performance
- **`get_monitoring_stats()`**: Get performance statistics
- **`clear()`**: Clear all indexed documents
- **`get_stats()`**: Get indexing statistics

### FlipResponse

```python
@dataclass
class FlipResponse:
    answer: str                    # Generated answer
    citations: List[Dict]          # Source citations
    context_chunks: List[str]      # Retrieved chunks
    metadata: Dict[str, Any]       # Additional metadata
```

### FlipConfig

```python
config = FlipConfig(
    # LLM Configuration
    llm_provider="openai",         # openai, azure-openai, anthropic, google, huggingface, ollama
    llm_model=None,                # Auto-selected if None
    llm_temperature=0.7,
    llm_max_tokens=1024,
    
    # Embedding Configuration
    embedding_provider="openai",   # openai, azure-openai, google, sentence-transformers, huggingface
    embedding_model=None,          # Auto-selected if None
    
    # Azure OpenAI (if using azure-openai provider)
    azure_openai_api_key=None,
    azure_openai_endpoint=None,
    azure_openai_api_version="2024-02-15-preview",
    
    # Vector Store
    vector_store="chroma",         # chroma (more coming soon)
    vector_store_path=None,        # Auto: ./flip_data
    
    # Chunking
    chunking_strategy="semantic",  # token, sentence, semantic, recursive
    chunk_size=512,
    chunk_overlap=50,
    
    # Retrieval
    retrieval_top_k=5,
    use_hybrid_search=True,
    use_reranking=True,
    
    # Cache
    enable_cache=True,
    cache_dir=None,                # Auto: ./flip_cache
)
```

## 🌟 Examples

### Example 1: Simple Q&A

```python
from flip import Flip

flip = Flip(directory="./company_docs")
response = flip.query("What is our refund policy?")

print(f"Answer: {response.answer}")
print(f"Sources: {len(response.citations)} documents")
```

### Example 2: Multiple Providers

```python
from flip import Flip

# Try different providers
providers = ["openai", "azure-openai", "anthropic", "google"]

for provider in providers:
    flip = Flip(directory="./docs", llm_provider=provider)
    response = flip.query("Summarize the main points")
    print(f"{provider}: {response.answer[:100]}...")
```

### Example 3: Incremental Updates

```python
from flip import Flip

flip = Flip(directory="./docs")

# Later, refresh with only changed documents
stats = flip.refresh_index()

print(f"Added: {stats['added']}")
print(f"Updated: {stats['updated']}")
print(f"Deleted: {stats['deleted']}")
```

### Example 4: Evaluation

```python
from flip import Flip

flip = Flip(directory="./docs")

# Evaluate a query
result = flip.evaluate(
    query="What is AI?",
    relevant_doc_ids=["chunk_1", "chunk_2"],  # Ground truth
    k=5
)

print(f"Precision: {result.retrieval_precision:.3f}")
print(f"Overall Score: {result.overall_score:.3f}")
```

## 🔑 Environment Variables

Set API keys as environment variables:

```bash
export OPENAI_API_KEY="your-key-here"
export AZURE_OPENAI_API_KEY="your-azure-key-here"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export ANTHROPIC_API_KEY="your-key-here"
export GOOGLE_API_KEY="your-key-here"
export HUGGINGFACE_API_KEY="your-key-here"
```

Or use a `.env` file:

```env
OPENAI_API_KEY=your-key-here
AZURE_OPENAI_API_KEY=your-azure-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
ANTHROPIC_API_KEY=your-key-here
GOOGLE_API_KEY=your-key-here
HUGGINGFACE_API_KEY=your-key-here
```

## 🗄️ Multi-Database Support

Flip supports **9 major vector and NoSQL databases**, giving you complete flexibility to choose the perfect storage backend for your use case:

### Vector Databases

#### Pinecone (Cloud-Native)
```python
config = FlipConfig(
    vector_store="pinecone",
    pinecone_api_key="your-key",
    pinecone_environment="gcp-starter",
    pinecone_index_name="flip-index"
)
flip = Flip(directory="./docs", config=config)
```

#### Qdrant (Local/Cloud)
```python
config = FlipConfig(
    vector_store="qdrant",
    qdrant_url="http://localhost:6333",  # or cloud URL
    qdrant_api_key="your-key"  # for cloud
)
flip = Flip(directory="./docs", config=config)
```

#### Weaviate (GraphQL)
```python
config = FlipConfig(
    vector_store="weaviate",
    weaviate_url="http://localhost:8080",
    weaviate_api_key="your-key"  # for cloud
)
flip = Flip(directory="./docs", config=config)
```

#### Milvus (Enterprise)
```python
config = FlipConfig(
    vector_store="milvus",
    milvus_host="localhost",
    milvus_port=19530
)
flip = Flip(directory="./docs", config=config)
```

#### FAISS (Local, Ultra-Fast)
```python
config = FlipConfig(
    vector_store="faiss",
    faiss_index_type="HNSW",  # or IVF, Flat
    persist_directory="./faiss_data"
)
flip = Flip(directory="./docs", config=config)
```

#### Pgvector (PostgreSQL)
```python
config = FlipConfig(
    vector_store="pgvector",
    pgvector_host="localhost",
    pgvector_database="flip_db",
    pgvector_user="postgres",
    pgvector_password="password"
)
flip = Flip(directory="./docs", config=config)
```

### NoSQL/Hybrid Databases

#### Redis (In-Memory + Cache)
```python
config = FlipConfig(
    vector_store="redis",
    redis_host="localhost",
    redis_port=6379
)
flip = Flip(directory="./docs", config=config)
```

#### Elasticsearch (Search Engine)
```python
config = FlipConfig(
    vector_store="elasticsearch",
    elasticsearch_url="http://localhost:9200",
    elasticsearch_api_key="your-key"
)
flip = Flip(directory="./docs", config=config)
```

#### MongoDB (Document Store)
```python
config = FlipConfig(
    vector_store="mongodb",
    mongodb_uri="mongodb://localhost:27017/",
    mongodb_database="flip_db"
)
flip = Flip(directory="./docs", config=config)
```

### Database Comparison

| Database | Type | Best For | Deployment |
|----------|------|----------|------------|
| **Pinecone** | Vector | Production, cloud-native | Cloud |
| **Qdrant** | Vector | Flexibility, snapshots | Local/Cloud |
| **Weaviate** | Vector | GraphQL, schema management | Local/Cloud |
| **Milvus** | Vector | Enterprise, scalability | Local/Cloud |
| **FAISS** | Vector | Local, speed, GPU support | Local |
| **Pgvector** | Vector | PostgreSQL users, ACID | Local/Cloud |
| **Redis** | Hybrid | Caching, real-time | Local/Cloud |
| **Elasticsearch** | Hybrid | Full-text + vectors | Local/Cloud |
| **MongoDB** | NoSQL | Metadata-rich storage | Local/Cloud |

### Key Features Across All Databases

- ✅ **Unified API**: Same interface for all databases
- ✅ **Lazy Loading**: Optional dependencies (install only what you need)
- ✅ **Health Checks**: Monitor database status
- ✅ **Batch Operations**: Efficient bulk inserts
- ✅ **Metadata Filtering**: Filter by custom metadata
- ✅ **Statistics**: Track vector counts and performance

See individual database examples in the `examples/` directory.

## 🎯 Design Philosophy

Flip is designed around three core principles:

1. **Simplicity First**: The API should be so simple that you can get started in 2 lines of code
2. **Automation**: Handle all the complexity internally - chunking, embedding, retrieval, generation
3. **Flexibility**: Support multiple providers and allow customization when needed

## 🛠️ Advanced Features

### Custom Providers

You can register custom LLM or embedding providers:

```python
from flip.generation.factory import LLMFactory
from flip.generation.base import BaseLLM

class MyCustomLLM(BaseLLM):
    # Implement required methods
    pass

LLMFactory.register_provider("mycustom", MyCustomLLM)

# Use it
flip = Flip(directory="./docs", llm_provider="mycustom")
```

## 📊 Performance

Flip is optimized for performance:

- **Batch Processing**: Embeddings generated in batches
- **Caching**: Automatic caching of embeddings and queries
- **Efficient Storage**: Support for 9 optimized vector databases
- **Incremental Updates**: Only re-index changed documents
- **Performance Testing**: Built-in benchmarking framework

## 🧪 Testing

Flip includes comprehensive testing infrastructure:

```python
# Use test fixtures for any database
from tests.fixtures import all_vector_stores

def test_my_feature(all_vector_stores):
    # Test runs against all available databases
    pass

# Benchmark performance
from tests.performance import PerformanceTester

tester = PerformanceTester(vector_store)
tester.run_full_benchmark()
tester.print_results()
```

## 🤝 Contributing

Contributions are welcome! Please see our contributing guidelines.

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

Flip builds on the shoulders of giants:
- OpenAI, Azure OpenAI, Anthropic, Google for LLM APIs
- Pinecone, Qdrant, Weaviate, Milvus, FAISS, Pgvector, Redis, Elasticsearch, MongoDB for vector storage
- Sentence Transformers for local embeddings
- And many other open-source projects

## 📞 Support

- **Documentation**: [https://flip-rag.readthedocs.io](https://flip-rag.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/yourusername/flip/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/flip/discussions)

---

**Made with ❤️ by the Flip Team**

