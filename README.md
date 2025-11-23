# Flip RAG SDK 🚀

**Fully Automated Retrieval-Augmented Generation**

Flip is the simplest yet most powerful RAG SDK. Initialize with a directory and start querying - that's it!

## ✨ Features

- 🎯 **Ultra-Simple API**: Just `Flip(directory="./docs")` to get started
- 🔄 **Fully Automated**: Handles everything from document loading to response generation
- 🤖 **Multiple LLM Providers**: OpenAI, Anthropic, Google, HuggingFace, Meta, Ollama
- 📊 **Smart Chunking**: Token, sentence, semantic, and recursive strategies
- 🔍 **Hybrid Search**: Vector + keyword search for better accuracy
- 💾 **Persistent Storage**: ChromaDB for zero-config vector storage
- 📝 **Rich File Support**: PDF, DOCX, TXT, MD, JSON, CSV, HTML, code files
- 🎨 **Extensible**: Easy to add custom providers and strategies

## 🚀 Advanced Features

Flip includes production-ready advanced features:

- **🔀 Hybrid Search**: Combines vector similarity + BM25 keyword search with reciprocal rank fusion
- **📊 Re-ranking**: Cross-encoder models for improved retrieval accuracy
- **🧠 Query Enhancement**: Automatic query classification, rewriting, decomposition, and HyDE
- **💾 Intelligent Caching**: LRU caching for embeddings and queries with disk persistence
- **⚡ Pipeline Orchestration**: Seamless integration of all components

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

## 📚 Supported LLM Providers

| Provider | Models | API Key Required |
|----------|--------|------------------|
| **OpenAI** | GPT-4, GPT-3.5 | Yes (`OPENAI_API_KEY`) |
| **Anthropic** | Claude 3 (Opus, Sonnet, Haiku) | Yes (`ANTHROPIC_API_KEY`) |
| **Google** | Gemini Pro | Yes (`GOOGLE_API_KEY`) |
| **HuggingFace** | Llama 2, Mistral, etc. | Yes (`HUGGINGFACE_API_KEY`) |
| **Meta** | Llama models via HuggingFace | Yes (`HUGGINGFACE_API_KEY`) |
| **Ollama** | Any local model | No (runs locally) |

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
    llm_provider="openai",         # openai, anthropic, google, huggingface, ollama
    llm_model=None,                # Auto-selected if None
    llm_temperature=0.7,
    llm_max_tokens=1024,
    
    # Embedding Configuration
    embedding_provider="openai",   # openai, google, sentence-transformers, huggingface
    embedding_model=None,          # Auto-selected if None
    
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
providers = ["openai", "anthropic", "google"]

for provider in providers:
    flip = Flip(directory="./docs", llm_provider=provider)
    response = flip.query("Summarize the main points")
    print(f"{provider}: {response.answer[:100]}...")
```

### Example 3: Custom Chunking

```python
from flip import Flip, FlipConfig

# Use different chunking strategies
strategies = ["token", "sentence", "semantic", "recursive"]

for strategy in strategies:
    config = FlipConfig(chunking_strategy=strategy)
    flip = Flip(directory="./docs", config=config)
    stats = flip.get_stats()
    print(f"{strategy}: {stats['chunk_count']} chunks")
```

### Example 4: Add Documents Incrementally

```python
from flip import Flip

flip = Flip()  # Initialize without directory

# Add documents as needed
flip.add_documents(["doc1.pdf", "doc2.txt"])
flip.add_documents(["doc3.md"])

response = flip.query("What do these documents say?")
```

## 🔑 Environment Variables

Set API keys as environment variables:

```bash
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
export GOOGLE_API_KEY="your-key-here"
export HUGGINGFACE_API_KEY="your-key-here"
```

Or use a `.env` file:

```env
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here
GOOGLE_API_KEY=your-key-here
HUGGINGFACE_API_KEY=your-key-here
```

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

### Metadata Filtering

```python
# Coming soon: Filter by metadata
response = flip.query(
    "What is the policy?",
    filter={"document_type": "policy", "year": 2024}
)
```

## 📊 Performance

Flip is optimized for performance:

- **Batch Processing**: Embeddings generated in batches
- **Caching**: Automatic caching of embeddings and queries
- **Efficient Storage**: ChromaDB for fast vector search
- **Streaming**: Support for streaming responses (coming soon)

## 🤝 Contributing

Contributions are welcome! Please see our contributing guidelines.

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

Flip builds on the shoulders of giants:
- OpenAI, Anthropic, Google for LLM APIs
- ChromaDB for vector storage
- Sentence Transformers for local embeddings
- And many other open-source projects

## 📞 Support

- **Documentation**: [https://flip-rag.readthedocs.io](https://flip-rag.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/yourusername/flip/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/flip/discussions)

---

**Made with ❤️ by the Flip Team**
