# Using google/embeddinggemma-300m with Flip SDK

## Quick Start

```python
from flip import Flip, FlipConfig

# Configure to use EmbeddingGemma
config = FlipConfig(
    embedding_provider="huggingface",
    embedding_model="google/embeddinggemma-300m",
    llm_provider="openai",  # or any other provider
)

# Initialize and use
flip = Flip(directory="./docs", config=config)
response = flip.query("What is this about?")
print(response.answer)
```

## Complete Example

```python
from flip import Flip, FlipConfig
import os

# Set API keys
os.environ["HUGGINGFACE_API_KEY"] = "your-hf-key"
os.environ["OPENAI_API_KEY"] = "your-openai-key"

# Configure Flip
config = FlipConfig(
    # Use HuggingFace for embeddings
    embedding_provider="huggingface",
    embedding_model="google/embeddinggemma-300m",
    
    # Use any LLM provider
    llm_provider="openai",
    llm_model="gpt-4-turbo-preview",
    
    # Advanced features
    use_hybrid_search=True,
    use_reranking=True,
    enable_cache=True,
)

# Initialize with documents
flip = Flip(directory="./sample_docs", config=config)

# Query
response = flip.query("What are the key points?")
print(response.answer)
```

## Alternative: Use with Different LLM Providers

### With Anthropic Claude

```python
config = FlipConfig(
    embedding_provider="huggingface",
    embedding_model="google/embeddinggemma-300m",
    llm_provider="anthropic",
    llm_model="claude-3-sonnet-20240229",
)
```

### With Google Gemini

```python
config = FlipConfig(
    embedding_provider="huggingface",
    embedding_model="google/embeddinggemma-300m",
    llm_provider="google",
    llm_model="gemini-pro",
)
```

### With Local Ollama

```python
config = FlipConfig(
    embedding_provider="huggingface",
    embedding_model="google/embeddinggemma-300m",
    llm_provider="ollama",
    llm_model="llama2",
)
```

## Environment Variables

Create a `.env` file:

```env
HUGGINGFACE_API_KEY=your-huggingface-api-key
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
GOOGLE_API_KEY=your-google-api-key
```

## Installation

Make sure you have the HuggingFace dependencies:

```bash
pip install huggingface-hub transformers
```

## Notes

- **EmbeddingGemma** is a 300M parameter embedding model from Google
- It produces high-quality embeddings for semantic search
- Requires HuggingFace API key for inference
- Works with any LLM provider in Flip

## Performance

- **Embedding dimension**: 768 (typical for this model)
- **Speed**: Fast inference via HuggingFace API
- **Quality**: High-quality embeddings optimized for retrieval

## Full Example Script

See `examples/embeddinggemma_example.py` for a complete working example.

## Troubleshooting

### API Key Issues

```python
# Set API key directly in config
config = FlipConfig(
    embedding_provider="huggingface",
    embedding_model="google/embeddinggemma-300m",
    huggingface_api_key="your-key-here",
)
```

### Model Loading Issues

If you encounter issues with the HuggingFace API, you can also use local embeddings:

```python
# Alternative: Use local sentence-transformers
config = FlipConfig(
    embedding_provider="sentence-transformers",
    embedding_model="all-MiniLM-L6-v2",  # Local, no API key needed
)
```

## Comparison with Other Embedding Models

| Model | Provider | Dimension | API Key | Quality |
|-------|----------|-----------|---------|---------|
| google/embeddinggemma-300m | HuggingFace | 768 | Yes | High |
| text-embedding-3-small | OpenAI | 1536 | Yes | High |
| all-MiniLM-L6-v2 | Local | 384 | No | Good |
| models/embedding-001 | Google | 768 | Yes | High |
