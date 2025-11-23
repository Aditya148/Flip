# Flip SDK - Quick Start Guide

Get started with Flip in 5 minutes!

## Step 1: Install Dependencies

```bash
cd FlipV2
python setup.py
```

This will install all required packages and create sample documents.

## Step 2: Set Up API Keys

Edit the `.env` file and add your API keys:

```env
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here
GOOGLE_API_KEY=your-google-key-here
```

> **Note**: You only need the API key for the provider you want to use. For local models (Ollama + Sentence Transformers), no API keys are needed!

## Step 3: Run Your First Query

### Option A: Use Sample Documents

```bash
python examples/basic_usage.py
```

### Option B: Use Your Own Documents

Create a Python script:

```python
from flip import Flip

# Initialize with your documents directory
flip = Flip(directory="./your_documents")

# Query your documents
response = flip.query("What is the main topic?")

print("Answer:", response.answer)
print("\nSources:", len(response.citations))
```

## Step 4: Try Different Providers

```python
from flip import Flip

# OpenAI (default)
flip = Flip(directory="./docs", llm_provider="openai")

# Anthropic Claude
flip = Flip(directory="./docs", llm_provider="anthropic")

# Google Gemini
flip = Flip(directory="./docs", llm_provider="google")

# Local Ollama (no API key needed!)
flip = Flip(directory="./docs", llm_provider="ollama")
```

## Step 5: Customize Configuration

```python
from flip import Flip, FlipConfig

config = FlipConfig(
    llm_provider="openai",
    llm_model="gpt-4-turbo-preview",
    chunking_strategy="semantic",  # or "token", "sentence", "recursive"
    chunk_size=512,
    retrieval_top_k=5,
)

flip = Flip(directory="./docs", config=config)
```

## Supported File Types

Flip automatically handles:
- **Documents**: PDF, DOCX, TXT, MD
- **Data**: JSON, CSV
- **Web**: HTML
- **Code**: Python, JavaScript, TypeScript, Java, C++, Go, Rust, Ruby

Just point it to a directory and it handles the rest!

## Common Use Cases

### 1. Document Q&A

```python
flip = Flip(directory="./company_docs")
response = flip.query("What is our refund policy?")
```

### 2. Code Documentation

```python
flip = Flip(directory="./src")
response = flip.query("How does the authentication system work?")
```

### 3. Research Papers

```python
flip = Flip(directory="./papers")
response = flip.query("Summarize the key findings")
```

### 4. Knowledge Base

```python
flip = Flip(directory="./knowledge_base")
response = flip.query("How do I troubleshoot error X?")
```

## Tips

1. **Start Small**: Test with a few documents first
2. **Check Stats**: Use `flip.get_stats()` to see indexing info
3. **Try Different Strategies**: Experiment with chunking strategies
4. **Use Local Models**: Try Ollama + Sentence Transformers for free usage
5. **Check Citations**: Always review `response.citations` for sources

## Troubleshooting

### "API key not found"
- Make sure you've set the API key in `.env`
- Or pass it directly: `flip = Flip(directory="./docs", config=FlipConfig(openai_api_key="your-key"))`

### "No documents found"
- Check that your directory exists
- Verify file extensions are supported
- Use `DocumentLoader.get_supported_extensions()` to see all supported types

### "Import errors"
- Run `python setup.py` to install dependencies
- Or manually: `pip install -e .`

## Next Steps

- Check out [examples/advanced_config.py](examples/advanced_config.py) for more features
- Read the full [README.md](README.md) for detailed documentation
- Review the [walkthrough.md](walkthrough.md) for implementation details

## Need Help?

- Check the examples in `examples/`
- Read the full documentation in `README.md`
- Review the implementation in `walkthrough.md`

Happy querying! 🚀
