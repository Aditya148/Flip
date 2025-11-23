# Flip SDK Examples

This directory contains examples demonstrating various features of the Flip SDK.

## Examples

### 1. Basic Usage (`basic_usage.py`)

The simplest way to use Flip:

```python
from flip import Flip

flip = Flip(directory="./sample_docs")
response = flip.query("What is the main topic?")
print(response.answer)
```

**Run it:**
```bash
python examples/basic_usage.py
```

### 2. Advanced Configuration (`advanced_config.py`)

Demonstrates:
- Custom configuration
- Multiple LLM providers
- Different chunking strategies
- Incremental indexing
- Local embeddings

**Run it:**
```bash
python examples/advanced_config.py
```

## Sample Documents

Create a `sample_docs` directory with some test documents:

```bash
mkdir sample_docs
echo "This is a test document about AI." > sample_docs/doc1.txt
echo "Machine learning is a subset of AI." > sample_docs/doc2.txt
echo "RAG combines retrieval and generation." > sample_docs/doc3.txt
```

## Environment Setup

Make sure you have API keys set:

```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

Or create a `.env` file in the project root.

## Next Steps

1. Try the basic example first
2. Experiment with different LLM providers
3. Test different chunking strategies
4. Build your own application!
