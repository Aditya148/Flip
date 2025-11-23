# Installation Guide

## Quick Install

Install all core dependencies:

```bash
pip install -r requirements.txt
```

## Development Install

For development with testing and linting tools:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Optional Vector Stores

If you want to use additional vector stores (Pinecone, Qdrant, Weaviate):

```bash
pip install -r requirements-optional.txt
```

## Editable Install

For development, install Flip in editable mode:

```bash
pip install -e .
```

This uses the dependencies defined in `pyproject.toml`.

## Minimal Install

If you want a minimal installation (local models only, no API dependencies):

```bash
# Install only core dependencies
pip install pydantic tenacity tqdm python-dotenv

# Vector store
pip install chromadb

# Document processing
pip install pypdf2 pdfplumber python-docx beautifulsoup4 lxml markdown

# Local embeddings and LLM
pip install sentence-transformers transformers torch

# Retrieval
pip install rank-bm25 tiktoken numpy

# Utilities
pip install requests
```

## Troubleshooting

### PyTorch Installation

If you have issues with PyTorch, install it separately first:

```bash
# CPU version
pip install torch --index-url https://download.pytorch.org/whl/cpu

# GPU version (CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Then install the rest:

```bash
pip install -r requirements.txt
```

### ChromaDB Issues

If ChromaDB installation fails, try:

```bash
pip install --upgrade pip setuptools wheel
pip install chromadb
```

### Windows-Specific

On Windows, you may need to install Visual C++ Build Tools for some packages.

## Verify Installation

Test that everything is installed correctly:

```bash
python -c "from flip import Flip; print('Flip SDK installed successfully!')"
```

Or run the quick test:

```bash
python tests/quick_test.py
```

## Version Information

- Python: 3.9+
- Tested on: Python 3.9, 3.10, 3.11, 3.12
- OS: Windows, macOS, Linux
