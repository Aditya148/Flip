"""
Setup script for Flip SDK development.
"""

import subprocess
import sys
from pathlib import Path


def install_dependencies():
    """Install all dependencies."""
    print("📦 Installing Flip SDK dependencies...")
    print("=" * 60)
    
    # Install in development mode
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-e", ".",
            "--quiet"
        ])
        print("✅ Core dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    
    # Install dev dependencies
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-e", ".[dev]",
            "--quiet"
        ])
        print("✅ Development dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Dev dependencies failed (optional): {e}")
    
    return True


def create_sample_docs():
    """Create sample documents for testing."""
    print("\n📄 Creating sample documents...")
    print("=" * 60)
    
    sample_dir = Path("sample_docs")
    sample_dir.mkdir(exist_ok=True)
    
    # Create sample text files
    samples = {
        "ai_overview.txt": """Artificial Intelligence Overview
        
Artificial Intelligence (AI) is the simulation of human intelligence by machines.
It encompasses various subfields including machine learning, natural language processing,
computer vision, and robotics.

Machine learning is a subset of AI that enables systems to learn from data without
being explicitly programmed. Deep learning, a subset of machine learning, uses
neural networks with multiple layers to process complex patterns.

Applications of AI include:
- Virtual assistants (Siri, Alexa)
- Recommendation systems (Netflix, Amazon)
- Autonomous vehicles
- Medical diagnosis
- Financial trading
""",
        
        "rag_explained.txt": """Retrieval-Augmented Generation (RAG)

RAG is a technique that combines information retrieval with text generation.
It works by first retrieving relevant documents from a knowledge base, then
using those documents as context for a language model to generate responses.

Key Components:
1. Document Store: Contains the knowledge base
2. Retriever: Finds relevant documents
3. Generator: Creates responses using retrieved context

Benefits of RAG:
- Reduces hallucinations
- Provides up-to-date information
- Enables citation of sources
- More cost-effective than fine-tuning
""",
        
        "flip_intro.md": """# Flip SDK

Flip is a fully automated RAG SDK that makes it easy to build
question-answering systems over your documents.

## Features

- Simple API
- Multiple LLM providers
- Automatic chunking
- Vector search
- Citation support

## Quick Start

```python
from flip import Flip

flip = Flip(directory="./docs")
response = flip.query("What is RAG?")
print(response.answer)
```
""",
    }
    
    for filename, content in samples.items():
        file_path = sample_dir / filename
        file_path.write_text(content.strip())
        print(f"✅ Created {filename}")
    
    print(f"\n📁 Sample documents created in: {sample_dir.absolute()}")
    return True


def create_env_file():
    """Create .env file if it doesn't exist."""
    print("\n🔑 Setting up environment...")
    print("=" * 60)
    
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    env_example = Path(".env.example")
    if env_example.exists():
        env_file.write_text(env_example.read_text())
        print("✅ Created .env file from template")
        print("⚠️  Please edit .env and add your API keys")
    else:
        print("⚠️  No .env.example found")
    
    return True


def main():
    """Run setup."""
    print("\n" + "=" * 60)
    print("Flip SDK - Setup")
    print("=" * 60 + "\n")
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed")
        return 1
    
    # Create sample documents
    create_sample_docs()
    
    # Create env file
    create_env_file()
    
    print("\n" + "=" * 60)
    print("✅ Setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Edit .env and add your API keys")
    print("2. Run: python examples/basic_usage.py")
    print("3. Check out examples/advanced_config.py for more options")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
