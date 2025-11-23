"""Configuration for pytest."""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """
    Artificial Intelligence (AI) is the simulation of human intelligence by machines.
    Machine learning is a subset of AI that enables systems to learn from data.
    Deep learning uses neural networks with multiple layers.
    """


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "content": "AI is artificial intelligence.",
            "metadata": {"source": "doc1.txt", "id": 1}
        },
        {
            "content": "Machine learning learns from data.",
            "metadata": {"source": "doc2.txt", "id": 2}
        },
        {
            "content": "Deep learning uses neural networks.",
            "metadata": {"source": "doc3.txt", "id": 3}
        }
    ]


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as benchmarks"
    )
