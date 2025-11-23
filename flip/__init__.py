"""
Flip - Fully Automated RAG SDK

A simple yet powerful SDK for Retrieval-Augmented Generation.
Initialize with a directory and start querying - that's it!

Example:
    >>> from flip import Flip
    >>> flip = Flip(directory="./docs")
    >>> response = flip.query("What is the main topic?")
    >>> print(response.answer)
    >>> print(response.citations)
"""

from flip.core.flip import Flip
from flip.core.config import FlipConfig
from flip.core.exceptions import FlipException

__version__ = "0.1.0"
__all__ = ["Flip", "FlipConfig", "FlipException"]
