"""Image embedding module."""

from flip.embedding.image.base import BaseImageEmbedder
from flip.embedding.image.factory import ImageEmbedderFactory

__all__ = [
    'BaseImageEmbedder',
    'ImageEmbedderFactory',
]
