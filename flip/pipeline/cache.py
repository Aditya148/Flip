"""Caching layer for embeddings and queries."""

import hashlib
import json
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any
from collections import OrderedDict


class EmbeddingCache:
    """Cache for text embeddings to avoid recomputation."""
    
    def __init__(self, cache_dir: Optional[Path] = None, max_size: int = 10000):
        """
        Initialize embedding cache.
        
        Args:
            cache_dir: Directory to persist cache (None for in-memory only)
            max_size: Maximum number of cached embeddings (LRU eviction)
        """
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.cache: OrderedDict[str, List[float]] = OrderedDict()
        
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_file = cache_dir / "embeddings.pkl"
            self._load_cache()
    
    def _get_key(self, text: str) -> str:
        """Generate cache key from text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, text: str) -> Optional[List[float]]:
        """
        Get cached embedding for text.
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Cached embedding or None if not found
        """
        key = self._get_key(text)
        
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        
        return None
    
    def put(self, text: str, embedding: List[float]):
        """
        Cache an embedding.
        
        Args:
            text: Text
            embedding: Embedding vector
        """
        key = self._get_key(text)
        
        # Add to cache
        self.cache[key] = embedding
        self.cache.move_to_end(key)
        
        # Evict oldest if over max size
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
    
    def get_batch(self, texts: List[str]) -> Dict[str, List[float]]:
        """
        Get cached embeddings for multiple texts.
        
        Args:
            texts: List of texts
            
        Returns:
            Dictionary mapping text to embedding (only cached items)
        """
        cached = {}
        for text in texts:
            embedding = self.get(text)
            if embedding is not None:
                cached[text] = embedding
        return cached
    
    def put_batch(self, texts: List[str], embeddings: List[List[float]]):
        """
        Cache multiple embeddings.
        
        Args:
            texts: List of texts
            embeddings: List of embeddings
        """
        for text, embedding in zip(texts, embeddings):
            self.put(text, embedding)
    
    def _load_cache(self):
        """Load cache from disk."""
        if self.cache_file and self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
            except:
                self.cache = OrderedDict()
    
    def save(self):
        """Save cache to disk."""
        if self.cache_file:
            try:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(self.cache, f)
            except:
                pass
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        if self.cache_file and self.cache_file.exists():
            self.cache_file.unlink()


class QueryCache:
    """Cache for query results."""
    
    def __init__(self, cache_dir: Optional[Path] = None, max_size: int = 1000):
        """
        Initialize query cache.
        
        Args:
            cache_dir: Directory to persist cache (None for in-memory only)
            max_size: Maximum number of cached queries (LRU eviction)
        """
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.cache: OrderedDict[str, Any] = OrderedDict()
        
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_file = cache_dir / "queries.pkl"
            self._load_cache()
    
    def _get_key(self, query: str, **kwargs) -> str:
        """Generate cache key from query and parameters."""
        # Include parameters in key
        key_data = {"query": query, **kwargs}
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, query: str, **kwargs) -> Optional[Any]:
        """
        Get cached result for query.
        
        Args:
            query: Query text
            **kwargs: Additional parameters (e.g., top_k)
            
        Returns:
            Cached result or None if not found
        """
        key = self._get_key(query, **kwargs)
        
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        
        return None
    
    def put(self, query: str, result: Any, **kwargs):
        """
        Cache a query result.
        
        Args:
            query: Query text
            result: Query result
            **kwargs: Additional parameters
        """
        key = self._get_key(query, **kwargs)
        
        # Add to cache
        self.cache[key] = result
        self.cache.move_to_end(key)
        
        # Evict oldest if over max size
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
    
    def _load_cache(self):
        """Load cache from disk."""
        if self.cache_file and self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
            except:
                self.cache = OrderedDict()
    
    def save(self):
        """Save cache to disk."""
        if self.cache_file:
            try:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(self.cache, f)
            except:
                pass
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        if self.cache_file and self.cache_file.exists():
            self.cache_file.unlink()
