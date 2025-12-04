"""Image embedding cache for performance optimization."""

import hashlib
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any
import json

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class ImageEmbeddingCache:
    """Cache for image embeddings to avoid recomputation."""
    
    def __init__(self, cache_dir: str = "./.image_cache", max_size_mb: int = 500):
        """
        Initialize image embedding cache.
        
        Args:
            cache_dir: Directory to store cache files
            max_size_mb: Maximum cache size in megabytes
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        
        # Metadata file
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"entries": {}, "total_size": 0}
    
    def _save_metadata(self):
        """Save cache metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _get_image_hash(self, image: Image.Image) -> str:
        """
        Generate hash for image.
        
        Args:
            image: PIL Image
            
        Returns:
            Hash string
        """
        # Convert image to bytes
        import io
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()
        
        # Generate hash
        return hashlib.sha256(img_bytes).hexdigest()
    
    def get(self, image: Image.Image, model_name: str) -> Optional[List[float]]:
        """
        Get cached embedding for image.
        
        Args:
            image: PIL Image
            model_name: Name of embedding model
            
        Returns:
            Cached embedding or None if not found
        """
        img_hash = self._get_image_hash(image)
        cache_key = f"{img_hash}_{model_name}"
        
        if cache_key in self.metadata["entries"]:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        embedding = pickle.load(f)
                    
                    # Update access time
                    self.metadata["entries"][cache_key]["access_count"] += 1
                    self._save_metadata()
                    
                    return embedding
                except Exception as e:
                    print(f"Error loading cached embedding: {e}")
        
        return None
    
    def put(self, image: Image.Image, model_name: str, embedding: List[float]):
        """
        Cache embedding for image.
        
        Args:
            image: PIL Image
            model_name: Name of embedding model
            embedding: Embedding vector
        """
        img_hash = self._get_image_hash(image)
        cache_key = f"{img_hash}_{model_name}"
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # Save embedding
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
            
            # Update metadata
            file_size = cache_file.stat().st_size
            
            self.metadata["entries"][cache_key] = {
                "size": file_size,
                "access_count": 1,
                "model": model_name
            }
            self.metadata["total_size"] += file_size
            
            self._save_metadata()
            
            # Check if cache is too large
            self._cleanup_if_needed()
            
        except Exception as e:
            print(f"Error caching embedding: {e}")
    
    def _cleanup_if_needed(self):
        """Remove least recently used entries if cache is too large."""
        if self.metadata["total_size"] > self.max_size_bytes:
            # Sort by access count (ascending)
            entries = sorted(
                self.metadata["entries"].items(),
                key=lambda x: x[1]["access_count"]
            )
            
            # Remove entries until under limit
            for cache_key, entry in entries:
                if self.metadata["total_size"] <= self.max_size_bytes * 0.8:
                    break
                
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                if cache_file.exists():
                    cache_file.unlink()
                
                self.metadata["total_size"] -= entry["size"]
                del self.metadata["entries"][cache_key]
            
            self._save_metadata()
    
    def clear(self):
        """Clear all cached embeddings."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        
        self.metadata = {"entries": {}, "total_size": 0}
        self._save_metadata()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        return {
            "total_entries": len(self.metadata["entries"]),
            "total_size_mb": self.metadata["total_size"] / (1024 * 1024),
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "usage_percent": (self.metadata["total_size"] / self.max_size_bytes) * 100
        }
