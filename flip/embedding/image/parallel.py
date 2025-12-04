"""Parallel image processing utilities."""

from typing import List, Callable, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count
import time


class ParallelImageProcessor:
    """Process images in parallel for better performance."""
    
    def __init__(self, max_workers: int = None, use_processes: bool = False):
        """
        Initialize parallel processor.
        
        Args:
            max_workers: Maximum number of workers (None = auto)
            use_processes: Use processes instead of threads
        """
        if max_workers is None:
            max_workers = min(cpu_count(), 8)  # Cap at 8
        
        self.max_workers = max_workers
        self.use_processes = use_processes
    
    def process_batch(
        self,
        items: List[Any],
        process_func: Callable,
        show_progress: bool = False
    ) -> List[Any]:
        """
        Process items in parallel.
        
        Args:
            items: List of items to process
            process_func: Function to apply to each item
            show_progress: Whether to show progress
            
        Returns:
            List of processed results
        """
        if len(items) == 0:
            return []
        
        # Choose executor
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        results = []
        
        with executor_class(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(process_func, item) for item in items]
            
            # Collect results
            if show_progress:
                try:
                    from tqdm import tqdm
                    iterator = tqdm(futures, desc="Processing", total=len(futures))
                except ImportError:
                    iterator = futures
            else:
                iterator = futures
            
            for future in iterator:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing item: {e}")
                    results.append(None)
        
        return results
    
    def extract_images_parallel(
        self,
        file_paths: List[str],
        extractor_func: Callable,
        show_progress: bool = True
    ) -> List[List[Any]]:
        """
        Extract images from multiple files in parallel.
        
        Args:
            file_paths: List of file paths
            extractor_func: Function to extract images from file
            show_progress: Whether to show progress
            
        Returns:
            List of lists of extracted images
        """
        return self.process_batch(file_paths, extractor_func, show_progress)
    
    def embed_images_parallel(
        self,
        images: List[Any],
        embedder_func: Callable,
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for images in parallel batches.
        
        Args:
            images: List of images
            embedder_func: Function to generate embeddings
            batch_size: Size of each batch
            show_progress: Whether to show progress
            
        Returns:
            List of embeddings
        """
        # Split into batches
        batches = [
            images[i:i + batch_size]
            for i in range(0, len(images), batch_size)
        ]
        
        # Process batches in parallel
        batch_results = self.process_batch(batches, embedder_func, show_progress)
        
        # Flatten results
        embeddings = []
        for batch_result in batch_results:
            if batch_result:
                embeddings.extend(batch_result)
        
        return embeddings


class ProgressTracker:
    """Track progress of image processing operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        """
        Initialize progress tracker.
        
        Args:
            total: Total number of items
            description: Description of operation
        """
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = time.time()
        
        try:
            from tqdm import tqdm
            self.pbar = tqdm(total=total, desc=description)
            self.use_tqdm = True
        except ImportError:
            self.pbar = None
            self.use_tqdm = False
    
    def update(self, n: int = 1):
        """Update progress by n items."""
        self.current += n
        
        if self.use_tqdm and self.pbar:
            self.pbar.update(n)
        else:
            # Simple text progress
            percent = (self.current / self.total) * 100
            elapsed = time.time() - self.start_time
            rate = self.current / elapsed if elapsed > 0 else 0
            
            print(f"\r{self.description}: {self.current}/{self.total} "
                  f"({percent:.1f}%) - {rate:.1f} items/sec", end='')
    
    def close(self):
        """Close progress tracker."""
        if self.use_tqdm and self.pbar:
            self.pbar.close()
        else:
            print()  # New line
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
