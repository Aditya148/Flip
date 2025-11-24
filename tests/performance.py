"""Performance testing framework for vector stores."""

import time
import statistics
from typing import List, Dict, Any, Callable
from dataclasses import dataclass
import numpy as np

from flip.vector_store.base import BaseVectorStore


@dataclass
class PerformanceMetrics:
    """Performance metrics for vector store operations."""
    operation: str
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    std_dev: float
    throughput: float  # operations per second
    vector_count: int
    
    def __str__(self) -> str:
        return f"""
{self.operation} Performance:
  Total Time: {self.total_time:.3f}s
  Average Time: {self.avg_time:.3f}s
  Min Time: {self.min_time:.3f}s
  Max Time: {self.max_time:.3f}s
  Std Dev: {self.std_dev:.3f}s
  Throughput: {self.throughput:.1f} ops/sec
  Vector Count: {self.vector_count}
"""


class PerformanceTester:
    """Performance testing framework for vector stores."""
    
    def __init__(self, vector_store: BaseVectorStore, dimension: int = 384):
        """
        Initialize performance tester.
        
        Args:
            vector_store: Vector store to test
            dimension: Vector dimension
        """
        self.vector_store = vector_store
        self.dimension = dimension
        self.results: Dict[str, PerformanceMetrics] = {}
    
    def generate_random_vectors(self, count: int) -> List[List[float]]:
        """Generate random vectors for testing."""
        return np.random.rand(count, self.dimension).tolist()
    
    def benchmark_operation(
        self,
        operation_name: str,
        operation_func: Callable,
        iterations: int = 10,
        vector_count: int = 100
    ) -> PerformanceMetrics:
        """
        Benchmark a vector store operation.
        
        Args:
            operation_name: Name of the operation
            operation_func: Function to benchmark
            iterations: Number of iterations
            vector_count: Number of vectors per iteration
            
        Returns:
            Performance metrics
        """
        times = []
        
        for i in range(iterations):
            start = time.time()
            operation_func()
            elapsed = time.time() - start
            times.append(elapsed)
        
        total_time = sum(times)
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
        throughput = (iterations * vector_count) / total_time if total_time > 0 else 0
        
        metrics = PerformanceMetrics(
            operation=operation_name,
            total_time=total_time,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            std_dev=std_dev,
            throughput=throughput,
            vector_count=vector_count
        )
        
        self.results[operation_name] = metrics
        return metrics
    
    def benchmark_add(self, vector_count: int = 100, iterations: int = 10) -> PerformanceMetrics:
        """Benchmark add operation."""
        def add_op():
            vectors = self.generate_random_vectors(vector_count)
            ids = [f"vec_{i}" for i in range(vector_count)]
            texts = [f"Text {i}" for i in range(vector_count)]
            self.vector_store.add(ids, vectors, texts)
        
        return self.benchmark_operation("add", add_op, iterations, vector_count)
    
    def benchmark_search(self, top_k: int = 5, iterations: int = 100) -> PerformanceMetrics:
        """Benchmark search operation."""
        # First add some vectors
        vectors = self.generate_random_vectors(1000)
        ids = [f"vec_{i}" for i in range(1000)]
        texts = [f"Text {i}" for i in range(1000)]
        self.vector_store.add(ids, vectors, texts)
        
        query_vector = self.generate_random_vectors(1)[0]
        
        def search_op():
            self.vector_store.search(query_vector, top_k=top_k)
        
        return self.benchmark_operation("search", search_op, iterations, 1)
    
    def benchmark_batch_add(self, batch_size: int = 100, num_batches: int = 10) -> PerformanceMetrics:
        """Benchmark batch add operation."""
        total_vectors = batch_size * num_batches
        vectors = self.generate_random_vectors(total_vectors)
        ids = [f"batch_vec_{i}" for i in range(total_vectors)]
        texts = [f"Batch text {i}" for i in range(total_vectors)]
        
        def batch_add_op():
            self.vector_store.batch_add(ids, vectors, texts, batch_size=batch_size)
        
        return self.benchmark_operation("batch_add", batch_add_op, 1, total_vectors)
    
    def benchmark_delete(self, vector_count: int = 100, iterations: int = 10) -> PerformanceMetrics:
        """Benchmark delete operation."""
        def delete_op():
            # Add vectors first
            vectors = self.generate_random_vectors(vector_count)
            ids = [f"del_vec_{i}" for i in range(vector_count)]
            texts = [f"Text {i}" for i in range(vector_count)]
            self.vector_store.add(ids, vectors, texts)
            
            # Delete them
            self.vector_store.delete(ids)
        
        return self.benchmark_operation("delete", delete_op, iterations, vector_count)
    
    def benchmark_get_by_ids(self, vector_count: int = 100, iterations: int = 100) -> PerformanceMetrics:
        """Benchmark get_by_ids operation."""
        # Add vectors first
        vectors = self.generate_random_vectors(1000)
        ids = [f"get_vec_{i}" for i in range(1000)]
        texts = [f"Text {i}" for i in range(1000)]
        self.vector_store.add(ids, vectors, texts)
        
        # Select random IDs to retrieve
        import random
        retrieve_ids = random.sample(ids, vector_count)
        
        def get_op():
            self.vector_store.get_by_ids(retrieve_ids)
        
        return self.benchmark_operation("get_by_ids", get_op, iterations, vector_count)
    
    def run_full_benchmark(self) -> Dict[str, PerformanceMetrics]:
        """
        Run full benchmark suite.
        
        Returns:
            Dictionary of performance metrics
        """
        print("Running full benchmark suite...")
        
        # Clear store first
        self.vector_store.clear()
        
        # Run benchmarks
        print("Benchmarking add...")
        self.benchmark_add(vector_count=100, iterations=10)
        
        print("Benchmarking batch_add...")
        self.benchmark_batch_add(batch_size=100, num_batches=10)
        
        print("Benchmarking search...")
        self.benchmark_search(top_k=5, iterations=100)
        
        print("Benchmarking get_by_ids...")
        self.benchmark_get_by_ids(vector_count=100, iterations=100)
        
        print("Benchmarking delete...")
        self.benchmark_delete(vector_count=100, iterations=10)
        
        return self.results
    
    def print_results(self):
        """Print all benchmark results."""
        print("\n" + "=" * 60)
        print(f"Performance Results for {self.vector_store.provider_name}")
        print("=" * 60)
        
        for operation, metrics in self.results.items():
            print(metrics)
    
    def compare_with(self, other_tester: 'PerformanceTester') -> Dict[str, float]:
        """
        Compare performance with another tester.
        
        Args:
            other_tester: Another performance tester
            
        Returns:
            Dictionary of speedup ratios (>1 means this is faster)
        """
        comparisons = {}
        
        for operation in self.results:
            if operation in other_tester.results:
                this_time = self.results[operation].avg_time
                other_time = other_tester.results[operation].avg_time
                speedup = other_time / this_time if this_time > 0 else 0
                comparisons[operation] = speedup
        
        return comparisons


def benchmark_all_stores(stores: List[BaseVectorStore], dimension: int = 384) -> Dict[str, Dict[str, PerformanceMetrics]]:
    """
    Benchmark multiple vector stores.
    
    Args:
        stores: List of vector stores to benchmark
        dimension: Vector dimension
        
    Returns:
        Dictionary mapping store names to their metrics
    """
    results = {}
    
    for store in stores:
        print(f"\nBenchmarking {store.provider_name}...")
        tester = PerformanceTester(store, dimension)
        tester.run_full_benchmark()
        tester.print_results()
        results[store.provider_name] = tester.results
    
    return results
