"""Performance benchmarking for Flip SDK."""

import time
import tempfile
from pathlib import Path
import statistics
from flip import Flip, FlipConfig


class PerformanceBenchmark:
    """Performance benchmarking utilities."""
    
    def __init__(self, output_file="benchmark_results.txt"):
        """Initialize benchmark."""
        self.output_file = output_file
        self.results = []
    
    def benchmark_indexing(self, num_docs=100, doc_size=1000):
        """Benchmark document indexing speed."""
        print(f"\n{'='*60}")
        print(f"Benchmarking Indexing ({num_docs} docs, {doc_size} chars each)")
        print(f"{'='*60}")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create test documents
            for i in range(num_docs):
                doc = tmp_path / f"doc{i}.txt"
                content = f"Document {i}. " + ("Test content. " * (doc_size // 15))
                doc.write_text(content)
            
            # Benchmark
            config = FlipConfig(
                embedding_provider="sentence-transformers",
                embedding_model="all-MiniLM-L6-v2",
                show_progress=False
            )
            
            start = time.time()
            flip = Flip(directory=str(tmp_path), config=config)
            duration = time.time() - start
            
            stats = flip.get_stats()
            
            result = {
                "test": "indexing",
                "num_docs": num_docs,
                "doc_size": doc_size,
                "duration": duration,
                "docs_per_sec": num_docs / duration,
                "chunks": stats["chunk_count"]
            }
            
            self.results.append(result)
            
            print(f"Duration: {duration:.2f}s")
            print(f"Docs/sec: {result['docs_per_sec']:.2f}")
            print(f"Total chunks: {stats['chunk_count']}")
    
    def benchmark_query_latency(self, num_queries=50):
        """Benchmark query latency."""
        print(f"\n{'='*60}")
        print(f"Benchmarking Query Latency ({num_queries} queries)")
        print(f"{'='*60}")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create test documents
            for i in range(10):
                doc = tmp_path / f"doc{i}.txt"
                doc.write_text(f"Document {i} about AI and machine learning. " * 50)
            
            config = FlipConfig(
                embedding_provider="sentence-transformers",
                show_progress=False
            )
            
            flip = Flip(directory=str(tmp_path), config=config)
            
            # Benchmark queries
            queries = [
                "What is AI?",
                "Explain machine learning",
                "How does deep learning work?",
                "What are neural networks?",
                "Describe artificial intelligence"
            ]
            
            times = []
            
            for i in range(num_queries):
                query = queries[i % len(queries)]
                
                start = time.time()
                response = flip.query(query)
                duration = time.time() - start
                
                times.append(duration)
            
            result = {
                "test": "query_latency",
                "num_queries": num_queries,
                "mean": statistics.mean(times),
                "median": statistics.median(times),
                "min": min(times),
                "max": max(times),
                "stdev": statistics.stdev(times) if len(times) > 1 else 0
            }
            
            self.results.append(result)
            
            print(f"Mean: {result['mean']:.3f}s")
            print(f"Median: {result['median']:.3f}s")
            print(f"Min: {result['min']:.3f}s")
            print(f"Max: {result['max']:.3f}s")
            print(f"Std Dev: {result['stdev']:.3f}s")
    
    def benchmark_chunking_strategies(self):
        """Benchmark different chunking strategies."""
        print(f"\n{'='*60}")
        print(f"Benchmarking Chunking Strategies")
        print(f"{'='*60}")
        
        strategies = ["token", "sentence", "semantic", "recursive"]
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create test document
            doc = tmp_path / "test.txt"
            doc.write_text("Test sentence. " * 1000)
            
            for strategy in strategies:
                config = FlipConfig(
                    embedding_provider="sentence-transformers",
                    chunking_strategy=strategy,
                    chunk_size=512,
                    show_progress=False
                )
                
                start = time.time()
                flip = Flip(directory=str(tmp_path), config=config)
                duration = time.time() - start
                
                stats = flip.get_stats()
                
                print(f"{strategy:12s}: {duration:.3f}s, {stats['chunk_count']} chunks")
    
    def benchmark_hybrid_vs_vector(self):
        """Benchmark hybrid search vs vector-only search."""
        print(f"\n{'='*60}")
        print(f"Benchmarking Hybrid vs Vector Search")
        print(f"{'='*60}")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create test documents
            for i in range(20):
                doc = tmp_path / f"doc{i}.txt"
                doc.write_text(f"Document {i} content. " * 100)
            
            # Vector only
            config_vector = FlipConfig(
                embedding_provider="sentence-transformers",
                use_hybrid_search=False,
                show_progress=False
            )
            
            flip_vector = Flip(directory=str(tmp_path), config=config_vector)
            
            start = time.time()
            for _ in range(10):
                flip_vector.query("test query")
            vector_time = time.time() - start
            
            # Hybrid
            config_hybrid = FlipConfig(
                embedding_provider="sentence-transformers",
                use_hybrid_search=True,
                show_progress=False
            )
            
            flip_hybrid = Flip(directory=str(tmp_path), config=config_hybrid)
            
            start = time.time()
            for _ in range(10):
                flip_hybrid.query("test query")
            hybrid_time = time.time() - start
            
            print(f"Vector only: {vector_time:.3f}s")
            print(f"Hybrid: {hybrid_time:.3f}s")
            print(f"Overhead: {((hybrid_time - vector_time) / vector_time * 100):.1f}%")
    
    def benchmark_cache_performance(self):
        """Benchmark caching performance."""
        print(f"\n{'='*60}")
        print(f"Benchmarking Cache Performance")
        print(f"{'='*60}")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create test documents
            doc = tmp_path / "test.txt"
            doc.write_text("Test content. " * 100)
            
            config = FlipConfig(
                embedding_provider="sentence-transformers",
                enable_cache=True,
                show_progress=False
            )
            
            flip = Flip(directory=str(tmp_path), config=config)
            
            query = "test query"
            
            # First query (not cached)
            start = time.time()
            flip.query(query)
            uncached_time = time.time() - start
            
            # Second query (cached)
            start = time.time()
            flip.query(query)
            cached_time = time.time() - start
            
            speedup = uncached_time / cached_time if cached_time > 0 else 0
            
            print(f"Uncached: {uncached_time:.3f}s")
            print(f"Cached: {cached_time:.3f}s")
            print(f"Speedup: {speedup:.1f}x")
    
    def save_results(self):
        """Save benchmark results to file."""
        with open(self.output_file, 'w') as f:
            f.write("Flip SDK Performance Benchmark Results\n")
            f.write("=" * 60 + "\n\n")
            
            for result in self.results:
                f.write(f"Test: {result['test']}\n")
                for key, value in result.items():
                    if key != 'test':
                        f.write(f"  {key}: {value}\n")
                f.write("\n")
        
        print(f"\n✅ Results saved to {self.output_file}")


def main():
    """Run all benchmarks."""
    print("\n🚀 Flip SDK Performance Benchmarks\n")
    
    benchmark = PerformanceBenchmark()
    
    try:
        benchmark.benchmark_indexing(num_docs=50, doc_size=1000)
    except Exception as e:
        print(f"Error in indexing benchmark: {e}")
    
    try:
        benchmark.benchmark_query_latency(num_queries=20)
    except Exception as e:
        print(f"Error in query latency benchmark: {e}")
    
    try:
        benchmark.benchmark_chunking_strategies()
    except Exception as e:
        print(f"Error in chunking benchmark: {e}")
    
    try:
        benchmark.benchmark_hybrid_vs_vector()
    except Exception as e:
        print(f"Error in hybrid vs vector benchmark: {e}")
    
    try:
        benchmark.benchmark_cache_performance()
    except Exception as e:
        print(f"Error in cache benchmark: {e}")
    
    benchmark.save_results()
    
    print("\n✅ All benchmarks complete!")


if __name__ == "__main__":
    main()
