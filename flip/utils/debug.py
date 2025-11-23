"""Debugging utilities for Flip SDK."""

import time
import traceback
import sys
from typing import Any, Dict, List, Optional, Callable
from functools import wraps
from pathlib import Path
import json


class FlipDebugger:
    """Debugging utilities for Flip SDK."""
    
    def __init__(self, enabled: bool = False, output_dir: Optional[Path] = None):
        """
        Initialize debugger.
        
        Args:
            enabled: Whether debugging is enabled
            output_dir: Directory for debug output
        """
        self.enabled = enabled
        self.output_dir = output_dir or Path("./flip_debug")
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.traces = []
        self.timings = {}
    
    def enable(self):
        """Enable debugging."""
        self.enabled = True
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def disable(self):
        """Disable debugging."""
        self.enabled = False
    
    def trace(self, component: str, action: str, data: Optional[Dict[str, Any]] = None):
        """
        Record a trace event.
        
        Args:
            component: Component name (e.g., "retriever", "embedder")
            action: Action being performed
            data: Optional data to log
        """
        if not self.enabled:
            return
        
        trace_entry = {
            "timestamp": time.time(),
            "component": component,
            "action": action,
            "data": data or {}
        }
        
        self.traces.append(trace_entry)
    
    def start_timer(self, name: str):
        """
        Start a timer.
        
        Args:
            name: Timer name
        """
        if not self.enabled:
            return
        
        self.timings[name] = {"start": time.time(), "end": None, "duration": None}
    
    def stop_timer(self, name: str):
        """
        Stop a timer.
        
        Args:
            name: Timer name
        """
        if not self.enabled or name not in self.timings:
            return
        
        self.timings[name]["end"] = time.time()
        self.timings[name]["duration"] = self.timings[name]["end"] - self.timings[name]["start"]
    
    def get_timing(self, name: str) -> Optional[float]:
        """
        Get timing for a named timer.
        
        Args:
            name: Timer name
            
        Returns:
            Duration in seconds or None
        """
        if name in self.timings:
            return self.timings[name].get("duration")
        return None
    
    def dump_traces(self, filename: Optional[str] = None):
        """
        Dump traces to file.
        
        Args:
            filename: Output filename (default: traces_<timestamp>.json)
        """
        if not self.enabled:
            return
        
        if filename is None:
            filename = f"traces_{int(time.time())}.json"
        
        output_file = self.output_dir / filename
        
        with open(output_file, 'w') as f:
            json.dump(self.traces, f, indent=2)
        
        print(f"Traces dumped to: {output_file}")
    
    def dump_timings(self, filename: Optional[str] = None):
        """
        Dump timings to file.
        
        Args:
            filename: Output filename (default: timings_<timestamp>.json)
        """
        if not self.enabled:
            return
        
        if filename is None:
            filename = f"timings_{int(time.time())}.json"
        
        output_file = self.output_dir / filename
        
        with open(output_file, 'w') as f:
            json.dump(self.timings, f, indent=2)
        
        print(f"Timings dumped to: {output_file}")
    
    def print_timings(self):
        """Print timing summary."""
        if not self.enabled:
            return
        
        print("\n" + "=" * 60)
        print("Timing Summary")
        print("=" * 60)
        
        for name, timing in self.timings.items():
            duration = timing.get("duration")
            if duration is not None:
                print(f"{name}: {duration:.4f}s")
        
        print("=" * 60)
    
    def clear(self):
        """Clear all traces and timings."""
        self.traces.clear()
        self.timings.clear()


def debug_function(debugger: Optional[FlipDebugger] = None, component: str = "unknown"):
    """
    Decorator to debug function calls.
    
    Args:
        debugger: FlipDebugger instance
        component: Component name
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if debugger and debugger.enabled:
                func_name = func.__name__
                timer_name = f"{component}.{func_name}"
                
                # Trace function entry
                debugger.trace(component, f"enter_{func_name}", {
                    "args": str(args)[:100],
                    "kwargs": str(kwargs)[:100]
                })
                
                # Start timer
                debugger.start_timer(timer_name)
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Trace function exit
                    debugger.trace(component, f"exit_{func_name}", {
                        "result": str(result)[:100]
                    })
                    
                    return result
                    
                except Exception as e:
                    # Trace exception
                    debugger.trace(component, f"error_{func_name}", {
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    })
                    raise
                    
                finally:
                    # Stop timer
                    debugger.stop_timer(timer_name)
            else:
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def inspect_embeddings(embeddings: List[List[float]], output_file: Optional[Path] = None):
    """
    Inspect embeddings for debugging.
    
    Args:
        embeddings: List of embedding vectors
        output_file: Optional output file
    """
    import numpy as np
    
    if not embeddings:
        print("No embeddings to inspect")
        return
    
    embeddings_array = np.array(embeddings)
    
    info = {
        "count": len(embeddings),
        "dimension": len(embeddings[0]) if embeddings else 0,
        "mean": float(np.mean(embeddings_array)),
        "std": float(np.std(embeddings_array)),
        "min": float(np.min(embeddings_array)),
        "max": float(np.max(embeddings_array)),
        "shape": embeddings_array.shape
    }
    
    print("\n" + "=" * 60)
    print("Embedding Inspection")
    print("=" * 60)
    for key, value in info.items():
        print(f"{key}: {value}")
    print("=" * 60)
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(info, f, indent=2)


def inspect_chunks(chunks: List[Any], output_file: Optional[Path] = None):
    """
    Inspect text chunks for debugging.
    
    Args:
        chunks: List of text chunks
        output_file: Optional output file
    """
    if not chunks:
        print("No chunks to inspect")
        return
    
    chunk_lengths = [len(chunk.text) if hasattr(chunk, 'text') else len(str(chunk)) for chunk in chunks]
    
    info = {
        "count": len(chunks),
        "avg_length": sum(chunk_lengths) / len(chunk_lengths),
        "min_length": min(chunk_lengths),
        "max_length": max(chunk_lengths),
        "total_chars": sum(chunk_lengths),
    }
    
    print("\n" + "=" * 60)
    print("Chunk Inspection")
    print("=" * 60)
    for key, value in info.items():
        print(f"{key}: {value}")
    print("=" * 60)
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(info, f, indent=2)
    
    # Show sample chunks
    print("\nSample Chunks:")
    for i, chunk in enumerate(chunks[:3], 1):
        text = chunk.text if hasattr(chunk, 'text') else str(chunk)
        print(f"\n[{i}] {text[:200]}...")


def profile_query(flip_instance, query: str, iterations: int = 5):
    """
    Profile a query for performance analysis.
    
    Args:
        flip_instance: Flip instance
        query: Query to profile
        iterations: Number of iterations
    """
    import statistics
    
    times = []
    
    print(f"\nProfiling query: '{query}'")
    print(f"Iterations: {iterations}")
    print("-" * 60)
    
    for i in range(iterations):
        start = time.time()
        response = flip_instance.query(query)
        duration = time.time() - start
        times.append(duration)
        
        print(f"Iteration {i+1}: {duration:.4f}s")
    
    print("-" * 60)
    print(f"Mean: {statistics.mean(times):.4f}s")
    print(f"Median: {statistics.median(times):.4f}s")
    print(f"Std Dev: {statistics.stdev(times):.4f}s" if len(times) > 1 else "Std Dev: N/A")
    print(f"Min: {min(times):.4f}s")
    print(f"Max: {max(times):.4f}s")
