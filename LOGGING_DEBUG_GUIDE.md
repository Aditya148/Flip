# Logging and Debugging Guide

This guide covers the logging and debugging utilities in the Flip SDK.

## 📝 Logging

### Quick Start

```python
from flip.utils.logger import setup_logging

# Setup logging
logger = setup_logging(
    level="INFO",
    log_file="./flip_logs/flip.log",
    console=True
)

# Use logger
logger.info("Flip SDK initialized")
logger.debug("Debug information")
logger.warning("Warning message")
logger.error("Error occurred")
```

### Log Levels

- **DEBUG**: Detailed information for diagnosing problems
- **INFO**: General informational messages
- **WARNING**: Warning messages
- **ERROR**: Error messages
- **CRITICAL**: Critical errors

### Configuration

```python
from flip.utils.logger import setup_logging
from pathlib import Path

logger = setup_logging(
    level="DEBUG",                    # Log level
    log_file=Path("./logs/flip.log"), # Log file path
    console=True,                      # Log to console
    format_string="%(asctime)s - %(levelname)s - %(message)s"  # Custom format
)
```

### Module-Level Logging

```python
from flip.utils import logger

logger.info("This is an info message")
logger.debug("This is a debug message")
logger.error("This is an error message")
```

## 🐛 Debugging

### FlipDebugger

```python
from flip.utils.debug import FlipDebugger
from pathlib import Path

# Create debugger
debugger = FlipDebugger(
    enabled=True,
    output_dir=Path("./flip_debug")
)

# Trace events
debugger.trace("retriever", "search", {"query": "test", "top_k": 5})

# Time operations
debugger.start_timer("embedding_generation")
# ... do work ...
debugger.stop_timer("embedding_generation")

# Get timing
duration = debugger.get_timing("embedding_generation")
print(f"Took {duration:.4f}s")

# Print all timings
debugger.print_timings()

# Dump to files
debugger.dump_traces("traces.json")
debugger.dump_timings("timings.json")
```

### Function Decoration

```python
from flip.utils.debug import debug_function, FlipDebugger

debugger = FlipDebugger(enabled=True)

@debug_function(debugger=debugger, component="my_component")
def my_function(x, y):
    return x + y

result = my_function(1, 2)
debugger.print_timings()
```

### Query Profiling

```python
from flip import Flip
from flip.utils.debug import profile_query

flip = Flip(directory="./docs")

# Profile a query
profile_query(
    flip_instance=flip,
    query="What is AI?",
    iterations=10
)
```

Output:
```
Profiling query: 'What is AI?'
Iterations: 10
------------------------------------------------------------
Iteration 1: 2.3451s
Iteration 2: 0.1234s  (cached)
...
------------------------------------------------------------
Mean: 0.5234s
Median: 0.1234s
Std Dev: 0.8912s
Min: 0.1234s
Max: 2.3451s
```

### Component Inspection

#### Inspect Embeddings

```python
from flip.utils.debug import inspect_embeddings

embeddings = embedder.embed_batch(texts)

inspect_embeddings(
    embeddings,
    output_file=Path("./debug/embeddings.json")
)
```

Output:
```
============================================================
Embedding Inspection
============================================================
count: 100
dimension: 1536
mean: 0.0234
std: 0.1234
min: -0.5678
max: 0.6789
shape: (100, 1536)
============================================================
```

#### Inspect Chunks

```python
from flip.utils.debug import inspect_chunks

chunks = chunker.chunk_text(text, metadata)

inspect_chunks(
    chunks,
    output_file=Path("./debug/chunks.json")
)
```

Output:
```
============================================================
Chunk Inspection
============================================================
count: 25
avg_length: 487
min_length: 234
max_length: 512
total_chars: 12175
============================================================

Sample Chunks:

[1] This is the first chunk of text...
[2] This is the second chunk of text...
[3] This is the third chunk of text...
```

## 🔍 Debugging Workflow

### 1. Enable Logging and Debugging

```python
from flip import Flip, FlipConfig
from flip.utils.logger import setup_logging
from flip.utils.debug import FlipDebugger

# Setup logging
logger = setup_logging(level="DEBUG", log_file="./logs/flip.log")

# Setup debugging
debugger = FlipDebugger(enabled=True)

# Initialize Flip
flip = Flip(directory="./docs")
```

### 2. Run Your Code

```python
debugger.start_timer("query")

response = flip.query("What is AI?")

debugger.stop_timer("query")

logger.info(f"Query completed in {debugger.get_timing('query'):.4f}s")
```

### 3. Analyze Results

```python
# Print timings
debugger.print_timings()

# Dump debug data
debugger.dump_traces()
debugger.dump_timings()

# Check logs
# cat ./logs/flip.log
```

## 📊 Performance Analysis

### Timing Breakdown

```python
debugger.start_timer("full_pipeline")

debugger.start_timer("document_loading")
documents = loader.load_directory("./docs")
debugger.stop_timer("document_loading")

debugger.start_timer("chunking")
chunks = chunker.chunk_text(text, metadata)
debugger.stop_timer("chunking")

debugger.start_timer("embedding")
embeddings = embedder.embed_batch(texts)
debugger.stop_timer("embedding")

debugger.stop_timer("full_pipeline")

debugger.print_timings()
```

### Profiling Queries

```python
from flip.utils.debug import profile_query

# Profile different queries
queries = [
    "What is AI?",
    "Explain machine learning",
    "What is RAG?"
]

for query in queries:
    print(f"\nProfiling: {query}")
    profile_query(flip, query, iterations=5)
```

## 🎯 Best Practices

### 1. Use Appropriate Log Levels

```python
logger.debug("Detailed variable values")  # Development only
logger.info("Normal operation")           # Production
logger.warning("Potential issue")         # Production
logger.error("Error occurred")            # Production
```

### 2. Enable Debugging for Development

```python
# Development
debugger = FlipDebugger(enabled=True)

# Production
debugger = FlipDebugger(enabled=False)
```

### 3. Profile Before Optimizing

```python
# Always profile first to find bottlenecks
profile_query(flip, "test query", iterations=10)

# Then optimize the slow parts
```

### 4. Inspect Components

```python
# Inspect embeddings to verify quality
inspect_embeddings(embeddings)

# Inspect chunks to verify chunking strategy
inspect_chunks(chunks)
```

## 📁 Output Files

### Log Files

```
./flip_logs/
├── flip.log          # Main log file
└── combined.log      # Combined logs
```

### Debug Files

```
./flip_debug/
├── traces_<timestamp>.json    # Trace events
├── timings_<timestamp>.json   # Timing data
├── embeddings.json            # Embedding inspection
└── chunks.json                # Chunk inspection
```

## 🔧 Troubleshooting

### Enable Verbose Logging

```python
logger = setup_logging(level="DEBUG", console=True)
```

### Check Timing Bottlenecks

```python
debugger.print_timings()
# Look for operations taking > 1s
```

### Inspect Failed Queries

```python
try:
    response = flip.query("test")
except Exception as e:
    logger.exception("Query failed")
    debugger.dump_traces()
```

## 📚 Examples

See [examples/logging_and_debugging.py](file:///C:/Users/adih4/OneDrive/Documents/Projects/FlipV2/examples/logging_and_debugging.py) for complete examples.

## 🎓 Advanced Usage

### Custom Log Format

```python
logger = setup_logging(
    level="INFO",
    format_string="[%(levelname)s] %(asctime)s - %(message)s"
)
```

### Conditional Debugging

```python
import os

debug_enabled = os.getenv("FLIP_DEBUG", "false").lower() == "true"
debugger = FlipDebugger(enabled=debug_enabled)
```

### Performance Monitoring

```python
# Combine with monitoring
stats = flip.get_monitoring_stats()
logger.info(f"Avg query time: {stats['avg_total_time']:.3f}s")

# Profile slow queries
slow_queries = flip.monitor.get_slow_queries(threshold=2.0)
for query in slow_queries:
    logger.warning(f"Slow query: {query['query']} ({query['total_time']:.2f}s)")
```
