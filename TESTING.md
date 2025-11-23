# Flip SDK Testing Guide

## Running Tests

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test Files

```bash
# Unit tests
pytest tests/test_document_processing.py -v
pytest tests/test_embedding.py -v
pytest tests/test_vector_store.py -v
pytest tests/test_retrieval.py -v

# Integration tests
pytest tests/test_integration.py -v
```

### Run Tests by Marker

```bash
# Run only integration tests
pytest -m integration -v

# Skip slow tests
pytest -m "not slow" -v

# Run only benchmarks
pytest -m benchmark -v
```

### Run with Coverage

```bash
pytest tests/ --cov=flip --cov-report=html
```

View coverage report: `open htmlcov/index.html`

## Performance Benchmarking

### Run Benchmarks

```bash
python tests/benchmark.py
```

This will run:
- Indexing speed benchmark
- Query latency benchmark
- Chunking strategy comparison
- Hybrid vs vector search comparison
- Cache performance analysis

Results are saved to `benchmark_results.txt`.

### Custom Benchmarks

```python
from tests.benchmark import PerformanceBenchmark

benchmark = PerformanceBenchmark()
benchmark.benchmark_indexing(num_docs=100, doc_size=1000)
benchmark.benchmark_query_latency(num_queries=50)
benchmark.save_results()
```

## Test Structure

```
tests/
├── conftest.py                    # Pytest configuration and fixtures
├── test_document_processing.py    # Document loader and chunker tests
├── test_embedding.py              # Embedding provider tests
├── test_vector_store.py           # Vector store tests
├── test_retrieval.py              # Retrieval and re-ranking tests
├── test_integration.py            # End-to-end integration tests
└── benchmark.py                   # Performance benchmarks
```

## Writing New Tests

### Unit Test Template

```python
import pytest
from flip.your_module import YourClass

class TestYourClass:
    """Test your class."""
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        obj = YourClass()
        result = obj.method()
        assert result is not None
    
    def test_edge_case(self):
        """Test edge case."""
        obj = YourClass()
        with pytest.raises(ValueError):
            obj.method(invalid_input)
```

### Integration Test Template

```python
import pytest
from flip import Flip, FlipConfig

class TestFeature:
    """Test feature integration."""
    
    @pytest.fixture
    def setup_flip(self, tmp_path):
        """Setup Flip instance."""
        config = FlipConfig(
            embedding_provider="sentence-transformers"
        )
        return Flip(directory=str(tmp_path), config=config)
    
    def test_feature(self, setup_flip):
        """Test feature."""
        flip = setup_flip
        result = flip.some_method()
        assert result is not None
```

## Continuous Integration

### GitHub Actions

Create `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: pytest tests/ -v --cov=flip
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## Test Coverage Goals

- **Unit Tests**: >80% coverage for all modules
- **Integration Tests**: Cover all major workflows
- **Benchmarks**: Track performance over time

## Troubleshooting

### Tests Fail Due to Missing Dependencies

```bash
pip install -r requirements-dev.txt
```

### Tests Timeout

Increase timeout for slow tests:

```python
@pytest.mark.timeout(60)
def test_slow_operation():
    pass
```

### Skip Tests Requiring API Keys

```python
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="API key not available"
)
def test_openai_integration():
    pass
```

## Best Practices

1. **Isolate Tests**: Each test should be independent
2. **Use Fixtures**: Reuse common setup code
3. **Mock External APIs**: Don't make real API calls in tests
4. **Test Edge Cases**: Include error conditions
5. **Keep Tests Fast**: Use markers for slow tests
6. **Document Tests**: Clear docstrings for each test

## Performance Benchmarking Best Practices

1. **Consistent Environment**: Run benchmarks on same hardware
2. **Multiple Runs**: Average results over multiple runs
3. **Warm-up**: Run once before timing to warm caches
4. **Track Over Time**: Compare with previous benchmarks
5. **Document Changes**: Note any configuration changes
