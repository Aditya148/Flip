"""Database-specific test fixtures for integration testing."""

import pytest
import os
from typing import Generator

# Import all vector stores
from flip.vector_store.chroma import ChromaVectorStore
from flip.vector_store.mock import MockVectorStore

# Optional imports with availability checks
try:
    from flip.vector_store.pinecone import PineconeVectorStore, PINECONE_AVAILABLE
except ImportError:
    PINECONE_AVAILABLE = False

try:
    from flip.vector_store.qdrant import QdrantVectorStore, QDRANT_AVAILABLE
except ImportError:
    QDRANT_AVAILABLE = False

try:
    from flip.vector_store.weaviate import WeaviateVectorStore, WEAVIATE_AVAILABLE
except ImportError:
    WEAVIATE_AVAILABLE = False

try:
    from flip.vector_store.milvus import MilvusVectorStore, MILVUS_AVAILABLE
except ImportError:
    MILVUS_AVAILABLE = False

try:
    from flip.vector_store.faiss import FAISSVectorStore, FAISS_AVAILABLE
except ImportError:
    FAISS_AVAILABLE = False

try:
    from flip.vector_store.pgvector import PgvectorVectorStore, PGVECTOR_AVAILABLE
except ImportError:
    PGVECTOR_AVAILABLE = False

try:
    from flip.vector_store.redis import RedisVectorStore, REDIS_AVAILABLE
except ImportError:
    REDIS_AVAILABLE = False

try:
    from flip.vector_store.elasticsearch import ElasticsearchVectorStore, ELASTICSEARCH_AVAILABLE
except ImportError:
    ELASTICSEARCH_AVAILABLE = False

try:
    from flip.vector_store.mongodb import MongoDBVectorStore, MONGODB_AVAILABLE
except ImportError:
    MONGODB_AVAILABLE = False


@pytest.fixture
def mock_vector_store() -> Generator[MockVectorStore, None, None]:
    """Fixture for MockVectorStore."""
    store = MockVectorStore(collection_name="test_mock", dimension=384)
    yield store
    store.clear()


@pytest.fixture
def chroma_vector_store() -> Generator[ChromaVectorStore, None, None]:
    """Fixture for ChromaVectorStore."""
    store = ChromaVectorStore(collection_name="test_chroma", dimension=384)
    yield store
    store.clear()


@pytest.fixture
@pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not installed")
def faiss_vector_store() -> Generator[FAISSVectorStore, None, None]:
    """Fixture for FAISSVectorStore."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        store = FAISSVectorStore(
            collection_name="test_faiss",
            dimension=384,
            persist_directory=tmpdir
        )
        yield store
        store.clear()


@pytest.fixture
@pytest.mark.skipif(not PINECONE_AVAILABLE, reason="Pinecone not installed")
def pinecone_vector_store() -> Generator[PineconeVectorStore, None, None]:
    """Fixture for PineconeVectorStore."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        pytest.skip("PINECONE_API_KEY not set")
    
    store = PineconeVectorStore(
        collection_name="test-pinecone",
        api_key=api_key,
        environment=os.getenv("PINECONE_ENVIRONMENT", "gcp-starter"),
        dimension=384
    )
    yield store
    store.clear()


@pytest.fixture
@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="Qdrant not installed")
def qdrant_vector_store() -> Generator[QdrantVectorStore, None, None]:
    """Fixture for QdrantVectorStore (local)."""
    store = QdrantVectorStore(
        collection_name="test_qdrant",
        host="localhost",
        port=6333,
        dimension=384
    )
    yield store
    store.clear()


@pytest.fixture
@pytest.mark.skipif(not WEAVIATE_AVAILABLE, reason="Weaviate not installed")
def weaviate_vector_store() -> Generator[WeaviateVectorStore, None, None]:
    """Fixture for WeaviateVectorStore (local)."""
    store = WeaviateVectorStore(
        collection_name="test_weaviate",
        url="http://localhost:8080",
        dimension=384
    )
    yield store
    store.clear()


@pytest.fixture
@pytest.mark.skipif(not MILVUS_AVAILABLE, reason="Milvus not installed")
def milvus_vector_store() -> Generator[MilvusVectorStore, None, None]:
    """Fixture for MilvusVectorStore (local)."""
    store = MilvusVectorStore(
        collection_name="test_milvus",
        host="localhost",
        port=19530,
        dimension=384
    )
    yield store
    store.clear()


@pytest.fixture
@pytest.mark.skipif(not PGVECTOR_AVAILABLE, reason="Pgvector not installed")
def pgvector_vector_store() -> Generator[PgvectorVectorStore, None, None]:
    """Fixture for PgvectorVectorStore (local)."""
    store = PgvectorVectorStore(
        collection_name="test_pgvector",
        host="localhost",
        port=5432,
        database="test_db",
        user="postgres",
        password=os.getenv("POSTGRES_PASSWORD", ""),
        dimension=384
    )
    yield store
    store.clear()


@pytest.fixture
@pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not installed")
def redis_vector_store() -> Generator[RedisVectorStore, None, None]:
    """Fixture for RedisVectorStore (local)."""
    store = RedisVectorStore(
        collection_name="test_redis",
        host="localhost",
        port=6379,
        dimension=384
    )
    yield store
    store.clear()


@pytest.fixture
@pytest.mark.skipif(not ELASTICSEARCH_AVAILABLE, reason="Elasticsearch not installed")
def elasticsearch_vector_store() -> Generator[ElasticsearchVectorStore, None, None]:
    """Fixture for ElasticsearchVectorStore (local)."""
    store = ElasticsearchVectorStore(
        collection_name="test_elasticsearch",
        url="http://localhost:9200",
        dimension=384
    )
    yield store
    store.clear()


@pytest.fixture
@pytest.mark.skipif(not MONGODB_AVAILABLE, reason="MongoDB not installed")
def mongodb_vector_store() -> Generator[MongoDBVectorStore, None, None]:
    """Fixture for MongoDBVectorStore (local)."""
    store = MongoDBVectorStore(
        collection_name="test_mongodb",
        uri="mongodb://localhost:27017/",
        database="test_db",
        dimension=384
    )
    yield store
    store.clear()


# Parametrized fixture for testing across all available databases
@pytest.fixture(params=[
    "mock",
    "chroma",
    pytest.param("faiss", marks=pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not installed")),
    pytest.param("pinecone", marks=pytest.mark.skipif(not PINECONE_AVAILABLE, reason="Pinecone not installed")),
    pytest.param("qdrant", marks=pytest.mark.skipif(not QDRANT_AVAILABLE, reason="Qdrant not installed")),
    pytest.param("weaviate", marks=pytest.mark.skipif(not WEAVIATE_AVAILABLE, reason="Weaviate not installed")),
    pytest.param("milvus", marks=pytest.mark.skipif(not MILVUS_AVAILABLE, reason="Milvus not installed")),
    pytest.param("pgvector", marks=pytest.mark.skipif(not PGVECTOR_AVAILABLE, reason="Pgvector not installed")),
    pytest.param("redis", marks=pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not installed")),
    pytest.param("elasticsearch", marks=pytest.mark.skipif(not ELASTICSEARCH_AVAILABLE, reason="Elasticsearch not installed")),
    pytest.param("mongodb", marks=pytest.mark.skipif(not MONGODB_AVAILABLE, reason="MongoDB not installed")),
])
def all_vector_stores(request):
    """Parametrized fixture that provides all available vector stores."""
    fixture_name = f"{request.param}_vector_store"
    return request.getfixturevalue(fixture_name)
