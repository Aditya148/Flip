"""Pgvector (PostgreSQL) vector store implementation."""

from typing import List, Dict, Any, Optional
import time
import json

try:
    import psycopg2
    from psycopg2 import pool
    from psycopg2.extras import execute_values
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False

from flip.vector_store.base import (
    BaseVectorStore,
    SearchResult,
    VectorStoreStats,
    HealthCheckResult,
    HealthStatus
)
from flip.core.exceptions import VectorStoreError


class PgvectorVectorStore(BaseVectorStore):
    """Pgvector (PostgreSQL with vector extension) vector store implementation."""
    
    def __init__(
        self,
        collection_name: str,
        host: str = "localhost",
        port: int = 5432,
        database: str = "postgres",
        user: str = "postgres",
        password: str = "",
        dimension: int = 1536,
        index_type: str = "ivfflat",
        distance_metric: str = "cosine",
        pool_size: int = 5,
        **kwargs
    ):
        """
        Initialize Pgvector vector store.
        
        Args:
            collection_name: Name of the table
            host: PostgreSQL host
            port: PostgreSQL port
            database: Database name
            user: Username
            password: Password
            dimension: Vector dimension
            index_type: Index type (ivfflat, hnsw)
            distance_metric: Distance metric (cosine, l2, inner_product)
            pool_size: Connection pool size
            **kwargs: Additional PostgreSQL settings
        """
        if not PGVECTOR_AVAILABLE:
            raise VectorStoreError(
                "Pgvector is not installed. Install with: pip install psycopg2-binary pgvector"
            )
        
        super().__init__(collection_name, **kwargs)
        
        self._dimension = dimension
        self.index_type = index_type
        self.distance_metric = distance_metric
        self.table_name = collection_name.lower().replace("-", "_")
        
        # Create connection pool
        try:
            self.pool = psycopg2.pool.SimpleConnectionPool(
                1, pool_size,
                host=host,
                port=port,
                database=database,
                user=user,
                password=password
            )
            
            # Initialize database
            self._init_database()
            
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize Pgvector: {str(e)}")
    
    def _get_connection(self):
        """Get connection from pool."""
        return self.pool.getconn()
    
    def _return_connection(self, conn):
        """Return connection to pool."""
        self.pool.putconn(conn)
    
    def _init_database(self):
        """Initialize database with vector extension and table."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Enable vector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                
                # Create table
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id TEXT PRIMARY KEY,
                        embedding vector({self._dimension}),
                        text TEXT,
                        metadata JSONB
                    )
                """)
                
                # Create index if not exists
                self._create_index_if_needed(cur)
                
                conn.commit()
        finally:
            self._return_connection(conn)
    
    def _create_index_if_needed(self, cur):
        """Create vector index if it doesn't exist."""
        index_name = f"{self.table_name}_embedding_idx"
        
        # Check if index exists
        cur.execute(f"""
            SELECT 1 FROM pg_indexes 
            WHERE tablename = '{self.table_name}' 
            AND indexname = '{index_name}'
        """)
        
        if cur.fetchone():
            return  # Index already exists
        
        # Get distance operator
        if self.distance_metric == "cosine":
            ops = "vector_cosine_ops"
        elif self.distance_metric == "l2":
            ops = "vector_l2_ops"
        elif self.distance_metric == "inner_product":
            ops = "vector_ip_ops"
        else:
            ops = "vector_cosine_ops"
        
        # Create index
        if self.index_type == "ivfflat":
            cur.execute(f"""
                CREATE INDEX {index_name} ON {self.table_name} 
                USING ivfflat (embedding {ops})
                WITH (lists = 100)
            """)
        elif self.index_type == "hnsw":
            cur.execute(f"""
                CREATE INDEX {index_name} ON {self.table_name} 
                USING hnsw (embedding {ops})
            """)
    
    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """Add vectors to Pgvector."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Prepare data
                data = []
                for i, id in enumerate(ids):
                    metadata = metadatas[i] if metadatas else {}
                    data.append((
                        id,
                        embeddings[i],
                        texts[i],
                        json.dumps(metadata)
                    ))
                
                # Insert with ON CONFLICT to handle duplicates
                execute_values(
                    cur,
                    f"""
                    INSERT INTO {self.table_name} (id, embedding, text, metadata)
                    VALUES %s
                    ON CONFLICT (id) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        text = EXCLUDED.text,
                        metadata = EXCLUDED.metadata
                    """,
                    data
                )
                
                conn.commit()
        except Exception as e:
            conn.rollback()
            raise VectorStoreError(f"Failed to add to Pgvector: {str(e)}")
        finally:
            self._return_connection(conn)
    
    def batch_add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 100
    ):
        """Add vectors in batches."""
        total = len(ids)
        for i in range(0, total, batch_size):
            end_idx = min(i + batch_size, total)
            self.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                texts=texts[i:end_idx],
                metadatas=metadatas[i:end_idx] if metadatas else None
            )
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search Pgvector for similar vectors."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Get distance operator
                if self.distance_metric == "cosine":
                    op = "<=>"
                elif self.distance_metric == "l2":
                    op = "<->"
                elif self.distance_metric == "inner_product":
                    op = "<#>"
                else:
                    op = "<=>"
                
                # Build query
                query = f"""
                    SELECT id, text, metadata, embedding {op} %s::vector AS distance
                    FROM {self.table_name}
                """
                
                params = [query_embedding]
                
                # Add filter if provided
                if filter_dict:
                    conditions = []
                    for key, value in filter_dict.items():
                        conditions.append(f"metadata->>'{key}' = %s")
                        params.append(str(value))
                    query += " WHERE " + " AND ".join(conditions)
                
                query += f" ORDER BY distance LIMIT {top_k}"
                
                cur.execute(query, params)
                results = cur.fetchall()
                
                # Convert to SearchResult objects
                search_results = []
                for row in results:
                    id, text, metadata_json, distance = row
                    metadata = json.loads(metadata_json) if metadata_json else {}
                    
                    # Convert distance to similarity score
                    score = 1.0 / (1.0 + distance) if self.distance_metric == "l2" else 1.0 - distance
                    
                    search_results.append(SearchResult(
                        id=id,
                        text=text,
                        score=float(score),
                        metadata=metadata
                    ))
                
                return search_results
                
        except Exception as e:
            raise VectorStoreError(f"Failed to search Pgvector: {str(e)}")
        finally:
            self._return_connection(conn)
    
    def filter_search(
        self,
        query_embedding: List[float],
        filters: Dict[str, Any],
        top_k: int = 5
    ) -> List[SearchResult]:
        """Search with metadata filtering."""
        return self.search(query_embedding, top_k, filter_dict=filters)
    
    def get_by_ids(
        self,
        ids: List[str]
    ) -> List[SearchResult]:
        """Retrieve vectors by their IDs."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT id, text, metadata FROM {self.table_name} WHERE id = ANY(%s)",
                    (ids,)
                )
                results = cur.fetchall()
                
                search_results = []
                for row in results:
                    id, text, metadata_json = row
                    metadata = json.loads(metadata_json) if metadata_json else {}
                    
                    search_results.append(SearchResult(
                        id=id,
                        text=text,
                        score=1.0,
                        metadata=metadata
                    ))
                
                return search_results
                
        except Exception as e:
            raise VectorStoreError(f"Failed to get by IDs from Pgvector: {str(e)}")
        finally:
            self._return_connection(conn)
    
    def delete(self, ids: List[str]):
        """Delete vectors from Pgvector."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self.table_name} WHERE id = ANY(%s)",
                    (ids,)
                )
                conn.commit()
        except Exception as e:
            conn.rollback()
            raise VectorStoreError(f"Failed to delete from Pgvector: {str(e)}")
        finally:
            self._return_connection(conn)
    
    def update(
        self,
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None,
        texts: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """Update vectors in Pgvector."""
        if not embeddings:
            raise VectorStoreError("Pgvector update requires embeddings")
        
        # Use upsert (add handles this)
        self.add(ids, embeddings, texts or [""] * len(ids), metadatas)
    
    def count(self) -> int:
        """Get count of vectors in Pgvector."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                return cur.fetchone()[0]
        except Exception as e:
            raise VectorStoreError(f"Failed to count Pgvector vectors: {str(e)}")
        finally:
            self._return_connection(conn)
    
    def clear(self):
        """Clear all vectors from Pgvector table."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {self.table_name}")
                conn.commit()
        except Exception as e:
            conn.rollback()
            raise VectorStoreError(f"Failed to clear Pgvector: {str(e)}")
        finally:
            self._return_connection(conn)
    
    def health_check(self) -> HealthCheckResult:
        """Check Pgvector health."""
        try:
            start = time.time()
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
                latency_ms = (time.time() - start) * 1000
                
                return HealthCheckResult(
                    status=HealthStatus.HEALTHY,
                    latency_ms=latency_ms,
                    message=f"Pgvector table '{self.table_name}' is healthy",
                    details={"connected": True}
                )
            finally:
                self._return_connection(conn)
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                latency_ms=0.0,
                message=f"Pgvector health check failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def get_stats(self) -> VectorStoreStats:
        """Get Pgvector statistics."""
        try:
            count = self.count()
            
            return VectorStoreStats(
                total_vectors=count,
                dimension=self._dimension,
                metadata={
                    "provider": self.provider_name,
                    "table_name": self.table_name,
                    "index_type": self.index_type,
                    "distance_metric": self.distance_metric
                }
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to get Pgvector stats: {str(e)}")
    
    def __del__(self):
        """Close connection pool."""
        try:
            if hasattr(self, 'pool'):
                self.pool.closeall()
        except:
            pass
    
    @property
    def dimension(self) -> int:
        """Get vector dimension."""
        return self._dimension
    
    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "pgvector"
