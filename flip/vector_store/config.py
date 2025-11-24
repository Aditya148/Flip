"""Enhanced vector store configuration."""

from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator
from dataclasses import dataclass
import os


@dataclass
class VectorStoreConnectionParams:
    """Connection parameters for vector stores."""
    
    # Common parameters
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    
    # Database-specific
    host: Optional[str] = None
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = None
    
    # Cloud-specific
    environment: Optional[str] = None
    region: Optional[str] = None
    project_id: Optional[str] = None
    
    # Collection/Index naming
    collection_name: Optional[str] = None
    index_name: Optional[str] = None
    namespace: Optional[str] = None
    
    # Additional params
    extra_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}


class VectorStoreConfig(BaseModel):
    """Configuration for vector store providers."""
    
    # Provider selection
    provider: Literal[
        "chroma",
        "pinecone",
        "qdrant",
        "weaviate",
        "milvus",
        "faiss",
        "pgvector",
        "redis",
        "elasticsearch"
    ] = Field(default="chroma", description="Vector store provider")
    
    # Connection parameters
    connection_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific connection parameters"
    )
    
    # Performance parameters
    batch_size: int = Field(
        default=100,
        gt=0,
        description="Batch size for operations"
    )
    
    timeout: int = Field(
        default=30,
        gt=0,
        description="Operation timeout in seconds"
    )
    
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum number of retries for failed operations"
    )
    
    # Feature flags
    enable_filtering: bool = Field(
        default=True,
        description="Enable metadata filtering"
    )
    
    enable_hybrid_search: bool = Field(
        default=False,
        description="Enable hybrid search (if supported)"
    )
    
    # Persistence
    persist_directory: Optional[str] = Field(
        default=None,
        description="Directory for local persistence"
    )
    
    # Collection/Index naming
    collection_name: str = Field(
        default="flip_collection",
        description="Name of collection/index"
    )
    
    @field_validator('connection_params')
    @classmethod
    def load_from_env(cls, v: Dict[str, Any], info) -> Dict[str, Any]:
        """Load connection params from environment variables if not provided."""
        provider = info.data.get('provider', 'chroma')
        
        # Environment variable mappings
        env_mappings = {
            'pinecone': {
                'api_key': 'PINECONE_API_KEY',
                'environment': 'PINECONE_ENVIRONMENT',
                'index_name': 'PINECONE_INDEX_NAME',
            },
            'qdrant': {
                'url': 'QDRANT_URL',
                'api_key': 'QDRANT_API_KEY',
            },
            'weaviate': {
                'url': 'WEAVIATE_URL',
                'api_key': 'WEAVIATE_API_KEY',
            },
            'milvus': {
                'host': 'MILVUS_HOST',
                'port': 'MILVUS_PORT',
            },
            'pgvector': {
                'host': 'PGVECTOR_HOST',
                'port': 'PGVECTOR_PORT',
                'database': 'PGVECTOR_DATABASE',
                'user': 'PGVECTOR_USER',
                'password': 'PGVECTOR_PASSWORD',
            },
            'redis': {
                'host': 'REDIS_HOST',
                'port': 'REDIS_PORT',
                'password': 'REDIS_PASSWORD',
            },
            'elasticsearch': {
                'url': 'ELASTICSEARCH_URL',
                'api_key': 'ELASTICSEARCH_API_KEY',
            }
        }
        
        if provider in env_mappings:
            for param, env_var in env_mappings[provider].items():
                if param not in v and os.getenv(env_var):
                    v[param] = os.getenv(env_var)
        
        return v
    
    def validate_config(self) -> tuple[bool, str]:
        """
        Validate configuration for the selected provider.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        provider = self.provider
        params = self.connection_params
        
        # Provider-specific validation
        if provider == 'pinecone':
            if not params.get('api_key'):
                return False, "Pinecone requires 'api_key' in connection_params or PINECONE_API_KEY env var"
            if not params.get('environment'):
                return False, "Pinecone requires 'environment' in connection_params or PINECONE_ENVIRONMENT env var"
        
        elif provider == 'qdrant':
            if not params.get('url') and not params.get('host'):
                return False, "Qdrant requires 'url' or 'host' in connection_params"
        
        elif provider == 'weaviate':
            if not params.get('url'):
                return False, "Weaviate requires 'url' in connection_params"
        
        elif provider == 'milvus':
            if not params.get('host'):
                return False, "Milvus requires 'host' in connection_params"
        
        elif provider == 'pgvector':
            required = ['host', 'database', 'user', 'password']
            missing = [p for p in required if not params.get(p)]
            if missing:
                return False, f"Pgvector requires: {', '.join(missing)}"
        
        elif provider == 'redis':
            if not params.get('host'):
                return False, "Redis requires 'host' in connection_params"
        
        elif provider == 'elasticsearch':
            if not params.get('url'):
                return False, "Elasticsearch requires 'url' in connection_params"
        
        return True, ""
    
    class Config:
        """Pydantic config."""
        validate_assignment = True
        extra = "allow"


def parse_connection_string(connection_string: str) -> VectorStoreConnectionParams:
    """
    Parse connection string into connection parameters.
    
    Supports formats like:
    - chroma://path/to/directory
    - pinecone://api-key@environment/index-name
    - qdrant://host:port/collection-name
    - postgresql://user:pass@host:port/database
    
    Args:
        connection_string: Connection string to parse
        
    Returns:
        Parsed connection parameters
    """
    params = VectorStoreConnectionParams()
    
    if "://" in connection_string:
        protocol, rest = connection_string.split("://", 1)
        
        if protocol == "chroma":
            params.endpoint = rest
        elif protocol == "pinecone":
            # Format: api-key@environment/index-name
            if "@" in rest:
                api_key, rest = rest.split("@", 1)
                params.api_key = api_key
            if "/" in rest:
                environment, index_name = rest.split("/", 1)
                params.environment = environment
                params.index_name = index_name
        elif protocol in ["qdrant", "weaviate"]:
            # Format: host:port/collection-name
            if "/" in rest:
                host_port, collection = rest.split("/", 1)
                params.collection_name = collection
                if ":" in host_port:
                    host, port = host_port.split(":", 1)
                    params.host = host
                    params.port = int(port)
                else:
                    params.host = host_port
            else:
                params.host = rest
        elif protocol == "postgresql":
            # Format: user:pass@host:port/database
            if "@" in rest:
                user_pass, host_db = rest.split("@", 1)
                if ":" in user_pass:
                    params.username, params.password = user_pass.split(":", 1)
                if "/" in host_db:
                    host_port, database = host_db.split("/", 1)
                    params.database = database
                    if ":" in host_port:
                        host, port = host_port.split(":", 1)
                        params.host = host
                        params.port = int(port)
    
    return params


def migrate_config(old_config: Dict[str, Any], new_provider: str) -> VectorStoreConfig:
    """
    Migrate configuration from one provider to another.
    
    Args:
        old_config: Existing configuration dictionary
        new_provider: Target provider name
        
    Returns:
        New VectorStoreConfig for the target provider
    """
    # Extract common parameters
    new_config = {
        'provider': new_provider,
        'collection_name': old_config.get('collection_name', 'flip_collection'),
        'batch_size': old_config.get('batch_size', 100),
        'timeout': old_config.get('timeout', 30),
        'max_retries': old_config.get('max_retries', 3),
    }
    
    # Map provider-specific parameters
    old_provider = old_config.get('provider', 'chroma')
    connection_params = {}
    
    # Try to preserve API keys and endpoints
    if 'api_key' in old_config.get('connection_params', {}):
        connection_params['api_key'] = old_config['connection_params']['api_key']
    
    new_config['connection_params'] = connection_params
    
    return VectorStoreConfig(**new_config)

