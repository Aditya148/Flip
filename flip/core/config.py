"""Configuration management for Flip SDK."""

from typing import Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, field_validator
from pathlib import Path


class FlipConfig(BaseModel):
    """Configuration for Flip SDK with sensible defaults."""
    
    # LLM Configuration
    llm_provider: Literal[
        "openai", "azure-openai", "anthropic", "google", "huggingface", "meta", "ollama"
    ] = Field(default="openai", description="LLM provider to use")
    
    llm_model: Optional[str] = Field(
        default=None,
        description="Specific model name. If None, uses provider default"
    )
    
    llm_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for generation"
    )
    
    llm_max_tokens: int = Field(
        default=1024,
        gt=0,
        description="Maximum tokens to generate"
    )
    
    # Embedding Configuration
    embedding_provider: Literal[
        "openai", "azure-openai", "huggingface", "sentence-transformers", "google"
    ] = Field(default="openai", description="Embedding provider to use")
    
    embedding_model: Optional[str] = Field(
        default=None,
        description="Specific embedding model. If None, uses provider default"
    )
    
    # Vector Store Configuration
    vector_store: Literal[
        "chroma", "pinecone", "qdrant", "weaviate", "faiss"
    ] = Field(default="chroma", description="Vector store to use")
    
    vector_store_path: Optional[Path] = Field(
        default=None,
        description="Path for local vector store. If None, uses ./flip_data"
    )
    
    # Chunking Configuration
    chunking_strategy: Literal[
        "token", "sentence", "semantic", "recursive"
    ] = Field(default="semantic", description="Chunking strategy")
    
    chunk_size: int = Field(
        default=512,
        gt=0,
        description="Target chunk size in tokens"
    )
    
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        description="Overlap between chunks in tokens"
    )
    
    # Retrieval Configuration
    retrieval_top_k: int = Field(
        default=5,
        gt=0,
        description="Number of chunks to retrieve"
    )
    
    use_hybrid_search: bool = Field(
        default=True,
        description="Use hybrid search (vector + keyword)"
    )
    
    use_reranking: bool = Field(
        default=True,
        description="Use reranking for better results"
    )
    
    reranker_model: Optional[str] = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Reranker model to use"
    )
    
    # Cache Configuration
    enable_cache: bool = Field(
        default=True,
        description="Enable caching for embeddings and queries"
    )
    
    cache_dir: Optional[Path] = Field(
        default=None,
        description="Cache directory. If None, uses ./flip_cache"
    )
    
    # Processing Configuration
    batch_size: int = Field(
        default=32,
        gt=0,
        description="Batch size for processing"
    )
    
    show_progress: bool = Field(
        default=True,
        description="Show progress bars"
    )
    
    # API Keys (optional, can also use environment variables)
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    azure_openai_api_key: Optional[str] = Field(default=None, description="Azure OpenAI API key")
    azure_openai_endpoint: Optional[str] = Field(default=None, description="Azure OpenAI endpoint")
    azure_openai_api_version: str = Field(default="2024-02-15-preview", description="Azure OpenAI API version")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    google_api_key: Optional[str] = Field(default=None, description="Google API key")
    huggingface_api_key: Optional[str] = Field(default=None, description="HuggingFace API key")
    pinecone_api_key: Optional[str] = Field(default=None, description="Pinecone API key")
    
    # Advanced Configuration
    custom_prompt_template: Optional[str] = Field(
        default=None,
        description="Custom prompt template for generation"
    )
    
    metadata_fields: list[str] = Field(
        default_factory=lambda: ["source", "page", "chunk_id"],
        description="Metadata fields to track"
    )
    
    @field_validator("vector_store_path", "cache_dir")
    @classmethod
    def validate_paths(cls, v: Optional[Path]) -> Optional[Path]:
        """Ensure paths are Path objects."""
        if v is not None and not isinstance(v, Path):
            return Path(v)
        return v
    
    def get_llm_default_model(self) -> str:
        """Get default model for the selected LLM provider."""
        defaults = {
            "openai": "gpt-4-turbo-preview",
            "azure-openai": "gpt-4",
            "anthropic": "claude-3-sonnet-20240229",
            "google": "gemini-pro",
            "huggingface": "meta-llama/Llama-2-70b-chat-hf",
            "meta": "meta-llama/Llama-2-70b-chat-hf",
            "ollama": "llama2",
        }
        return self.llm_model or defaults.get(self.llm_provider, "gpt-4-turbo-preview")
    
    def get_embedding_default_model(self) -> str:
        """Get default embedding model for the selected provider."""
        defaults = {
            "openai": "text-embedding-3-small",
            "azure-openai": "text-embedding-ada-002",
            "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers": "all-MiniLM-L6-v2",
            "google": "models/embedding-001",
        }
        return self.embedding_model or defaults.get(
            self.embedding_provider, "text-embedding-3-small"
        )
    
    def get_vector_store_path(self) -> Path:
        """Get the vector store path, using default if not set."""
        return self.vector_store_path or Path("./flip_data")
    
    def get_cache_dir(self) -> Path:
        """Get the cache directory, using default if not set."""
        return self.cache_dir or Path("./flip_cache")
    
    class Config:
        """Pydantic config."""
        validate_assignment = True
        extra = "allow"  # Allow extra fields for extensibility
