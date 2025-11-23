"""Main Flip SDK class."""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass
from tqdm import tqdm

from flip.core.config import FlipConfig
from flip.core.exceptions import FlipException, ConfigurationError
from flip.document_processing.loader import DocumentLoader
from flip.document_processing.chunker import TextChunker
from flip.document_processing.preprocessor import TextPreprocessor
from flip.embedding.factory import EmbedderFactory
from flip.vector_store.factory import VectorStoreFactory
from flip.generation.factory import LLMFactory


@dataclass
class FlipResponse:
    """Response from Flip query."""
    
    answer: str
    """Generated answer."""
    
    citations: List[Dict[str, Any]]
    """Source citations used in the answer."""
    
    context_chunks: List[str]
    """Retrieved context chunks."""
    
    metadata: Dict[str, Any]
    """Additional metadata (tokens used, model, etc.)."""


class Flip:
    """
    Flip - Fully Automated RAG SDK
    
    Initialize with a directory and start querying - that's it!
    
    Example:
        >>> flip = Flip(directory="./docs")
        >>> response = flip.query("What is the main topic?")
        >>> print(response.answer)
        >>> print(response.citations)
    """
    
    def __init__(
        self,
        directory: Optional[str] = None,
        config: Optional[FlipConfig] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Flip SDK.
        
        Args:
            directory: Directory containing documents to index
            config: FlipConfig object (optional, uses defaults if not provided)
            llm_provider: LLM provider to use (overrides config)
            llm_model: LLM model to use (overrides config)
            **kwargs: Additional configuration options
        """
        # Initialize configuration
        if config is None:
            config = FlipConfig(**kwargs)
        
        # Override with direct parameters
        if llm_provider:
            config.llm_provider = llm_provider
        if llm_model:
            config.llm_model = llm_model
        
        self.config = config
        self.directory = directory
        
        # Initialize components
        self._initialize_components()
        
        # Auto-index if directory provided
        if directory:
            self.index_directory(directory)
    
    def _initialize_components(self):
        """Initialize all SDK components."""
        # Document processing
        self.loader = DocumentLoader(show_progress=self.config.show_progress)
        self.chunker = TextChunker(
            strategy=self.config.chunking_strategy,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        self.preprocessor = TextPreprocessor()
        
        # Embedding
        embedding_model = self.config.get_embedding_default_model()
        self.embedder = EmbedderFactory.create(
            provider=self.config.embedding_provider,
            model=embedding_model,
            api_key=self._get_api_key(self.config.embedding_provider)
        )
        
        # Vector store
        persist_dir = str(self.config.get_vector_store_path())
        self.vector_store = VectorStoreFactory.create(
            provider=self.config.vector_store,
            collection_name="flip_collection",
            persist_directory=persist_dir
        )
        
        # LLM
        llm_model = self.config.get_llm_default_model()
        self.llm = LLMFactory.create(
            provider=self.config.llm_provider,
            model=llm_model,
            api_key=self._get_api_key(self.config.llm_provider)
        )
        
        # Pipeline orchestrator with advanced features
        from flip.pipeline.orchestrator import PipelineOrchestrator
        
        cache_dir = self.config.get_cache_dir() if self.config.enable_cache else None
        
        self.pipeline = PipelineOrchestrator(
            loader=self.loader,
            chunker=self.chunker,
            preprocessor=self.preprocessor,
            embedder=self.embedder,
            vector_store=self.vector_store,
            llm=self.llm,
            use_hybrid_search=self.config.use_hybrid_search,
            use_reranking=self.config.use_reranking,
            use_query_enhancement=True,  # Enable query enhancement
            enable_cache=self.config.enable_cache,
            cache_dir=cache_dir
        )
        
        # Incremental update components
        from flip.pipeline.incremental import DocumentTracker, IncrementalUpdater
        
        tracking_file = self.config.get_vector_store_path() / "document_tracking.json"
        self.document_tracker = DocumentTracker(tracking_file=tracking_file)
        self.incremental_updater = IncrementalUpdater(
            loader=self.loader,
            tracker=self.document_tracker,
            pipeline_orchestrator=self.pipeline
        )
        
        # Monitoring
        from flip.evaluation.monitor import RAGMonitor
        
        monitor_file = self.config.get_vector_store_path() / "rag_monitor.jsonl"
        self.monitor = RAGMonitor(log_file=monitor_file)
        
        # State
        self._indexed = False
        self._document_count = 0
        self._chunk_count = 0
    
    def _get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a provider."""
        key_map = {
            "openai": self.config.openai_api_key or os.getenv("OPENAI_API_KEY"),
            "azure-openai": self.config.azure_openai_api_key or os.getenv("AZURE_OPENAI_API_KEY"),
            "anthropic": self.config.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"),
            "google": self.config.google_api_key or os.getenv("GOOGLE_API_KEY"),
            "huggingface": self.config.huggingface_api_key or os.getenv("HUGGINGFACE_API_KEY"),
        }
        return key_map.get(provider)
    
    def index_directory(
        self,
        directory: str,
        recursive: bool = True,
        file_extensions: Optional[List[str]] = None
    ):
        """
        Index all documents in a directory.
        
        Args:
            directory: Directory path
            recursive: Whether to search subdirectories
            file_extensions: Optional list of file extensions to filter
        """
        print(f"🔍 Indexing documents from: {directory}")
        
        # Load documents
        documents = self.loader.load_directory(
            directory,
            recursive=recursive,
            file_extensions=file_extensions
        )
        
        if not documents:
            print("⚠️  No documents found!")
            return
        
        self._document_count = len(documents)
        print(f"📄 Loaded {self._document_count} documents")
        
        # Process documents through pipeline
        print("⚙️  Processing documents...")
        all_ids, all_embeddings, all_texts, all_metadatas = self.pipeline.process_documents(documents)
        
        self._chunk_count = len(all_ids)
        print(f"✂️  Created {self._chunk_count} chunks")
        
        # Add to vector store
        print("💾 Storing in vector database...")
        self.vector_store.add(
            ids=all_ids,
            embeddings=all_embeddings,
            texts=all_texts,
            metadatas=all_metadatas
        )
        
        # Save caches
        if self.config.enable_cache:
            self.pipeline.save_caches()
        
        self._indexed = True
        print(f"✅ Indexing complete! {self._chunk_count} chunks indexed.")
        
        # Print advanced features status
        if self.config.use_hybrid_search:
            print("🔀 Hybrid search enabled (vector + keyword)")
        if self.config.use_reranking:
            print("📊 Re-ranking enabled for better accuracy")
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False
    ) -> FlipResponse:
        """
        Query the indexed documents.
        
        Args:
            question: User's question
            top_k: Number of context chunks to retrieve (uses config default if None)
            temperature: LLM temperature (uses config default if None)
            stream: Whether to stream the response
            
        Returns:
            FlipResponse object with answer and citations
        """
        if not self._indexed:
            raise FlipException("No documents indexed. Call index_directory() first.")
        
        top_k = top_k or self.config.retrieval_top_k
        temperature = temperature or self.config.llm_temperature
        
        if stream:
            # TODO: Implement streaming
            raise NotImplementedError("Streaming not yet implemented")
        
        # Use pipeline for retrieval and generation
        result = self.pipeline.retrieve_and_generate(
            query=question,
            top_k=top_k,
            temperature=temperature,
            max_tokens=self.config.llm_max_tokens
        )
        
        return FlipResponse(
            answer=result["answer"],
            citations=result["citations"],
            context_chunks=result["context_chunks"],
            metadata=result["metadata"]
        )
    
    def add_documents(self, file_paths: List[str]):
        """
        Add specific documents to the index.
        
        Args:
            file_paths: List of file paths to add
        """
        documents = []
        for path in file_paths:
            doc = self.loader.load_file(path)
            documents.append(doc)
        
        # Process through pipeline
        all_ids, all_embeddings, all_texts, all_metadatas = self.pipeline.process_documents(documents)
        
        # Add to vector store
        self.vector_store.add(
            ids=all_ids,
            embeddings=all_embeddings,
            texts=all_texts,
            metadatas=all_metadatas
        )
        
        self._chunk_count += len(all_ids)
        self._indexed = True
        
        # Save caches
        if self.config.enable_cache:
            self.pipeline.save_caches()
    
    def clear(self):
        """Clear all indexed documents and caches."""
        self.vector_store.clear()
        self._indexed = False
        self._document_count = 0
        self._chunk_count = 0
        
        # Clear caches
        if self.config.enable_cache:
            self.pipeline.clear_caches()
        
        # Reset hybrid search index
        if self.config.use_hybrid_search:
            self.pipeline.retriever.reset_index()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get indexing statistics and configuration."""
        return {
            "indexed": self._indexed,
            "document_count": self._document_count,
            "chunk_count": self._chunk_count,
            "vector_store_count": self.vector_store.count(),
            "llm_provider": self.config.llm_provider,
            "llm_model": self.config.get_llm_default_model(),
            "embedding_provider": self.config.embedding_provider,
            "embedding_model": self.config.get_embedding_default_model(),
            "hybrid_search": self.config.use_hybrid_search,
            "reranking": self.config.use_reranking,
            "caching": self.config.enable_cache,
        }
    
    def refresh_index(
        self,
        directory: Optional[str] = None,
        recursive: bool = True,
        file_extensions: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """
        Incrementally update the index with changed documents.
        
        Args:
            directory: Directory to refresh from (uses initial directory if None)
            recursive: Whether to search subdirectories
            file_extensions: Optional file extensions filter
            
        Returns:
            Dictionary with counts of added, updated, deleted documents
        """
        dir_to_refresh = directory or self.directory
        
        if not dir_to_refresh:
            raise FlipException("No directory specified for refresh")
        
        print(f"🔄 Refreshing index from: {dir_to_refresh}")
        
        stats = self.incremental_updater.update_index(
            directory=dir_to_refresh,
            vector_store=self.vector_store,
            recursive=recursive,
            file_extensions=file_extensions,
            show_progress=self.config.show_progress
        )
        
        # Update counts
        self._chunk_count = self.vector_store.count()
        
        return stats
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """
        Get monitoring statistics.
        
        Returns:
            Dictionary of monitoring statistics
        """
        return self.monitor.get_statistics()
    
    def get_recent_queries(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent queries.
        
        Args:
            n: Number of recent queries to return
            
        Returns:
            List of recent query logs
        """
        return self.monitor.get_recent_queries(n)
    
    def evaluate(
        self,
        query: str,
        relevant_doc_ids: List[str],
        k: int = 5
    ):
        """
        Evaluate RAG performance for a query.
        
        Args:
            query: Query text
            relevant_doc_ids: List of relevant document IDs (ground truth)
            k: Number of top results to consider
            
        Returns:
            EvaluationResult object
        """
        from flip.evaluation.metrics import RAGMetrics
        import time
        
        # Perform query
        start_time = time.time()
        response = self.query(query, top_k=k)
        query_time = time.time() - start_time
        
        # Extract retrieved IDs
        retrieved_ids = [c["chunk_id"] for c in response.citations]
        
        # Evaluate
        result = RAGMetrics.evaluate_rag(
            query=query,
            answer=response.answer,
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_doc_ids,
            context_chunks=response.context_chunks,
            k=k
        )
        
        # Log to monitor
        self.monitor.log_query(
            query=query,
            answer=response.answer,
            num_results=len(response.citations),
            tokens_used=response.metadata.get("tokens_used", 0),
            retrieval_time=query_time * 0.3,  # Estimate
            generation_time=query_time * 0.7,  # Estimate
            metadata={
                "evaluation": {
                    "precision": result.retrieval_precision,
                    "recall": result.retrieval_recall,
                    "f1": result.retrieval_f1,
                    "overall_score": result.overall_score
                }
            }
        )
        
        return result

