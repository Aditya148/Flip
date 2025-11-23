"""Pipeline orchestrator for managing the RAG workflow."""

from typing import List, Optional, Dict, Any
from pathlib import Path

from flip.document_processing.loader import DocumentLoader, Document
from flip.document_processing.chunker import TextChunker, TextChunk
from flip.document_processing.preprocessor import TextPreprocessor
from flip.embedding.base import BaseEmbedder
from flip.vector_store.base import BaseVectorStore
from flip.retrieval.hybrid_search import HybridSearchRetriever
from flip.retrieval.reranker import Reranker
from flip.retrieval.query_processor import QueryProcessor
from flip.generation.base import BaseLLM
from flip.pipeline.cache import EmbeddingCache, QueryCache


class PipelineOrchestrator:
    """Orchestrate the end-to-end RAG pipeline."""
    
    def __init__(
        self,
        loader: DocumentLoader,
        chunker: TextChunker,
        preprocessor: TextPreprocessor,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        llm: BaseLLM,
        use_hybrid_search: bool = True,
        use_reranking: bool = True,
        use_query_enhancement: bool = True,
        enable_cache: bool = True,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize pipeline orchestrator.
        
        Args:
            loader: Document loader
            chunker: Text chunker
            preprocessor: Text preprocessor
            embedder: Embedder
            vector_store: Vector store
            llm: LLM
            use_hybrid_search: Whether to use hybrid search
            use_reranking: Whether to use reranking
            use_query_enhancement: Whether to enhance queries
            enable_cache: Whether to enable caching
            cache_dir: Cache directory
        """
        self.loader = loader
        self.chunker = chunker
        self.preprocessor = preprocessor
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm
        
        # Advanced features
        self.use_hybrid_search = use_hybrid_search
        self.use_reranking = use_reranking
        self.use_query_enhancement = use_query_enhancement
        
        # Initialize retriever
        if use_hybrid_search:
            self.retriever = HybridSearchRetriever(
                vector_store=vector_store,
                embedder=embedder
            )
        else:
            from flip.retrieval.retriever import Retriever
            self.retriever = Retriever(
                vector_store=vector_store,
                embedder=embedder
            )
        
        # Initialize reranker
        self.reranker = Reranker() if use_reranking else None
        
        # Initialize query processor
        self.query_processor = QueryProcessor(llm=llm) if use_query_enhancement else None
        
        # Initialize caches
        self.enable_cache = enable_cache
        if enable_cache:
            self.embedding_cache = EmbeddingCache(cache_dir=cache_dir)
            self.query_cache = QueryCache(cache_dir=cache_dir)
        else:
            self.embedding_cache = None
            self.query_cache = None
    
    def process_documents(self, documents: List[Document]) -> tuple[List[str], List[List[float]], List[str], List[Dict]]:
        """
        Process documents through the pipeline.
        
        Args:
            documents: List of documents
            
        Returns:
            Tuple of (ids, embeddings, texts, metadatas)
        """
        all_ids = []
        all_texts = []
        all_metadatas = []
        
        # Process each document
        for doc in documents:
            # Preprocess
            cleaned_text = self.preprocessor.clean_text(
                doc.content,
                preserve_formatting=True
            )
            
            # Chunk
            chunks = self.chunker.chunk_text(cleaned_text, doc.metadata)
            
            for chunk in chunks:
                all_ids.append(chunk.chunk_id)
                all_texts.append(chunk.text)
                all_metadatas.append(chunk.metadata)
        
        # Generate embeddings with caching
        all_embeddings = self._embed_with_cache(all_texts)
        
        return all_ids, all_embeddings, all_texts, all_metadatas
    
    def _embed_with_cache(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings with caching."""
        if not self.enable_cache or not self.embedding_cache:
            return self.embedder.embed_batch(texts)
        
        # Check cache
        cached = self.embedding_cache.get_batch(texts)
        
        # Find texts that need embedding
        texts_to_embed = [t for t in texts if t not in cached]
        
        if texts_to_embed:
            # Generate embeddings for uncached texts
            new_embeddings = self.embedder.embed_batch(texts_to_embed)
            
            # Cache new embeddings
            self.embedding_cache.put_batch(texts_to_embed, new_embeddings)
            
            # Combine cached and new
            text_to_embedding = {**cached}
            for text, emb in zip(texts_to_embed, new_embeddings):
                text_to_embedding[text] = emb
        else:
            text_to_embedding = cached
        
        # Return in original order
        return [text_to_embedding[t] for t in texts]
    
    def retrieve_and_generate(
        self,
        query: str,
        top_k: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        """
        Retrieve relevant documents and generate response.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            temperature: LLM temperature
            max_tokens: Max tokens to generate
            
        Returns:
            Dictionary with answer, citations, and metadata
        """
        # Check query cache
        if self.enable_cache and self.query_cache:
            cached_result = self.query_cache.get(query, top_k=top_k)
            if cached_result:
                return cached_result
        
        # Query enhancement
        processed_query = query
        if self.use_query_enhancement and self.query_processor:
            # Check if retrieval is needed
            if not self.query_processor.needs_retrieval(query):
                # Direct response without retrieval
                response = self.llm.generate(
                    prompt=query,
                    context=[],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return {
                    "answer": response.answer,
                    "citations": [],
                    "context_chunks": [],
                    "metadata": {"retrieval_used": False}
                }
            
            # Optionally rewrite query
            # processed_query = self.query_processor.rewrite_query(query)
        
        # Retrieve
        results = self.retriever.retrieve(processed_query, top_k=top_k * 2)
        
        # Rerank
        if self.use_reranking and self.reranker and results:
            results = self.reranker.rerank(query, results, top_k=top_k)
        else:
            results = results[:top_k]
        
        # Extract context
        context_chunks = [r.text for r in results]
        
        # Generate
        response = self.llm.generate(
            prompt=query,
            context=context_chunks,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Prepare result
        result = {
            "answer": response.answer,
            "citations": [
                {
                    "source": r.metadata.get("filepath", "unknown"),
                    "chunk_id": r.id,
                    "score": r.score,
                    "text_preview": r.text[:200] + "..." if len(r.text) > 200 else r.text
                }
                for r in results
            ],
            "context_chunks": context_chunks,
            "metadata": {
                "tokens_used": response.tokens_used,
                "model": response.model,
                "retrieval_used": True,
                "reranking_used": self.use_reranking,
                "hybrid_search_used": self.use_hybrid_search
            }
        }
        
        # Cache result
        if self.enable_cache and self.query_cache:
            self.query_cache.put(query, result, top_k=top_k)
        
        return result
    
    def save_caches(self):
        """Save caches to disk."""
        if self.enable_cache:
            if self.embedding_cache:
                self.embedding_cache.save()
            if self.query_cache:
                self.query_cache.save()
    
    def clear_caches(self):
        """Clear all caches."""
        if self.enable_cache:
            if self.embedding_cache:
                self.embedding_cache.clear()
            if self.query_cache:
                self.query_cache.clear()
