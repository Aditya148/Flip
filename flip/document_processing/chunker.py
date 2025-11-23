"""Text chunking strategies."""

from typing import List, Dict, Any, Literal
from dataclasses import dataclass
import tiktoken
import re

from flip.core.exceptions import DocumentProcessingError


@dataclass
class TextChunk:
    """Represents a chunk of text."""
    
    text: str
    """Chunk text content."""
    
    metadata: Dict[str, Any]
    """Chunk metadata."""
    
    chunk_id: str
    """Unique chunk identifier."""


class TextChunker:
    """Chunk text using various strategies."""
    
    def __init__(
        self,
        strategy: Literal["token", "sentence", "semantic", "recursive"] = "semantic",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        encoding_name: str = "cl100k_base"
    ):
        """
        Initialize text chunker.
        
        Args:
            strategy: Chunking strategy to use
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            encoding_name: Tokenizer encoding name
        """
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except:
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def chunk_text(
        self,
        text: str,
        metadata: Dict[str, Any] = None
    ) -> List[TextChunk]:
        """
        Chunk text using the configured strategy.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of TextChunk objects
        """
        if not text or not text.strip():
            return []
        
        metadata = metadata or {}
        
        if self.strategy == "token":
            chunks = self._chunk_by_tokens(text)
        elif self.strategy == "sentence":
            chunks = self._chunk_by_sentences(text)
        elif self.strategy == "semantic":
            chunks = self._chunk_semantic(text)
        elif self.strategy == "recursive":
            chunks = self._chunk_recursive(text)
        else:
            raise DocumentProcessingError(f"Unknown chunking strategy: {self.strategy}")
        
        # Create TextChunk objects with metadata
        text_chunks = []
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = {
                **metadata,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'strategy': self.strategy,
            }
            
            text_chunks.append(TextChunk(
                text=chunk_text,
                metadata=chunk_metadata,
                chunk_id=f"{metadata.get('source', 'unknown')}_{i}"
            ))
        
        return text_chunks
    
    def _chunk_by_tokens(self, text: str) -> List[str]:
        """Chunk text by token count with overlap."""
        tokens = self.encoding.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start >= len(tokens):
                break
        
        return chunks
    
    def _chunk_by_sentences(self, text: str) -> List[str]:
        """Chunk text by sentences, respecting token limits."""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.encoding.encode(sentence))
            
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_sentences = []
                overlap_tokens = 0
                for s in reversed(current_chunk):
                    s_tokens = len(self.encoding.encode(s))
                    if overlap_tokens + s_tokens <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_tokens += s_tokens
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_tokens = overlap_tokens
            
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _chunk_semantic(self, text: str) -> List[str]:
        """
        Chunk text semantically by paragraphs and sections.
        This is a simplified version - production would use LLM for better results.
        """
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\n+', text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_tokens = len(self.encoding.encode(para))
            
            # If single paragraph exceeds chunk size, split it
            if para_tokens > self.chunk_size:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                
                # Split large paragraph by sentences
                sub_chunks = self._chunk_by_sentences(para)
                chunks.extend(sub_chunks)
                continue
            
            if current_tokens + para_tokens > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append('\n\n'.join(current_chunk))
                
                # Start new chunk with last paragraph as overlap if it fits
                if para_tokens <= self.chunk_overlap:
                    current_chunk = [current_chunk[-1]] if current_chunk else []
                    current_tokens = len(self.encoding.encode(current_chunk[0])) if current_chunk else 0
                else:
                    current_chunk = []
                    current_tokens = 0
            
            current_chunk.append(para)
            current_tokens += para_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def _chunk_recursive(self, text: str) -> List[str]:
        """
        Recursively chunk text using multiple separators.
        Tries to split on paragraphs, then sentences, then tokens.
        """
        separators = ['\n\n', '\n', '. ', ' ']
        return self._recursive_split(text, separators)
    
    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using separators."""
        if not separators:
            # Base case: split by tokens
            return self._chunk_by_tokens(text)
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # Split by current separator
        splits = text.split(separator)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for split in splits:
            if not split.strip():
                continue
            
            split_tokens = len(self.encoding.encode(split))
            
            # If split is too large, recursively split it
            if split_tokens > self.chunk_size:
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                
                sub_chunks = self._recursive_split(split, remaining_separators)
                chunks.extend(sub_chunks)
                continue
            
            if current_tokens + split_tokens > self.chunk_size and current_chunk:
                chunks.append(separator.join(current_chunk))
                current_chunk = []
                current_tokens = 0
            
            current_chunk.append(split)
            current_tokens += split_tokens
        
        if current_chunk:
            chunks.append(separator.join(current_chunk))
        
        return chunks
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
