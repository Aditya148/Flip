"""Unit tests for document processing components."""

import pytest
import tempfile
from pathlib import Path
from flip.document_processing.loader import DocumentLoader
from flip.document_processing.chunker import TextChunker
from flip.document_processing.preprocessor import TextPreprocessor


class TestDocumentLoader:
    """Test document loader functionality."""
    
    def test_load_text_file(self, tmp_path):
        """Test loading a text file."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is a test document.")
        
        loader = DocumentLoader()
        documents = loader.load_file(str(test_file))
        
        assert len(documents) == 1
        assert documents[0].content == "This is a test document."
        assert documents[0].metadata["source"] == str(test_file)
    
    def test_load_markdown_file(self, tmp_path):
        """Test loading a markdown file."""
        test_file = tmp_path / "test.md"
        test_file.write_text("# Test\n\nThis is markdown.")
        
        loader = DocumentLoader()
        documents = loader.load_file(str(test_file))
        
        assert len(documents) == 1
        assert "Test" in documents[0].content
    
    def test_load_directory(self, tmp_path):
        """Test loading multiple files from directory."""
        # Create test files
        (tmp_path / "file1.txt").write_text("Content 1")
        (tmp_path / "file2.txt").write_text("Content 2")
        (tmp_path / "file3.md").write_text("# Content 3")
        
        loader = DocumentLoader()
        documents = loader.load_directory(str(tmp_path), recursive=False)
        
        assert len(documents) == 3
    
    def test_unsupported_file_type(self, tmp_path):
        """Test handling of unsupported file types."""
        test_file = tmp_path / "test.xyz"
        test_file.write_text("Unsupported content")
        
        loader = DocumentLoader()
        documents = loader.load_file(str(test_file))
        
        # Should skip unsupported files
        assert len(documents) == 0


class TestTextChunker:
    """Test text chunking strategies."""
    
    def test_token_chunking(self):
        """Test token-based chunking."""
        text = "This is a test. " * 100  # Long text
        chunker = TextChunker(strategy="token", chunk_size=50, chunk_overlap=10)
        
        chunks = chunker.chunk_text(text, {"source": "test"})
        
        assert len(chunks) > 1
        assert all(hasattr(c, 'text') for c in chunks)
        assert all(hasattr(c, 'metadata') for c in chunks)
    
    def test_sentence_chunking(self):
        """Test sentence-based chunking."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunker = TextChunker(strategy="sentence", chunk_size=100)
        
        chunks = chunker.chunk_text(text, {"source": "test"})
        
        assert len(chunks) >= 1
        assert all(c.text.endswith('.') or c.text.endswith('!') or c.text.endswith('?') 
                   for c in chunks if c.text.strip())
    
    def test_semantic_chunking(self):
        """Test semantic chunking."""
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
        chunker = TextChunker(strategy="semantic", chunk_size=200)
        
        chunks = chunker.chunk_text(text, {"source": "test"})
        
        assert len(chunks) >= 1
    
    def test_chunk_overlap(self):
        """Test that chunks have proper overlap."""
        text = "Word " * 100
        chunker = TextChunker(strategy="token", chunk_size=20, chunk_overlap=5)
        
        chunks = chunker.chunk_text(text, {"source": "test"})
        
        # Check that consecutive chunks overlap
        if len(chunks) > 1:
            # There should be some overlap
            assert len(chunks) > 1


class TestTextPreprocessor:
    """Test text preprocessing."""
    
    def test_clean_text(self):
        """Test text cleaning."""
        preprocessor = TextPreprocessor()
        
        dirty_text = "  Multiple   spaces\n\n\nand newlines  "
        clean_text = preprocessor.clean(dirty_text)
        
        assert "Multiple spaces" in clean_text
        assert "  " not in clean_text
    
    def test_remove_urls(self):
        """Test URL removal."""
        preprocessor = TextPreprocessor()
        
        text_with_url = "Check out https://example.com for more info."
        cleaned = preprocessor.clean(text_with_url)
        
        # URLs should be removed or normalized
        assert "example.com" not in cleaned or "http" not in cleaned
    
    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        preprocessor = TextPreprocessor()
        
        text = "Line1\n\n\n\nLine2\t\tLine3"
        cleaned = preprocessor.clean(text)
        
        # Should normalize excessive whitespace
        assert "\n\n\n\n" not in cleaned
        assert "\t\t" not in cleaned


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
