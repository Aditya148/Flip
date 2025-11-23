"""Document loader for various file formats."""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from tqdm import tqdm

from flip.document_processing.parser import DocumentParser
from flip.core.exceptions import DocumentProcessingError, UnsupportedFileTypeError


@dataclass
class Document:
    """Represents a loaded document."""
    
    content: str
    """Document text content."""
    
    metadata: Dict[str, Any]
    """Document metadata (filename, path, etc.)."""
    
    source: str
    """Source file path."""


class DocumentLoader:
    """Load documents from files and directories."""
    
    SUPPORTED_EXTENSIONS = {
        '.txt', '.md', '.pdf', '.docx', '.doc',
        '.html', '.htm', '.json', '.csv',
        '.py', '.js', '.java', '.cpp', '.c', '.h',
        '.ts', '.tsx', '.jsx', '.go', '.rs', '.rb'
    }
    
    def __init__(self, show_progress: bool = True):
        """
        Initialize document loader.
        
        Args:
            show_progress: Whether to show progress bars
        """
        self.show_progress = show_progress
        self.parser = DocumentParser()
    
    def load_directory(
        self,
        directory: str,
        recursive: bool = True,
        file_extensions: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory: Directory path to load from
            recursive: Whether to search subdirectories
            file_extensions: Optional list of file extensions to filter
            
        Returns:
            List of Document objects
            
        Raises:
            DocumentProcessingError: If directory doesn't exist or can't be read
        """
        dir_path = Path(directory)
        
        if not dir_path.exists():
            raise DocumentProcessingError(f"Directory not found: {directory}")
        
        if not dir_path.is_dir():
            raise DocumentProcessingError(f"Path is not a directory: {directory}")
        
        # Get all files
        files = self._discover_files(dir_path, recursive, file_extensions)
        
        # Load documents
        documents = []
        iterator = tqdm(files, desc="Loading documents") if self.show_progress else files
        
        for file_path in iterator:
            try:
                doc = self.load_file(str(file_path))
                documents.append(doc)
            except Exception as e:
                if self.show_progress:
                    tqdm.write(f"Warning: Failed to load {file_path}: {str(e)}")
                continue
        
        return documents
    
    def load_file(self, file_path: str) -> Document:
        """
        Load a single file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Document object
            
        Raises:
            UnsupportedFileTypeError: If file type is not supported
            DocumentProcessingError: If file can't be read
        """
        path = Path(file_path)
        
        if not path.exists():
            raise DocumentProcessingError(f"File not found: {file_path}")
        
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"Unsupported file type: {path.suffix}. "
                f"Supported types: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
            )
        
        # Parse the file
        content = self.parser.parse_file(str(path))
        
        # Create metadata
        metadata = {
            'filename': path.name,
            'filepath': str(path.absolute()),
            'extension': path.suffix,
            'size_bytes': path.stat().st_size,
            'modified_time': path.stat().st_mtime,
        }
        
        return Document(
            content=content,
            metadata=metadata,
            source=str(path.absolute())
        )
    
    def _discover_files(
        self,
        directory: Path,
        recursive: bool,
        file_extensions: Optional[List[str]]
    ) -> List[Path]:
        """
        Discover all supported files in a directory.
        
        Args:
            directory: Directory to search
            recursive: Whether to search subdirectories
            file_extensions: Optional list of extensions to filter
            
        Returns:
            List of file paths
        """
        extensions = set(file_extensions) if file_extensions else self.SUPPORTED_EXTENSIONS
        
        # Normalize extensions (ensure they start with .)
        extensions = {ext if ext.startswith('.') else f'.{ext}' for ext in extensions}
        
        files = []
        pattern = '**/*' if recursive else '*'
        
        for path in directory.glob(pattern):
            if path.is_file() and path.suffix.lower() in extensions:
                files.append(path)
        
        return sorted(files)
    
    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Get list of supported file extensions."""
        return sorted(cls.SUPPORTED_EXTENSIONS)
