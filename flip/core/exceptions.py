"""Custom exceptions for the Flip SDK."""


class FlipException(Exception):
    """Base exception for all Flip-related errors."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DocumentProcessingError(FlipException):
    """Raised when document processing fails."""
    pass


class EmbeddingError(FlipException):
    """Raised when embedding generation fails."""
    pass


class RetrievalError(FlipException):
    """Raised when retrieval fails."""
    pass


class GenerationError(FlipException):
    """Raised when LLM generation fails."""
    pass


class ConfigurationError(FlipException):
    """Raised when configuration is invalid."""
    pass


class VectorStoreError(FlipException):
    """Raised when vector store operations fail."""
    pass


class UnsupportedFileTypeError(DocumentProcessingError):
    """Raised when file type is not supported."""
    pass


class APIKeyMissingError(ConfigurationError):
    """Raised when required API key is missing."""
    pass
