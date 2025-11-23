"""Text preprocessing utilities."""

import re
from typing import Optional


class TextPreprocessor:
    """Preprocess text before chunking and embedding."""
    
    @staticmethod
    def clean_text(text: str, preserve_formatting: bool = False) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Text to clean
            preserve_formatting: Whether to preserve code blocks and lists
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but preserve punctuation
        if not preserve_formatting:
            text = re.sub(r'[^\w\s.,!?;:()\[\]{}\-\'\"]+', '', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    @staticmethod
    def extract_metadata_from_text(text: str) -> dict:
        """
        Extract metadata from text (e.g., title, headers).
        
        Args:
            text: Text to extract metadata from
            
        Returns:
            Dictionary of metadata
        """
        metadata = {}
        
        # Try to extract title (first line if it looks like a title)
        lines = text.split('\n')
        if lines:
            first_line = lines[0].strip()
            if len(first_line) < 100 and not first_line.endswith('.'):
                metadata['title'] = first_line
        
        # Extract headers (markdown style)
        headers = re.findall(r'^#+\s+(.+)$', text, re.MULTILINE)
        if headers:
            metadata['headers'] = headers[:5]  # First 5 headers
        
        return metadata
    
    @staticmethod
    def remove_urls(text: str) -> str:
        """Remove URLs from text."""
        return re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    @staticmethod
    def remove_emails(text: str) -> str:
        """Remove email addresses from text."""
        return re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
