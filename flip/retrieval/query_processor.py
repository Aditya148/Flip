"""Query processing and enhancement."""

from typing import List, Optional, Literal
from flip.generation.base import BaseLLM


class QueryProcessor:
    """Process and enhance queries for better retrieval."""
    
    def __init__(self, llm: Optional[BaseLLM] = None):
        """
        Initialize query processor.
        
        Args:
            llm: Optional LLM for query enhancement
        """
        self.llm = llm
    
    def classify_query(self, query: str) -> Literal["factual", "analytical", "creative", "conversational"]:
        """
        Classify query type.
        
        Args:
            query: Query text
            
        Returns:
            Query type classification
        """
        query_lower = query.lower()
        
        # Simple rule-based classification
        if any(word in query_lower for word in ["what", "who", "when", "where", "which"]):
            return "factual"
        elif any(word in query_lower for word in ["why", "how", "explain", "analyze"]):
            return "analytical"
        elif any(word in query_lower for word in ["create", "generate", "write", "design"]):
            return "creative"
        else:
            return "conversational"
    
    def needs_retrieval(self, query: str) -> bool:
        """
        Determine if query needs document retrieval.
        
        Args:
            query: Query text
            
        Returns:
            True if retrieval is needed
        """
        query_lower = query.lower()
        
        # Queries that likely don't need retrieval
        no_retrieval_patterns = [
            "hello", "hi", "hey", "thanks", "thank you",
            "how are you", "what's your name", "who are you"
        ]
        
        if any(pattern in query_lower for pattern in no_retrieval_patterns):
            return False
        
        return True
    
    def rewrite_query(self, query: str) -> str:
        """
        Rewrite query for better retrieval.
        
        Args:
            query: Original query
            
        Returns:
            Rewritten query
        """
        if not self.llm:
            return query
        
        prompt = f"""Rewrite the following query to be more specific and better suited for document retrieval.
Keep it concise and focused on the key information needed.

Original query: {query}

Rewritten query:"""
        
        try:
            response = self.llm.generate(
                prompt=prompt,
                context=[],
                temperature=0.3,
                max_tokens=100
            )
            rewritten = response.answer.strip()
            return rewritten if rewritten else query
        except:
            return query
    
    def decompose_query(self, query: str) -> List[str]:
        """
        Decompose complex query into sub-queries.
        
        Args:
            query: Complex query
            
        Returns:
            List of sub-queries
        """
        if not self.llm:
            return [query]
        
        prompt = f"""Break down the following complex question into simpler sub-questions.
Each sub-question should be answerable independently.
Return only the sub-questions, one per line.

Complex question: {query}

Sub-questions:"""
        
        try:
            response = self.llm.generate(
                prompt=prompt,
                context=[],
                temperature=0.3,
                max_tokens=200
            )
            
            # Parse sub-questions
            lines = response.answer.strip().split('\n')
            sub_queries = []
            
            for line in lines:
                line = line.strip()
                # Remove numbering and bullet points
                line = line.lstrip('0123456789.-) ')
                if line and len(line) > 10:
                    sub_queries.append(line)
            
            return sub_queries if sub_queries else [query]
        except:
            return [query]
    
    def generate_hypothetical_document(self, query: str) -> str:
        """
        Generate hypothetical document for HyDE (Hypothetical Document Embeddings).
        
        Args:
            query: Query text
            
        Returns:
            Hypothetical document text
        """
        if not self.llm:
            return query
        
        prompt = f"""Write a short, factual paragraph that would answer the following question.
This should read like an excerpt from a relevant document.

Question: {query}

Hypothetical document excerpt:"""
        
        try:
            response = self.llm.generate(
                prompt=prompt,
                context=[],
                temperature=0.5,
                max_tokens=150
            )
            return response.answer.strip()
        except:
            return query
