"""RAG monitoring and logging."""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json
from collections import deque


class RAGMonitor:
    """Monitor RAG system performance and usage."""
    
    def __init__(
        self,
        log_file: Optional[Path] = None,
        max_history: int = 1000
    ):
        """
        Initialize RAG monitor.
        
        Args:
            log_file: Path to log file (default: ./flip_data/rag_monitor.jsonl)
            max_history: Maximum number of queries to keep in memory
        """
        self.log_file = log_file or Path("./flip_data/rag_monitor.jsonl")
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.max_history = max_history
        self.query_history = deque(maxlen=max_history)
        
        # Statistics
        self.total_queries = 0
        self.total_tokens = 0
        self.total_retrieval_time = 0.0
        self.total_generation_time = 0.0
    
    def log_query(
        self,
        query: str,
        answer: str,
        num_results: int,
        tokens_used: int,
        retrieval_time: float = 0.0,
        generation_time: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a query.
        
        Args:
            query: Query text
            answer: Generated answer
            num_results: Number of retrieved results
            tokens_used: Tokens used for generation
            retrieval_time: Time spent on retrieval (seconds)
            generation_time: Time spent on generation (seconds)
            metadata: Additional metadata
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer": answer,
            "num_results": num_results,
            "tokens_used": tokens_used,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_time": retrieval_time + generation_time,
            "metadata": metadata or {}
        }
        
        # Add to history
        self.query_history.append(log_entry)
        
        # Update statistics
        self.total_queries += 1
        self.total_tokens += tokens_used
        self.total_retrieval_time += retrieval_time
        self.total_generation_time += generation_time
        
        # Write to file
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except:
            pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get monitoring statistics.
        
        Returns:
            Dictionary of statistics
        """
        avg_tokens = self.total_tokens / max(self.total_queries, 1)
        avg_retrieval_time = self.total_retrieval_time / max(self.total_queries, 1)
        avg_generation_time = self.total_generation_time / max(self.total_queries, 1)
        avg_total_time = avg_retrieval_time + avg_generation_time
        
        return {
            "total_queries": self.total_queries,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_query": avg_tokens,
            "avg_retrieval_time": avg_retrieval_time,
            "avg_generation_time": avg_generation_time,
            "avg_total_time": avg_total_time,
            "total_retrieval_time": self.total_retrieval_time,
            "total_generation_time": self.total_generation_time,
        }
    
    def get_recent_queries(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent queries.
        
        Args:
            n: Number of recent queries to return
            
        Returns:
            List of recent query logs
        """
        return list(self.query_history)[-n:]
    
    def get_slow_queries(self, threshold: float = 5.0, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get slow queries.
        
        Args:
            threshold: Time threshold in seconds
            n: Maximum number of queries to return
            
        Returns:
            List of slow queries
        """
        slow_queries = [
            q for q in self.query_history
            if q.get("total_time", 0) > threshold
        ]
        
        # Sort by total time (slowest first)
        slow_queries.sort(key=lambda x: x.get("total_time", 0), reverse=True)
        
        return slow_queries[:n]
    
    def get_token_usage_stats(self) -> Dict[str, Any]:
        """
        Get token usage statistics.
        
        Returns:
            Token usage statistics
        """
        if not self.query_history:
            return {
                "min_tokens": 0,
                "max_tokens": 0,
                "avg_tokens": 0,
                "total_tokens": 0
            }
        
        tokens_list = [q.get("tokens_used", 0) for q in self.query_history]
        
        return {
            "min_tokens": min(tokens_list),
            "max_tokens": max(tokens_list),
            "avg_tokens": sum(tokens_list) / len(tokens_list),
            "total_tokens": self.total_tokens
        }
    
    def clear_history(self):
        """Clear query history."""
        self.query_history.clear()
    
    def export_logs(self, output_file: Path):
        """
        Export logs to a file.
        
        Args:
            output_file: Path to output file
        """
        try:
            with open(output_file, 'w') as f:
                for entry in self.query_history:
                    f.write(json.dumps(entry) + '\n')
        except Exception as e:
            raise Exception(f"Failed to export logs: {str(e)}")
