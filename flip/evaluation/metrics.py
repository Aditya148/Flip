"""RAG evaluation metrics."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from collections import Counter
import re


@dataclass
class EvaluationResult:
    """Result of RAG evaluation."""
    
    # Retrieval metrics
    retrieval_precision: float
    retrieval_recall: float
    retrieval_f1: float
    mrr: float  # Mean Reciprocal Rank
    ndcg: float  # Normalized Discounted Cumulative Gain
    
    # Generation metrics
    answer_relevance: float
    faithfulness: float
    
    # Overall
    overall_score: float
    
    # Details
    metadata: Dict[str, Any]


class RAGMetrics:
    """Metrics for evaluating RAG system performance."""
    
    @staticmethod
    def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        Calculate precision@k.
        
        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: List of relevant document IDs
            k: Number of top results to consider
            
        Returns:
            Precision@k score
        """
        if not retrieved_ids or k == 0:
            return 0.0
        
        top_k = retrieved_ids[:k]
        relevant_set = set(relevant_ids)
        
        relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant_set)
        return relevant_retrieved / k
    
    @staticmethod
    def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        Calculate recall@k.
        
        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: List of relevant document IDs
            k: Number of top results to consider
            
        Returns:
            Recall@k score
        """
        if not relevant_ids:
            return 0.0
        
        top_k = retrieved_ids[:k]
        relevant_set = set(relevant_ids)
        
        relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant_set)
        return relevant_retrieved / len(relevant_ids)
    
    @staticmethod
    def f1_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        Calculate F1@k.
        
        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: List of relevant document IDs
            k: Number of top results to consider
            
        Returns:
            F1@k score
        """
        precision = RAGMetrics.precision_at_k(retrieved_ids, relevant_ids, k)
        recall = RAGMetrics.recall_at_k(retrieved_ids, relevant_ids, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: List of relevant document IDs
            
        Returns:
            MRR score
        """
        relevant_set = set(relevant_ids)
        
        for i, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_set:
                return 1.0 / i
        
        return 0.0
    
    @staticmethod
    def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@k).
        
        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: List of relevant document IDs
            k: Number of top results to consider
            
        Returns:
            NDCG@k score
        """
        if not relevant_ids:
            return 0.0
        
        # Create relevance scores (1 for relevant, 0 for not)
        relevance_scores = [1 if doc_id in relevant_ids else 0 for doc_id in retrieved_ids[:k]]
        
        # Calculate DCG
        dcg = relevance_scores[0] if relevance_scores else 0.0
        for i in range(1, len(relevance_scores)):
            dcg += relevance_scores[i] / np.log2(i + 1)
        
        # Calculate ideal DCG
        ideal_relevance = sorted([1] * min(len(relevant_ids), k), reverse=True)
        idcg = ideal_relevance[0] if ideal_relevance else 0.0
        for i in range(1, len(ideal_relevance)):
            idcg += ideal_relevance[i] / np.log2(i + 1)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def answer_relevance_score(answer: str, query: str) -> float:
        """
        Calculate answer relevance to query (simple keyword overlap).
        
        Args:
            answer: Generated answer
            query: Original query
            
        Returns:
            Relevance score (0-1)
        """
        # Simple keyword-based relevance
        answer_words = set(re.findall(r'\w+', answer.lower()))
        query_words = set(re.findall(r'\w+', query.lower()))
        
        if not query_words:
            return 0.0
        
        overlap = len(answer_words & query_words)
        return overlap / len(query_words)
    
    @staticmethod
    def faithfulness_score(answer: str, context_chunks: List[str]) -> float:
        """
        Calculate faithfulness (how well answer is grounded in context).
        
        Args:
            answer: Generated answer
            context_chunks: Retrieved context chunks
            
        Returns:
            Faithfulness score (0-1)
        """
        if not context_chunks:
            return 0.0
        
        # Simple approach: check what fraction of answer words appear in context
        answer_words = set(re.findall(r'\w+', answer.lower()))
        
        # Combine all context
        context_text = " ".join(context_chunks).lower()
        context_words = set(re.findall(r'\w+', context_text))
        
        if not answer_words:
            return 0.0
        
        # Calculate overlap
        overlap = len(answer_words & context_words)
        return overlap / len(answer_words)
    
    @staticmethod
    def evaluate_retrieval(
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int = 5
    ) -> Dict[str, float]:
        """
        Evaluate retrieval performance.
        
        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: List of relevant document IDs
            k: Number of top results to consider
            
        Returns:
            Dictionary of retrieval metrics
        """
        return {
            "precision@k": RAGMetrics.precision_at_k(retrieved_ids, relevant_ids, k),
            "recall@k": RAGMetrics.recall_at_k(retrieved_ids, relevant_ids, k),
            "f1@k": RAGMetrics.f1_at_k(retrieved_ids, relevant_ids, k),
            "mrr": RAGMetrics.mean_reciprocal_rank(retrieved_ids, relevant_ids),
            "ndcg@k": RAGMetrics.ndcg_at_k(retrieved_ids, relevant_ids, k),
        }
    
    @staticmethod
    def evaluate_generation(
        answer: str,
        query: str,
        context_chunks: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate generation quality.
        
        Args:
            answer: Generated answer
            query: Original query
            context_chunks: Retrieved context chunks
            
        Returns:
            Dictionary of generation metrics
        """
        return {
            "answer_relevance": RAGMetrics.answer_relevance_score(answer, query),
            "faithfulness": RAGMetrics.faithfulness_score(answer, context_chunks),
        }
    
    @staticmethod
    def evaluate_rag(
        query: str,
        answer: str,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        context_chunks: List[str],
        k: int = 5
    ) -> EvaluationResult:
        """
        Comprehensive RAG evaluation.
        
        Args:
            query: Original query
            answer: Generated answer
            retrieved_ids: List of retrieved document IDs
            relevant_ids: List of relevant document IDs (ground truth)
            context_chunks: Retrieved context chunks
            k: Number of top results to consider
            
        Returns:
            EvaluationResult object
        """
        # Retrieval metrics
        retrieval_metrics = RAGMetrics.evaluate_retrieval(retrieved_ids, relevant_ids, k)
        
        # Generation metrics
        generation_metrics = RAGMetrics.evaluate_generation(answer, query, context_chunks)
        
        # Overall score (weighted average)
        overall_score = (
            0.4 * retrieval_metrics["f1@k"] +
            0.3 * generation_metrics["answer_relevance"] +
            0.3 * generation_metrics["faithfulness"]
        )
        
        return EvaluationResult(
            retrieval_precision=retrieval_metrics["precision@k"],
            retrieval_recall=retrieval_metrics["recall@k"],
            retrieval_f1=retrieval_metrics["f1@k"],
            mrr=retrieval_metrics["mrr"],
            ndcg=retrieval_metrics["ndcg@k"],
            answer_relevance=generation_metrics["answer_relevance"],
            faithfulness=generation_metrics["faithfulness"],
            overall_score=overall_score,
            metadata={
                "k": k,
                "num_retrieved": len(retrieved_ids),
                "num_relevant": len(relevant_ids),
                "answer_length": len(answer),
            }
        )
