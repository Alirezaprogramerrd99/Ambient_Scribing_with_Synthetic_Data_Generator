"""
RAG Quality Metrics for Synthetic Data Validation

Evaluates the quality of RAG retrieval and response generation.
Supports RAGAS metrics and custom evaluation methods.

Author: Alireza Rashidi
MSc Project: Trustworthy SLMs for Ambient Clinical Scribing
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from src.models import (
    SyntheticSample,
    RAGMetadata,
    ValidationError,
    ErrorSeverity,
)
from src.knowledge_base import RetrievalResponse, RetrievalResult

logger = logging.getLogger(__name__)


# =============================================================================
# RAG Metric Types
# =============================================================================

class RAGMetricType(str, Enum):
    """Types of RAG quality metrics"""
    
    FAITHFULNESS = "faithfulness"         # Is answer faithful to retrieved context?
    ANSWER_RELEVANCY = "answer_relevancy" # Is answer relevant to question?
    CONTEXT_PRECISION = "context_precision"  # Are retrieved docs relevant?
    CONTEXT_RECALL = "context_recall"     # Are all relevant docs retrieved?
    CONTEXT_UTILIZATION = "context_utilization"  # How much context is used?


@dataclass
class RAGMetricResult:
    """Result of a single RAG metric evaluation"""
    
    metric_type: RAGMetricType
    score: float  # 0.0 to 1.0
    details: str = ""
    
    @property
    def is_good(self) -> bool:
        """Check if score is above threshold"""
        thresholds = {
            RAGMetricType.FAITHFULNESS: 0.7,
            RAGMetricType.ANSWER_RELEVANCY: 0.6,
            RAGMetricType.CONTEXT_PRECISION: 0.5,
            RAGMetricType.CONTEXT_RECALL: 0.5,
            RAGMetricType.CONTEXT_UTILIZATION: 0.3,
        }
        return self.score >= thresholds.get(self.metric_type, 0.5)


@dataclass
class RAGEvaluationResult:
    """Complete RAG evaluation result"""
    
    metrics: Dict[RAGMetricType, RAGMetricResult] = field(default_factory=dict)
    overall_score: float = 0.0
    is_acceptable: bool = True
    warnings: List[str] = field(default_factory=list)
    
    def add_metric(self, result: RAGMetricResult):
        """Add a metric result"""
        self.metrics[result.metric_type] = result
        self._update_overall_score()
    
    def _update_overall_score(self):
        """Update overall score as weighted average"""
        if not self.metrics:
            self.overall_score = 0.0
            return
        
        # Weights for different metrics
        weights = {
            RAGMetricType.FAITHFULNESS: 0.35,
            RAGMetricType.ANSWER_RELEVANCY: 0.25,
            RAGMetricType.CONTEXT_PRECISION: 0.20,
            RAGMetricType.CONTEXT_RECALL: 0.10,
            RAGMetricType.CONTEXT_UTILIZATION: 0.10,
        }
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for metric_type, result in self.metrics.items():
            weight = weights.get(metric_type, 0.1)
            weighted_sum += result.score * weight
            total_weight += weight
        
        self.overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        self.is_acceptable = self.overall_score >= 0.5
    
    def get_metric(self, metric_type: RAGMetricType) -> Optional[RAGMetricResult]:
        """Get a specific metric result"""
        return self.metrics.get(metric_type)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "overall_score": self.overall_score,
            "is_acceptable": self.is_acceptable,
            "metrics": {
                k.value: {"score": v.score, "details": v.details}
                for k, v in self.metrics.items()
            },
            "warnings": self.warnings,
        }


# =============================================================================
# Simple RAG Evaluator (Rule-Based)
# =============================================================================

class SimpleRAGEvaluator:
    """
    Rule-based RAG quality evaluator
    
    Performs basic RAG quality checks without requiring external LLMs.
    Good for quick validation and when RAGAS is not available.
    """
    
    def __init__(self):
        pass
    
    def evaluate(
        self,
        sample: SyntheticSample,
        retrieval_response: Optional[RetrievalResponse] = None,
    ) -> RAGEvaluationResult:
        """
        Evaluate RAG quality for a sample
        
        Args:
            sample: SyntheticSample with RAG metadata
            retrieval_response: Optional full retrieval response
            
        Returns:
            RAGEvaluationResult
        """
        result = RAGEvaluationResult()
        
        # Check if RAG was used
        if not sample.rag.rag_enabled:
            result.warnings.append("RAG was not enabled for this sample")
            return result
        
        # Evaluate context utilization
        context_util = self._evaluate_context_utilization(sample)
        result.add_metric(context_util)
        
        # Evaluate answer relevancy (based on scenario and summary)
        answer_relevancy = self._evaluate_answer_relevancy(sample)
        result.add_metric(answer_relevancy)
        
        # Evaluate faithfulness (basic check)
        faithfulness = self._evaluate_faithfulness(sample)
        result.add_metric(faithfulness)
        
        # Evaluate context precision if we have retrieval scores
        if sample.rag.retrieval_scores:
            context_precision = self._evaluate_context_precision(sample)
            result.add_metric(context_precision)
        
        return result
    
    def _evaluate_context_utilization(
        self,
        sample: SyntheticSample,
    ) -> RAGMetricResult:
        """Evaluate how much of the retrieved context was meaningfully used"""
        
        rag_meta = sample.rag
        
        if not rag_meta.context_used:
            return RAGMetricResult(
                metric_type=RAGMetricType.CONTEXT_UTILIZATION,
                score=0.0,
                details="No context was recorded",
            )
        
        # Get context and comprehensive summary text
        context = rag_meta.context_used.lower()
        summary_text = " ".join(filter(None, [
            sample.summary.chief_complaint,
            sample.summary.history_of_present_illness,
            sample.summary.assessment,
            sample.summary.plan,
            sample.summary.safety_netting,
        ])).lower()
        
        # Stop words to filter
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 'and', 
            'in', 'for', 'with', 'patient', 'should', 'may', 'can', 'will',
            'have', 'has', 'been', 'be', 'this', 'that', 'their', 'from',
            'also', 'other', 'such', 'more', 'most', 'some', 'any', 'all'
        }
        
        # Extract meaningful clinical terms from context (5+ chars, not stop words)
        context_words = {w for w in context.split() if len(w) >= 5 and w not in stop_words}
        
        # Count how many context terms appear in summary
        used_words = [word for word in context_words if word in summary_text]
        
        # Calculate utilization
        if context_words:
            utilization = len(used_words) / len(context_words)
        else:
            utilization = 0.0
        
        # Scale appropriately (perfect match unlikely, 30%+ is good)
        scaled_score = min(1.0, utilization * 2.5)
        
        return RAGMetricResult(
            metric_type=RAGMetricType.CONTEXT_UTILIZATION,
            score=scaled_score,
            details=f"Used {len(used_words)}/{len(context_words)} clinical terms from context",
        )
    
    def _evaluate_answer_relevancy(
        self,
        sample: SyntheticSample,
    ) -> RAGMetricResult:
        """Evaluate if the answer comprehensively addresses the scenario"""
        
        scenario = sample.scenario.scenario_text.lower()
        
        # Get comprehensive summary including plan
        summary_text = " ".join(filter(None, [
            sample.summary.chief_complaint,
            sample.summary.history_of_present_illness,
            sample.summary.assessment,
            sample.summary.plan,
        ])).lower()
        
        # Stop words to filter
        stop_words = {
            'the', 'a', 'an', 'with', 'for', 'of', 'and', 'or', 'in', 'on', 
            'to', 'is', 'has', 'been', 'year', 'old', 'male', 'female',
            'patient', 'presenting', 'history', 'associated', 'reports'
        }
        
        # Extract key clinical terms from scenario
        scenario_words = {w for w in scenario.split() if w not in stop_words and len(w) >= 4}
        
        # Check how many scenario terms appear in summary
        relevant_count = sum(1 for word in scenario_words if word in summary_text)
        
        # Calculate base relevancy
        if scenario_words:
            relevancy = relevant_count / len(scenario_words)
        else:
            relevancy = 0.0
        
        # Bonus: Check if main complaint is addressed
        bonus = 0.0
        main_symptoms = ['pain', 'cough', 'fever', 'breath', 'dizz', 'nausea', 'head', 'chest', 'abdom']
        for symptom in main_symptoms:
            if symptom in scenario and symptom in summary_text:
                bonus += 0.1
        
        final_score = min(1.0, relevancy * 1.3 + bonus)
        
        return RAGMetricResult(
            metric_type=RAGMetricType.ANSWER_RELEVANCY,
            score=final_score,
            details=f"Answer addresses {relevant_count}/{len(scenario_words)} scenario terms",
        )
    
    def _evaluate_faithfulness(
        self,
        sample: SyntheticSample,
    ) -> RAGMetricResult:
        """
        Evaluate faithfulness of generated content to retrieved context
        
        Checks multiple aspects:
        1. Clinical terms overlap
        2. Medication alignment
        3. Investigation alignment
        4. Red flag/safety terms
        """
        
        rag_meta = sample.rag
        
        if not rag_meta.context_used:
            return RAGMetricResult(
                metric_type=RAGMetricType.FAITHFULNESS,
                score=0.5,  # Neutral score when no context
                details="Cannot evaluate faithfulness without context",
            )
        
        import re
        
        context = rag_meta.context_used.lower()
        
        # Get all relevant summary text
        summary_parts = [
            sample.summary.assessment or "",
            sample.summary.plan or "",
            sample.summary.safety_netting or "",
            sample.summary.history_of_present_illness or "",
        ]
        summary_text = " ".join(summary_parts).lower()
        
        if not summary_text.strip():
            return RAGMetricResult(
                metric_type=RAGMetricType.FAITHFULNESS,
                score=0.5,
                details="No summary text to evaluate",
            )
        
        scores = []
        details = []
        
        # 1. Check medication overlap
        med_pattern = r'\b\w+(?:mab|nib|zole|pril|sartan|statin|mycin|cillin|pam|lol|pine|ine|ide|ate|one)\b'
        summary_meds = set(re.findall(med_pattern, summary_text))
        context_meds = set(re.findall(med_pattern, context))
        
        if summary_meds:
            med_score = len(summary_meds & context_meds) / len(summary_meds)
            scores.append(med_score)
            details.append(f"Meds: {len(summary_meds & context_meds)}/{len(summary_meds)}")
        
        # 2. Check investigation/test overlap
        test_pattern = r'\b(?:x-ray|xray|ct|mri|ultrasound|ecg|ekg|blood test|fbc|cbc|crp|esr|lfts|ufes|troponin|d-dimer|bnp|hba1c|glucose|urinalysis|culture|biopsy|endoscopy|colonoscopy|spirometry|peak flow)\b'
        summary_tests = set(re.findall(test_pattern, summary_text))
        context_tests = set(re.findall(test_pattern, context))
        
        if summary_tests:
            test_score = len(summary_tests & context_tests) / len(summary_tests)
            scores.append(test_score)
            details.append(f"Tests: {len(summary_tests & context_tests)}/{len(summary_tests)}")
        
        # 3. Check red flag terms overlap
        red_flag_terms = [
            'red flag', 'warning', 'emergency', 'urgent', 'immediately', 
            'severe', 'sudden', 'worst', 'chest pain', 'shortness of breath',
            'weakness', 'numbness', 'vision', 'confusion', 'fever'
        ]
        summary_flags = sum(1 for term in red_flag_terms if term in summary_text)
        context_flags = sum(1 for term in red_flag_terms if term in context)
        
        if summary_flags > 0 and context_flags > 0:
            flag_score = min(1.0, context_flags / max(summary_flags, 1))
            scores.append(flag_score)
            details.append(f"RedFlags: {min(summary_flags, context_flags)}/{summary_flags}")
        
        # 4. Check clinical term overlap (broader)
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 'and', 
            'in', 'for', 'with', 'patient', 'doctor', 'will', 'should', 'may',
            'can', 'have', 'has', 'been', 'be', 'or', 'if', 'no', 'not', 'any'
        }
        
        # Extract meaningful clinical words (4+ chars, not stop words)
        context_words = {w for w in context.split() if len(w) >= 4 and w not in stop_words}
        summary_words = {w for w in summary_text.split() if len(w) >= 4 and w not in stop_words}
        
        if summary_words and context_words:
            # What proportion of summary clinical terms appear in context?
            term_overlap = len(summary_words & context_words) / len(summary_words)
            scores.append(term_overlap * 0.5 + 0.5)  # Scale to 0.5-1.0 range
            details.append(f"Terms: {len(summary_words & context_words)}/{len(summary_words)}")
        
        # Calculate final score
        if scores:
            final_score = sum(scores) / len(scores)
        else:
            final_score = 0.5  # Default neutral
        
        return RAGMetricResult(
            metric_type=RAGMetricType.FAITHFULNESS,
            score=final_score,
            details="; ".join(details) if details else "Basic evaluation",
        )
    
    def _evaluate_context_precision(
        self,
        sample: SyntheticSample,
    ) -> RAGMetricResult:
        """Evaluate precision of retrieved contexts"""
        
        scores = sample.rag.retrieval_scores
        
        if not scores:
            return RAGMetricResult(
                metric_type=RAGMetricType.CONTEXT_PRECISION,
                score=0.5,
                details="No retrieval scores available",
            )
        
        # Average retrieval score as proxy for precision
        avg_score = sum(scores) / len(scores)
        
        # Count high-quality retrievals (score > 0.7)
        high_quality = sum(1 for s in scores if s > 0.7)
        
        return RAGMetricResult(
            metric_type=RAGMetricType.CONTEXT_PRECISION,
            score=avg_score,
            details=f"Avg retrieval score: {avg_score:.3f}, High quality: {high_quality}/{len(scores)}",
        )


# =============================================================================
# RAGAS Evaluator (Advanced)
# =============================================================================

class RAGASEvaluator:
    """
    RAG evaluation using RAGAS framework
    
    Provides comprehensive RAG quality metrics using LLM-based evaluation.
    
    Note: Requires RAGAS installation: pip install ragas
    """
    
    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
    ):
        """
        Initialize RAGAS evaluator
        
        Args:
            llm_model: LLM for evaluation (OpenAI model name)
            embedding_model: Embedding model for semantic similarity
        """
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        
        self._ragas = None
        self._metrics = None
        
        self._initialize_ragas()
    
    def _initialize_ragas(self):
        """Initialize RAGAS framework"""
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )
            
            self._ragas = evaluate
            self._metrics = {
                RAGMetricType.FAITHFULNESS: faithfulness,
                RAGMetricType.ANSWER_RELEVANCY: answer_relevancy,
                RAGMetricType.CONTEXT_PRECISION: context_precision,
                RAGMetricType.CONTEXT_RECALL: context_recall,
            }
            
            logger.info("RAGAS framework initialized")
            
        except ImportError:
            logger.warning(
                "RAGAS not installed. Install with: pip install ragas. "
                "Falling back to simple evaluation."
            )
            self._ragas = None
    
    def evaluate(
        self,
        sample: SyntheticSample,
        retrieval_response: Optional[RetrievalResponse] = None,
    ) -> RAGEvaluationResult:
        """
        Evaluate RAG quality using RAGAS
        
        Args:
            sample: SyntheticSample to evaluate
            retrieval_response: Optional retrieval response with contexts
            
        Returns:
            RAGEvaluationResult
        """
        # Fall back to simple evaluator if RAGAS not available
        if not self._ragas:
            simple_eval = SimpleRAGEvaluator()
            return simple_eval.evaluate(sample, retrieval_response)
        
        try:
            return self._evaluate_with_ragas(sample, retrieval_response)
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            simple_eval = SimpleRAGEvaluator()
            return simple_eval.evaluate(sample, retrieval_response)
    
    def _evaluate_with_ragas(
        self,
        sample: SyntheticSample,
        retrieval_response: Optional[RetrievalResponse],
    ) -> RAGEvaluationResult:
        """Perform RAGAS evaluation"""
        from datasets import Dataset
        
        # Prepare data for RAGAS
        question = sample.scenario.scenario_text
        
        # Get answer from summary
        answer = " ".join(filter(None, [
            sample.summary.assessment,
            sample.summary.plan,
        ]))
        
        # Get contexts
        if retrieval_response:
            contexts = [r.text for r in retrieval_response.results]
        elif sample.rag.context_used:
            contexts = [sample.rag.context_used]
        else:
            contexts = []
        
        # Create dataset
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
        dataset = Dataset.from_dict(data)
        
        # Run RAGAS evaluation
        ragas_result = self._ragas(
            dataset,
            metrics=list(self._metrics.values()),
        )
        
        # Convert to our format
        result = RAGEvaluationResult()
        
        for metric_type, metric in self._metrics.items():
            metric_name = metric.name
            if metric_name in ragas_result:
                score = ragas_result[metric_name]
                result.add_metric(RAGMetricResult(
                    metric_type=metric_type,
                    score=score,
                    details=f"RAGAS {metric_name}",
                ))
        
        return result
    
    def evaluate_batch(
        self,
        samples: List[SyntheticSample],
    ) -> Tuple[RAGEvaluationResult, List[RAGEvaluationResult]]:
        """
        Evaluate multiple samples
        
        Args:
            samples: List of samples to evaluate
            
        Returns:
            Tuple of (aggregate result, individual results)
        """
        individual_results = []
        
        for sample in samples:
            result = self.evaluate(sample)
            individual_results.append(result)
        
        # Compute aggregate
        aggregate = RAGEvaluationResult()
        
        for metric_type in RAGMetricType:
            scores = [
                r.metrics[metric_type].score
                for r in individual_results
                if metric_type in r.metrics
            ]
            
            if scores:
                avg_score = sum(scores) / len(scores)
                aggregate.add_metric(RAGMetricResult(
                    metric_type=metric_type,
                    score=avg_score,
                    details=f"Average over {len(scores)} samples",
                ))
        
        return aggregate, individual_results


# =============================================================================
# Factory Function
# =============================================================================

def create_rag_evaluator(
    use_ragas: bool = False,
    **kwargs,
) -> SimpleRAGEvaluator:
    """
    Create RAG evaluator
    
    Args:
        use_ragas: Whether to use RAGAS framework
        **kwargs: Additional arguments for RAGAS evaluator
        
    Returns:
        RAG evaluator instance
    """
    if use_ragas:
        return RAGASEvaluator(**kwargs)
    return SimpleRAGEvaluator()


# =============================================================================
# Utility Functions
# =============================================================================

def quick_rag_score(sample: SyntheticSample) -> float:
    """
    Quick RAG quality score
    
    Returns a single score (0-1) for RAG quality.
    """
    if not sample.rag.rag_enabled:
        return 0.0
    
    evaluator = SimpleRAGEvaluator()
    result = evaluator.evaluate(sample)
    return result.overall_score


def evaluate_retrieval_quality(
    retrieval_response: RetrievalResponse,
    expected_keywords: List[str],
) -> Dict[str, float]:
    """
    Evaluate retrieval quality against expected keywords
    
    Args:
        retrieval_response: Retrieval response to evaluate
        expected_keywords: Keywords that should appear in results
        
    Returns:
        Dictionary with precision, recall, and coverage metrics
    """
    # Combine all retrieved text
    retrieved_text = " ".join(r.text.lower() for r in retrieval_response.results)
    
    # Check keyword coverage
    found_keywords = [kw for kw in expected_keywords if kw.lower() in retrieved_text]
    
    coverage = len(found_keywords) / len(expected_keywords) if expected_keywords else 0.0
    
    return {
        "coverage": coverage,
        "keywords_found": len(found_keywords),
        "keywords_total": len(expected_keywords),
        "avg_retrieval_score": retrieval_response.average_score,
        "num_results": retrieval_response.num_results,
    }


if __name__ == "__main__":
    print("RAG Metrics Module")
    print("=" * 60)
    
    print("\nAvailable metrics:")
    for metric in RAGMetricType:
        print(f"  - {metric.value}")
    
    print("\nSimple RAG Evaluator: Rule-based, no external dependencies")
    print("RAGAS Evaluator: LLM-based, requires 'pip install ragas'")
    
    print("\n✓ RAG metrics module ready!")