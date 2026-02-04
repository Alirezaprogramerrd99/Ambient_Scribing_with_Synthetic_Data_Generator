"""
Evaluation Module

Comprehensive benchmarking and evaluation metrics for synthetic data generation.

Metrics Categories:
1. Generation Quality - BLEU, ROUGE, BERTScore
2. Clinical Accuracy - Entity extraction, Hallucination detection
3. Dialogue Quality - Coherence, Completeness, Naturalness
4. RAG Performance - Retrieval precision, Context relevance
5. RAGAS Metrics - Faithfulness, Answer relevancy, Context precision/recall
6. Efficiency - Speed, Cost, Token usage
"""

from .metrics import (
    # Text quality metrics
    compute_bleu,
    compute_rouge,
    compute_bertscore,
    
    # Dialogue metrics
    compute_dialogue_coherence,
    compute_dialogue_completeness,
    
    # Clinical metrics
    compute_clinical_accuracy,
    compute_entity_extraction_f1,
    
    # RAG metrics
    compute_retrieval_precision,
    compute_context_relevance,
    
    # RAGAS metrics
    compute_ragas_metrics,
    RAGASResult,
    
    # Aggregate
    MetricResult,
    TeacherBenchmark,
    BenchmarkResult,
    run_benchmark,
)

from .reporter import (
    BenchmarkReporter,
    ComparisonResult,
    generate_benchmark_report,
)

__all__ = [
    # Metrics
    "compute_bleu",
    "compute_rouge", 
    "compute_bertscore",
    "compute_dialogue_coherence",
    "compute_dialogue_completeness",
    "compute_clinical_accuracy",
    "compute_entity_extraction_f1",
    "compute_retrieval_precision",
    "compute_context_relevance",
    # RAGAS
    "compute_ragas_metrics",
    "RAGASResult",
    # Benchmark
    "MetricResult",
    "TeacherBenchmark",
    "BenchmarkResult",
    "run_benchmark",
    # Reporting
    "BenchmarkReporter",
    "ComparisonResult",
    "generate_benchmark_report",
]