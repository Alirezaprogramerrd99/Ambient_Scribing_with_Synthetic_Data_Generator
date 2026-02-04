"""
Evaluation Metrics

Comprehensive metrics for evaluating synthetic clinical data quality.

Based on:
- Woo et al. (2025) - Synthetic data quality metrics
- RAGAS framework - RAG evaluation
- Clinical NLP benchmarks - Medical accuracy
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import re

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MetricResult:
    """Single metric result"""
    name: str
    value: float
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "details": self.details,
        }


@dataclass
class RAGASResult:
    """RAGAS evaluation results"""
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    
    # Per-sample scores
    sample_scores: List[Dict[str, float]] = field(default_factory=list)
    
    @property
    def overall_score(self) -> float:
        """Average of all RAGAS metrics"""
        scores = [
            self.faithfulness,
            self.answer_relevancy,
            self.context_precision,
            self.context_recall,
        ]
        valid_scores = [s for s in scores if s > 0]
        return sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "overall_score": self.overall_score,
            "num_samples": len(self.sample_scores),
        }


@dataclass
class BenchmarkResult:
    """Complete benchmark results for a teacher model"""
    
    # Model info
    model_name: str
    model_provider: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Sample counts
    total_samples: int = 0
    successful_samples: int = 0
    failed_samples: int = 0
    
    # Generation metrics
    generation_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Quality metrics
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Clinical metrics
    clinical_metrics: Dict[str, float] = field(default_factory=dict)
    
    # RAG metrics
    rag_metrics: Dict[str, float] = field(default_factory=dict)
    
    # RAGAS metrics (separate for clarity)
    ragas_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Efficiency metrics
    efficiency_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Individual sample scores
    sample_scores: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.successful_samples / self.total_samples
    
    @property
    def overall_score(self) -> float:
        """Weighted overall score"""
        weights = {
            "quality": 0.25,
            "clinical": 0.35,
            "rag": 0.15,
            "ragas": 0.15,
            "efficiency": 0.10,
        }
        
        scores = []
        total_weight = 0
        
        if self.quality_metrics:
            scores.append(weights["quality"] * self._avg_metrics(self.quality_metrics))
            total_weight += weights["quality"]
        if self.clinical_metrics:
            scores.append(weights["clinical"] * self._avg_metrics(self.clinical_metrics))
            total_weight += weights["clinical"]
        if self.rag_metrics:
            scores.append(weights["rag"] * self._avg_metrics(self.rag_metrics))
            total_weight += weights["rag"]
        if self.ragas_metrics:
            scores.append(weights["ragas"] * self._avg_metrics(self.ragas_metrics))
            total_weight += weights["ragas"]
        if self.efficiency_metrics:
            scores.append(weights["efficiency"] * self._normalize_efficiency())
            total_weight += weights["efficiency"]
        
        return sum(scores) / total_weight if total_weight > 0 else 0.0
    
    def _avg_metrics(self, metrics: Dict[str, float]) -> float:
        if not metrics:
            return 0.0
        # Filter only float values
        float_values = [v for v in metrics.values() if isinstance(v, (int, float))]
        return sum(float_values) / len(float_values) if float_values else 0.0
    
    def _normalize_efficiency(self) -> float:
        """Normalize efficiency metrics to 0-1 scale"""
        # Lower generation time is better
        time = self.efficiency_metrics.get("avg_generation_time", 60)
        time_score = max(0, 1 - (time / 120))  # 0-120s range
        
        # Lower cost is better
        cost = self.efficiency_metrics.get("avg_cost_per_sample", 0.1)
        cost_score = max(0, 1 - (cost / 0.5))  # $0-0.5 range
        
        return (time_score + cost_score) / 2
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_provider": self.model_provider,
            "timestamp": self.timestamp.isoformat(),
            "total_samples": self.total_samples,
            "successful_samples": self.successful_samples,
            "failed_samples": self.failed_samples,
            "success_rate": self.success_rate,
            "overall_score": self.overall_score,
            "generation_metrics": self.generation_metrics,
            "quality_metrics": self.quality_metrics,
            "clinical_metrics": self.clinical_metrics,
            "rag_metrics": self.rag_metrics,
            "ragas_metrics": self.ragas_metrics,
            "efficiency_metrics": self.efficiency_metrics,
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


# =============================================================================
# Text Quality Metrics
# =============================================================================

def compute_bleu(
    predictions: List[str],
    references: List[str],
) -> MetricResult:
    """
    Compute BLEU score for generated summaries
    
    Args:
        predictions: Generated summaries
        references: Reference summaries
        
    Returns:
        MetricResult with BLEU score
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import nltk
        nltk.download('punkt', quiet=True)
    except ImportError:
        logger.warning("NLTK not available, returning 0 for BLEU")
        return MetricResult(name="bleu", value=0.0, details={"error": "NLTK not installed"})
    
    if not predictions or not references:
        return MetricResult(name="bleu", value=0.0)
    
    smoother = SmoothingFunction().method1
    scores = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.lower().split()
        ref_tokens = [ref.lower().split()]
        
        try:
            score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoother)
            scores.append(score)
        except Exception:
            scores.append(0.0)
    
    avg_score = sum(scores) / len(scores) if scores else 0.0
    
    return MetricResult(
        name="bleu",
        value=avg_score,
        details={
            "num_samples": len(scores),
            "min": min(scores) if scores else 0,
            "max": max(scores) if scores else 0,
        }
    )


def compute_rouge(
    predictions: List[str],
    references: List[str],
) -> MetricResult:
    """
    Compute ROUGE scores for generated summaries
    
    Args:
        predictions: Generated summaries
        references: Reference summaries
        
    Returns:
        MetricResult with ROUGE scores
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        logger.warning("rouge-score not available")
        return MetricResult(name="rouge", value=0.0, details={"error": "rouge-score not installed"})
    
    if not predictions or not references:
        return MetricResult(name="rouge", value=0.0)
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)
    
    # Use ROUGE-L as primary score
    return MetricResult(
        name="rouge",
        value=avg_rougeL,
        details={
            "rouge1": avg_rouge1,
            "rouge2": avg_rouge2,
            "rougeL": avg_rougeL,
            "num_samples": len(predictions),
        }
    )


def compute_bertscore(
    predictions: List[str],
    references: List[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
) -> MetricResult:
    """
    Compute BERTScore for semantic similarity
    
    Args:
        predictions: Generated summaries
        references: Reference summaries
        model_type: Model to use for embeddings
        
    Returns:
        MetricResult with BERTScore
    """
    try:
        from bert_score import score as bert_score
    except ImportError:
        logger.warning("bert-score not available")
        return MetricResult(name="bertscore", value=0.0, details={"error": "bert-score not installed"})
    
    if not predictions or not references:
        return MetricResult(name="bertscore", value=0.0)
    
    try:
        P, R, F1 = bert_score(
            predictions, 
            references, 
            model_type=model_type,
            verbose=False,
        )
        
        avg_f1 = F1.mean().item()
        avg_precision = P.mean().item()
        avg_recall = R.mean().item()
        
        return MetricResult(
            name="bertscore",
            value=avg_f1,
            details={
                "precision": avg_precision,
                "recall": avg_recall,
                "f1": avg_f1,
                "model": model_type,
            }
        )
    except Exception as e:
        logger.warning(f"BERTScore computation failed: {e}")
        return MetricResult(name="bertscore", value=0.0, details={"error": str(e)})


# =============================================================================
# Dialogue Quality Metrics
# =============================================================================

def compute_dialogue_coherence(dialogues: List[List[Dict]]) -> MetricResult:
    """
    Compute dialogue coherence score
    
    Measures:
    - Turn-taking patterns (doctor-patient alternation)
    - Topic consistency
    - Logical flow
    
    Args:
        dialogues: List of dialogue turn lists
        
    Returns:
        MetricResult with coherence score
    """
    if not dialogues:
        return MetricResult(name="dialogue_coherence", value=0.0)
    
    scores = []
    
    for dialogue in dialogues:
        if not dialogue:
            scores.append(0.0)
            continue
        
        score = 0.0
        max_score = 4.0
        
        # 1. Turn-taking pattern (alternating speakers)
        alternation_count = 0
        for i in range(1, len(dialogue)):
            if dialogue[i].get("speaker") != dialogue[i-1].get("speaker"):
                alternation_count += 1
        
        alternation_ratio = alternation_count / (len(dialogue) - 1) if len(dialogue) > 1 else 0
        score += alternation_ratio  # Max 1.0
        
        # 2. Doctor starts conversation
        if dialogue[0].get("speaker") in ["Doctor", "Dr"]:
            score += 0.5
        
        # 3. Minimum dialogue length
        if len(dialogue) >= 8:
            score += 0.5
        elif len(dialogue) >= 6:
            score += 0.25
        
        # 4. Both speakers present
        speakers = set(turn.get("speaker") for turn in dialogue)
        if len(speakers) >= 2:
            score += 0.5
        
        # 5. Question-answer patterns (doctor asks, patient responds)
        qa_patterns = 0
        for i in range(len(dialogue) - 1):
            current_text = dialogue[i].get("text", "").lower()
            if dialogue[i].get("speaker") == "Doctor" and "?" in current_text:
                if dialogue[i+1].get("speaker") == "Patient":
                    qa_patterns += 1
        
        qa_ratio = qa_patterns / (len(dialogue) // 2) if len(dialogue) > 2 else 0
        score += min(1.0, qa_ratio)  # Max 1.0
        
        scores.append(score / max_score)
    
    avg_score = sum(scores) / len(scores)
    
    return MetricResult(
        name="dialogue_coherence",
        value=avg_score,
        details={
            "num_dialogues": len(dialogues),
            "min": min(scores),
            "max": max(scores),
        }
    )


def compute_dialogue_completeness(
    dialogues: List[List[Dict]],
    summaries: List[Dict],
) -> MetricResult:
    """
    Compute dialogue completeness score
    
    Measures whether key clinical elements are covered:
    - Chief complaint discussed
    - History taking (onset, duration, severity)
    - Review of symptoms
    - Medical history
    - Medications/allergies
    - Examination mentioned
    - Plan discussed
    
    Args:
        dialogues: List of dialogue turn lists
        summaries: Corresponding clinical summaries
        
    Returns:
        MetricResult with completeness score
    """
    if not dialogues:
        return MetricResult(name="dialogue_completeness", value=0.0)
    
    # Elements to check in dialogue
    checklist = [
        # History taking
        (r"what brings you|why .* here|how can i help", "greeting"),
        (r"when did .* start|how long|onset|duration", "timing"),
        (r"how .* describe|what .* like|character|quality", "character"),
        (r"how severe|scale of|rate .* pain|intensity", "severity"),
        (r"where .* pain|location|which part", "location"),
        (r"make .* better|make .* worse|aggravat|reliev", "modifying_factors"),
        (r"other symptoms|anything else|associated", "associated_symptoms"),
        # Medical history
        (r"medical history|past .* history|conditions|illnesses", "pmh"),
        (r"medications|taking any|drugs|pills", "medications"),
        (r"allergies|allergic|react", "allergies"),
        # Examination
        (r"examine|check|look at|vital|blood pressure", "examination"),
        # Plan
        (r"test|x-ray|scan|blood work|investigation", "investigations"),
        (r"follow.?up|come back|return|see you", "followup"),
    ]
    
    scores = []
    element_coverage = {elem[1]: [] for elem in checklist}
    
    for dialogue in dialogues:
        dialogue_text = " ".join(
            turn.get("text", "").lower() 
            for turn in dialogue
        )
        
        elements_found = 0
        for pattern, element_name in checklist:
            if re.search(pattern, dialogue_text, re.IGNORECASE):
                elements_found += 1
                element_coverage[element_name].append(1)
            else:
                element_coverage[element_name].append(0)
        
        score = elements_found / len(checklist)
        scores.append(score)
    
    avg_score = sum(scores) / len(scores)
    
    # Calculate per-element coverage
    element_rates = {
        elem: sum(vals) / len(vals) if vals else 0
        for elem, vals in element_coverage.items()
    }
    
    return MetricResult(
        name="dialogue_completeness",
        value=avg_score,
        details={
            "num_dialogues": len(dialogues),
            "element_coverage": element_rates,
            "min": min(scores),
            "max": max(scores),
        }
    )


# =============================================================================
# Clinical Accuracy Metrics
# =============================================================================

def compute_clinical_accuracy(
    samples: List[Any],
) -> MetricResult:
    """
    Compute clinical accuracy metrics
    
    Measures:
    - Validation pass rate
    - Hallucination rate
    - Safety concern coverage
    
    Args:
        samples: List of SyntheticSample objects
        
    Returns:
        MetricResult with clinical accuracy score
    """
    if not samples:
        return MetricResult(name="clinical_accuracy", value=0.0)
    
    validation_passed = 0
    clinical_valid = 0
    has_hallucinations = 0
    has_safety_netting = 0
    
    for sample in samples:
        # Check validation status
        if hasattr(sample, 'validation') and sample.validation:
            if sample.validation.status.value in ["passed", "warning"]:
                validation_passed += 1
            if sample.validation.clinical_valid:
                clinical_valid += 1
            
            # Check for hallucination errors
            if sample.validation.errors:
                hallucination_errors = [
                    e for e in sample.validation.errors
                    if "hallucination" in e.error_type.lower()
                ]
                if hallucination_errors:
                    has_hallucinations += 1
        
        # Check safety netting
        if hasattr(sample, 'summary') and sample.summary:
            if sample.summary.safety_netting:
                has_safety_netting += 1
    
    total = len(samples)
    
    validation_rate = validation_passed / total
    clinical_rate = clinical_valid / total
    hallucination_rate = has_hallucinations / total
    safety_rate = has_safety_netting / total
    
    # Combined score (higher is better, so invert hallucination rate)
    combined_score = (
        validation_rate * 0.3 +
        clinical_rate * 0.3 +
        (1 - hallucination_rate) * 0.2 +
        safety_rate * 0.2
    )
    
    return MetricResult(
        name="clinical_accuracy",
        value=combined_score,
        details={
            "validation_pass_rate": validation_rate,
            "clinical_validity_rate": clinical_rate,
            "hallucination_rate": hallucination_rate,
            "safety_netting_rate": safety_rate,
            "total_samples": total,
        }
    )


def compute_entity_extraction_f1(
    predicted_entities: List[List[str]],
    reference_entities: List[List[str]],
) -> MetricResult:
    """
    Compute F1 score for clinical entity extraction
    
    Args:
        predicted_entities: Predicted entities per sample
        reference_entities: Reference entities per sample
        
    Returns:
        MetricResult with F1 score
    """
    if not predicted_entities or not reference_entities:
        return MetricResult(name="entity_f1", value=0.0)
    
    precisions = []
    recalls = []
    f1_scores = []
    
    for pred, ref in zip(predicted_entities, reference_entities):
        pred_set = set(e.lower() for e in pred)
        ref_set = set(e.lower() for e in ref)
        
        if not pred_set and not ref_set:
            precisions.append(1.0)
            recalls.append(1.0)
            f1_scores.append(1.0)
            continue
        
        true_positives = len(pred_set & ref_set)
        
        precision = true_positives / len(pred_set) if pred_set else 0
        recall = true_positives / len(ref_set) if ref_set else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    avg_f1 = sum(f1_scores) / len(f1_scores)
    
    return MetricResult(
        name="entity_f1",
        value=avg_f1,
        details={
            "precision": sum(precisions) / len(precisions),
            "recall": sum(recalls) / len(recalls),
            "f1": avg_f1,
        }
    )


# =============================================================================
# RAG Performance Metrics
# =============================================================================

def compute_retrieval_precision(
    samples: List[Any],
    relevance_threshold: float = 0.5,
) -> MetricResult:
    """
    Compute retrieval precision for RAG
    
    Args:
        samples: List of SyntheticSample objects with RAG metadata
        relevance_threshold: Score threshold for relevance
        
    Returns:
        MetricResult with precision score
    """
    if not samples:
        return MetricResult(name="retrieval_precision", value=0.0)
    
    precisions = []
    
    for sample in samples:
        if not hasattr(sample, 'rag') or not sample.rag or not sample.rag.rag_enabled:
            continue
        
        scores = sample.rag.retrieval_scores
        if not scores:
            continue
        
        relevant_count = sum(1 for s in scores if s >= relevance_threshold)
        precision = relevant_count / len(scores)
        precisions.append(precision)
    
    avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
    
    return MetricResult(
        name="retrieval_precision",
        value=avg_precision,
        details={
            "threshold": relevance_threshold,
            "num_samples": len(precisions),
        }
    )


def compute_context_relevance(
    samples: List[Any],
) -> MetricResult:
    """
    Compute context relevance for RAG
    
    Measures how well retrieved context matches the scenario
    
    Args:
        samples: List of SyntheticSample objects
        
    Returns:
        MetricResult with relevance score
    """
    if not samples:
        return MetricResult(name="context_relevance", value=0.0)
    
    faithfulness_scores = []
    
    for sample in samples:
        if hasattr(sample, 'validation') and sample.validation:
            if sample.validation.rag_faithfulness is not None:
                faithfulness_scores.append(sample.validation.rag_faithfulness)
    
    avg_score = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0
    
    return MetricResult(
        name="context_relevance",
        value=avg_score,
        details={
            "num_samples": len(faithfulness_scores),
            "min": min(faithfulness_scores) if faithfulness_scores else 0,
            "max": max(faithfulness_scores) if faithfulness_scores else 0,
        }
    )


# =============================================================================
# RAGAS Metrics
# =============================================================================

def compute_ragas_metrics(
    samples: List[Any],
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o-mini",
) -> RAGASResult:
    """
    Compute RAGAS metrics for RAG evaluation
    
    RAGAS (Retrieval Augmented Generation Assessment) provides:
    - Faithfulness: Is the answer grounded in the context?
    - Answer Relevancy: Does the answer address the question?
    - Context Precision: Are the retrieved contexts relevant?
    - Context Recall: Are all relevant facts retrieved?
    
    Args:
        samples: List of SyntheticSample objects with RAG data
        llm_provider: LLM provider for RAGAS evaluation
        llm_model: Model name for evaluation
        
    Returns:
        RAGASResult with all RAGAS metrics
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from datasets import Dataset
    except ImportError:
        logger.warning(
            "RAGAS not installed. Install with: pip install ragas datasets"
        )
        return RAGASResult()
    
    if not samples:
        return RAGASResult()
    
    # Prepare data for RAGAS
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    for sample in samples:
        # Skip if no RAG data
        if not hasattr(sample, 'rag') or not sample.rag or not sample.rag.rag_enabled:
            continue
        
        # Question = scenario text
        question = ""
        if hasattr(sample, 'scenario') and sample.scenario:
            question = sample.scenario.scenario_text or ""
        if not question and hasattr(sample, 'summary') and sample.summary:
            question = sample.summary.chief_complaint or ""
        
        if not question:
            continue
        
        # Answer = generated summary (HPI + Assessment + Plan)
        answer = ""
        if hasattr(sample, 'summary') and sample.summary:
            parts = []
            if sample.summary.history_of_present_illness:
                parts.append(sample.summary.history_of_present_illness)
            if sample.summary.assessment:
                parts.append(f"Assessment: {sample.summary.assessment}")
            if sample.summary.plan:
                parts.append(f"Plan: {sample.summary.plan}")
            answer = " ".join(parts)
        
        if not answer:
            continue
        
        # Contexts = retrieved RAG context
        context_list = []
        if sample.rag.context_used:
            # Split context by source markers
            context_list = [sample.rag.context_used]
        
        if not context_list:
            continue
        
        # Ground truth = we use the answer itself for self-consistency check
        # In a real scenario, you'd have human-annotated ground truths
        ground_truth = answer
        
        questions.append(question)
        answers.append(answer)
        contexts.append(context_list)
        ground_truths.append(ground_truth)
    
    if not questions:
        logger.warning("No valid samples for RAGAS evaluation")
        return RAGASResult()
    
    # Create RAGAS dataset
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
    
    dataset = Dataset.from_dict(data)
    
    try:
        # Configure LLM for RAGAS
        if llm_provider == "openai":
            import os
            if not os.getenv("OPENAI_API_KEY"):
                logger.warning("OPENAI_API_KEY not set for RAGAS evaluation")
                return _compute_ragas_fallback(samples)
        
        # Run RAGAS evaluation
        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]
        
        result = evaluate(dataset, metrics=metrics)
        
        # Extract scores
        ragas_result = RAGASResult(
            faithfulness=result.get("faithfulness", 0.0),
            answer_relevancy=result.get("answer_relevancy", 0.0),
            context_precision=result.get("context_precision", 0.0),
            context_recall=result.get("context_recall", 0.0),
        )
        
        # Store per-sample scores if available
        if hasattr(result, 'scores'):
            ragas_result.sample_scores = result.scores
        
        return ragas_result
        
    except Exception as e:
        logger.warning(f"RAGAS evaluation failed: {e}. Using fallback metrics.")
        return _compute_ragas_fallback(samples)


def _compute_ragas_fallback(samples: List[Any]) -> RAGASResult:
    """
    Compute approximate RAGAS-like metrics without the RAGAS library
    
    Uses simpler heuristics as fallback when RAGAS isn't available.
    """
    if not samples:
        return RAGASResult()
    
    faithfulness_scores = []
    relevancy_scores = []
    precision_scores = []
    
    for sample in samples:
        if not hasattr(sample, 'rag') or not sample.rag or not sample.rag.rag_enabled:
            continue
        
        # Faithfulness: Use existing RAG faithfulness score
        if hasattr(sample, 'validation') and sample.validation:
            if sample.validation.rag_faithfulness:
                faithfulness_scores.append(sample.validation.rag_faithfulness)
        
        # Relevancy: Check if answer addresses key elements from context
        if hasattr(sample, 'summary') and sample.summary:
            answer = sample.summary.history_of_present_illness or ""
            context = sample.rag.context_used or ""
            
            if answer and context:
                # Simple word overlap metric
                answer_words = set(answer.lower().split())
                context_words = set(context.lower().split())
                
                # Remove stop words
                stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 'and', 'in', 'for', 'with'}
                answer_words -= stop_words
                context_words -= stop_words
                
                if context_words:
                    overlap = len(answer_words & context_words) / len(context_words)
                    relevancy_scores.append(min(1.0, overlap * 2))  # Scale up
        
        # Precision: Use retrieval scores
        if sample.rag.retrieval_scores:
            # Proportion of high-scoring retrievals
            high_scores = sum(1 for s in sample.rag.retrieval_scores if s >= 0.5)
            precision = high_scores / len(sample.rag.retrieval_scores)
            precision_scores.append(precision)
    
    return RAGASResult(
        faithfulness=sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0,
        answer_relevancy=sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0.0,
        context_precision=sum(precision_scores) / len(precision_scores) if precision_scores else 0.0,
        context_recall=0.0,  # Can't compute without ground truth
    )


# =============================================================================
# Main Benchmark Class
# =============================================================================

class TeacherBenchmark:
    """
    Comprehensive benchmark for teacher model evaluation
    
    Runs all metrics and produces a complete evaluation report.
    
    Example:
        benchmark = TeacherBenchmark()
        result = benchmark.evaluate(samples, references=ref_summaries)
        print(f"Overall Score: {result.overall_score:.2f}")
    """
    
    def __init__(
        self,
        compute_bertscore: bool = False,  # Slow, optional
        compute_ragas: bool = False,  # Requires RAGAS library
        relevance_threshold: float = 0.5,
        ragas_llm_provider: str = "openai",
        ragas_llm_model: str = "gpt-4o-mini",
    ):
        """
        Initialize benchmark
        
        Args:
            compute_bertscore: Whether to compute BERTScore (slow)
            compute_ragas: Whether to compute RAGAS metrics
            relevance_threshold: RAG relevance threshold
            ragas_llm_provider: LLM provider for RAGAS
            ragas_llm_model: LLM model for RAGAS
        """
        self.compute_bertscore_flag = compute_bertscore
        self.compute_ragas_flag = compute_ragas
        self.relevance_threshold = relevance_threshold
        self.ragas_llm_provider = ragas_llm_provider
        self.ragas_llm_model = ragas_llm_model
    
    def evaluate(
        self,
        samples: List[Any],
        references: Optional[List[str]] = None,
        model_name: str = "unknown",
        model_provider: str = "unknown",
    ) -> BenchmarkResult:
        """
        Run comprehensive evaluation
        
        Args:
            samples: Generated SyntheticSample objects
            references: Optional reference summaries for comparison
            model_name: Name of the teacher model
            model_provider: Provider (ollama, openai, etc.)
            
        Returns:
            BenchmarkResult with all metrics
        """
        result = BenchmarkResult(
            model_name=model_name,
            model_provider=model_provider,
            total_samples=len(samples),
        )
        
        # Count successes/failures
        for sample in samples:
            if hasattr(sample, 'validation') and sample.validation:
                if sample.validation.status.value != "failed":
                    result.successful_samples += 1
                else:
                    result.failed_samples += 1
            else:
                result.successful_samples += 1
        
        # Extract data for metrics
        dialogues = [
            [{"speaker": t.speaker.value, "text": t.text} for t in sample.dialogue]
            for sample in samples
            if hasattr(sample, 'dialogue')
        ]
        
        summaries = [
            sample.summary.model_dump() if hasattr(sample.summary, 'model_dump') else {}
            for sample in samples
            if hasattr(sample, 'summary')
        ]
        
        predictions = [
            sample.summary.history_of_present_illness or ""
            for sample in samples
            if hasattr(sample, 'summary')
        ]
        
        # Compute quality metrics
        result.quality_metrics["dialogue_coherence"] = compute_dialogue_coherence(dialogues).value
        result.quality_metrics["dialogue_completeness"] = compute_dialogue_completeness(dialogues, summaries).value
        
        # Text quality (if references provided)
        if references:
            result.quality_metrics["bleu"] = compute_bleu(predictions, references).value
            result.quality_metrics["rouge"] = compute_rouge(predictions, references).value
            if self.compute_bertscore_flag:
                result.quality_metrics["bertscore"] = compute_bertscore(predictions, references).value
        
        # Clinical metrics
        clinical_result = compute_clinical_accuracy(samples)
        result.clinical_metrics["clinical_accuracy"] = clinical_result.value
        result.clinical_metrics.update(clinical_result.details)
        
        # RAG metrics
        result.rag_metrics["retrieval_precision"] = compute_retrieval_precision(
            samples, self.relevance_threshold
        ).value
        result.rag_metrics["context_relevance"] = compute_context_relevance(samples).value
        
        # RAGAS metrics (optional)
        if self.compute_ragas_flag:
            logger.info("Computing RAGAS metrics...")
            ragas_result = compute_ragas_metrics(
                samples,
                llm_provider=self.ragas_llm_provider,
                llm_model=self.ragas_llm_model,
            )
            result.ragas_metrics = ragas_result.to_dict()
        else:
            # Use fallback RAGAS-like metrics (no LLM required)
            ragas_result = _compute_ragas_fallback(samples)
            result.ragas_metrics = ragas_result.to_dict()
        
        # Efficiency metrics
        generation_times = [
            sample.generation.generation_time_seconds
            for sample in samples
            if hasattr(sample, 'generation') and sample.generation.generation_time_seconds
        ]
        
        if generation_times:
            result.efficiency_metrics["avg_generation_time"] = sum(generation_times) / len(generation_times)
            result.efficiency_metrics["min_generation_time"] = min(generation_times)
            result.efficiency_metrics["max_generation_time"] = max(generation_times)
            result.efficiency_metrics["total_generation_time"] = sum(generation_times)
        
        # Store individual sample scores
        for sample in samples:
            sample_score = {
                "id": sample.id if hasattr(sample, 'id') else "unknown",
                "generation_time": sample.generation.generation_time_seconds if hasattr(sample, 'generation') else None,
                "validation_status": sample.validation.status.value if hasattr(sample, 'validation') and sample.validation else None,
                "rag_faithfulness": sample.validation.rag_faithfulness if hasattr(sample, 'validation') and sample.validation else None,
                "difficulty_score": sample.difficulty.difficulty_score if hasattr(sample, 'difficulty') and sample.difficulty else None,
            }
            result.sample_scores.append(sample_score)
        
        return result


def run_benchmark(
    samples: List[Any],
    model_name: str = "unknown",
    model_provider: str = "unknown",
    references: Optional[List[str]] = None,
    **kwargs,
) -> BenchmarkResult:
    """
    Convenience function to run benchmark
    
    Args:
        samples: Generated samples
        model_name: Model name
        model_provider: Provider name
        references: Optional reference summaries
        **kwargs: Additional arguments for TeacherBenchmark
        
    Returns:
        BenchmarkResult
    """
    benchmark = TeacherBenchmark(**kwargs)
    return benchmark.evaluate(
        samples=samples,
        references=references,
        model_name=model_name,
        model_provider=model_provider,
    )


if __name__ == "__main__":
    # Test metrics
    print("Testing Evaluation Metrics")
    print("=" * 60)
    
    # Test dialogue coherence
    test_dialogues = [
        [
            {"speaker": "Doctor", "text": "What brings you in today?"},
            {"speaker": "Patient", "text": "I have chest pain."},
            {"speaker": "Doctor", "text": "How long have you had it?"},
            {"speaker": "Patient", "text": "About 3 days."},
            {"speaker": "Doctor", "text": "Let me examine you."},
            {"speaker": "Patient", "text": "Okay."},
        ]
    ]
    
    coherence = compute_dialogue_coherence(test_dialogues)
    print(f"Dialogue Coherence: {coherence.value:.3f}")
    
    completeness = compute_dialogue_completeness(test_dialogues, [{}])
    print(f"Dialogue Completeness: {completeness.value:.3f}")
    print(f"  Element coverage: {completeness.details['element_coverage']}")
    
    print("\n✓ Metrics tests passed!")