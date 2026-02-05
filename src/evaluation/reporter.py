"""
Benchmark Reporter

Generates reports and visualizations for benchmark results.

Author: Alireza Rashidi
MSc Project: Trustworthy SLMs for Ambient Clinical Scribing
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from .metrics import BenchmarkResult


@dataclass
class ComparisonResult:
    """Result of comparing multiple models"""
    models: List[str]
    metrics: Dict[str, Dict[str, float]]
    best_model: str
    best_score: float


class BenchmarkReporter:
    """
    Generate reports from benchmark results
    
    Supports:
    - JSON export
    - Markdown reports
    - Model comparison tables
    - RAGAS metrics reporting
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize reporter
        
        Args:
            output_dir: Directory for report output
        """
        self.output_dir = Path(output_dir) if output_dir else Path("./benchmark_reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_json(self, result: BenchmarkResult, filename: Optional[str] = None) -> Path:
        """Save benchmark result as JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Sanitize model name for filename
            model_name_safe = result.model_name.replace(":", "-").replace("/", "-")
            filename = f"benchmark_{result.model_provider}_{model_name_safe}_{timestamp}.json"
        
        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(result.to_json())
        
        return filepath
    
    def generate_markdown_report(
        self,
        result: BenchmarkResult,
        filename: Optional[str] = None,
    ) -> Path:
        """Generate a markdown report"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_report_{timestamp}.md"
        
        report = self._build_markdown_report(result)
        
        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report)
        
        return filepath
    
    def _build_markdown_report(self, result: BenchmarkResult) -> str:
        """Build markdown report content"""
        lines = [
            f"# Teacher Model Benchmark Report",
            f"",
            f"**Model:** {result.model_provider}/{result.model_name}",
            f"**Date:** {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Overall Score:** {result.overall_score:.3f}",
            f"",
            f"---",
            f"",
            f"## Executive Summary",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Samples | {result.total_samples} |",
            f"| Successful | {result.successful_samples} |",
            f"| Failed | {result.failed_samples} |",
            f"| Success Rate | {result.success_rate:.1%} |",
            f"| Overall Score | {result.overall_score:.3f} |",
            f"",
        ]
        
        # Quality Metrics
        if result.quality_metrics:
            lines.extend([
                f"---",
                f"",
                f"## Quality Metrics",
                f"",
                f"These metrics measure the quality of generated dialogues and summaries.",
                f"",
                f"| Metric | Score | Description |",
                f"|--------|-------|-------------|",
            ])
            
            metric_descriptions = {
                "dialogue_coherence": "Turn-taking patterns, Q&A flow",
                "dialogue_completeness": "Coverage of clinical elements",
                "bleu": "N-gram overlap with references",
                "rouge": "Recall-oriented summary evaluation",
                "bertscore": "Semantic similarity score",
            }
            
            for metric, value in result.quality_metrics.items():
                desc = metric_descriptions.get(metric, "")
                if isinstance(value, (int, float)):
                    lines.append(f"| {metric} | {value:.3f} | {desc} |")
                else:
                    lines.append(f"| {metric} | {value} | {desc} |")
            
            lines.append(f"")
        
        # Clinical Metrics
        if result.clinical_metrics:
            lines.extend([
                f"---",
                f"",
                f"## Clinical Metrics",
                f"",
                f"These metrics assess the clinical accuracy and safety of generated content.",
                f"",
                f"| Metric | Score |",
                f"|--------|-------|",
            ])
            
            for metric, value in result.clinical_metrics.items():
                if isinstance(value, float):
                    lines.append(f"| {metric} | {value:.3f} |")
                elif isinstance(value, int):
                    lines.append(f"| {metric} | {value} |")
                # Skip non-numeric values in main table
            
            lines.append(f"")
        
        # RAG Metrics
        if result.rag_metrics:
            lines.extend([
                f"---",
                f"",
                f"## RAG Metrics",
                f"",
                f"These metrics evaluate the retrieval-augmented generation performance.",
                f"",
                f"| Metric | Score | Description |",
                f"|--------|-------|-------------|",
            ])
            
            rag_descriptions = {
                "retrieval_precision": "Proportion of relevant retrieved docs",
                "context_relevance": "How well context matches the query",
            }
            
            for metric, value in result.rag_metrics.items():
                desc = rag_descriptions.get(metric, "")
                if isinstance(value, (int, float)):
                    lines.append(f"| {metric} | {value:.3f} | {desc} |")
            
            lines.append(f"")
        
        # RAGAS Metrics (NEW)
        if result.ragas_metrics:
            lines.extend([
                f"---",
                f"",
                f"## RAGAS Metrics",
                f"",
                f"[RAGAS](https://github.com/explodinggradients/ragas) (Retrieval Augmented Generation Assessment) provides standardized evaluation metrics for RAG systems.",
                f"",
                f"| Metric | Score | Description |",
                f"|--------|-------|-------------|",
            ])
            
            ragas_descriptions = {
                "faithfulness": "Is the answer grounded in the retrieved context?",
                "answer_relevancy": "Does the answer address the question?",
                "context_precision": "Are the retrieved contexts relevant?",
                "context_recall": "Are all relevant facts retrieved?",
                "overall_score": "Average of all RAGAS metrics",
                "num_samples": "Number of samples evaluated",
            }
            
            # Order metrics for better readability
            metric_order = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "overall_score"]
            
            for metric in metric_order:
                if metric in result.ragas_metrics:
                    value = result.ragas_metrics[metric]
                    desc = ragas_descriptions.get(metric, "")
                    if isinstance(value, (int, float)):
                        lines.append(f"| {metric} | {value:.3f} | {desc} |")
            
            # Add any other metrics not in the order
            for metric, value in result.ragas_metrics.items():
                if metric not in metric_order and metric != "num_samples":
                    desc = ragas_descriptions.get(metric, "")
                    if isinstance(value, (int, float)):
                        lines.append(f"| {metric} | {value:.3f} | {desc} |")
            
            lines.append(f"")
            
            # RAGAS interpretation guide
            lines.extend([
                f"### RAGAS Score Interpretation",
                f"",
                f"| Score Range | Interpretation |",
                f"|-------------|----------------|",
                f"| 0.8 - 1.0 | Excellent - High quality RAG responses |",
                f"| 0.6 - 0.8 | Good - Acceptable quality with room for improvement |",
                f"| 0.4 - 0.6 | Fair - Needs improvement in retrieval or generation |",
                f"| 0.0 - 0.4 | Poor - Significant issues with RAG pipeline |",
                f"",
            ])
        
        # Efficiency Metrics
        if result.efficiency_metrics:
            lines.extend([
                f"---",
                f"",
                f"## Efficiency Metrics",
                f"",
                f"| Metric | Value |",
                f"|--------|-------|",
            ])
            
            for metric, value in result.efficiency_metrics.items():
                if "time" in metric.lower():
                    lines.append(f"| {metric} | {value:.2f}s |")
                elif "cost" in metric.lower():
                    lines.append(f"| {metric} | ${value:.4f} |")
                else:
                    lines.append(f"| {metric} | {value:.3f} |")
            
            lines.append(f"")
        
        # Individual Sample Scores
        if result.sample_scores:
            lines.extend([
                f"---",
                f"",
                f"## Individual Sample Scores",
                f"",
                f"| ID | Gen Time | Validation | RAG Score | Difficulty |",
                f"|----|----------|------------|-----------|------------|",
            ])
            
            for sample in result.sample_scores[:20]:  # Limit to 20 samples
                gen_time = f"{sample.get('generation_time', 0):.1f}s" if sample.get('generation_time') else "N/A"
                validation = sample.get('validation_status', 'N/A')
                rag = f"{sample.get('rag_faithfulness', 0):.2f}" if sample.get('rag_faithfulness') else "N/A"
                difficulty = sample.get('difficulty_score', 'N/A')
                sample_id = str(sample.get('id', 'N/A'))[:20]
                lines.append(f"| {sample_id} | {gen_time} | {validation} | {rag} | {difficulty} |")
            
            if len(result.sample_scores) > 20:
                lines.append(f"| ... | ... | ... | ... | ... |")
                lines.append(f"| *(showing 20 of {len(result.sample_scores)} samples)* | | | | |")
            
            lines.append(f"")
        
        # Recommendations
        lines.extend([
            f"---",
            f"",
            f"## Recommendations",
            f"",
        ])
        
        recommendations = self._generate_recommendations(result)
        for rec in recommendations:
            lines.append(f"- {rec}")
        
        lines.append(f"")
        
        # Footer
        lines.extend([
            f"---",
            f"",
            f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            f"",
            f"*MSc Project: Trustworthy SLMs for Ambient Clinical Scribing*",
        ])
        
        return "\n".join(lines)
    
    def _generate_recommendations(self, result: BenchmarkResult) -> List[str]:
        """Generate recommendations based on benchmark results"""
        recommendations = []
        
        # Check success rate
        if result.success_rate < 0.8:
            recommendations.append(
                f"Success rate ({result.success_rate:.1%}) is below 80%. "
                f"Consider adjusting prompts or increasing max_tokens."
            )
        
        # Check quality metrics
        if result.quality_metrics:
            coherence = result.quality_metrics.get("dialogue_coherence", 1.0)
            if coherence < 0.7:
                recommendations.append(
                    f"Dialogue coherence ({coherence:.2f}) is low. "
                    f"Review prompt instructions for conversation flow."
                )
            
            completeness = result.quality_metrics.get("dialogue_completeness", 1.0)
            if completeness < 0.6:
                recommendations.append(
                    f"Dialogue completeness ({completeness:.2f}) is low. "
                    f"Ensure prompts request all clinical elements."
                )
        
        # Check clinical metrics
        if result.clinical_metrics:
            clinical_acc = result.clinical_metrics.get("clinical_accuracy", 1.0)
            if clinical_acc < 0.7:
                recommendations.append(
                    f"Clinical accuracy ({clinical_acc:.2f}) needs improvement. "
                    f"Consider adding more clinical validation rules."
                )
            
            hallucination_rate = result.clinical_metrics.get("hallucination_rate", 0.0)
            if hallucination_rate > 0.2:
                recommendations.append(
                    f"Hallucination rate ({hallucination_rate:.1%}) is high. "
                    f"Improve RAG context or add fact-checking."
                )
        
        # Check RAG metrics
        if result.rag_metrics:
            precision = result.rag_metrics.get("retrieval_precision", 1.0)
            if precision < 0.5:
                recommendations.append(
                    f"Retrieval precision ({precision:.2f}) is low. "
                    f"Consider improving knowledge base or embeddings."
                )
        
        # Check RAGAS metrics
        if result.ragas_metrics:
            faithfulness = result.ragas_metrics.get("faithfulness", 1.0)
            if faithfulness < 0.6:
                recommendations.append(
                    f"RAGAS faithfulness ({faithfulness:.2f}) is below threshold. "
                    f"Responses may not be well-grounded in retrieved context."
                )
            
            relevancy = result.ragas_metrics.get("answer_relevancy", 1.0)
            if relevancy < 0.6:
                recommendations.append(
                    f"RAGAS answer relevancy ({relevancy:.2f}) is low. "
                    f"Generated answers may not fully address the queries."
                )
        
        # Check efficiency
        if result.efficiency_metrics:
            avg_time = result.efficiency_metrics.get("avg_generation_time", 0)
            if avg_time > 60:
                recommendations.append(
                    f"Average generation time ({avg_time:.1f}s) is high. "
                    f"Consider using a faster model or reducing max_tokens."
                )
        
        if not recommendations:
            recommendations.append("All metrics are within acceptable ranges. Good job! 🎉")
        
        return recommendations
    
    def compare_models(
        self,
        results: List[BenchmarkResult],
    ) -> ComparisonResult:
        """
        Compare multiple model benchmark results
        
        Args:
            results: List of BenchmarkResult objects
            
        Returns:
            ComparisonResult with comparison data
        """
        models = [f"{r.model_provider}/{r.model_name}" for r in results]
        
        # Aggregate metrics
        metrics = {
            "overall_score": {},
            "success_rate": {},
            "quality": {},
            "clinical": {},
            "rag": {},
            "ragas": {},
            "efficiency": {},
        }
        
        for result in results:
            model_key = f"{result.model_provider}/{result.model_name}"
            metrics["overall_score"][model_key] = result.overall_score
            metrics["success_rate"][model_key] = result.success_rate
            
            if result.quality_metrics:
                float_vals = [v for v in result.quality_metrics.values() if isinstance(v, (int, float))]
                if float_vals:
                    metrics["quality"][model_key] = sum(float_vals) / len(float_vals)
            
            if result.clinical_metrics:
                clinical_floats = [v for v in result.clinical_metrics.values() if isinstance(v, float)]
                if clinical_floats:
                    metrics["clinical"][model_key] = sum(clinical_floats) / len(clinical_floats)
            
            if result.rag_metrics:
                rag_floats = [v for v in result.rag_metrics.values() if isinstance(v, (int, float))]
                if rag_floats:
                    metrics["rag"][model_key] = sum(rag_floats) / len(rag_floats)
            
            if result.ragas_metrics:
                ragas_overall = result.ragas_metrics.get("overall_score", 0)
                if ragas_overall:
                    metrics["ragas"][model_key] = ragas_overall
            
            if result.efficiency_metrics:
                # Lower time is better, so invert
                avg_time = result.efficiency_metrics.get("avg_generation_time", 60)
                metrics["efficiency"][model_key] = max(0, 1 - (avg_time / 120))
        
        # Find best model
        best_model = max(metrics["overall_score"].items(), key=lambda x: x[1])
        
        return ComparisonResult(
            models=models,
            metrics=metrics,
            best_model=best_model[0],
            best_score=best_model[1],
        )
    
    def generate_comparison_report(
        self,
        results: List[BenchmarkResult],
        filename: Optional[str] = None,
    ) -> Path:
        """Generate a comparison report for multiple models"""
        comparison = self.compare_models(results)
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_comparison_{timestamp}.md"
        
        lines = [
            f"# Model Comparison Report",
            f"",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Models Compared:** {len(comparison.models)}",
            f"**Best Model:** {comparison.best_model} (score: {comparison.best_score:.3f})",
            f"",
            f"---",
            f"",
            f"## Overall Comparison",
            f"",
            f"| Model | Overall | Success | Quality | Clinical | RAG | RAGAS | Efficiency |",
            f"|-------|---------|---------|---------|----------|-----|-------|------------|",
        ]
        
        for model in comparison.models:
            overall = comparison.metrics["overall_score"].get(model, 0)
            success = comparison.metrics["success_rate"].get(model, 0)
            quality = comparison.metrics["quality"].get(model, 0)
            clinical = comparison.metrics["clinical"].get(model, 0)
            rag = comparison.metrics["rag"].get(model, 0)
            ragas = comparison.metrics["ragas"].get(model, 0)
            efficiency = comparison.metrics["efficiency"].get(model, 0)
            
            lines.append(
                f"| {model} | {overall:.3f} | {success:.1%} | {quality:.3f} | "
                f"{clinical:.3f} | {rag:.3f} | {ragas:.3f} | {efficiency:.3f} |"
            )
        
        lines.extend([
            f"",
            f"---",
            f"",
            f"## Detailed Results by Model",
            f"",
        ])
        
        for result in results:
            lines.extend([
                f"### {result.model_provider}/{result.model_name}",
                f"",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Samples | {result.total_samples} |",
                f"| Success Rate | {result.success_rate:.1%} |",
                f"| Overall Score | {result.overall_score:.3f} |",
            ])
            
            if result.efficiency_metrics:
                avg_time = result.efficiency_metrics.get("avg_generation_time", 0)
                lines.append(f"| Avg Generation Time | {avg_time:.1f}s |")
            
            if result.ragas_metrics:
                ragas_overall = result.ragas_metrics.get("overall_score", 0)
                lines.append(f"| RAGAS Score | {ragas_overall:.3f} |")
            
            lines.append(f"")
        
        # Winner analysis
        lines.extend([
            f"---",
            f"",
            f"## Recommendation",
            f"",
            f"Based on the comparison, **{comparison.best_model}** achieves the highest overall score "
            f"({comparison.best_score:.3f}) and is recommended for production use.",
            f"",
            f"---",
            f"",
            f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        ])
        
        report = "\n".join(lines)
        
        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report)
        
        return filepath


def generate_benchmark_report(
    result: BenchmarkResult,
    output_dir: Optional[Path] = None,
    format: str = "markdown",
) -> Path:
    """
    Convenience function to generate a benchmark report
    
    Args:
        result: BenchmarkResult to report
        output_dir: Output directory
        format: Report format ("markdown" or "json")
        
    Returns:
        Path to generated report
    """
    reporter = BenchmarkReporter(output_dir)
    
    if format == "json":
        return reporter.save_json(result)
    else:
        return reporter.generate_markdown_report(result)


if __name__ == "__main__":
    # Test reporter
    print("Testing Benchmark Reporter")
    print("=" * 60)
    
    # Create mock result with RAGAS metrics
    result = BenchmarkResult(
        model_name="gpt-4o-mini",
        model_provider="openai",
        total_samples=10,
        successful_samples=9,
        failed_samples=1,
        quality_metrics={
            "dialogue_coherence": 0.85,
            "dialogue_completeness": 0.78,
        },
        clinical_metrics={
            "clinical_accuracy": 0.82,
            "validation_pass_rate": 0.90,
            "hallucination_rate": 0.10,
            "safety_netting_rate": 0.95,
        },
        rag_metrics={
            "retrieval_precision": 0.72,
            "context_relevance": 0.68,
        },
        ragas_metrics={
            "faithfulness": 0.75,
            "answer_relevancy": 0.82,
            "context_precision": 0.70,
            "context_recall": 0.65,
            "overall_score": 0.73,
        },
        efficiency_metrics={
            "avg_generation_time": 25.5,
            "min_generation_time": 18.2,
            "max_generation_time": 35.8,
        },
        sample_scores=[
            {"id": "sample_001", "generation_time": 22.5, "validation_status": "passed", "rag_faithfulness": 0.75, "difficulty_score": 5},
            {"id": "sample_002", "generation_time": 28.3, "validation_status": "warning", "rag_faithfulness": 0.62, "difficulty_score": 7},
        ],
    )
    
    reporter = BenchmarkReporter(Path("./test_reports"))
    
    # Generate markdown report
    md_path = reporter.generate_markdown_report(result)
    print(f"Markdown report: {md_path}")
    
    # Generate JSON
    json_path = reporter.save_json(result)
    print(f"JSON report: {json_path}")
    
    # Show report content
    print("\n" + "=" * 60)
    print("Report Preview:")
    print("=" * 60)
    with open(md_path, "r") as f:
        content = f.read()
        # Show first 50 lines
        lines = content.split("\n")[:50]
        print("\n".join(lines))
        if len(content.split("\n")) > 50:
            print("\n... (truncated)")
    
    print("\n✓ Reporter tests passed!")