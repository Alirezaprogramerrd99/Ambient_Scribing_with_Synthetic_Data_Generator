from src.evaluation import BenchmarkResult

# Simulate your metrics
result = BenchmarkResult(
    model_name="test",
    model_provider="test",
    total_samples=7,
    successful_samples=7,
    clinical_metrics={
        "clinical_accuracy": 0.929,
        "validation_pass_rate": 1.0,
        "hallucination_rate": 0.143,
        "total_samples": 7,  # This should now be excluded
    }
)

print(f"Overall Score: {result.overall_score}")  # Should be < 1.0