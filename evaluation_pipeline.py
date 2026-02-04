from src.evaluation import run_benchmark, generate_benchmark_report

# Load your generated samples
from src.pipeline import load_samples
samples = load_samples("data/synthetic_output/synthetic_data_xxx.jsonl")

# Run benchmark
result = run_benchmark(
    samples=samples,
    model_name="gpt-4o-mini",
    model_provider="openai",
)

# Print summary
print(f"Overall Score: {result.overall_score:.3f}")
print(f"Success Rate: {result.success_rate:.1%}")
print(f"Quality Metrics: {result.quality_metrics}")
print(f"Clinical Metrics: {result.clinical_metrics}")

# Generate report
report_path = generate_benchmark_report(result, output_dir="./benchmark_reports")
print(f"Report saved to: {report_path}")



##### Comparison between models (optional) #####


# from src.evaluation import BenchmarkReporter, run_benchmark

# # Run benchmarks for each model
# ollama_result = run_benchmark(ollama_samples, "llama3.1:8b", "ollama")
# openai_result = run_benchmark(openai_samples, "gpt-4o-mini", "openai")

# # Compare
# reporter = BenchmarkReporter(output_dir="./benchmark_reports")
# comparison_path = reporter.generate_comparison_report([ollama_result, openai_result])
# print(f"Comparison report: {comparison_path}")