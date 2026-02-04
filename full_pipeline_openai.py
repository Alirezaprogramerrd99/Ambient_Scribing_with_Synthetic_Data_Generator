"""Run full pipeline with OpenAI"""
import os
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()

from src.pipeline import SyntheticDataPipeline, PipelineConfig

config = PipelineConfig(
    num_scenarios=7,
    rag_top_k=7,
    # OpenAI settings
    teacher_provider="openai",
    teacher_model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=4096,
    
    # RAG
    use_rag=True,
    knowledge_base_path=Path("./medical_knowledge"),
    
    # Output
    output_dir = Path("./data/synthetic_output"),
    
    # Validation
    enable_validation=True,
    filter_invalid=True,
    
    
     # Benchmarking (NEW)
    enable_benchmarking=True,       # Enable benchmarks
    compute_bertscore=False,        # Skip BERTScore (slow)
    compute_ragas=True,             # Enable full RAGAS metrics
    generate_benchmark_report=True, # Generate markdown report
)

pipeline = SyntheticDataPipeline(config)
result = pipeline.run()

print(f"\n✓ Generated {result.total_valid} valid samples")
print(f"✓ Output: {result.output_path}")