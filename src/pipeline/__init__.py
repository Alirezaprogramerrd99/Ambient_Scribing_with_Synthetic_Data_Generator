"""
Pipeline Module

End-to-end orchestration for synthetic clinical data generation.

The pipeline integrates:
- Scenario generation
- Knowledge base and RAG retrieval
- Teacher model generation
- Validation and filtering
- Export and experiment tracking

Example:
    from src.pipeline import SyntheticDataPipeline, PipelineConfig
    
    config = PipelineConfig(
        num_scenarios=100,
        teacher_model="llama3.1:8b",
        use_rag=True,
        output_dir="./output",
    )
    
    pipeline = SyntheticDataPipeline(config)
    result = pipeline.run()
    
    print(f"Generated {result.total_valid} valid samples")
    print(f"Success rate: {result.success_rate:.1%}")

CLI Usage:
    python -m src.pipeline.synthetic_data_pipeline \\
        --num-scenarios 100 \\
        --provider ollama \\
        --model llama3.1:8b \\
        --use-rag \\
        --output-dir ./output
"""

from .synthetic_data_pipeline import (
    # Configuration
    PipelineConfig,
    # Results
    PipelineResult,
    # Main pipeline
    SyntheticDataPipeline,
    # CLI
    main,
)

__all__ = [
    "PipelineConfig",
    "PipelineResult",
    "SyntheticDataPipeline",
    "main",
]