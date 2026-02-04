"""
Ambient Scribe Teacher - Synthetic Clinical Data Generation

A comprehensive framework for generating synthetic clinical dialogue-summary pairs
for training Small Language Models (SLMs) in ambient clinical scribing.

Based on: Woo et al. (2025) - Synthetic data distillation enables the
extraction of clinical information at scale

Modules:
    config      - Configuration management
    models      - Pydantic schemas and enums
    utils       - Utilities (retry, logging)
    knowledge_base - RAG components (indexing, retrieval)
    teacher     - LLM teacher models (Ollama, OpenAI, Anthropic)
    validation  - Data quality validation
    scenarios   - Scenario generation
    pipeline    - End-to-end orchestration

Quick Start:
    from src.pipeline import SyntheticDataPipeline, PipelineConfig
    
    config = PipelineConfig(
        num_scenarios=100,
        teacher_model="llama3.1:8b",
        use_rag=True,
    )
    
    pipeline = SyntheticDataPipeline(config)
    result = pipeline.run()
"""

__version__ = "0.1.0"
__author__ = "Alireza Rashidi"

# Convenience imports for common usage
from .pipeline import SyntheticDataPipeline, PipelineConfig, PipelineResult
from .teacher import create_teacher, OllamaTeacher
from .knowledge_base import RAGFactory, RAGConfig, create_rag_system
from .scenarios import ScenarioGenerator, generate_scenarios
from .validation import validate_sample, ValidationResult
from .models import SyntheticSample, DialogueTurn, ClinicalSummary

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Pipeline
    "SyntheticDataPipeline",
    "PipelineConfig",
    "PipelineResult",
    # Teacher
    "create_teacher",
    "OllamaTeacher",
    # Knowledge Base
    "RAGFactory",
    "RAGConfig",
    "create_rag_system",
    # Scenarios
    "ScenarioGenerator",
    "generate_scenarios",
    # Validation
    "validate_sample",
    "ValidationResult",
    # Models
    "SyntheticSample",
    "DialogueTurn",
    "ClinicalSummary",
]