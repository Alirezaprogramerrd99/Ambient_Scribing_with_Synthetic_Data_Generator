"""
Student Model Module - Fine-tuning and Inference

This module implements the Student side of the Teacher-Student pipeline:
1. Data Preparation  - Filter and format synthetic data for instruction tuning
2. Training          - QLoRA fine-tuning with Unsloth
3. Export            - Convert to GGUF and register with Ollama
4. Inference         - RAG-augmented generation with the fine-tuned model
5. Evaluation        - Multi-dimensional evaluation with LLM-as-a-Judge

Author: Alireza Rashidi
MSc Project: Trustworthy SLMs for Ambient Clinical Scribing
"""
import patch_torch


from .data_prep import TrainingDataPreparator, DataPrepConfig
from .trainer import StudentTrainer, TrainingConfig
from .exporter import ModelExporter, ExportConfig
from .inference_fixed import ClinicalScribeInference, InferenceConfig
from .evaluator import StudentEvaluator, EvaluationConfig, LLMJudge


__all__ = [
    "TrainingDataPreparator",
    "DataPrepConfig",
    "StudentTrainer",
    "TrainingConfig",
    "ModelExporter",
    "ExportConfig",
    "ClinicalScribeInference",
    "InferenceConfig",
    "StudentEvaluator",
    "EvaluationConfig",
    "LLMJudge",
]
