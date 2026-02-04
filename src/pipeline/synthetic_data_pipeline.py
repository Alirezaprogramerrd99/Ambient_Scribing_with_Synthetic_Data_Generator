"""
Synthetic Data Generation Pipeline

End-to-end orchestration for generating synthetic clinical dialogue-summary pairs.
Integrates all components: scenarios, RAG, teacher model, validation, and export.

"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from src.config import get_settings, Settings
from src.models import (
    SyntheticSample,
    SyntheticDataBatch,
    DifficultyLevel,
    ClinicalSpecialty,
    OutputFormat,
)
from src.knowledge_base import (
    RAGFactory,
    RAGConfig,
    RAGBackend,
    BaseRetriever,
)
from src.teacher import (
    BaseTeacher,
    create_teacher,
    GenerationConfig,
    BatchGenerationResult,
)
from src.validation import (
    validate_sample,
    validate_batch_comprehensive,
    ValidationResult,
    ValidationStatus,
)
from src.scenarios import (
    ScenarioGenerator,
    ClinicalScenario,
    generate_scenarios,
    load_scenarios,
    save_scenarios,
)
from src.utils import (
    ExperimentTracker,
    progress_context,
    print_header,
    print_success,
    print_warning,
    print_error,
    print_info,
    logger,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
pipeline_logger = logging.getLogger(__name__)


# =============================================================================
# Pipeline Configuration
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the synthetic data generation pipeline"""
    
    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("./data/synthetic_output"))
    output_format: OutputFormat = OutputFormat.JSONL
    experiment_name: str = "synthetic_data_generation"
    
    # Scenario settings
    num_scenarios: int = 100
    scenario_file: Optional[Path] = None  # Use existing scenarios file
    specialties: Optional[List[ClinicalSpecialty]] = None
    complexity_distribution: Optional[Dict[DifficultyLevel, float]] = None
    scenario_seed: Optional[int] = 42
    
    # RAG settings
    use_rag: bool = True
    rag_backend: RAGBackend = RAGBackend.MANUAL
    knowledge_base_path: Optional[Path] = None
    rag_top_k: int = 5
    
    # Teacher settings
    teacher_provider: str = "ollama"
    teacher_model: str = "llama3.1:8b"
    temperature: float = 0.7
    max_tokens: int = 4096
    
    # Validation settings
    enable_validation: bool = True
    enable_clinical_validation: bool = True
    enable_rag_validation: bool = True
    filter_invalid: bool = True  # Only keep valid samples
    
    # Processing settings
    batch_size: int = 10
    save_interval: int = 10
    max_retries: int = 3
    
    # Experiment tracking
    use_mlflow: bool = True
    use_wandb: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "output_format": self.output_format.value,
            "num_scenarios": self.num_scenarios,
            "use_rag": self.use_rag,
            "rag_backend": self.rag_backend.value,
            "teacher_provider": self.teacher_provider,
            "teacher_model": self.teacher_model,
            "temperature": self.temperature,
            "enable_validation": self.enable_validation,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Create config from dictionary"""
        # Handle path conversions
        if "output_dir" in data:
            data["output_dir"] = Path(data["output_dir"])
        if "scenario_file" in data and data["scenario_file"]:
            data["scenario_file"] = Path(data["scenario_file"])
        if "knowledge_base_path" in data and data["knowledge_base_path"]:
            data["knowledge_base_path"] = Path(data["knowledge_base_path"])
        
        # Handle enum conversions
        if "output_format" in data:
            data["output_format"] = OutputFormat(data["output_format"])
        if "rag_backend" in data:
            data["rag_backend"] = RAGBackend(data["rag_backend"])
        
        return cls(**data)


# =============================================================================
# Pipeline Results
# =============================================================================

@dataclass
class PipelineResult:
    """Result of pipeline execution"""
    
    # Counts
    total_scenarios: int = 0
    total_generated: int = 0
    total_valid: int = 0
    total_failed: int = 0
    
    # Samples
    samples: List[SyntheticSample] = field(default_factory=list)
    failed_scenarios: List[str] = field(default_factory=list)
    
    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_seconds: float = 0.0
    
    # Metrics
    validation_stats: Dict[str, Any] = field(default_factory=dict)
    
    # Paths
    output_path: Optional[Path] = None
    
    @property
    def success_rate(self) -> float:
        if self.total_scenarios == 0:
            return 0.0
        return self.total_valid / self.total_scenarios
    
    @property
    def samples_per_minute(self) -> float:
        if self.total_seconds == 0:
            return 0.0
        return (self.total_valid / self.total_seconds) * 60
    
    def to_summary(self) -> Dict[str, Any]:
        return {
            "total_scenarios": self.total_scenarios,
            "total_generated": self.total_generated,
            "total_valid": self.total_valid,
            "total_failed": self.total_failed,
            "success_rate": f"{self.success_rate:.1%}",
            "samples_per_minute": f"{self.samples_per_minute:.2f}",
            "total_time": f"{self.total_seconds:.1f}s",
            "output_path": str(self.output_path) if self.output_path else None,
        }


# =============================================================================
# Main Pipeline
# =============================================================================

class SyntheticDataPipeline:
    """
    End-to-end pipeline for synthetic clinical data generation
    
    Orchestrates:
    1. Scenario generation or loading
    2. Knowledge base setup and RAG retrieval
    3. Teacher model generation
    4. Validation and filtering
    5. Export and experiment tracking
    
    Example:
        config = PipelineConfig(
            num_scenarios=100,
            teacher_model="llama3.1:8b",
            use_rag=True,
        )
        
        pipeline = SyntheticDataPipeline(config)
        result = pipeline.run()
        
        print(f"Generated {result.total_valid} valid samples")
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize pipeline
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        
        # Components (initialized lazily)
        self._retriever: Optional[BaseRetriever] = None
        self._teacher: Optional[BaseTeacher] = None
        self._tracker: Optional[ExperimentTracker] = None
        
        # State
        self._initialized = False
    
    def initialize(self):
        """Initialize all pipeline components"""
        if self._initialized:
            return
        
        print_header("Initializing Pipeline")
        
        # Setup output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        print_success(f"Output directory: {self.config.output_dir}")
        
        # Initialize RAG if enabled
        if self.config.use_rag:
            self._initialize_rag()
        
        # Initialize teacher model
        self._initialize_teacher()
        
        # Initialize experiment tracking
        if self.config.use_mlflow or self.config.use_wandb:
            self._initialize_tracking()
        
        self._initialized = True
        print_success("Pipeline initialized")
    
    def _initialize_rag(self):
        """Initialize RAG retriever"""
        print_info("Setting up RAG retriever...")
        
        rag_config = RAGConfig(
            backend=self.config.rag_backend,
            persist_dir=str(self.config.output_dir / "vector_store"),
        )
        
        rag_factory = RAGFactory(rag_config)
        
        # Build knowledge base if path provided
        if self.config.knowledge_base_path:
            if self.config.knowledge_base_path.exists():
                print_info(f"Building knowledge base from: {self.config.knowledge_base_path}")
                rag_factory.build_knowledge_base(self.config.knowledge_base_path)
                print_success("Knowledge base built")
            else:
                print_warning(f"Knowledge base path not found: {self.config.knowledge_base_path}")
        
        try:
            self._retriever = rag_factory.get_retriever()
            print_success(f"RAG retriever ready ({self.config.rag_backend.value})")
        except Exception as e:
            print_warning(f"RAG initialization failed: {e}. Continuing without RAG.")
            self._retriever = None
    
    def _initialize_teacher(self):
        """Initialize teacher model"""
        print_info(f"Initializing teacher: {self.config.teacher_provider}/{self.config.teacher_model}")
        
        self._teacher = create_teacher(
            provider=self.config.teacher_provider,
            model_name=self.config.teacher_model,
            retriever=self._retriever,
            temperature=self.config.temperature,
            use_rag=self.config.use_rag and self._retriever is not None,
            max_tokens=self.config.max_tokens,
            max_retries=self.config.max_retries,
        )
        
        print_success(f"Teacher ready: {self._teacher.model_name}")
    
    def _initialize_tracking(self):
        """Initialize experiment tracking"""
        self._tracker = ExperimentTracker(
            experiment_name=self.config.experiment_name,
            use_mlflow=self.config.use_mlflow,
            use_wandb=self.config.use_wandb,
        )
        print_success("Experiment tracking initialized")
    
    def run(self) -> PipelineResult:
        """
        Run the complete pipeline
        
        Returns:
            PipelineResult with generated samples and statistics
        """
        # Initialize if needed
        self.initialize()
        
        print_header("Running Synthetic Data Pipeline")
        
        result = PipelineResult()
        result.start_time = datetime.now()
        
        # Start tracking run
        tracking_context = None
        if self._tracker:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            tracking_context = self._tracker.start_run(run_name=run_name)
            tracking_context.__enter__()
            self._tracker.log_params(self.config.to_dict())
        
        try:
            # Step 1: Generate or load scenarios
            scenarios = self._get_scenarios()
            result.total_scenarios = len(scenarios)
            print_info(f"Processing {len(scenarios)} scenarios")
            
            # Step 2: Generate samples
            samples, failed = self._generate_samples(scenarios)
            result.total_generated = len(samples)
            result.failed_scenarios = failed
            
            # Step 3: Validate samples
            if self.config.enable_validation:
                samples, validation_stats = self._validate_samples(samples)
                result.validation_stats = validation_stats
            
            result.total_valid = len(samples)
            result.total_failed = len(failed) + (result.total_generated - result.total_valid)
            result.samples = samples
            
            # Log final metrics (timing will be 0 here, but that's ok)
            if self._tracker:
                self._tracker.log_batch_summary(
                    total_generated=result.total_generated,
                    total_valid=result.total_valid,
                    total_failed=result.total_failed,
                    generation_time_seconds=result.total_seconds,
                )
            
        finally:
            result.end_time = datetime.now()
            result.total_seconds = (result.end_time - result.start_time).total_seconds()
            
            if tracking_context:
                tracking_context.__exit__(None, None, None)
        
        # Step 5: Export results (AFTER timing is calculated)
        output_path = self._export_results(result.samples, result)
        result.output_path = output_path
        
        # Print summary
        self._print_summary(result)
        
        return result
    
    def _get_scenarios(self) -> List[str]:
        """Get scenarios (load from file or generate)"""
        
        # Load from file if specified
        if self.config.scenario_file and self.config.scenario_file.exists():
            print_info(f"Loading scenarios from: {self.config.scenario_file}")
            return load_scenarios(self.config.scenario_file)
        
        # Generate new scenarios
        print_info(f"Generating {self.config.num_scenarios} scenarios...")
        
        generator = ScenarioGenerator(seed=self.config.scenario_seed)
        
        scenario_objects = generator.generate_batch(
            count=self.config.num_scenarios,
            specialties=self.config.specialties,
            complexity_distribution=self.config.complexity_distribution,
        )
        
        # Save generated scenarios
        scenario_file = self.config.output_dir / "scenarios.jsonl"
        save_scenarios(scenario_objects, scenario_file)
        print_success(f"Scenarios saved to: {scenario_file}")
        
        return [s.to_text() for s in scenario_objects]
    
    def _generate_samples(
        self,
        scenarios: List[str],
    ) -> tuple[List[SyntheticSample], List[str]]:
        """Generate samples from scenarios"""
        
        samples = []
        failed = []
        
        print_info(f"Generating samples with {self._teacher.model_name}...")
        
        with progress_context(len(scenarios), "Generating") as (progress, task):
            for i, scenario in enumerate(scenarios):
                try:
                    result = self._teacher.generate(scenario)
                    
                    if result.success:
                        samples.append(result.sample)
                        
                        # Log to tracker
                        if self._tracker:
                            self._tracker.log_metric("samples_generated", len(samples), step=i)
                    else:
                        failed.append(scenario)
                        pipeline_logger.warning(f"Generation failed for scenario {i}: {result.error}")
                    
                except Exception as e:
                    failed.append(scenario)
                    pipeline_logger.error(f"Error generating sample {i}: {e}")
                
                progress.update(task, advance=1)
                
                # Save intermediate results
                if (i + 1) % self.config.save_interval == 0:
                    self._save_intermediate(samples, "intermediate")
        
        return samples, failed
    
    def _validate_samples(
        self,
        samples: List[SyntheticSample],
    ) -> tuple[List[SyntheticSample], Dict[str, Any]]:
        """Validate generated samples"""
        
        print_info("Validating samples...")
        
        valid_samples = []
        
        stats = {
            "total": len(samples),
            "passed": 0,
            "failed": 0,
            "warnings": 0,
        }
        
        with progress_context(len(samples), "Validating") as (progress, task):
            for sample in samples:
                result = validate_sample(
                    sample,
                    clinical_validation=self.config.enable_clinical_validation,
                    rag_validation=self.config.enable_rag_validation,
                )
                
                # Update sample with validation result
                sample.validation = result
                
                if result.status == ValidationStatus.PASSED:
                    stats["passed"] += 1
                    valid_samples.append(sample)
                elif result.status == ValidationStatus.WARNING:
                    stats["warnings"] += 1
                    # Warnings should NOT filter out samples - they're still valid
                    valid_samples.append(sample)
                else:
                    stats["failed"] += 1
                    if not self.config.filter_invalid:
                        valid_samples.append(sample)
                
                progress.update(task, advance=1)
        
        print_success(f"Validation complete: {stats['passed']} passed, "
                     f"{stats['warnings']} warnings, {stats['failed']} failed")
        
        return valid_samples, stats
    
    def _export_results(
        self,
        samples: List[SyntheticSample],
        result: PipelineResult,
    ) -> Path:
        """Export results to files"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine output path
        if self.config.output_format == OutputFormat.JSONL:
            output_path = self.config.output_dir / f"synthetic_data_{timestamp}.jsonl"
            self._export_jsonl(samples, output_path)
        elif self.config.output_format == OutputFormat.JSON:
            output_path = self.config.output_dir / f"synthetic_data_{timestamp}.json"
            self._export_json(samples, output_path)
        elif self.config.output_format == OutputFormat.CSV:
            output_path = self.config.output_dir / f"synthetic_data_{timestamp}.csv"
            self._export_csv(samples, output_path)
        else:
            output_path = self.config.output_dir / f"synthetic_data_{timestamp}.jsonl"
            self._export_jsonl(samples, output_path)
        
        # Save summary
        summary_path = self.config.output_dir / f"summary_{timestamp}.json"
        with open(summary_path, "w") as f:
            json.dump(result.to_summary(), f, indent=2)
        
        print_success(f"Results exported to: {output_path}")
        return output_path
    
    def _export_jsonl(self, samples: List[SyntheticSample], path: Path):
        """Export as JSONL"""
        with open(path, "w") as f:
            for sample in samples:
                f.write(sample.model_dump_json() + "\n")
    
    def _export_json(self, samples: List[SyntheticSample], path: Path):
        """Export as JSON"""
        with open(path, "w") as f:
            json.dump([sample.model_dump() for sample in samples], f, indent=2, default=str)
    
    def _export_csv(self, samples: List[SyntheticSample], path: Path):
        """Export as CSV (simplified format)"""
        import csv
        
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "id", "dialogue", "chief_complaint", "assessment", "plan",
                "specialty", "difficulty"
            ])
            
            for sample in samples:
                writer.writerow([
                    sample.id,
                    sample.dialogue_text[:500],  # Truncate for CSV
                    sample.summary.chief_complaint,
                    sample.summary.assessment,
                    sample.summary.plan[:500],
                    sample.scenario.specialty.value,
                    sample.difficulty.difficulty_level.value if sample.difficulty else "unknown",
                ])
    
    def _save_intermediate(self, samples: List[SyntheticSample], prefix: str):
        """Save intermediate results"""
        path = self.config.output_dir / f"{prefix}_{len(samples)}.jsonl"
        self._export_jsonl(samples, path)
    
    def _print_summary(self, result: PipelineResult):
        """Print pipeline summary"""
        print_header("Pipeline Complete")
        
        print(f"  Total scenarios:     {result.total_scenarios}")
        print(f"  Generated samples:   {result.total_generated}")
        print(f"  Valid samples:       {result.total_valid}")
        print(f"  Failed:              {result.total_failed}")
        print(f"  Success rate:        {result.success_rate:.1%}")
        print(f"  Total time:          {result.total_seconds:.1f}s")
        print(f"  Samples/minute:      {result.samples_per_minute:.2f}")
        print(f"  Output:              {result.output_path}")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line interface for the pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate synthetic clinical dialogue-summary pairs"
    )
    
    # Required arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/synthetic_output",
        help="Output directory for generated data",
    )
    
    # Scenario arguments
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=100,
        help="Number of scenarios to generate",
    )
    parser.add_argument(
        "--scenario-file",
        type=str,
        help="Path to existing scenarios file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    # Model arguments
    parser.add_argument(
        "--provider",
        type=str,
        default="ollama",
        choices=["ollama", "openai", "anthropic"],
        help="LLM provider",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.1:8b",
        help="Model name",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature",
    )
    
    # RAG arguments
    parser.add_argument(
        "--use-rag",
        action="store_true",
        default=True,
        help="Enable RAG retrieval",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG retrieval",
    )
    parser.add_argument(
        "--knowledge-base",
        type=str,
        help="Path to knowledge base documents",
    )
    
    # Validation arguments
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Disable validation",
    )
    parser.add_argument(
        "--keep-invalid",
        action="store_true",
        help="Keep invalid samples in output",
    )
    
    # Output format
    parser.add_argument(
        "--format",
        type=str,
        default="jsonl",
        choices=["jsonl", "json", "csv"],
        help="Output format",
    )
    
    args = parser.parse_args()
    
    # Build config
    config = PipelineConfig(
        output_dir=Path(args.output_dir),
        output_format=OutputFormat(args.format),
        num_scenarios=args.num_scenarios,
        scenario_file=Path(args.scenario_file) if args.scenario_file else None,
        scenario_seed=args.seed,
        use_rag=not args.no_rag,
        knowledge_base_path=Path(args.knowledge_base) if args.knowledge_base else None,
        teacher_provider=args.provider,
        teacher_model=args.model,
        temperature=args.temperature,
        enable_validation=not args.no_validation,
        filter_invalid=not args.keep_invalid,
    )
    
    # Run pipeline
    pipeline = SyntheticDataPipeline(config)
    result = pipeline.run()
    
    # Exit code based on success
    exit(0 if result.success_rate > 0.5 else 1)


if __name__ == "__main__":
    main()