"""
Logging and Experiment Tracking Utilities

Provides unified logging with MLflow and Weights & Biases integration
for tracking synthetic data generation experiments.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from contextlib import contextmanager

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# Initialize rich console
console = Console()


# =============================================================================
# Logger Setup
# =============================================================================

def setup_logger(
    name: str = "ambient_scribe",
    level: Union[str, int] = "INFO",
    log_file: Optional[str] = None,
    use_rich: bool = True,
) -> logging.Logger:
    """
    Setup a configured logger with optional file output and rich formatting
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
        use_rich: Whether to use rich formatting for console output
        
    Returns:
        Configured logger instance
    """
    
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler with rich formatting
    if use_rich:
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
        )
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


# Create default logger
logger = setup_logger()


# =============================================================================
# Experiment Tracker
# =============================================================================

class ExperimentTracker:
    """
    Unified experiment tracking with MLflow and Weights & Biases
    
    Provides a single interface for logging metrics, parameters, and artifacts
    to both MLflow and W&B (if enabled).
    
    Example:
        tracker = ExperimentTracker(
            experiment_name="synthetic-data-gen",
            use_mlflow=True,
            use_wandb=True
        )
        
        with tracker.start_run(run_name="batch_001"):
            tracker.log_params({"model": "llama3.1:8b", "temperature": 0.7})
            tracker.log_metrics({"accuracy": 0.95, "samples": 100})
            tracker.log_artifact("output.json")
    """
    
    def __init__(
        self,
        experiment_name: str = "ambient-scribe-teacher",
        use_mlflow: bool = True,
        use_wandb: bool = False,
        mlflow_tracking_uri: Optional[str] = None,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize experiment tracker
        
        Args:
            experiment_name: Name of the experiment
            use_mlflow: Whether to use MLflow
            use_wandb: Whether to use Weights & Biases
            mlflow_tracking_uri: MLflow tracking server URI
            wandb_project: W&B project name
            wandb_entity: W&B entity (username or team)
            tags: Optional tags for the experiment
        """
        
        self.experiment_name = experiment_name
        self.use_mlflow = use_mlflow
        self.use_wandb = use_wandb
        self.tags = tags or {}
        
        self._mlflow_run = None
        self._wandb_run = None
        
        # Initialize MLflow
        if self.use_mlflow:
            try:
                import mlflow
                self.mlflow = mlflow
                
                if mlflow_tracking_uri:
                    mlflow.set_tracking_uri(mlflow_tracking_uri)
                
                mlflow.set_experiment(experiment_name)
                logger.info(f"MLflow initialized: experiment='{experiment_name}'")
                
            except ImportError:
                logger.warning("MLflow not installed. Disabling MLflow tracking.")
                self.use_mlflow = False
        
        # Initialize W&B
        if self.use_wandb:
            try:
                import wandb
                self.wandb = wandb
                
                self._wandb_project = wandb_project or experiment_name
                self._wandb_entity = wandb_entity
                logger.info(f"W&B initialized: project='{self._wandb_project}'")
                
            except ImportError:
                logger.warning("wandb not installed. Disabling W&B tracking.")
                self.use_wandb = False
    
    @contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
        run_tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
    ):
        """
        Context manager to start a tracking run
        
        Args:
            run_name: Name for this run
            run_tags: Additional tags for this run
            description: Run description
            
        Yields:
            Self for chaining
        """
        
        combined_tags = {**self.tags, **(run_tags or {})}
        
        try:
            # Start MLflow run
            if self.use_mlflow:
                self._mlflow_run = self.mlflow.start_run(
                    run_name=run_name,
                    tags=combined_tags,
                    description=description,
                )
                logger.info(f"MLflow run started: {run_name}")
            
            # Start W&B run
            if self.use_wandb:
                self._wandb_run = self.wandb.init(
                    project=self._wandb_project,
                    entity=self._wandb_entity,
                    name=run_name,
                    tags=list(combined_tags.values()),
                    notes=description,
                    reinit=True,
                )
                logger.info(f"W&B run started: {run_name}")
            
            yield self
            
        finally:
            # End runs
            if self.use_mlflow and self._mlflow_run:
                self.mlflow.end_run()
                self._mlflow_run = None
            
            if self.use_wandb and self._wandb_run:
                self.wandb.finish()
                self._wandb_run = None
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters
        
        Args:
            params: Dictionary of parameter names and values
        """
        
        if self.use_mlflow:
            # MLflow requires string values for params
            mlflow_params = {k: str(v) for k, v in params.items()}
            self.mlflow.log_params(mlflow_params)
        
        if self.use_wandb and self._wandb_run:
            self.wandb.config.update(params)
        
        logger.debug(f"Logged params: {list(params.keys())}")
    
    def log_metrics(
        self,
        metrics: Dict[str, Union[int, float]],
        step: Optional[int] = None,
    ):
        """
        Log metrics
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for time-series metrics
        """
        
        if self.use_mlflow:
            self.mlflow.log_metrics(metrics, step=step)
        
        if self.use_wandb and self._wandb_run:
            log_dict = metrics.copy()
            if step is not None:
                log_dict["step"] = step
            self.wandb.log(log_dict)
        
        logger.debug(f"Logged metrics: {metrics}")
    
    def log_metric(
        self,
        key: str,
        value: Union[int, float],
        step: Optional[int] = None,
    ):
        """Log a single metric"""
        self.log_metrics({key: value}, step=step)
    
    def log_artifact(
        self,
        local_path: str,
        artifact_path: Optional[str] = None,
    ):
        """
        Log an artifact (file or directory)
        
        Args:
            local_path: Path to local file or directory
            artifact_path: Optional path within artifact storage
        """
        
        if self.use_mlflow:
            if os.path.isdir(local_path):
                self.mlflow.log_artifacts(local_path, artifact_path)
            else:
                self.mlflow.log_artifact(local_path, artifact_path)
        
        if self.use_wandb and self._wandb_run:
            artifact = self.wandb.Artifact(
                name=Path(local_path).stem,
                type="output",
            )
            if os.path.isdir(local_path):
                artifact.add_dir(local_path)
            else:
                artifact.add_file(local_path)
            self._wandb_run.log_artifact(artifact)
        
        logger.debug(f"Logged artifact: {local_path}")
    
    def log_dict(
        self,
        data: Dict[str, Any],
        filename: str,
    ):
        """
        Log a dictionary as a JSON artifact
        
        Args:
            data: Dictionary to log
            filename: Name for the JSON file
        """
        
        if self.use_mlflow:
            self.mlflow.log_dict(data, filename)
        
        if self.use_wandb and self._wandb_run:
            # Save to temp file and log
            import tempfile
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.json',
                delete=False
            ) as f:
                json.dump(data, f, indent=2, default=str)
                temp_path = f.name
            
            self.log_artifact(temp_path)
            os.unlink(temp_path)
    
    def log_sample(
        self,
        sample: Dict[str, Any],
        sample_id: str,
        step: Optional[int] = None,
    ):
        """
        Log a synthetic sample with its metadata
        
        Args:
            sample: The synthetic sample dictionary
            sample_id: Unique identifier for the sample
            step: Optional step number
        """
        
        # Extract key metrics from sample
        metrics = {
            "dialogue_turns": len(sample.get("dialogue", [])),
        }
        
        # Add difficulty if present
        if "difficulty" in sample and sample["difficulty"]:
            metrics["difficulty_score"] = sample["difficulty"].get("difficulty_score", 0)
        
        # Add validation metrics if present
        if "validation" in sample and sample["validation"]:
            val = sample["validation"]
            metrics["validation_passed"] = 1 if val.get("status") == "passed" else 0
            if "rag_faithfulness" in val:
                metrics["rag_faithfulness"] = val["rag_faithfulness"]
        
        self.log_metrics(metrics, step=step)
        
        # Log full sample as artifact every N samples
        if step and step % 100 == 0:
            self.log_dict(sample, f"samples/sample_{sample_id}.json")
    
    def log_batch_summary(
        self,
        total_generated: int,
        total_valid: int,
        total_failed: int,
        generation_time_seconds: float,
        additional_metrics: Optional[Dict[str, Any]] = None,
    ):
        """
        Log summary metrics for a batch generation
        
        Args:
            total_generated: Total samples attempted
            total_valid: Successfully validated samples
            total_failed: Failed samples
            generation_time_seconds: Total generation time
            additional_metrics: Optional additional metrics
        """
        
        metrics = {
            "total_generated": total_generated,
            "total_valid": total_valid,
            "total_failed": total_failed,
            "success_rate": total_valid / total_generated if total_generated > 0 else 0,
            "generation_time_seconds": generation_time_seconds,
            "samples_per_minute": (total_generated / generation_time_seconds * 60) 
                                  if generation_time_seconds > 0 else 0,
        }
        
        if additional_metrics:
            metrics.update(additional_metrics)
        
        self.log_metrics(metrics)
        
        # Pretty print summary
        console.print("\n[bold green]Batch Generation Summary[/bold green]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in metrics.items():
            if isinstance(value, float):
                table.add_row(key, f"{value:.4f}")
            else:
                table.add_row(key, str(value))
        
        console.print(table)


# =============================================================================
# Progress Tracking
# =============================================================================

def create_progress_bar(
    total: int,
    description: str = "Processing",
) -> Progress:
    """
    Create a rich progress bar for batch processing
    
    Args:
        total: Total number of items
        description: Description text
        
    Returns:
        Configured Progress instance
    """
    
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
    )


@contextmanager
def progress_context(
    total: int,
    description: str = "Processing",
):
    """
    Context manager for progress tracking
    
    Example:
        with progress_context(100, "Generating samples") as (progress, task):
            for i in range(100):
                do_work()
                progress.update(task, advance=1)
    """
    
    progress = create_progress_bar(total, description)
    
    with progress:
        task = progress.add_task(description, total=total)
        yield progress, task


# =============================================================================
# Utility Functions
# =============================================================================

def print_header(title: str, width: int = 60):
    """Print a formatted header"""
    console.print(f"\n[bold blue]{'=' * width}[/bold blue]")
    console.print(f"[bold white]{title.center(width)}[/bold white]")
    console.print(f"[bold blue]{'=' * width}[/bold blue]\n")


def print_success(message: str):
    """Print a success message"""
    console.print(f"[bold green]✓[/bold green] {message}")


def print_warning(message: str):
    """Print a warning message"""
    console.print(f"[bold yellow]⚠[/bold yellow] {message}")


def print_error(message: str):
    """Print an error message"""
    console.print(f"[bold red]✗[/bold red] {message}")


def print_info(message: str):
    """Print an info message"""
    console.print(f"[bold blue]ℹ[/bold blue] {message}")


def print_config(config: Dict[str, Any], title: str = "Configuration"):
    """Print configuration in a formatted table"""
    
    print_header(title)
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in config.items():
        table.add_row(str(key), str(value))
    
    console.print(table)


def log_generation_start(
    model_name: str,
    num_scenarios: int,
    use_rag: bool,
    output_path: str,
):
    """Log the start of a generation run"""
    
    print_header("Synthetic Data Generation")
    
    config = {
        "Model": model_name,
        "Scenarios": num_scenarios,
        "RAG Enabled": use_rag,
        "Output": output_path,
        "Started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    table = Table(show_header=False)
    table.add_column("", style="cyan")
    table.add_column("", style="green")
    
    for key, value in config.items():
        table.add_row(key, str(value))
    
    console.print(table)
    console.print()


if __name__ == "__main__":
    # Test logging utilities
    print_header("Testing Logging Utilities")
    
    # Test logger
    test_logger = setup_logger("test", level="DEBUG")
    test_logger.debug("This is a debug message")
    test_logger.info("This is an info message")
    test_logger.warning("This is a warning message")
    
    # Test print functions
    print_success("Operation completed successfully")
    print_warning("This is a warning")
    print_error("This is an error")
    print_info("This is informational")
    
    # Test config printing
    print_config({
        "model": "llama3.1:8b",
        "temperature": 0.7,
        "max_tokens": 2048,
        "rag_enabled": True,
    })
    
    # Test progress bar
    print_info("Testing progress bar...")
    with progress_context(10, "Processing items") as (progress, task):
        import time
        for i in range(10):
            time.sleep(0.1)
            progress.update(task, advance=1)
    
    print_success("All logging tests passed!")