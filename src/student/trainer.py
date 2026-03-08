"""
Student Model Trainer - QLoRA Fine-Tuning with Unsloth

Fine-tunes SLMs (Phi-3.5, Qwen2.5, Llama-3.2, etc.) using QLoRA (4-bit)
via Unsloth for clinical scribing tasks. Supports curriculum learning using
difficulty metadata from the teacher pipeline.

Prerequisites:
    pip install unsloth
    pip install trl datasets transformers peft accelerate bitsandbytes

Author: Alireza Rashidi
MSc Project: Trustworthy SLMs for Ambient Clinical Scribing
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for QLoRA fine-tuning."""
    
    # Model
    base_model: str = "unsloth/Phi-3.5-mini-instruct"
    max_seq_length: int = 4096
    load_in_4bit: bool = True
    dtype: Optional[str] = None  # Auto-detect (bf16 on Ampere+, fp16 otherwise)
    
    # LoRA
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    
    # Training
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4  # Effective batch = 16
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Evaluation & Saving
    eval_steps: int = 50
    save_steps: int = 100
    logging_steps: int = 10
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Optimisation
    gradient_checkpointing: bool = True
    optim: str = "adamw_8bit"
    
    # Data
    training_data_dir: str = "./data/training_data"
    output_dir: str = "./checkpoints/phi35_clinical_scribe"
    
    # Curriculum learning
    use_curriculum: bool = False
    
    # Tracking
    report_to: str = "none"  # "mlflow", "wandb", or "none"
    run_name: Optional[str] = None
    
    # Reproducibility
    seed: int = 42


# =============================================================================
# Trainer Class
# =============================================================================

class StudentTrainer:
    """
    Fine-tunes an SLM for clinical scribing using Unsloth + QLoRA.
    
    Example:
        config = TrainingConfig(
            base_model="unsloth/Qwen2.5-3B-Instruct",
            training_data_dir="./data/training_data_qwen",
            output_dir="./checkpoints/qwen25_clinical_scribe",
            num_epochs=3,
        )
        trainer = StudentTrainer(config)
        trainer.train()
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self._training_start = None
    
    def train(self) -> Dict[str, Any]:
        """
        Execute the full training pipeline.
        
        Returns:
            Dictionary with training results and metrics.
        """
        logger.info("=" * 60)
        logger.info("Starting Student Model Training")
        logger.info("=" * 60)
        
        self._training_start = time.time()
        
        # Step 1: Load model
        self._load_model()
        
        # Step 2: Load data
        train_dataset, eval_dataset = self._load_datasets()
        
        # Step 3: Configure trainer
        self._setup_trainer(train_dataset, eval_dataset)
        
        # Step 4: Train
        logger.info("Starting training...")
        train_result = self.trainer.train()
        
        # Step 5: Save LoRA adapters
        final_dir = Path(self.config.output_dir) / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(final_dir))
        self.tokenizer.save_pretrained(str(final_dir))
        logger.info(f"LoRA adapters saved to {final_dir}")
        
        # Step 5b: Merge adapters into base model for direct inference
        # This produces a standalone model that inference_fixed.py can load
        # without needing to download the base model from HuggingFace.
        merged_dir = Path(self.config.output_dir) / "hf_merged"
        logger.info(f"Merging LoRA adapters into base model → {merged_dir}")
        try:
            self.model.save_pretrained_merged(
                str(merged_dir),
                self.tokenizer,
                save_method="merged_16bit",
            )
            logger.info(f"Merged model saved to {merged_dir}")
        except Exception as e:
            logger.error(f"Merge failed: {e}")
            logger.error("You can merge manually later. The LoRA adapters in /final are intact.")
        
        # Step 6: Save training results
        elapsed = time.time() - self._training_start
        results = {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", elapsed),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
            "total_steps": train_result.global_step,
            "num_epochs": self.config.num_epochs,
            "model": self.config.base_model,
            "lora_r": self.config.lora_r,
            "lora_alpha": self.config.lora_alpha,
            "learning_rate": self.config.learning_rate,
            "effective_batch_size": (
                self.config.per_device_train_batch_size 
                * self.config.gradient_accumulation_steps
            ),
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset) if eval_dataset else 0,
            "elapsed_seconds": elapsed,
            "timestamp": datetime.now().isoformat(),
        }
        
        results_path = Path(self.config.output_dir) / "training_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\nTraining complete in {elapsed / 60:.1f} minutes")
        logger.info(f"  Final train loss: {results['train_loss']:.4f}")
        logger.info(f"  Total steps: {results['total_steps']}")
        
        return results
    
    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------
    
    def _load_model(self):
        """Load base model with Unsloth and add LoRA adapters."""
        try:
            import os
            os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"  # Bypass unstable Windows Triton compiler import
            os.environ["TORCH_COMPILE_DISABLE"] = "1"    # Bypass PyTorch's internal compiler
            os.environ["TORCHDYNAMO_DISABLE"] = "1"      # Force PyTorch into safe 'eager' mode
            import torch
            import torch._inductor.config                # Pre-load the missing PyTorch module

            # # --- PYTORCH 2.4 COMPATIBILITY PATCH ---
            # # torchao expects PyTorch 2.6+ which has int1-int7.
            # # We mock them to prevent the import crash.
            # for i in range (1, 8):
            #     if not hasattr(torch, f"int{i}"):
            #         setattr(torch, f"int{i}", torch.int8)

            from unsloth import FastLanguageModel

        except ImportError:
            raise ImportError(
                "Unsloth is not installed. Install with:\n"
                "  pip install unsloth\n"
                "See: https://github.com/unslothai/unsloth"
            )
        
        logger.info(f"Loading model: {self.config.base_model}")
        logger.info(f"  4-bit quantisation: {self.config.load_in_4bit}")
        logger.info(f"  Max sequence length: {self.config.max_seq_length}")
        
        # Load model with Unsloth (handles quantisation automatically)
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.base_model,
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=self.config.load_in_4bit,
            dtype=self.config.dtype,
        )
        
        # Add LoRA adapters
        logger.info(f"  Adding LoRA adapters (r={self.config.lora_r}, alpha={self.config.lora_alpha})")
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
            use_gradient_checkpointing="unsloth" if self.config.gradient_checkpointing else False,
            random_state=self.config.seed,
        )
        
        # Log trainable parameters
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"  Trainable parameters: {trainable:,} / {total:,} ({trainable/total:.2%})")
    
    def _load_datasets(self):
        """Load train and validation datasets from JSONL files."""
        from datasets import load_dataset
        
        data_dir = Path(self.config.training_data_dir)
        
        train_file = data_dir / "train.jsonl"
        val_file = data_dir / "val.jsonl"
        
        if not train_file.exists():
            raise FileNotFoundError(
                f"Training data not found at {train_file}. "
                "Run data_prep.py first."
            )
        
        logger.info(f"Loading training data from {data_dir}")
        
        train_dataset = load_dataset("json", data_files=str(train_file), split="train")
        
        eval_dataset = None
        if val_file.exists():
            eval_dataset = load_dataset("json", data_files=str(val_file), split="train")
        
        logger.info(f"  Train: {len(train_dataset)} examples")
        if eval_dataset:
            logger.info(f"  Val:   {len(eval_dataset)} examples")
        
        # Apply curriculum learning if enabled
        if self.config.use_curriculum:
            train_dataset = self._apply_curriculum(train_dataset)
        
        return train_dataset, eval_dataset
    
    def _apply_curriculum(self, dataset):
        """
        Reorder training data by difficulty for curriculum learning.
        
        Orders: easy (1-4) -> medium (5-7) -> hard (8-10).
        Within each group, order is shuffled.
        """
        import random as rng
        rng.seed(self.config.seed)
        
        logger.info("Applying curriculum learning (easy → medium → hard)")
        
        # Convert to list for sorting
        examples = list(dataset)
        
        easy = [e for e in examples if e.get("difficulty", 5) <= 4]
        medium = [e for e in examples if 5 <= e.get("difficulty", 5) <= 7]
        hard = [e for e in examples if e.get("difficulty", 5) >= 8]
        
        rng.shuffle(easy)
        rng.shuffle(medium)
        rng.shuffle(hard)
        
        ordered = easy + medium + hard
        
        logger.info(f"  Curriculum: {len(easy)} easy, {len(medium)} medium, {len(hard)} hard")
        
        from datasets import Dataset
        return Dataset.from_list(ordered)
    
    def _setup_trainer(self, train_dataset, eval_dataset):
        """Configure the SFTTrainer."""
        from trl import SFTTrainer, SFTConfig
        
        # Derive short model name for run naming (e.g. "phi35", "qwen25", "llama32")
        _model_short = Path(self.config.base_model).name.lower()
        _model_short = _model_short.replace("-instruct", "").replace("-", "").replace(".", "")[:10]
        
        run_name = self.config.run_name or (
            f"{_model_short}-clinical-r{self.config.lora_r}-"
            f"lr{self.config.learning_rate}-"
            f"e{self.config.num_epochs}"
        )
        
        sft_config = SFTConfig(
            output_dir=self.config.output_dir,
            
            # Training hyperparameters
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            lr_scheduler_type=self.config.lr_scheduler_type,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            optim=self.config.optim,
            
            # Sequence
            max_seq_length=self.config.max_seq_length,
            dataset_text_field="text",
            packing=False,  # Don't pack sequences - clinical examples need boundaries
            
            # Evaluation & Saving
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=self.config.eval_steps if eval_dataset else None,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end if eval_dataset else False,
            metric_for_best_model=self.config.metric_for_best_model if eval_dataset else None,
            greater_is_better=self.config.greater_is_better if eval_dataset else None,
            
            # Logging
            logging_steps=self.config.logging_steps,
            report_to=self.config.report_to,
            run_name=run_name,
            
            # Reproducibility
            seed=self.config.seed,
            data_seed=self.config.seed,
            
            # Performance
            fp16=not self._supports_bf16(),
            bf16=self._supports_bf16(),
        )
        
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=sft_config,
        )
        
        logger.info(f"Trainer configured:")
        logger.info(f"  Effective batch size: {self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps}")
        logger.info(f"  Learning rate: {self.config.learning_rate}")
        logger.info(f"  Epochs: {self.config.num_epochs}")
        logger.info(f"  Scheduler: {self.config.lr_scheduler_type}")
    
    def _supports_bf16(self) -> bool:
        """Check if GPU supports bf16 (Ampere+ architecture)."""
        try:
            import torch
            if torch.cuda.is_available():
                capability = torch.cuda.get_device_capability()
                return capability[0] >= 8  # Ampere is SM 8.0+
        except Exception:
            pass
        return False


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    
    parser = argparse.ArgumentParser(description="Fine-tune an SLM for clinical scribing")
    parser.add_argument("--data-dir", default="./data/training_data", help="Training data directory")
    parser.add_argument("--output-dir", default="./checkpoints/phi35_clinical_scribe", help="Checkpoint output")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--curriculum", action="store_true", help="Enable curriculum learning")
    parser.add_argument("--report-to", default="none", choices=["none", "mlflow", "wandb"])
    parser.add_argument("--model", default="unsloth/Phi-3.5-mini-instruct", help="Base model")
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        base_model=args.model,
        training_data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        per_device_train_batch_size=args.batch_size,
        use_curriculum=args.curriculum,
        report_to=args.report_to,
    )
    
    trainer = StudentTrainer(config)
    results = trainer.train()
    
    print(f"\n✓ Training complete!")
    print(f"  Loss: {results['train_loss']:.4f}")
    print(f"  Time: {results['elapsed_seconds'] / 60:.1f} min")
    print(f"  Model saved to: {config.output_dir}/final")
