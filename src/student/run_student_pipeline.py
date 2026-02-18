"""
Student Pipeline Runner

Complete end-to-end runner for the student model pipeline:
    Step 1: Data Preparation
    Step 2: Fine-Tuning
    Step 3: Export to Ollama
    Step 4: Inference testing
    Step 5: Full Evaluation with LLM-as-a-Judge

Each step can be run independently via CLI flags.

Usage:
    # Run everything:
    python run_student_pipeline.py --all

    # Run individual steps:
    python run_student_pipeline.py --prepare-data
    python run_student_pipeline.py --train
    python run_student_pipeline.py --export
    python run_student_pipeline.py --test-inference
    python run_student_pipeline.py --evaluate

Author: Alireza Rashidi
MSc Project: Trustworthy SLMs for Ambient Clinical Scribing
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Step 1: Data Preparation
# =============================================================================

def step_prepare_data(args):
    """Prepare training data from synthetic teacher pipeline output."""
    from src.student.data_prep import TrainingDataPreparator, DataPrepConfig
    
    config = DataPrepConfig(
        raw_data_dirs=args.data_dirs,
        output_dir=args.training_data_dir,
        rag_context_ratio=args.rag_ratio,
        min_dialogue_turns=args.min_turns,
        seed=args.seed,
    )
    
    prep = TrainingDataPreparator(config)
    stats = prep.run()
    
    print(f"\n{'='*60}")
    print(f"STEP 1 COMPLETE: Data Preparation")
    print(f"{'='*60}")
    print(f"  Train: {stats.get('train_count', 0)}")
    print(f"  Val:   {stats.get('val_count', 0)}")
    print(f"  Test:  {stats.get('test_count', 0)}")
    print(f"  Output: {args.training_data_dir}")
    
    return stats


# =============================================================================
# Step 2: Training
# =============================================================================

def step_train(args):
    """Fine-tune Phi-3.5-mini with QLoRA via Unsloth."""
    from src.student.trainer import StudentTrainer, TrainingConfig
    
    config = TrainingConfig(
        base_model=args.base_model,
        training_data_dir=args.training_data_dir,
        output_dir=args.checkpoint_dir,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        per_device_train_batch_size=args.batch_size,
        use_curriculum=args.curriculum,
        report_to=args.report_to,
        max_seq_length=args.max_seq_length,
    )
    
    trainer = StudentTrainer(config)
    results = trainer.train()
    
    print(f"\n{'='*60}")
    print(f"STEP 2 COMPLETE: Fine-Tuning")
    print(f"{'='*60}")
    print(f"  Loss:       {results['train_loss']:.4f}")
    print(f"  Steps:      {results['total_steps']}")
    print(f"  Time:       {results['elapsed_seconds'] / 60:.1f} min")
    print(f"  Checkpoint: {args.checkpoint_dir}/final")
    
    return results


# =============================================================================
# Step 3: Export to Ollama
# =============================================================================

def step_export(args):
    """Export fine-tuned model to GGUF and register with Ollama."""
    from src.student.exporter import ModelExporter, ExportConfig
    
    config = ExportConfig(
        checkpoint_dir=f"{args.checkpoint_dir}/final",
        base_model=args.base_model,
        merged_dir=f"{args.model_dir}/merged",
        gguf_dir=f"{args.model_dir}/gguf",
        quantisation_method=args.quant,
        ollama_model_name=args.ollama_model,
    )
    
    exporter = ModelExporter(config)
    results = exporter.export()
    
    print(f"\n{'='*60}")
    print(f"STEP 3 COMPLETE: Export")
    print(f"{'='*60}")
    print(f"  GGUF:     {results.get('gguf_path')}")
    print(f"  Ollama:   {results.get('ollama_model_name')}")
    print(f"  Verified: {results.get('verified', False)}")
    
    return results


# =============================================================================
# Step 4: Test Inference
# =============================================================================

def step_test_inference(args):
    """Quick inference test with the deployed model."""
    from src.student.inference import ClinicalScribeInference, InferenceConfig
    
    config = InferenceConfig(
        model_name=args.ollama_model,
        use_rag=True,
        rag_backend=args.rag_backend,
    )
    
    scribe = ClinicalScribeInference(config)
    
    if not scribe.is_model_available():
        print(f"\n! Model '{args.ollama_model}' not found in Ollama.")
        print(f"  Run step 3 first: python run_student_pipeline.py --export")
        return None
    
    # Test dialogue
    test_dialogue = (
        "Doctor: Good afternoon. What can I help you with today?\n"
        "Patient: I've been having this persistent cough for about two weeks now.\n"
        "Doctor: Is it a dry cough or are you producing any phlegm?\n"
        "Patient: It started dry but now I'm coughing up some yellowish mucus.\n"
        "Doctor: Any fever, shortness of breath, or chest pain?\n"
        "Patient: I had a mild fever last week, around 37.8. No chest pain though.\n"
        "Doctor: Are you a smoker?\n"
        "Patient: No, never smoked.\n"
        "Doctor: Any other medical conditions or regular medications?\n"
        "Patient: I have asthma. I use a blue inhaler when needed.\n"
        "Doctor: Let me listen to your chest. Take some deep breaths.\n"
        "Patient: Okay.\n"
        "Doctor: I can hear some crackles in your lower right lung. I think you may "
        "have a chest infection. I'll prescribe some amoxicillin and arrange a chest X-ray."
    )
    
    print(f"\nTesting inference with model '{args.ollama_model}'...")
    
    # Test with RAG
    result_rag = scribe.generate_summary(test_dialogue)
    
    # Test without RAG
    result_no_rag = scribe.generate_summary_no_rag(test_dialogue)
    
    print(f"\n{'='*60}")
    print(f"STEP 4 COMPLETE: Inference Test")
    print(f"{'='*60}")
    print(f"\n--- With RAG ({args.rag_backend}) ---")
    print(f"Time: {result_rag['generation_time']:.1f}s")
    print(f"Sources: {result_rag['rag_sources'][:3]}")
    print(f"Output preview: {result_rag['raw_output'][:300]}...")
    print(f"\n--- Without RAG ---")
    print(f"Time: {result_no_rag['generation_time']:.1f}s")
    print(f"Output preview: {result_no_rag['raw_output'][:300]}...")
    
    return {"with_rag": result_rag, "without_rag": result_no_rag}


# =============================================================================
# Step 5: Full Evaluation
# =============================================================================

def step_evaluate(args):
    """Run comprehensive evaluation with LLM-as-a-Judge."""
    from src.student.evaluator import StudentEvaluator, EvaluationConfig
    
    config = EvaluationConfig(
        test_data_path=f"{args.training_data_dir}/test.jsonl",
        student_model=args.ollama_model,
        base_model=args.base_model_ollama,
        teacher_model=args.teacher_model,
        teacher_provider=args.teacher_provider,
        judge_model=args.judge_model,
        judge_provider=args.judge_provider,
        enable_llm_judge=not args.no_judge,
        output_dir=args.eval_output_dir,
        max_samples=args.max_eval_samples,
        compute_bertscore=args.bertscore,
    )
    
    evaluator = StudentEvaluator(config)
    results = evaluator.run_full_evaluation()
    
    print(f"\n{'='*60}")
    print(f"STEP 5 COMPLETE: Evaluation")
    print(f"{'='*60}")
    print(f"  Results: {args.eval_output_dir}")
    if results.get("report_path"):
        print(f"  Report:  {results['report_path']}")
    
    # Print comparison summary
    comp = results.get("comparative", {})
    if comp:
        print(f"\n  Configuration Comparison:")
        for name, data in comp.items():
            if data.get("error"):
                print(f"    {name}: SKIPPED ({data['error']})")
                continue
            metrics = data.get("metrics", {})
            rouge = metrics.get("rouge_l", "N/A")
            judge = metrics.get("llm_judge", {})
            overall = judge.get("avg_overall", "N/A")
            
            rouge_str = f"{rouge:.3f}" if isinstance(rouge, float) else str(rouge)
            overall_str = f"{overall:.2f}/5" if isinstance(overall, float) else str(overall)
            
            print(f"    {name:12s}: ROUGE-L={rouge_str}  Judge={overall_str}")
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Student Model Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline:
    python run_student_pipeline.py --all --data-dirs ./data/batch_1 ./data/batch_2

    # Just prepare data:
    python run_student_pipeline.py --prepare-data --data-dirs ./data/synthetic_output

    # Just train:
    python run_student_pipeline.py --train --epochs 3 --curriculum

    # Just evaluate:
    python run_student_pipeline.py --evaluate --max-eval-samples 20
        """
    )
    
    # Step flags
    step_group = parser.add_argument_group("Pipeline Steps")
    step_group.add_argument("--all", action="store_true", help="Run all steps")
    step_group.add_argument("--prepare-data", action="store_true", help="Step 1: Prepare training data")
    step_group.add_argument("--train", action="store_true", help="Step 2: Fine-tune model")
    step_group.add_argument("--export", action="store_true", help="Step 3: Export to Ollama")
    step_group.add_argument("--test-inference", action="store_true", help="Step 4: Test inference")
    step_group.add_argument("--evaluate", action="store_true", help="Step 5: Full evaluation")
    
    # Data preparation args
    data_group = parser.add_argument_group("Data Preparation (Step 1)")
    data_group.add_argument("--data-dirs", nargs="+", default=["./data/synthetic_output_llama_index"],
                           help="Directories with synthetic data from teacher pipeline")
    data_group.add_argument("--training-data-dir", default="./data/training_data",
                           help="Output directory for prepared training data")
    data_group.add_argument("--rag-ratio", type=float, default=0.5,
                           help="Fraction of examples with RAG context (default: 0.5)")
    data_group.add_argument("--min-turns", type=int, default=8,
                           help="Minimum dialogue turns to keep sample")
    
    # Training args
    train_group = parser.add_argument_group("Training (Step 2)")
    train_group.add_argument("--base-model", default="unsloth/Phi-3.5-mini-instruct",
                            help="Base model for fine-tuning")
    train_group.add_argument("--checkpoint-dir", default="./checkpoints/phi35_clinical_scribe",
                            help="Directory for training checkpoints")
    train_group.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    train_group.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    train_group.add_argument("--lora-r", type=int, default=32, help="LoRA rank")
    train_group.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    train_group.add_argument("--max-seq-length", type=int, default=4096, help="Max sequence length")
    train_group.add_argument("--curriculum", action="store_true", help="Enable curriculum learning")
    train_group.add_argument("--report-to", default="none", choices=["none", "mlflow", "wandb"])
    
    # Export args
    export_group = parser.add_argument_group("Export (Step 3)")
    export_group.add_argument("--model-dir", default="./models/phi35_clinical",
                             help="Directory for exported model files")
    export_group.add_argument("--ollama-model", default="clinical-scribe",
                             help="Ollama model name")
    export_group.add_argument("--quant", default="q4_k_m",
                             choices=["q4_k_m", "q5_k_m", "q8_0"],
                             help="GGUF quantisation method")
    
    # Inference args
    inf_group = parser.add_argument_group("Inference (Step 4)")
    inf_group.add_argument("--rag-backend", default="llama_index",
                          choices=["llama_index", "manual", "hybrid"])
    
    # Evaluation args
    eval_group = parser.add_argument_group("Evaluation (Step 5)")
    eval_group.add_argument("--base-model-ollama", default="phi3.5:3.8b-mini-instruct-q4_K_M",
                           help="Base (non-fine-tuned) model name in Ollama")
    eval_group.add_argument("--teacher-model", default="gpt-4o-mini", help="Teacher model for comparison")
    eval_group.add_argument("--teacher-provider", default="openai")
    eval_group.add_argument("--judge-model", default="gpt-4o-mini", help="LLM judge model")
    eval_group.add_argument("--judge-provider", default="openai")
    eval_group.add_argument("--no-judge", action="store_true", help="Disable LLM-as-a-Judge")
    eval_group.add_argument("--eval-output-dir", default="./evaluation_results")
    eval_group.add_argument("--max-eval-samples", type=int, default=None, help="Limit test samples")
    eval_group.add_argument("--bertscore", action="store_true", help="Compute BERTScore (slow)")
    
    # General
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Default to showing help
    if not any([args.all, args.prepare_data, args.train, args.export, 
                args.test_inference, args.evaluate]):
        parser.print_help()
        sys.exit(0)
    
    run_all = args.all
    
    print("=" * 60)
    print("STUDENT MODEL PIPELINE")
    print("Trustworthy SLMs for Ambient Clinical Scribing")
    print("=" * 60)
    
    try:
        if run_all or args.prepare_data:
            print("\n>>> STEP 1: Data Preparation")
            step_prepare_data(args)
        
        if run_all or args.train:
            print("\n>>> STEP 2: Fine-Tuning")
            step_train(args)
        
        if run_all or args.export:
            print("\n>>> STEP 3: Export to Ollama")
            step_export(args)
        
        if run_all or args.test_inference:
            print("\n>>> STEP 4: Test Inference")
            step_test_inference(args)
        
        if run_all or args.evaluate:
            print("\n>>> STEP 5: Full Evaluation")
            step_evaluate(args)
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nPipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
