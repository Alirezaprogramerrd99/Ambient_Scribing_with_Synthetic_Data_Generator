"""
Temperature Experiment Runner and Analysis

Runs the evaluator at different temperature settings for ft_rag config only,
then generates a comparison table and plot.

Scientific justification:
- Woo et al. (2025, npj Digital Medicine) tested temperature 0 vs 1
  and found temperature 0 slightly outperformed higher values.
- Renze & Guven (2024) found no significant difference in LLM 
  problem-solving for temperature 0-1.
- Chang et al. (2023) hypothesized lower temperature suits QA tasks.

Usage:
    # Run experiments for one model (creates 4 eval JSONs)
    python run_temperature_experiment.py \
        --test-data ./data/training_data/test.jsonl \
        --ft-model-path ./checkpoints/phi35_clinical_scribe/hf_merged \
        --base-model-hf unsloth/Phi-3.5-mini-instruct \
        --label "Phi-3.5" \
        --output-dir ./temp_experiment_phi \
        --max-samples 50

    # Analyze results from multiple models
    python run_temperature_experiment.py --analyze \
        --result-dirs ./temp_experiment_phi ./temp_experiment_llama3b ./temp_experiment_llama1b \
        --labels "Phi-3.5 (3.8B)" "Llama-3.2 (3B)" "Llama-3.2 (1B)" \
        --output-dir ./temp_analysis

Author: Alireza Rashidi
MSc Project: Trustworthy SLMs for Ambient Clinical Scribing
"""

import json
import argparse
import subprocess
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)

TEMPERATURES = [0.0, 0.3, 0.7, 1.0]
TOP_P_VALUES = [0.9]  # Fixed; expand if needed


def run_experiments(
    test_data: str,
    ft_model_path: str,
    base_model_hf: str,
    label: str,
    output_dir: str,
    max_samples: int = 50,
    judge_model: str = "gpt-4o-mini",
):
    """Run evaluator at each temperature setting."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for temp in TEMPERATURES:
        temp_str = f"{temp:.1f}".replace(".", "")
        run_dir = output_path / f"temp_{temp_str}"
        
        print(f"\n{'='*60}")
        print(f"Running: {label} | temperature={temp} | top_p=0.9")
        print(f"Output: {run_dir}")
        print(f"{'='*60}\n")
        
        cmd = [
            sys.executable, "-m", "src.student.evaluator",
            "--test-data", test_data,
            "--ft-model-path", ft_model_path,
            "--base-model-hf", base_model_hf,
            "--output-dir", str(run_dir),
            "--max-samples", str(max_samples),
            "--judge-model", judge_model,
            "--no-bertscore",
            "--temperature", str(temp),
            "--top-p", "0.9",
        ]
        
        result = subprocess.run(cmd, capture_output=False)
        
        if result.returncode != 0:
            print(f"WARNING: Temperature {temp} run failed!")
        else:
            print(f"Completed: temperature={temp}")
    
    print(f"\nAll temperature experiments complete for {label}")
    print(f"Results in: {output_path}")


def analyze_results(
    result_dirs: List[str],
    labels: List[str],
    output_dir: str,
):
    """Analyze temperature experiment results and generate comparison."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect results: label -> temp -> metrics
    all_results = {}
    
    for result_dir, label in zip(result_dirs, labels):
        all_results[label] = {}
        
        for temp in TEMPERATURES:
            temp_str = f"{temp:.1f}".replace(".", "")
            temp_dir = Path(result_dir) / f"temp_{temp_str}"
            
            # Find evaluation JSON
            json_files = list(temp_dir.glob("evaluation_*.json"))
            if not json_files:
                print(f"  WARNING: No results for {label} temp={temp}")
                continue
            
            with open(json_files[0]) as f:
                eval_data = json.load(f)
            
            # Extract ft_rag metrics
            ft_rag = eval_data.get("comparative", {}).get("ft_rag", {})
            if ft_rag.get("error"):
                print(f"  WARNING: {label} temp={temp} ft_rag errored")
                continue
            
            m = ft_rag.get("metrics", {})
            j = m.get("llm_judge", {})
            
            all_results[label][temp] = {
                "rouge_l": m.get("rouge_l", 0),
                "rouge_1": m.get("rouge_details", {}).get("rouge1", 0),
                "bleu_4": 0,  # Would need post_eval for this
                "judge_overall": j.get("avg_overall", 0),
                "judge_accuracy": j.get("avg_clinical_accuracy", 0),
                "judge_hallucination": j.get("avg_hallucination", 0),
                "judge_safety": j.get("avg_clinical_safety", 0),
                "judge_coherence": j.get("avg_coherence", 0),
                "avg_time": m.get("avg_generation_time", 0),
                "critical_errors": j.get("total_critical_errors", 0),
            }
    
    # Generate markdown table
    lines = [
        "# Temperature Experiment Results",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## FT+RAG Configuration — Effect of Generation Temperature",
        "",
        "Scientific references:",
        "- Woo et al. (2025): tested temp 0 vs 1, found temp 0 slightly better",
        "- Renze & Guven (2024): no significant difference for temp 0-1",
        "- Chang et al. (2023): lower temp better for QA with attribution",
        "",
    ]
    
    # Per-model tables
    for label in labels:
        temps = all_results.get(label, {})
        if not temps:
            continue
        
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| Metric | " + " | ".join(f"T={t}" for t in TEMPERATURES) + " |")
        lines.append("|---|" + "---|" * len(TEMPERATURES))
        
        metric_rows = [
            ("ROUGE-L", "rouge_l"),
            ("Judge Overall", "judge_overall"),
            ("Accuracy", "judge_accuracy"),
            ("Hallucination", "judge_hallucination"),
            ("Safety", "judge_safety"),
            ("Coherence", "judge_coherence"),
            ("Avg Time (s)", "avg_time"),
            ("Critical Errors", "critical_errors"),
        ]
        
        for display, key in metric_rows:
            row = f"| {display} |"
            values = []
            for temp in TEMPERATURES:
                val = temps.get(temp, {}).get(key, "—")
                if isinstance(val, float):
                    if key == "avg_time":
                        row += f" {val:.1f} |"
                    else:
                        row += f" {val:.4f} |"
                    values.append(val)
                elif isinstance(val, int):
                    row += f" {val} |"
                    values.append(val)
                else:
                    row += f" {val} |"
                    values.append(None)
            
            # Bold the best value
            # Higher is better except for time and errors
            lines.append(row)
        
        lines.append("")
    
    # Save table
    table_path = output_path / "temperature_comparison.md"
    with open(table_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {table_path}")
    
    # Generate plot
    _plot_temperature(all_results, labels, output_path)
    
    # Save raw data
    data_path = output_path / "temperature_data.json"
    with open(data_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {data_path}")


def _plot_temperature(all_results: Dict, labels: List[str], output_dir: Path):
    """Generate temperature comparison plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available, skipping plots")
        return
    
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(labels), 3)))
    
    metrics_to_plot = [
        ("rouge_l", "ROUGE-L", False),
        ("judge_overall", "Judge Overall (/5)", False),
        ("judge_hallucination", "Hallucination (/5)", False),
        ("avg_time", "Avg Time (s)", True),  # Lower is better
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for ax_idx, (key, title, lower_better) in enumerate(metrics_to_plot):
        ax = axes[ax_idx // 2][ax_idx % 2]
        
        for i, label in enumerate(labels):
            temps = all_results.get(label, {})
            x_vals = []
            y_vals = []
            for temp in TEMPERATURES:
                if temp in temps:
                    x_vals.append(temp)
                    y_vals.append(temps[temp].get(key, 0))
            
            if x_vals:
                ax.plot(x_vals, y_vals, "o-", label=label, color=colors[i],
                        linewidth=2, markersize=8)
        
        ax.set_xlabel("Temperature")
        ax.set_ylabel(title)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_xticks(TEMPERATURES)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle("Effect of Generation Temperature on FT+RAG Performance",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "temperature_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: temperature_comparison.png")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    
    parser = argparse.ArgumentParser(description="Temperature experiment runner")
    
    # Mode
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze existing results instead of running experiments")
    
    # Run mode args
    parser.add_argument("--test-data", default="./data/training_data/test.jsonl")
    parser.add_argument("--ft-model-path")
    parser.add_argument("--base-model-hf")
    parser.add_argument("--label", default="Model")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    
    # Analyze mode args
    parser.add_argument("--result-dirs", nargs="+")
    parser.add_argument("--labels", nargs="+")
    
    parser.add_argument("--output-dir", required=True)
    
    args = parser.parse_args()
    
    if args.analyze:
        if not args.result_dirs or not args.labels:
            parser.error("--analyze requires --result-dirs and --labels")
        analyze_results(args.result_dirs, args.labels, args.output_dir)
    else:
        if not args.ft_model_path or not args.base_model_hf:
            parser.error("Run mode requires --ft-model-path and --base-model-hf")
        run_experiments(
            test_data=args.test_data,
            ft_model_path=args.ft_model_path,
            base_model_hf=args.base_model_hf,
            label=args.label,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            judge_model=args.judge_model,
        )
