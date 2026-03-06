"""
Model Comparison Visualisation

Generates comparison plots, tables, and a markdown report from
evaluation results of two SLMs (Phi-3.5-mini vs Qwen2.5-3B).

Usage:
    python compare_models.py \
        --phi-results ./evaluation_results_phi/evaluation_*.json \
        --qwen-results ./evaluation_results_qwen/evaluation_*.json \
        --output-dir ./comparison_results

Author: Alireza Rashidi
MSc Project: Trustworthy SLMs for Ambient Clinical Scribing
"""

import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def load_results(path: str) -> Dict[str, Any]:
    """Load evaluation results JSON."""
    with open(path) as f:
        return json.load(f)


def extract_metrics(results: Dict, config_key: str) -> Dict[str, Any]:
    """Extract key metrics from a comparative config."""
    comp = results.get("comparative", {})
    cfg = comp.get(config_key, {})
    
    if cfg.get("error"):
        return {"error": cfg["error"]}
    
    m = cfg.get("metrics", {})
    rouge = m.get("rouge_details", {})
    judge = m.get("llm_judge", {})
    clinical = m.get("clinical_structure", {})
    
    return {
        "rouge1": rouge.get("rouge1", 0),
        "rouge2": rouge.get("rouge2", 0),
        "rougeL": rouge.get("rougeL", 0),
        "bertscore_f1": m.get("bertscore_f1", None),
        "bertscore_precision": m.get("bertscore_precision", None),
        "bertscore_recall": m.get("bertscore_recall", None),
        "avg_generation_time": m.get("avg_generation_time", 0),
        "required_sections": clinical.get("required_section_coverage", 0),
        "optional_sections": clinical.get("optional_section_coverage", 0),
        "judge_overall": judge.get("avg_overall", None),
        "judge_accuracy": judge.get("avg_clinical_accuracy", None),
        "judge_completeness": judge.get("avg_completeness", None),
        "judge_hallucination": judge.get("avg_hallucination", None),
        "judge_safety": judge.get("avg_clinical_safety", None),
        "judge_coherence": judge.get("avg_coherence", None),
        "judge_conciseness": judge.get("avg_conciseness", None),
        "critical_errors": judge.get("total_critical_errors", None),
        "samples_with_errors": judge.get("samples_with_critical_errors", None),
    }


def fmt(val, decimals=3):
    """Format a value for display."""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def generate_comparison_table(
    phi_metrics: Dict, 
    qwen_metrics: Dict,
    teacher_metrics: Optional[Dict] = None,
    baseline_metrics: Optional[Dict] = None,
) -> str:
    """Generate a comprehensive markdown comparison table."""
    
    rows = [
        ("ROUGE-1", "rouge1", 3),
        ("ROUGE-2", "rouge2", 3),
        ("ROUGE-L", "rougeL", 3),
        ("BERTScore F1", "bertscore_f1", 3),
        ("BERTScore Precision", "bertscore_precision", 3),
        ("BERTScore Recall", "bertscore_recall", 3),
        ("Avg Gen Time (s)", "avg_generation_time", 1),
        ("Required Sections", "required_sections", 2),
        ("Optional Sections", "optional_sections", 2),
        ("Judge Overall", "judge_overall", 2),
        ("Judge Accuracy", "judge_accuracy", 2),
        ("Judge Completeness", "judge_completeness", 2),
        ("Judge Hallucination", "judge_hallucination", 2),
        ("Judge Safety", "judge_safety", 2),
        ("Judge Coherence", "judge_coherence", 2),
        ("Judge Conciseness", "judge_conciseness", 2),
        ("Critical Errors", "critical_errors", 0),
        ("Samples w/ Errors", "samples_with_errors", 0),
    ]
    
    lines = []
    
    # Build header
    header = "| Metric | Phi-3.5 (FT+RAG) | Qwen2.5 (FT+RAG) |"
    sep = "|---|---|---|"
    if teacher_metrics:
        header += " Teacher |"
        sep += "---|"
    if baseline_metrics:
        header += " Baseline |"
        sep += "---|"
    
    lines.append(header)
    lines.append(sep)
    
    for label, key, dec in rows:
        phi_val = phi_metrics.get(key)
        qwen_val = qwen_metrics.get(key)
        
        # Determine winner (higher is better for all except gen_time and critical_errors)
        lower_is_better = key in ("avg_generation_time", "critical_errors", "samples_with_errors")
        
        phi_str = fmt(phi_val, dec)
        qwen_str = fmt(qwen_val, dec)
        
        # Bold the winner
        if phi_val is not None and qwen_val is not None:
            if isinstance(phi_val, (int, float)) and isinstance(qwen_val, (int, float)):
                if lower_is_better:
                    if phi_val < qwen_val:
                        phi_str = f"**{phi_str}**"
                    elif qwen_val < phi_val:
                        qwen_str = f"**{qwen_str}**"
                else:
                    if phi_val > qwen_val:
                        phi_str = f"**{phi_str}**"
                    elif qwen_val > phi_val:
                        qwen_str = f"**{qwen_str}**"
        
        row = f"| {label} | {phi_str} | {qwen_str} |"
        
        if teacher_metrics:
            row += f" {fmt(teacher_metrics.get(key), dec)} |"
        if baseline_metrics:
            row += f" {fmt(baseline_metrics.get(key), dec)} |"
        
        lines.append(row)
    
    return "\n".join(lines)


def generate_config_comparison_table(
    phi_results: Dict, 
    qwen_results: Dict,
) -> str:
    """Generate a table comparing all 5 configurations for both models."""
    configs = ["baseline", "rag_only", "ft_only", "ft_rag", "teacher"]
    
    lines = [
        "| Configuration | Phi ROUGE-L | Qwen ROUGE-L | Phi Judge | Qwen Judge | Phi Halluc | Qwen Halluc |",
        "|---|---|---|---|---|---|---|",
    ]
    
    for cfg in configs:
        phi_m = extract_metrics(phi_results, cfg)
        qwen_m = extract_metrics(qwen_results, cfg)
        
        if phi_m.get("error") and qwen_m.get("error"):
            continue
        
        phi_desc = phi_results.get("comparative", {}).get(cfg, {}).get("description", cfg)
        
        row = (
            f"| {phi_desc} "
            f"| {fmt(phi_m.get('rougeL'))} "
            f"| {fmt(qwen_m.get('rougeL'))} "
            f"| {fmt(phi_m.get('judge_overall'), 2)} "
            f"| {fmt(qwen_m.get('judge_overall'), 2)} "
            f"| {fmt(phi_m.get('judge_hallucination'), 2)} "
            f"| {fmt(qwen_m.get('judge_hallucination'), 2)} |"
        )
        lines.append(row)
    
    return "\n".join(lines)


def generate_report(
    phi_results: Dict,
    qwen_results: Dict,
    output_dir: Path,
) -> Path:
    """Generate a comprehensive comparison report."""
    
    # Extract FT+RAG metrics for both
    phi_ft_rag = extract_metrics(phi_results, "ft_rag")
    qwen_ft_rag = extract_metrics(qwen_results, "ft_rag")
    
    # Teacher and baseline from Phi results (same for both)
    teacher = extract_metrics(phi_results, "teacher")
    baseline = extract_metrics(phi_results, "baseline")
    
    lines = [
        "# SLM Comparison Report: Phi-3.5-mini vs Qwen2.5-3B",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Phi Test Samples:** {phi_results.get('num_test_samples', 'N/A')}",
        f"**Qwen Test Samples:** {qwen_results.get('num_test_samples', 'N/A')}",
        "",
        "---",
        "",
        "## 1. Head-to-Head: FT+RAG Configuration",
        "",
        "Both models fine-tuned on the same training data with identical hyperparameters,",
        "evaluated on the same test set with GPT-4o-mini as judge.",
        "",
        generate_comparison_table(phi_ft_rag, qwen_ft_rag, teacher, baseline),
        "",
        "---",
        "",
        "## 2. All Configurations Comparison",
        "",
        generate_config_comparison_table(phi_results, qwen_results),
        "",
        "---",
        "",
        "## 3. Key Findings",
        "",
    ]
    
    # Auto-generate findings
    if not phi_ft_rag.get("error") and not qwen_ft_rag.get("error"):
        phi_rouge = phi_ft_rag.get("rougeL", 0)
        qwen_rouge = qwen_ft_rag.get("rougeL", 0)
        phi_halluc = phi_ft_rag.get("judge_hallucination")
        qwen_halluc = qwen_ft_rag.get("judge_hallucination")
        phi_overall = phi_ft_rag.get("judge_overall")
        qwen_overall = qwen_ft_rag.get("judge_overall")
        phi_time = phi_ft_rag.get("avg_generation_time", 0)
        qwen_time = qwen_ft_rag.get("avg_generation_time", 0)
        phi_bert = phi_ft_rag.get("bertscore_f1")
        qwen_bert = qwen_ft_rag.get("bertscore_f1")
        
        rouge_winner = "Phi-3.5" if phi_rouge > qwen_rouge else "Qwen2.5"
        lines.append(f"- **ROUGE-L:** {rouge_winner} leads ({fmt(phi_rouge)} vs {fmt(qwen_rouge)})")
        
        if phi_bert and qwen_bert:
            bert_winner = "Phi-3.5" if phi_bert > qwen_bert else "Qwen2.5"
            lines.append(f"- **BERTScore F1:** {bert_winner} leads ({fmt(phi_bert)} vs {fmt(qwen_bert)})")
        
        if phi_halluc and qwen_halluc:
            halluc_winner = "Phi-3.5" if phi_halluc > qwen_halluc else "Qwen2.5"
            lines.append(f"- **Hallucination:** {halluc_winner} is better ({fmt(phi_halluc, 2)} vs {fmt(qwen_halluc, 2)})")
        
        if phi_overall and qwen_overall:
            overall_winner = "Phi-3.5" if phi_overall > qwen_overall else "Qwen2.5"
            lines.append(f"- **Judge Overall:** {overall_winner} leads ({fmt(phi_overall, 2)} vs {fmt(qwen_overall, 2)})")
        
        speed_winner = "Phi-3.5" if phi_time < qwen_time else "Qwen2.5"
        lines.append(f"- **Speed:** {speed_winner} is faster ({fmt(phi_time, 1)}s vs {fmt(qwen_time, 1)}s)")
        
        phi_errors = phi_ft_rag.get("critical_errors", 0) or 0
        qwen_errors = qwen_ft_rag.get("critical_errors", 0) or 0
        error_winner = "Phi-3.5" if phi_errors < qwen_errors else "Qwen2.5"
        lines.append(f"- **Critical Errors:** {error_winner} has fewer ({phi_errors} vs {qwen_errors})")
    
    lines.extend([
        "",
        "---",
        "",
        "## 4. Dissertation Implications",
        "",
        "This comparison demonstrates that the teacher-student distillation pipeline",
        "is model-agnostic — the same training data and methodology can be applied to",
        "different SLM architectures. The relative performance differences highlight",
        "architectural trade-offs relevant to clinical deployment scenarios.",
        "",
    ])
    
    report_content = "\n".join(lines)
    report_path = output_dir / f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    with open(report_path, "w") as f:
        f.write(report_content)
    
    return report_path


def generate_plot_data(
    phi_results: Dict,
    qwen_results: Dict,
    output_dir: Path,
) -> Path:
    """
    Export comparison data as JSON for plotting.
    
    Can be loaded by matplotlib, plotly, or any visualisation tool.
    """
    configs = ["baseline", "ft_only", "ft_rag", "teacher"]
    
    plot_data = {
        "configs": configs,
        "phi": {},
        "qwen": {},
    }
    
    for cfg in configs:
        phi_m = extract_metrics(phi_results, cfg)
        qwen_m = extract_metrics(qwen_results, cfg)
        plot_data["phi"][cfg] = phi_m
        plot_data["qwen"][cfg] = qwen_m
    
    data_path = output_dir / "comparison_plot_data.json"
    with open(data_path, "w") as f:
        json.dump(plot_data, f, indent=2, default=str)
    
    return data_path


def generate_matplotlib_script(output_dir: Path) -> Path:
    """Generate a Python script that creates comparison bar charts."""
    
    script = '''"""
Auto-generated comparison plots for Phi-3.5 vs Qwen2.5
Run: python plot_comparison.py
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load data
with open("comparison_plot_data.json") as f:
    data = json.load(f)

phi = data["phi"]
qwen = data["qwen"]

# --- Plot 1: ROUGE Scores (FT+RAG) ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics_rouge = ["rouge1", "rouge2", "rougeL"]
labels_rouge = ["ROUGE-1", "ROUGE-2", "ROUGE-L"]

for ax, metric, label in zip(axes, metrics_rouge, labels_rouge):
    configs = ["baseline", "ft_only", "ft_rag", "teacher"]
    config_labels = ["Baseline", "FT Only", "FT+RAG", "Teacher"]
    
    phi_vals = [phi.get(c, {}).get(metric, 0) or 0 for c in configs]
    qwen_vals = [qwen.get(c, {}).get(metric, 0) or 0 for c in configs]
    
    x = np.arange(len(configs))
    width = 0.35
    
    ax.bar(x - width/2, phi_vals, width, label="Phi-3.5", color="#4C72B0", alpha=0.85)
    ax.bar(x + width/2, qwen_vals, width, label="Qwen2.5", color="#DD8452", alpha=0.85)
    
    ax.set_ylabel("Score")
    ax.set_title(label)
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("ROUGE Score Comparison: Phi-3.5-mini vs Qwen2.5-3B", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("comparison_rouge.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: comparison_rouge.png")

# --- Plot 2: LLM Judge Dimensions (FT+RAG only) ---
dimensions = [
    "judge_accuracy", "judge_completeness", "judge_hallucination",
    "judge_safety", "judge_coherence", "judge_conciseness"
]
dim_labels = ["Accuracy", "Completeness", "Hallucination", "Safety", "Coherence", "Conciseness"]

phi_judge = [phi.get("ft_rag", {}).get(d, 0) or 0 for d in dimensions]
qwen_judge = [qwen.get("ft_rag", {}).get(d, 0) or 0 for d in dimensions]

# Also get teacher and baseline for reference
teacher_judge = [phi.get("teacher", {}).get(d, 0) or 0 for d in dimensions]
baseline_judge = [phi.get("baseline", {}).get(d, 0) or 0 for d in dimensions]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(dimensions))
width = 0.2

ax.bar(x - 1.5*width, baseline_judge, width, label="Baseline", color="#8C8C8C", alpha=0.6)
ax.bar(x - 0.5*width, phi_judge, width, label="Phi-3.5 FT+RAG", color="#4C72B0", alpha=0.85)
ax.bar(x + 0.5*width, qwen_judge, width, label="Qwen2.5 FT+RAG", color="#DD8452", alpha=0.85)
ax.bar(x + 1.5*width, teacher_judge, width, label="Teacher", color="#55A868", alpha=0.6)

ax.set_ylabel("Score (1-5)")
ax.set_title("LLM Judge Scores by Dimension", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(dim_labels, rotation=30, ha="right")
ax.legend()
ax.set_ylim(0, 5.5)
ax.axhline(y=5, color="gray", linestyle="--", alpha=0.3)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("comparison_judge.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: comparison_judge.png")

# --- Plot 3: BERTScore (if available) ---
phi_bert = phi.get("ft_rag", {}).get("bertscore_f1")
qwen_bert = qwen.get("ft_rag", {}).get("bertscore_f1")

if phi_bert and qwen_bert:
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bert_metrics = ["bertscore_precision", "bertscore_recall", "bertscore_f1"]
    bert_labels = ["Precision", "Recall", "F1"]
    
    phi_bert_vals = [phi.get("ft_rag", {}).get(m, 0) or 0 for m in bert_metrics]
    qwen_bert_vals = [qwen.get("ft_rag", {}).get(m, 0) or 0 for m in bert_metrics]
    
    x = np.arange(len(bert_metrics))
    width = 0.35
    
    ax.bar(x - width/2, phi_bert_vals, width, label="Phi-3.5 FT+RAG", color="#4C72B0", alpha=0.85)
    ax.bar(x + width/2, qwen_bert_vals, width, label="Qwen2.5 FT+RAG", color="#DD8452", alpha=0.85)
    
    ax.set_ylabel("Score")
    ax.set_title("BERTScore Comparison (FT+RAG)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(bert_labels)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("comparison_bertscore.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: comparison_bertscore.png")

# --- Plot 4: Radar/Spider chart (FT+RAG) ---
categories = dim_labels
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

phi_vals = phi_judge + phi_judge[:1]
qwen_vals = qwen_judge + qwen_judge[:1]
teacher_vals = teacher_judge + teacher_judge[:1]

ax.plot(angles, phi_vals, "o-", label="Phi-3.5 FT+RAG", color="#4C72B0", linewidth=2)
ax.fill(angles, phi_vals, color="#4C72B0", alpha=0.1)
ax.plot(angles, qwen_vals, "s-", label="Qwen2.5 FT+RAG", color="#DD8452", linewidth=2)
ax.fill(angles, qwen_vals, color="#DD8452", alpha=0.1)
ax.plot(angles, teacher_vals, "^--", label="Teacher", color="#55A868", linewidth=1.5, alpha=0.7)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=10)
ax.set_ylim(0, 5)
ax.set_title("Quality Dimensions Radar Chart", fontsize=14, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

plt.tight_layout()
plt.savefig("comparison_radar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: comparison_radar.png")

print("\\nAll plots generated successfully!")
'''
    
    script_path = output_dir / "plot_comparison.py"
    with open(script_path, "w") as f:
        f.write(script)
    
    return script_path


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    
    parser = argparse.ArgumentParser(description="Compare two SLM evaluation results")
    parser.add_argument("--phi-results", required=True, help="Path to Phi-3.5 evaluation JSON")
    parser.add_argument("--qwen-results", required=True, help="Path to Qwen2.5 evaluation JSON")
    parser.add_argument("--output-dir", default="./comparison_results")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    phi_results = load_results(args.phi_results)
    qwen_results = load_results(args.qwen_results)
    
    # Generate comparison report
    report_path = generate_report(phi_results, qwen_results, output_dir)
    print(f"Report: {report_path}")
    
    # Generate plot data
    data_path = generate_plot_data(phi_results, qwen_results, output_dir)
    print(f"Plot data: {data_path}")
    
    # Generate matplotlib script
    script_path = generate_matplotlib_script(output_dir)
    print(f"Plot script: {script_path}")
    
    print(f"\nTo generate plots:")
    print(f"  cd {output_dir}")
    print(f"  python plot_comparison.py")
