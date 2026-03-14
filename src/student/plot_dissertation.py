"""
Comprehensive Dissertation Plots

Generates all plots needed for the dissertation from post_eval_metrics JSON.
Includes ROUGE variants, BLEU, MEDCON, BERTScore, judge dimensions,
radar charts, RAG backend comparison, and per-section clinical analysis.

Usage:
    python plot_dissertation.py --results ./post_eval_results/post_eval_metrics_*.json

    # With per-section analysis (requires evaluation JSONs)
    python plot_dissertation.py \
        --results ./post_eval_results/post_eval_metrics_*.json \
        --eval-jsons eval_phi.json eval_llama3b.json eval_llama1b.json \
        --labels "Phi-3.5 (3.8B)" "Llama-3.2 (3B)" "Llama-3.2 (1B)"

Author: Alireza Rashidi
MSc Project: Trustworthy SLMs for Ambient Clinical Scribing
"""

import json
import argparse
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

# Ensure non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# =============================================================================
# RAG Backend Label Mapping (Priority 3)
# =============================================================================

RAG_LABEL_MAP = {
    "llama_index": "Dense Retrieval\n(BGE + ChromaDB)",
    "manual": "Custom Dense\n(BGE + ChromaDB)",
    "hybrid": "Medical-Enhanced\n(Dense + Query Exp.\n+ Clinical Filter)",
}

RAG_SHORT_LABELS = {
    "llama_index": "Dense (LlamaIndex)",
    "manual": "Custom Dense",
    "hybrid": "Medical-Enhanced",
}

CONFIG_LABELS = {
    "baseline": "Base\n(no FT, no RAG)",
    "rag_only": "Base\n+ RAG",
    "ft_only": "Fine-tuned\n(no RAG)",
    "ft_rag": "Fine-tuned\n+ RAG",
    "teacher": "Teacher\n(GPT-4o-mini)",
}

CONFIG_SHORT = {
    "baseline": "Base",
    "rag_only": "Base+RAG",
    "ft_only": "FT",
    "ft_rag": "FT+RAG",
    "teacher": "Teacher",
}


# =============================================================================
# Data Loading
# =============================================================================

def load_post_eval(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def get_metric(data: Dict, label: str, config: str, *keys) -> float:
    """Safely navigate nested dict to get a metric value."""
    d = data.get("models", {}).get(label, {}).get(config, {})
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, 0)
        else:
            return 0
    return d if isinstance(d, (int, float)) else 0


def get_rag_metric(data: Dict, label: str, backend: str, *keys) -> float:
    """Get metric from RAG backend results."""
    d = data.get("models", {}).get(label, {}).get("_rag_backends", {}).get(backend, {})
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, 0)
        else:
            return 0
    return d if isinstance(d, (int, float)) else 0


# =============================================================================
# Plot 1: ROUGE Variants (1, 2, L) — Grouped Bar Chart
# =============================================================================

def plot_rouge_variants(data: Dict, labels: List[str], output_dir: Path):
    """ROUGE-1, ROUGE-2, ROUGE-L across all configs for all models."""
    configs = ["baseline", "rag_only", "ft_only", "ft_rag", "teacher"]
    rouge_types = [("rouge1", "ROUGE-1"), ("rouge2", "ROUGE-2"), ("rougeL", "ROUGE-L")]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(labels), 3)))
    
    for ax_idx, (rouge_key, rouge_name) in enumerate(rouge_types):
        ax = axes[ax_idx]
        x = np.arange(len(configs))
        w = 0.8 / len(labels)
        
        for i, label in enumerate(labels):
            vals = [get_metric(data, label, c, "rouge", rouge_key) for c in configs]
            bars = ax.bar(x + i * w, vals, w, label=label, color=colors[i])
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f"{val:.3f}", ha="center", va="bottom", fontsize=6, rotation=45)
        
        ax.set_title(rouge_name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Configuration")
        ax.set_xticks(x + w * (len(labels) - 1) / 2)
        ax.set_xticklabels([CONFIG_SHORT[c] for c in configs], fontsize=8)
        ax.set_ylim(0, 0.85)
        if ax_idx == 0:
            ax.set_ylabel("Score")
    
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("ROUGE Scores Across Models and Configurations", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "rouge_variants.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: rouge_variants.png")


# =============================================================================
# Plot 2: BLEU Variants (1, 2, 3, 4) — Grouped Bar Chart
# =============================================================================

def plot_bleu_variants(data: Dict, labels: List[str], output_dir: Path):
    """BLEU-1 through BLEU-4 for ft_only and ft_rag configs."""
    bleu_keys = [("avg_bleu1", "BLEU-1"), ("avg_bleu2", "BLEU-2"),
                 ("avg_bleu3", "BLEU-3"), ("avg_bleu4", "BLEU-4")]
    configs = ["ft_only", "ft_rag"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(labels), 3)))
    
    for ax_idx, config in enumerate(configs):
        ax = axes[ax_idx]
        x = np.arange(len(bleu_keys))
        w = 0.8 / len(labels)
        
        for i, label in enumerate(labels):
            vals = [get_metric(data, label, config, "bleu", bk) for bk, _ in bleu_keys]
            bars = ax.bar(x + i * w, vals, w, label=label, color=colors[i])
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f"{val:.3f}", ha="center", va="bottom", fontsize=7)
        
        ax.set_title(CONFIG_LABELS[config].replace("\n", " "), fontsize=12, fontweight="bold")
        ax.set_xlabel("BLEU Variant")
        ax.set_xticks(x + w * (len(labels) - 1) / 2)
        ax.set_xticklabels([bn for _, bn in bleu_keys], fontsize=9)
        ax.set_ylim(0, 0.85)
        if ax_idx == 0:
            ax.set_ylabel("Score")
    
    axes[0].legend(fontsize=8)
    fig.suptitle("BLEU Scores: Fine-tuned Configurations", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "bleu_variants.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: bleu_variants.png")


# =============================================================================
# Plot 3: MEDCON (Precision, Recall, F1) — Grouped Bar Chart
# =============================================================================

def plot_medcon(data: Dict, labels: List[str], output_dir: Path):
    """MEDCON Precision, Recall, F1 for key configs."""
    configs = ["baseline", "ft_only", "ft_rag", "teacher"]
    medcon_dims = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(labels), 3)))
    
    for ax_idx, (mk, mn) in enumerate(medcon_dims):
        ax = axes[ax_idx]
        x = np.arange(len(configs))
        w = 0.8 / len(labels)
        
        for i, label in enumerate(labels):
            vals = [get_metric(data, label, c, "medcon", mk) for c in configs]
            bars = ax.bar(x + i * w, vals, w, label=label, color=colors[i])
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f"{val:.3f}", ha="center", va="bottom", fontsize=7)
        
        ax.set_title(f"MEDCON {mn}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Configuration")
        ax.set_xticks(x + w * (len(labels) - 1) / 2)
        ax.set_xticklabels([CONFIG_SHORT[c] for c in configs], fontsize=8)
        ax.set_ylim(0, 1.0)
        if ax_idx == 0:
            ax.set_ylabel("Score")
    
    axes[0].legend(fontsize=8)
    fig.suptitle("MEDCON (Medical Concept Overlap) Scores", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "medcon_scores.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: medcon_scores.png")


# =============================================================================
# Plot 4: BERTScore (P, R, F1) — Only if available
# =============================================================================

def plot_bertscore(data: Dict, labels: List[str], output_dir: Path):
    """BERTScore P/R/F1 for key configs."""
    configs = ["baseline", "ft_only", "ft_rag", "teacher"]
    bert_dims = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]
    
    # Check if BERTScore data exists
    has_bert = False
    for label in labels:
        for c in configs:
            if get_metric(data, label, c, "bertscore", "f1") > 0:
                has_bert = True
                break
    
    if not has_bert:
        print("Skipping bertscore plot (no BERTScore data available)")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(labels), 3)))
    
    for ax_idx, (bk, bn) in enumerate(bert_dims):
        ax = axes[ax_idx]
        x = np.arange(len(configs))
        w = 0.8 / len(labels)
        
        for i, label in enumerate(labels):
            vals = [get_metric(data, label, c, "bertscore", bk) for c in configs]
            bars = ax.bar(x + i * w, vals, w, label=label, color=colors[i])
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f"{val:.3f}", ha="center", va="bottom", fontsize=7)
        
        ax.set_title(f"BERTScore {bn}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Configuration")
        ax.set_xticks(x + w * (len(labels) - 1) / 2)
        ax.set_xticklabels([CONFIG_SHORT[c] for c in configs], fontsize=8)
        ax.set_ylim(0.4, 1.0)
        if ax_idx == 0:
            ax.set_ylabel("Score")
    
    axes[0].legend(fontsize=8)
    fig.suptitle("BERTScore (Semantic Similarity) Scores", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "bertscore.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: bertscore.png")


# =============================================================================
# Plot 5: LLM Judge — All 6 Dimensions for FT+RAG
# =============================================================================

def plot_judge_dimensions(data: Dict, labels: List[str], output_dir: Path):
    """All LLM judge dimensions for ft_rag config."""
    dims = [
        ("avg_overall", "Overall"), ("avg_clinical_accuracy", "Accuracy"),
        ("avg_completeness", "Completeness"), ("avg_hallucination", "Hallucination"),
        ("avg_clinical_safety", "Safety"), ("avg_coherence", "Coherence"),
        ("avg_conciseness", "Conciseness"),
    ]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(labels), 3)))
    x = np.arange(len(dims))
    w = 0.8 / (len(labels) + 1)  # +1 for teacher
    
    for i, label in enumerate(labels):
        vals = [get_metric(data, label, "ft_rag", "llm_judge", dk) for dk, _ in dims]
        bars = ax.bar(x + i * w, vals, w, label=label, color=colors[i])
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7)
    
    # Add teacher as reference
    teacher_vals = [get_metric(data, labels[0], "teacher", "llm_judge", dk) for dk, _ in dims]
    bars = ax.bar(x + len(labels) * w, teacher_vals, w, label="Teacher (GPT-4o-mini)",
                  color="lightgray", edgecolor="gray", hatch="//")
    for bar, val in zip(bars, teacher_vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7, color="gray")
    
    ax.set_xlabel("Quality Dimension")
    ax.set_ylabel("Score (1-5)")
    ax.set_title("LLM-as-a-Judge Scores: Fine-tuned + RAG Configuration", fontsize=13, fontweight="bold")
    ax.set_xticks(x + w * len(labels) / 2)
    ax.set_xticklabels([dn for _, dn in dims], fontsize=9)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 5.5)
    plt.tight_layout()
    plt.savefig(output_dir / "judge_dimensions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: judge_dimensions.png")


# =============================================================================
# Plot 6: Radar Chart — FT+RAG Multi-Metric
# =============================================================================

def plot_radar(data: Dict, labels: List[str], output_dir: Path):
    """Radar chart combining all metric types for ft_rag."""
    categories = ["ROUGE-L", "BLEU-4", "MEDCON-F1", "Judge\nOverall", 
                   "Judge\nHalluc.", "Judge\nSafety"]
    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(labels), 3)))
    
    for i, label in enumerate(labels):
        vals = [
            get_metric(data, label, "ft_rag", "rouge", "rougeL"),
            get_metric(data, label, "ft_rag", "bleu", "avg_bleu4"),
            get_metric(data, label, "ft_rag", "medcon", "f1"),
            get_metric(data, label, "ft_rag", "llm_judge", "avg_overall") / 5.0,
            get_metric(data, label, "ft_rag", "llm_judge", "avg_hallucination") / 5.0,
            get_metric(data, label, "ft_rag", "llm_judge", "avg_clinical_safety") / 5.0,
        ]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", label=label, color=colors[i], linewidth=2)
        ax.fill(angles, vals, alpha=0.1, color=colors[i])
    
    # Teacher reference
    t_vals = [
        get_metric(data, labels[0], "teacher", "rouge", "rougeL"),
        get_metric(data, labels[0], "teacher", "bleu", "avg_bleu4"),
        get_metric(data, labels[0], "teacher", "medcon", "f1"),
        get_metric(data, labels[0], "teacher", "llm_judge", "avg_overall") / 5.0,
        get_metric(data, labels[0], "teacher", "llm_judge", "avg_hallucination") / 5.0,
        get_metric(data, labels[0], "teacher", "llm_judge", "avg_clinical_safety") / 5.0,
    ]
    t_vals += t_vals[:1]
    ax.plot(angles, t_vals, "^--", label="Teacher", color="gray", linewidth=1.5, alpha=0.5)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_title("Multi-Metric Radar: FT+RAG", fontsize=14, fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "radar_multimetric.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: radar_multimetric.png")


# =============================================================================
# Plot 7: RAG Backend Comparison (with descriptive labels)
# =============================================================================

def plot_rag_backends(data: Dict, labels: List[str], output_dir: Path):
    """RAG backend comparison with descriptive labels."""
    backends = ["llama_index", "manual", "hybrid"]
    metrics = [("rouge_l", "ROUGE-L"), ("bleu", "BLEU-4"), ("medcon", "MEDCON-F1")]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(labels), 3)))
    
    for ax_idx, (mk, mn) in enumerate(metrics):
        ax = axes[ax_idx]
        x = np.arange(len(backends))
        w = 0.8 / len(labels)
        
        for i, label in enumerate(labels):
            if mk == "rouge_l":
                vals = [get_rag_metric(data, label, b, "rouge_l") for b in backends]
            elif mk == "bleu":
                vals = [get_rag_metric(data, label, b, "bleu", "avg_bleu4") for b in backends]
            elif mk == "medcon":
                vals = [get_rag_metric(data, label, b, "medcon", "f1") for b in backends]
            else:
                vals = [0] * len(backends)
            
            bars = ax.bar(x + i * w, vals, w, label=label, color=colors[i])
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f"{val:.3f}", ha="center", va="bottom", fontsize=7)
        
        ax.set_title(mn, fontsize=12, fontweight="bold")
        ax.set_xlabel("RAG Strategy")
        ax.set_xticks(x + w * (len(labels) - 1) / 2)
        ax.set_xticklabels([RAG_SHORT_LABELS.get(b, b) for b in backends], fontsize=8)
        if ax_idx == 0:
            ax.set_ylabel("Score")
    
    axes[0].legend(fontsize=8)
    fig.suptitle("RAG Retrieval Strategy Comparison Across Models", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "rag_backend_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: rag_backend_comparison.png")


# =============================================================================
# Plot 8: Metrics Heatmap — Models x Configs
# =============================================================================

def plot_heatmap(data: Dict, labels: List[str], output_dir: Path):
    """Comprehensive heatmap of all key metrics."""
    configs = ["baseline", "rag_only", "ft_only", "ft_rag", "teacher"]
    metric_defs = [
        ("ROUGE-L", lambda d, l, c: get_metric(d, l, c, "rouge", "rougeL")),
        ("BLEU-4", lambda d, l, c: get_metric(d, l, c, "bleu", "avg_bleu4")),
        ("MEDCON-F1", lambda d, l, c: get_metric(d, l, c, "medcon", "f1")),
        ("Judge Overall", lambda d, l, c: get_metric(d, l, c, "llm_judge", "avg_overall") / 5.0),
        ("Hallucination", lambda d, l, c: get_metric(d, l, c, "llm_judge", "avg_hallucination") / 5.0),
        ("Safety", lambda d, l, c: get_metric(d, l, c, "llm_judge", "avg_clinical_safety") / 5.0),
    ]
    
    # Build rows: one row per (model, config) pair
    row_labels = []
    matrix = []
    for label in labels:
        for config in configs:
            row_labels.append(f"{label}\n{CONFIG_SHORT[config]}")
            row = [fn(data, label, config) for _, fn in metric_defs]
            matrix.append(row)
    
    matrix = np.array(matrix)
    col_labels = [mn for mn, _ in metric_defs]
    
    fig, ax = plt.subplots(figsize=(10, max(8, len(row_labels) * 0.5)))
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
    
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    
    # Add value annotations
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7, color=color)
    
    plt.colorbar(im, ax=ax, shrink=0.8, label="Score (normalised to 0-1)")
    ax.set_title("Comprehensive Metrics Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: metrics_heatmap.png")


# =============================================================================
# Plot 9: Per-Section Clinical Accuracy Table (Priority 4 - Approach A)
# =============================================================================

CLINICAL_SECTIONS = [
    ("chief_complaint", "Chief Complaint"),
    ("history_of_present_illness", "HPI"),
    ("past_medical_history", "Past Medical Hx"),
    ("medications", "Medications"),
    ("allergies", "Allergies"),
    ("physical_examination", "Examination"),
    ("assessment", "Assessment"),
    ("plan", "Plan"),
    ("safety_netting", "Safety Netting"),
]

SECTION_PATTERNS = {
    "chief_complaint": r"\*?\*?Chief Complaint\*?\*?\s*:?\s*(.*?)(?=\*?\*?(?:History|Past|Medications|Allergies|Examination|Physical|Assessment|Plan|Safety|$))",
    "history_of_present_illness": r"\*?\*?History of Present Illness\*?\*?\s*:?\s*(.*?)(?=\*?\*?(?:Past|Medications|Allergies|Examination|Physical|Assessment|Plan|Safety|$))",
    "past_medical_history": r"\*?\*?Past Medical History\*?\*?\s*:?\s*(.*?)(?=\*?\*?(?:Medications|Allergies|Examination|Physical|Assessment|Plan|Safety|$))",
    "medications": r"\*?\*?Medications\*?\*?\s*:?\s*(.*?)(?=\*?\*?(?:Allergies|Examination|Physical|Assessment|Plan|Safety|$))",
    "allergies": r"\*?\*?Allergies\*?\*?\s*:?\s*(.*?)(?=\*?\*?(?:Examination|Physical|Assessment|Plan|Safety|$))",
    "physical_examination": r"\*?\*?(?:Examination Findings|Physical Examination)\*?\*?\s*:?\s*(.*?)(?=\*?\*?(?:Assessment|Plan|Safety|$))",
    "assessment": r"\*?\*?Assessment\*?\*?\s*:?\s*(.*?)(?=\*?\*?(?:Plan|Safety|$))",
    "plan": r"\*?\*?Plan\*?\*?\s*:?\s*(.*?)(?=\*?\*?(?:Safety|$))",
    "safety_netting": r"\*?\*?Safety Netting\*?\*?\s*:?\s*(.*?)$",
}


def parse_sections(text: str) -> Dict[str, str]:
    """Parse clinical summary into sections."""
    sections = {}
    for key, pattern in SECTION_PATTERNS.items():
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            sections[key] = match.group(1).strip()
    return sections


def compute_section_rouge(generated: str, reference: str) -> float:
    """Compute ROUGE-L for a single section pair."""
    if not generated.strip() or not reference.strip():
        return 0.0
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference, generated)
        return scores['rougeL'].fmeasure
    except ImportError:
        # Simple fallback: word overlap
        gen_words = set(generated.lower().split())
        ref_words = set(reference.lower().split())
        if not ref_words:
            return 0.0
        overlap = gen_words & ref_words
        p = len(overlap) / len(gen_words) if gen_words else 0
        r = len(overlap) / len(ref_words)
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def plot_per_section_analysis(
    eval_json_paths: List[str],
    labels: List[str],
    output_dir: Path,
    config: str = "ft_rag",
):
    """
    Per-section ROUGE-L analysis (Priority 4 - Approach A).
    Parses generated and reference summaries into clinical sections,
    computes ROUGE-L per section per model.
    """
    all_section_scores = {}  # label -> section_key -> list of scores
    
    for eval_path, label in zip(eval_json_paths, labels):
        with open(eval_path) as f:
            eval_data = json.load(f)
        
        references = eval_data.get("references", [])
        comp = eval_data.get("comparative", {}).get(config, {})
        candidates = comp.get("raw_outputs", [])
        
        if not references or not candidates:
            print(f"  {label}: No raw_outputs for {config}, skipping per-section analysis")
            continue
        
        section_scores = {key: [] for key, _ in CLINICAL_SECTIONS}
        
        for ref, cand in zip(references, candidates):
            ref_sections = parse_sections(ref)
            cand_sections = parse_sections(cand)
            
            for key, _ in CLINICAL_SECTIONS:
                ref_sec = ref_sections.get(key, "")
                cand_sec = cand_sections.get(key, "")
                if ref_sec:  # Only score if reference has this section
                    score = compute_section_rouge(cand_sec, ref_sec)
                    section_scores[key].append(score)
        
        all_section_scores[label] = {
            key: (sum(scores) / len(scores) if scores else 0)
            for key, scores in section_scores.items()
        }
    
    if not all_section_scores:
        print("No per-section data available, skipping plot")
        return
    
    # Plot
    section_keys = [k for k, _ in CLINICAL_SECTIONS]
    section_names = [n for _, n in CLINICAL_SECTIONS]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(labels), 3)))
    x = np.arange(len(section_keys))
    w = 0.8 / len(labels)
    
    for i, label in enumerate(labels):
        vals = [all_section_scores.get(label, {}).get(k, 0) for k in section_keys]
        bars = ax.bar(x + i * w, vals, w, label=label, color=colors[i])
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7, rotation=45)
    
    ax.set_xlabel("Clinical Section")
    ax.set_ylabel("ROUGE-L")
    ax.set_title(f"Per-Section ROUGE-L: {CONFIG_LABELS.get(config, config).replace(chr(10), ' ')}", 
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x + w * (len(labels) - 1) / 2)
    ax.set_xticklabels(section_names, fontsize=9, rotation=30, ha="right")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(output_dir / "per_section_rouge.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: per_section_rouge.png")
    
    # Also save as markdown table
    table_path = output_dir / "per_section_table.md"
    lines = ["# Per-Section ROUGE-L Scores", "",
             "| Section |" + "".join(f" {l} |" for l in labels),
             "|---|" + "---|" * len(labels)]
    for key, name in CLINICAL_SECTIONS:
        row = f"| {name} |"
        for label in labels:
            val = all_section_scores.get(label, {}).get(key, 0)
            row += f" {val:.4f} |"
        lines.append(row)
    with open(table_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: per_section_table.md")


# =============================================================================
# Plot 10: Model Size vs Performance
# =============================================================================

def plot_size_vs_performance(data: Dict, labels: List[str], output_dir: Path):
    """Scatter plot of model size vs key metrics."""
    # Extract model sizes from labels (parse "3.8B", "3B", "1B")
    sizes = []
    for label in labels:
        match = re.search(r'(\d+\.?\d*)\s*B', label)
        sizes.append(float(match.group(1)) if match else 0)
    
    if not any(sizes):
        print("Cannot parse model sizes from labels, skipping size plot")
        return
    
    metrics = [
        ("ROUGE-L", lambda l: get_metric(data, l, "ft_rag", "rouge", "rougeL")),
        ("BLEU-4", lambda l: get_metric(data, l, "ft_rag", "bleu", "avg_bleu4")),
        ("Judge Overall", lambda l: get_metric(data, l, "ft_rag", "llm_judge", "avg_overall")),
        ("Hallucination", lambda l: get_metric(data, l, "ft_rag", "llm_judge", "avg_hallucination")),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(labels), 3)))
    
    for ax_idx, (mn, fn) in enumerate(metrics):
        ax = axes[ax_idx // 2][ax_idx % 2]
        vals = [fn(l) for l in labels]
        
        for i, (s, v, label) in enumerate(zip(sizes, vals, labels)):
            ax.scatter(s, v, s=150, color=colors[i], zorder=5, edgecolors="black")
            ax.annotate(label, (s, v), textcoords="offset points",
                       xytext=(10, 5), fontsize=8)
        
        # Trend line
        if len(sizes) >= 2:
            z = np.polyfit(sizes, vals, 1)
            p = np.poly1d(z)
            xs = np.linspace(min(sizes) * 0.8, max(sizes) * 1.2, 100)
            ax.plot(xs, p(xs), "--", color="gray", alpha=0.5)
        
        ax.set_xlabel("Model Size (B params)")
        ax.set_ylabel(mn)
        ax.set_title(f"Model Size vs {mn}", fontsize=11, fontweight="bold")
    
    fig.suptitle("Model Size-Performance Tradeoff (FT+RAG)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "size_vs_performance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: size_vs_performance.png")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dissertation plots")
    parser.add_argument("--results", required=True,
                        help="Path to post_eval_metrics JSON")
    parser.add_argument("--eval-jsons", nargs="+", default=None,
                        help="Paths to evaluation JSONs (for per-section analysis)")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Model labels (required with --eval-jsons)")
    parser.add_argument("--output-dir", default="./dissertation_plots")
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data = load_post_eval(args.results)
    labels = data.get("labels", [])
    
    print(f"Models: {labels}")
    print(f"Output: {output_dir}\n")
    
    plot_rouge_variants(data, labels, output_dir)
    plot_bleu_variants(data, labels, output_dir)
    plot_medcon(data, labels, output_dir)
    plot_bertscore(data, labels, output_dir)
    plot_judge_dimensions(data, labels, output_dir)
    plot_radar(data, labels, output_dir)
    plot_rag_backends(data, labels, output_dir)
    plot_heatmap(data, labels, output_dir)
    plot_size_vs_performance(data, labels, output_dir)
    
    # Per-section analysis (needs evaluation JSONs)
    if args.eval_jsons:
        eval_labels = args.labels or labels
        if len(args.eval_jsons) != len(eval_labels):
            print("ERROR: --eval-jsons count must match --labels count")
        else:
            plot_per_section_analysis(args.eval_jsons, eval_labels, output_dir)
    
    print(f"\nAll plots saved to {output_dir}/")
