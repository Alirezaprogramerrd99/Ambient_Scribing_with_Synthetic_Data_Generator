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
# Per-Section Multi-Metric Analysis
# =============================================================================

def compute_section_metrics(generated: str, reference: str) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-L, BLEU, and BERTScore for a section pair."""
    if not generated.strip() or not reference.strip():
        return {"rouge1": 0.0, "rougeL": 0.0, "bleu": 0.0, "bertscore": 0.0}
    
    result = {"rouge1": 0.0, "rougeL": 0.0, "bleu": 0.0, "bertscore": 0.0}
    
    # ROUGE
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, generated)
        result["rouge1"] = scores['rouge1'].fmeasure
        result["rougeL"] = scores['rougeL'].fmeasure
    except ImportError:
        gen_words = set(generated.lower().split())
        ref_words = set(reference.lower().split())
        if ref_words:
            overlap = gen_words & ref_words
            p = len(overlap) / len(gen_words) if gen_words else 0
            r = len(overlap) / len(ref_words)
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            result["rouge1"] = f1
            result["rougeL"] = f1
    
    # BLEU (sentence-level)
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        ref_tokens = reference.lower().split()
        gen_tokens = generated.lower().split()
        if len(ref_tokens) >= 4 and len(gen_tokens) >= 4:
            smooth = SmoothingFunction().method1
            result["bleu"] = sentence_bleu(
                [ref_tokens], gen_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smooth,
            )
    except ImportError:
        pass
    
    # BERTScore
    try:
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn(
            [generated], [reference],
            model_type="roberta-large",
            verbose=False,
        )
        result["bertscore"] = float(F1[0])
    except (ImportError, Exception):
        pass
    
    return result


def compute_all_per_section_metrics(
    eval_json_paths: List[str],
    labels: List[str],
    configs: List[str] = None,
) -> Dict:
    """
    Compute per-section metrics for all models, all configs.
    
    Returns:
        {label: {config: {section_key: {metric: avg_value}}}}
    """
    if configs is None:
        configs = ["ft_only", "ft_rag", "baseline", "rag_only", "teacher"]
    
    results = {}
    
    for eval_path, label in zip(eval_json_paths, labels):
        with open(eval_path) as f:
            eval_data = json.load(f)
        
        references = eval_data.get("references", [])
        results[label] = {}
        
        for config in configs:
            comp = eval_data.get("comparative", {}).get(config, {})
            candidates = comp.get("raw_outputs", [])
            
            if not references or not candidates:
                continue
            
            section_scores = {key: {"rouge1": [], "rougeL": [], "bleu": [], "bertscore": []} 
                            for key, _ in CLINICAL_SECTIONS}
            
            for ref, cand in zip(references, candidates):
                ref_sections = parse_sections(ref)
                cand_sections = parse_sections(cand)
                
                for key, _ in CLINICAL_SECTIONS:
                    ref_sec = ref_sections.get(key, "")
                    cand_sec = cand_sections.get(key, "")
                    if ref_sec:
                        metrics = compute_section_metrics(cand_sec, ref_sec)
                        for mk, mv in metrics.items():
                            if mk in section_scores[key]:
                                section_scores[key][mk].append(mv)
            
            # Average
            results[label][config] = {}
            for key, scores_dict in section_scores.items():
                results[label][config][key] = {
                    mk: (sum(vals) / len(vals) if vals else 0.0)
                    for mk, vals in scores_dict.items()
                }
        
        # Also process RAG ablation configs if present
        rag_backends = eval_data.get("rag_backends", {})
        for rag_name, rag_data in rag_backends.items():
            if rag_data.get("error") or not rag_data:
                continue
            candidates = rag_data.get("raw_outputs", [])
            if not candidates:
                continue
            
            cfg_key = f"rag_{rag_name}"
            section_scores = {key: {"rouge1": [], "rougeL": [], "bleu": [], "bertscore": []} 
                            for key, _ in CLINICAL_SECTIONS}
            
            for ref, cand in zip(references, candidates):
                ref_sections = parse_sections(ref)
                cand_sections = parse_sections(cand)
                for key, _ in CLINICAL_SECTIONS:
                    ref_sec = ref_sections.get(key, "")
                    cand_sec = cand_sections.get(key, "")
                    if ref_sec:
                        metrics = compute_section_metrics(cand_sec, ref_sec)
                        for mk, mv in metrics.items():
                            if mk in section_scores[key]:
                                section_scores[key][mk].append(mv)
            
            results[label][cfg_key] = {}
            for key, scores_dict in section_scores.items():
                results[label][cfg_key][key] = {
                    mk: (sum(vals) / len(vals) if vals else 0.0)
                    for mk, vals in scores_dict.items()
                }
    
    return results


# =============================================================================
# Table A: Combined Automated Metrics (All Models × Configs × Sections)
# =============================================================================

RAG_METHOD_LABELS = {
    "baseline": "None",
    "rag_only": "Dense+Rerank",
    "ft_only": "None",
    "ft_rag": "Dense+Rerank",
    "teacher": "None (API)",
    "rag_dense_only": "Dense Only",
    "rag_dense_rerank": "Dense+Rerank",
    "rag_dense_rerank_qe": "Dense+Rerank+QE",
    "rag_full_medical": "Full Medical",
}

def generate_table_a_automated(
    data: Dict,
    labels: List[str],
    per_section: Dict,
    medcon_data: Optional[Dict],
    output_dir: Path,
):
    """
    Table A: Combined Automated Evaluation Metrics.
    One table with all models, configs, overall + per-section.
    """
    configs = ["ft_only", "ft_rag", "baseline", "rag_only", "teacher"]
    
    lines = [
        "# Table A: Automated Evaluation Metrics",
        "",
        "All metrics computed on 50-sample test set. Per-section values show ROUGE-L.",
        "MEDCON uses QuickUMLS + UMLS 2025AB where available, regex fallback otherwise.",
        "",
        "| Model | Config | RAG | Scope | ROUGE-1 | ROUGE-L | BLEU-4 | MEDCON-F1 | BERTScore | Perplexity |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    
    for label in labels:
        first_label = True
        for config in configs:
            # Overall row
            r1 = get_metric(data, label, config, "rouge", "rouge1")
            rl = get_metric(data, label, config, "rouge", "rougeL")
            bl = get_metric(data, label, config, "bleu", "avg_bleu4")
            bs = get_metric(data, label, config, "bertscore", "f1")
            
            # MEDCON: prefer actual QuickUMLS data if available
            mc = 0.0
            if medcon_data and label in medcon_data:
                mc_cfg = medcon_data[label].get(config, {})
                mc = mc_cfg.get("medcon_f1", 0.0)
            if mc == 0.0:
                mc = get_metric(data, label, config, "medcon", "f1")
            
            ppl = get_metric(data, label, config, "perplexity")
            rag_method = RAG_METHOD_LABELS.get(config, "—")
            
            model_col = label if first_label else ""
            first_label = False
            config_label = CONFIG_SHORT.get(config, config)
            
            # Format with dashes for missing values
            def fmt(v, decimals=3):
                return f"{v:.{decimals}f}" if v > 0 else "—"
            
            lines.append(
                f"| {model_col} | {config_label} | {rag_method} | **Overall** | "
                f"{fmt(r1)} | {fmt(rl)} | {fmt(bl)} | {fmt(mc)} | {fmt(bs)} | "
                f"{fmt(ppl)} |"
            )
            
            # Per-section rows with all available metrics
            sec_data = per_section.get(label, {}).get(config, {})
            for key, name in CLINICAL_SECTIONS:
                sd = sec_data.get(key, {})
                sec_r1 = sd.get("rouge1", 0.0)
                sec_rl = sd.get("rougeL", 0.0)
                sec_bl = sd.get("bleu", 0.0)
                sec_bs = sd.get("bertscore", 0.0)
                if sec_rl > 0:
                    lines.append(
                        f"| | | | {name} | "
                        f"{fmt(sec_r1)} | {fmt(sec_rl)} | {fmt(sec_bl)} | — | {fmt(sec_bs)} | — |"
                    )
        
        # Separator between models
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
    
    table_path = output_dir / "table_a_automated_metrics.md"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved: table_a_automated_metrics.md")


# =============================================================================
# Table B: LLM Judge Scores
# =============================================================================

def generate_table_b_judge(
    data: Dict,
    labels: List[str],
    output_dir: Path,
):
    """Table B: LLM-as-Judge evaluation scores."""
    configs = ["ft_only", "ft_rag", "baseline", "rag_only", "teacher"]
    
    judge_dims = [
        ("avg_clinical_accuracy", "Clin.Acc"),
        ("avg_completeness", "Complete"),
        ("avg_hallucination", "Halluc.↑"),
        ("avg_clinical_safety", "Safety"),
        ("avg_coherence", "Coherence"),
        ("avg_conciseness", "Concise"),
        ("avg_overall", "Overall"),
    ]
    
    header = "| Model | Config | RAG | " + " | ".join(d[1] for d in judge_dims) + " | Critical Err |"
    sep = "|---|---|---|" + "---|" * len(judge_dims) + "---|"
    
    lines = [
        "# Table B: LLM-as-Judge Evaluation (GPT-4o-mini, 1-5 scale)",
        "",
        "Higher is better for all dimensions. Halluc.↑ means higher = less hallucination.",
        "",
        header, sep,
    ]
    
    for label in labels:
        first_label = True
        for config in configs:
            vals = []
            for key, _ in judge_dims:
                v = get_metric(data, label, config, "llm_judge", key)
                vals.append(f"{v:.2f}" if v > 0 else "—")
            
            # Critical errors - stored under llm_judge in the post_eval JSON
            n_err = get_metric(data, label, config, "llm_judge", "samples_with_critical_errors")
            n_total = get_metric(data, label, config, "llm_judge", "num_samples")
            if n_total == 0:
                # Try alternative: total from top level
                n_total = get_metric(data, label, config, "num_samples")
            err_str = f"{int(n_err)}/{int(n_total)}" if n_total > 0 else "—"
            
            model_col = label if first_label else ""
            first_label = False
            config_label = CONFIG_SHORT.get(config, config)
            rag_method = RAG_METHOD_LABELS.get(config, "—")
            
            lines.append(
                f"| {model_col} | {config_label} | {rag_method} | "
                + " | ".join(vals) + f" | {err_str} |"
            )
        lines.append(sep)
    
    table_path = output_dir / "table_b_judge_scores.md"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved: table_b_judge_scores.md")


# =============================================================================
# Table C: RAG Ablation Results
# =============================================================================

def generate_table_c_rag_ablation(
    eval_json_paths: List[str],
    labels: List[str],
    per_section: Dict,
    medcon_data: Optional[Dict],
    output_dir: Path,
):
    """
    Table C: RAG Ablation with per-section ROUGE-L.
    Includes teacher as reference row (no RAG).
    """
    # Support both old-style keys (llama_index, hybrid) and new-style (dense_only, dense_rerank, etc.)
    rag_configs_new = ["dense_only", "dense_rerank", "dense_rerank_qe", "full_medical"]
    rag_configs_old = ["llama_index", "manual", "hybrid"]
    
    # Map old keys to display names
    old_key_labels = {
        "llama_index": "Dense (LlamaIndex)",
        "manual": "Custom Dense",
        "hybrid": "Full Medical",
    }
    
    lines = [
        "# Table C: RAG Ablation Study",
        "",
        "All results use fine-tuned models. Teacher (GPT-4o-mini) shown as reference.",
        "Per-section values show all metrics for all clinical sections.",
        "",
        "| Model | RAG Config | Scope | ROUGE-L | ROUGE-1 | BLEU-4 | BERTScore | MEDCON-F1 | Judge Avg |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    
    for eval_path, label in zip(eval_json_paths, labels):
        with open(eval_path) as f:
            eval_data = json.load(f)
        
        first_label = True
        
        rag_backends = eval_data.get("rag_backends", {})
        
        # Determine which key format is present
        available_keys = list(rag_backends.keys())
        if any(k in available_keys for k in rag_configs_new):
            rag_keys_to_try = rag_configs_new
        else:
            rag_keys_to_try = rag_configs_old
        
        for rag_name in rag_keys_to_try:
            rag_data = rag_backends.get(rag_name, {})
            if not rag_data or rag_data.get("error"):
                continue
            
            metrics = rag_data.get("metrics", {})
            rouge_data = metrics.get("rouge", {})
            rl = rouge_data.get("rougeL", 0)
            r1 = rouge_data.get("rouge1", 0)
            bl = metrics.get("bleu", {}).get("avg_bleu4", 0)
            bs_val = metrics.get("bertscore", {}).get("f1", 0)
            
            mc = 0.0
            if medcon_data and label in medcon_data:
                mc_key = f"rag_{rag_name}"
                mc = medcon_data[label].get(mc_key, {}).get("medcon_f1", 0.0)
            if mc == 0.0:
                mc = metrics.get("medcon", {}).get("f1", 0)
            
            judge_avg = metrics.get("llm_judge", {}).get("avg_overall", 0)
            
            rag_label = RAG_METHOD_LABELS.get(f"rag_{rag_name}", 
                        old_key_labels.get(rag_name, rag_name))
            model_col = label if first_label else ""
            first_label = False
            
            def fmt_c(v, decimals=3):
                return f"{v:.{decimals}f}" if v > 0 else "—"
            
            lines.append(
                f"| {model_col} | {rag_label} | **Overall** | "
                f"{fmt_c(rl)} | {fmt_c(r1)} | {fmt_c(bl)} | {fmt_c(bs_val)} | {fmt_c(mc)} | {fmt_c(judge_avg, 2)} |"
            )
            
            # Per-section for ALL clinical sections
            sec_data = per_section.get(label, {}).get(f"rag_{rag_name}", {})
            if not sec_data:
                sec_data = per_section.get(label, {}).get(rag_name, {})
            for key, name in CLINICAL_SECTIONS:
                sd = sec_data.get(key, {})
                sec_rl = sd.get("rougeL", 0.0)
                sec_r1 = sd.get("rouge1", 0.0)
                sec_bl = sd.get("bleu", 0.0)
                sec_bs = sd.get("bertscore", 0.0)
                if sec_rl > 0:
                    lines.append(
                        f"| | | {name} | {fmt_c(sec_rl)} | {fmt_c(sec_r1)} | {fmt_c(sec_bl)} | {fmt_c(sec_bs)} | — | — |"
                    )
        # Teacher reference row
        comp = eval_data.get("comparative", {}).get("teacher", {})
        if comp and not comp.get("error"):
            t_metrics = comp.get("metrics", {})
            t_rl = t_metrics.get("rouge", {}).get("rougeL", 0)
            t_r1 = t_metrics.get("rouge", {}).get("rouge1", 0)
            t_bl = t_metrics.get("bleu", {}).get("avg_bleu4", 0)
            t_bs = t_metrics.get("bertscore", {}).get("f1", 0)
            t_mc = 0.0
            if medcon_data and label in medcon_data:
                t_mc = medcon_data[label].get("teacher", {}).get("medcon_f1", 0.0)
            if t_mc == 0.0:
                t_mc = t_metrics.get("medcon", {}).get("f1", 0)
            t_judge = t_metrics.get("llm_judge", {}).get("avg_overall", 0)
            
            def fmt_t(v, decimals=3):
                return f"{v:.{decimals}f}" if v > 0 else "—"
            
            lines.append(
                f"| | Teacher (ref.) | **Overall** | "
                f"{fmt_t(t_rl)} | {fmt_t(t_r1)} | {fmt_t(t_bl)} | {fmt_t(t_bs)} | {fmt_t(t_mc)} | {fmt_t(t_judge, 2)} |"
            )
        
        lines.append("|---|---|---|---|---|---|---|---|---|")
    
    table_path = output_dir / "table_c_rag_ablation.md"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved: table_c_rag_ablation.md")

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
    parser = argparse.ArgumentParser(
        description="Generate dissertation plots and tables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic plots only (from post_eval_metrics JSON)
    python plot_dissertation.py --results ./post_eval_metrics.json

    # Full analysis with per-section metrics, tables, and actual MEDCON
    python plot_dissertation.py \\
        --results ./post_eval_metrics.json \\
        --eval-jsons eval_phi.json eval_llama3b.json eval_llama1b.json \\
        --labels "Phi-3.5 (3.8B)" "Llama-3.2 (3B)" "Llama-3.2 (1B)" \\
        --medcon-json ./medcon_results/medcon_results.json \\
        --output-dir ./dissertation_plots
        """,
    )
    parser.add_argument("--results", required=True,
                        help="Path to post_eval_metrics JSON")
    parser.add_argument("--eval-jsons", nargs="+", default=None,
                        help="Paths to evaluation JSONs (for per-section + tables)")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Model labels (required with --eval-jsons)")
    parser.add_argument("--medcon-json", default=None,
                        help="Path to medcon_results.json (actual QuickUMLS MEDCON)")
    parser.add_argument("--output-dir", default="./dissertation_plots")
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data = load_post_eval(args.results)
    labels = data.get("labels", [])
    
    # Load actual MEDCON data if provided
    medcon_data = None
    if args.medcon_json:
        with open(args.medcon_json) as f:
            medcon_data = json.load(f)
        print(f"Loaded actual MEDCON data from {args.medcon_json}")
    
    print(f"Models: {labels}")
    print(f"Output: {output_dir}\n")
    
    # ---- Generate all plots ----
    print("=" * 50)
    print("GENERATING PLOTS")
    print("=" * 50)
    plot_rouge_variants(data, labels, output_dir)
    plot_bleu_variants(data, labels, output_dir)
    plot_medcon(data, labels, output_dir)
    plot_bertscore(data, labels, output_dir)
    plot_judge_dimensions(data, labels, output_dir)
    plot_radar(data, labels, output_dir)
    plot_rag_backends(data, labels, output_dir)
    plot_heatmap(data, labels, output_dir)
    plot_size_vs_performance(data, labels, output_dir)
    
    # ---- Per-section analysis + tables (needs evaluation JSONs) ----
    if args.eval_jsons:
        eval_labels = args.labels or labels
        if len(args.eval_jsons) != len(eval_labels):
            print("ERROR: --eval-jsons count must match --labels count")
        else:
            # Per-section ROUGE-L plot (existing)
            print("\n" + "=" * 50)
            print("PER-SECTION ANALYSIS")
            print("=" * 50)
            plot_per_section_analysis(args.eval_jsons, eval_labels, output_dir)
            
            # Compute comprehensive per-section metrics
            print("\nComputing per-section metrics for all configs...")
            per_section = compute_all_per_section_metrics(
                args.eval_jsons, eval_labels,
            )
            print(f"  Per-section data for {len(per_section)} models")
            
            # ---- Generate tables ----
            print("\n" + "=" * 50)
            print("GENERATING TABLES")
            print("=" * 50)
            
            generate_table_a_automated(
                data, eval_labels, per_section, medcon_data, output_dir,
            )
            
            generate_table_b_judge(data, eval_labels, output_dir)
            
            generate_table_c_rag_ablation(
                args.eval_jsons, eval_labels, per_section, medcon_data, output_dir,
            )
    else:
        print("\nSkipping per-section analysis and tables (no --eval-jsons provided)")
    
    print(f"\nAll outputs saved to {output_dir}/")