"""
RAG Ablation Study Plotter

Reads evaluation JSONs and plots the progressive RAG ablation results.
Shows how each RAG component (reranking, query expansion, clinical filtering)
contributes to overall performance.

Usage:
    # Single model
    python plot_rag_ablation.py \
        --eval-json ./rag_ablation_phi/evaluation_*.json \
        --label "Phi-3.5 (3.8B)"

    # Multiple models
    python plot_rag_ablation.py \
        --eval-json ./rag_ablation_phi/evaluation_*.json \
        --eval-json ./rag_ablation_llama3b/evaluation_*.json \
        --labels "Phi-3.5 (3.8B)" "Llama-3.2 (3B)" \
        --output-dir ./rag_ablation_plots

Author: Alireza Rashidi
MSc Project: Trustworthy SLMs for Ambient Clinical Scribing
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Descriptive Labels for RAG Configs
# =============================================================================

RAG_CONFIG_LABELS = {
    "dense_only": "Dense\nRetrieval",
    "dense_rerank": "Dense +\nReranker",
    "dense_rerank_qe": "Dense +\nReranker +\nQuery Exp.",
    "full_medical": "Full\nMedical\nPipeline",
    # Legacy
    "llama_index": "Dense\nRetrieval",
    "hybrid": "Full\nMedical\nPipeline",
}

RAG_SHORT_LABELS = {
    "dense_only": "Dense",
    "dense_rerank": "+Rerank",
    "dense_rerank_qe": "+Rerank+QE",
    "full_medical": "Full Medical",
}

RAG_DESCRIPTIONS = {
    "dense_only": "BGE embeddings + ChromaDB cosine similarity",
    "dense_rerank": "+ Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)",
    "dense_rerank_qe": "+ Medical query expansion (synonym injection)",
    "full_medical": "+ Clinical relevance post-filtering (entity matching)",
}


# =============================================================================
# Data Extraction
# =============================================================================

def load_rag_results(eval_path: str) -> Dict:
    """Load RAG ablation results from evaluation JSON."""
    with open(eval_path) as f:
        data = json.load(f)
    
    rag = data.get("rag_backends", {})
    if not rag:
        print(f"  WARNING: No rag_backends in {eval_path}")
    
    # Also include ft_rag from comparative as baseline reference
    ft_rag = data.get("comparative", {}).get("ft_rag", {})
    ft_only = data.get("comparative", {}).get("ft_only", {})
    
    return {
        "rag_ablation": rag,
        "ft_rag_ref": ft_rag,
        "ft_only_ref": ft_only,
        "config": data.get("config", {}),
    }


def get_metric(rag_data: Dict, config_name: str, *keys) -> float:
    """Safely extract metric from RAG results."""
    d = rag_data.get(config_name, {})
    if d.get("error"):
        return 0
    d = d.get("metrics", d)
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, 0)
        else:
            return 0
    return d if isinstance(d, (int, float)) else 0


# =============================================================================
# Plots
# =============================================================================

def plot_ablation(
    eval_json_paths: List[str],
    labels: List[str],
    output_dir: str,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load all results
    all_data = {}
    for ep, label in zip(eval_json_paths, labels):
        result = load_rag_results(ep)
        rag = result["rag_ablation"]
        if rag:
            all_data[label] = rag
            configs_found = [k for k in rag if not rag[k].get("error")]
            print(f"{label}: found configs {configs_found}")
        else:
            print(f"{label}: NO RAG ablation data found")
    
    if not all_data:
        print("ERROR: No RAG ablation data found in any file")
        return
    
    # Determine which configs are present
    all_configs = []
    for label_data in all_data.values():
        for cfg in label_data:
            if cfg not in all_configs and not label_data[cfg].get("error"):
                all_configs.append(cfg)
    
    # Order them logically
    desired_order = ["dense_only", "dense_rerank", "dense_rerank_qe", "full_medical"]
    configs = [c for c in desired_order if c in all_configs]
    # Add any others not in desired order
    configs += [c for c in all_configs if c not in configs]
    
    print(f"Configs to plot: {configs}")
    
    n_models = len(all_data)
    model_labels = list(all_data.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_models, 3)))
    
    # ---- Plot 1: ROUGE-L across RAG configs ----
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(configs))
    w = 0.8 / max(n_models, 1)
    
    for i, label in enumerate(model_labels):
        rag = all_data[label]
        vals = [get_metric(rag, c, "rouge_l") for c in configs]
        bars = ax.bar(x + i * w, vals, w, label=label, color=colors[i])
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    
    ax.set_xlabel("RAG Configuration")
    ax.set_ylabel("ROUGE-L")
    ax.set_title("RAG Ablation: ROUGE-L Score", fontsize=13, fontweight="bold")
    ax.set_xticks(x + w * (n_models - 1) / 2)
    ax.set_xticklabels([RAG_CONFIG_LABELS.get(c, c) for c in configs], fontsize=8)
    ax.legend(fontsize=9)
    ax.set_ylim(0, min(max(vals) * 1.3, 1.0) if vals else 1.0)
    plt.tight_layout()
    plt.savefig(output_path / "rag_ablation_rouge.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: rag_ablation_rouge.png")
    
    # ---- Plot 2: Judge dimensions across RAG configs ----
    judge_dims = [
        ("avg_overall", "Overall"),
        ("avg_clinical_accuracy", "Accuracy"),
        ("avg_hallucination", "Hallucination"),
        ("avg_clinical_safety", "Safety"),
        ("avg_coherence", "Coherence"),
    ]
    
    fig, axes = plt.subplots(1, len(judge_dims), figsize=(4 * len(judge_dims), 6), sharey=True)
    
    for ax_idx, (jkey, jname) in enumerate(judge_dims):
        ax = axes[ax_idx]
        x = np.arange(len(configs))
        
        for i, label in enumerate(model_labels):
            rag = all_data[label]
            vals = [get_metric(rag, c, "llm_judge", jkey) for c in configs]
            bars = ax.bar(x + i * w, vals, w, label=label, color=colors[i])
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                            f"{val:.2f}", ha="center", va="bottom", fontsize=6, rotation=45)
        
        ax.set_title(jname, fontsize=11, fontweight="bold")
        ax.set_xticks(x + w * (n_models - 1) / 2)
        ax.set_xticklabels([RAG_SHORT_LABELS.get(c, c) for c in configs], fontsize=7, rotation=30, ha="right")
        ax.set_ylim(0, 5.5)
        if ax_idx == 0:
            ax.set_ylabel("Score (/5)")
    
    axes[0].legend(fontsize=7)
    fig.suptitle("RAG Ablation: LLM Judge Scores by Dimension", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path / "rag_ablation_judge.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: rag_ablation_judge.png")
    
    # ---- Plot 3: Generation time comparison ----
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(configs))
    
    for i, label in enumerate(model_labels):
        rag = all_data[label]
        vals = [get_metric(rag, c, "avg_generation_time") for c in configs]
        bars = ax.bar(x + i * w, vals, w, label=label, color=colors[i])
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                        f"{val:.1f}s", ha="center", va="bottom", fontsize=8)
    
    ax.set_xlabel("RAG Configuration")
    ax.set_ylabel("Avg Generation Time (seconds)")
    ax.set_title("RAG Ablation: Latency Impact", fontsize=13, fontweight="bold")
    ax.set_xticks(x + w * (n_models - 1) / 2)
    ax.set_xticklabels([RAG_CONFIG_LABELS.get(c, c) for c in configs], fontsize=8)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path / "rag_ablation_latency.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: rag_ablation_latency.png")
    
    # ---- Plot 4: RAG retrieval score ----
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(configs))
    
    for i, label in enumerate(model_labels):
        rag = all_data[label]
        vals = [get_metric(rag, c, "avg_rag_score") for c in configs]
        bars = ax.bar(x + i * w, vals, w, label=label, color=colors[i])
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    
    ax.set_xlabel("RAG Configuration")
    ax.set_ylabel("Avg RAG Retrieval Score")
    ax.set_title("RAG Ablation: Retrieval Quality", fontsize=13, fontweight="bold")
    ax.set_xticks(x + w * (n_models - 1) / 2)
    ax.set_xticklabels([RAG_CONFIG_LABELS.get(c, c) for c in configs], fontsize=8)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path / "rag_ablation_retrieval.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: rag_ablation_retrieval.png")
    
    # ---- Plot 5: Combined summary table ----
    fig, ax = plt.subplots(figsize=(14, 3 + n_models * len(configs) * 0.35))
    ax.axis("off")
    
    col_labels = ["Model", "RAG Config", "ROUGE-L", "Judge Overall", "Hallucination", "Safety", "Time (s)"]
    table_data = []
    cell_colors = []
    
    for label in model_labels:
        rag = all_data[label]
        for ci, cfg in enumerate(configs):
            row = [
                label if ci == 0 else "",
                RAG_SHORT_LABELS.get(cfg, cfg),
                f"{get_metric(rag, cfg, 'rouge_l'):.3f}",
                f"{get_metric(rag, cfg, 'llm_judge', 'avg_overall'):.2f}",
                f"{get_metric(rag, cfg, 'llm_judge', 'avg_hallucination'):.2f}",
                f"{get_metric(rag, cfg, 'llm_judge', 'avg_clinical_safety'):.2f}",
                f"{get_metric(rag, cfg, 'avg_generation_time'):.1f}",
            ]
            table_data.append(row)
    
    table = ax.table(cellText=table_data, colLabels=col_labels,
                    loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.4)
    
    # Bold header
    for j in range(len(col_labels)):
        table[0, j].set_text_props(fontweight="bold")
    
    ax.set_title("RAG Ablation Study: Progressive Component Analysis",
                 fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(output_path / "rag_ablation_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: rag_ablation_summary.png")
    
    # ---- Save descriptions ----
    desc_path = output_path / "rag_ablation_descriptions.md"
    lines = [
        "# RAG Ablation Study: Component Descriptions", "",
        "| Config | Components | Description |",
        "|---|---|---|",
    ]
    components_progressive = ["Dense retrieval (BGE)", "Cross-encoder reranker", 
                               "Medical query expansion", "Clinical relevance filter"]
    for i, cfg in enumerate(desired_order):
        comp_str = " + ".join(components_progressive[:i+1])
        desc = RAG_DESCRIPTIONS.get(cfg, "")
        lines.append(f"| {RAG_SHORT_LABELS.get(cfg, cfg)} | {comp_str} | {desc} |")
    lines.extend(["", "## Scientific References", "",
                  "- Dense retrieval: Karpukhin et al. (2020) 'Dense Passage Retrieval'",
                  "- Cross-encoder reranking: Nogueira & Cho (2019) 'Passage Re-ranking with BERT'",
                  "- Query expansion: Jagerman et al. (2023) 'Query Expansion by Prompting LLMs'",
                  "- RAG for clinical NLP: Lewis et al. (2020) 'Retrieval-Augmented Generation'",
                  ])
    with open(desc_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved: rag_ablation_descriptions.md")
    
    print(f"\nAll RAG ablation plots saved to {output_path}/")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot RAG ablation results")
    parser.add_argument("--eval-json", action="append", required=True,
                        help="Path to evaluation JSON (repeat for multiple models)")
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output-dir", default="./rag_ablation_plots")
    
    args = parser.parse_args()
    
    if len(args.eval_json) != len(args.labels):
        parser.error(f"--eval-json count ({len(args.eval_json)}) != --labels ({len(args.labels)})")
    
    for ep in args.eval_json:
        if not Path(ep).exists():
            parser.error(f"File not found: {ep}")
    
    plot_ablation(
        eval_json_paths=args.eval_json,
        labels=args.labels,
        output_dir=args.output_dir,
    )
