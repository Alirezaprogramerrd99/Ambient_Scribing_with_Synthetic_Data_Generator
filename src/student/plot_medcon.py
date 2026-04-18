"""
MEDCON Metrics Plotter

Generates dissertation-quality plots from MEDCON results JSON.

Usage:
    python plot_medcon.py \
        --results medcon_results.json \
        --output-dir ./medcon_plots

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


CONFIG_LABELS = {
    "ft_only": "FT Only",
    "ft_rag": "FT + RAG",
    "baseline": "Base",
    "rag_only": "Base + RAG",
    "teacher": "Teacher\n(GPT-4o-mini)",
    "rag_dense_only": "RAG: Dense",
    "rag_dense_rerank": "RAG: Dense\n+ Rerank",
    "rag_dense_rerank_qe": "RAG: Dense\n+ Rerank + QE",
    "rag_full_medical": "RAG: Full\nMedical",
}


def load_results(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def plot_medcon(results: Dict, output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    labels = list(results.keys())
    n_models = len(labels)
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_models, 3)))
    
    # Separate comparative configs from RAG ablation configs
    comp_configs = ["baseline", "rag_only", "ft_only", "ft_rag", "teacher"]
    rag_configs = [k for k in next(iter(results.values())).keys() 
                   if k.startswith("rag_") and k not in comp_configs]
    rag_configs.sort()
    
    # ---- Plot 1: MEDCON F1 across comparative configs ----
    present_comp = [c for c in comp_configs if any(c in results[l] for l in labels)]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(present_comp))
    w = 0.8 / max(n_models, 1)
    
    for i, label in enumerate(labels):
        vals = [results[label].get(c, {}).get("medcon_f1", 0) for c in present_comp]
        bars = ax.bar(x + i * w, vals, w, label=label, color=colors[i],
                     edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 0:
                if val > 0.15:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.02,
                            f"{val:.3f}", ha="center", va="top", fontsize=8,
                            fontweight="bold", color="white")
                else:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                            f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    
    ax.set_ylabel("MEDCON F1 (UMLS Concept Overlap)", fontsize=12)
    ax.set_title("MEDCON F1: Medical Concept Fidelity Across Configurations",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x + w * (n_models - 1) / 2)
    ax.set_xticklabels([CONFIG_LABELS.get(c, c) for c in present_comp], fontsize=9)
    ax.legend(fontsize=9)
    ax.set_ylim(0.45, 0.85)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "medcon_f1_configs.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: medcon_f1_configs.png")
    
    # ---- Plot 2: MEDCON P/R/F1 for FT configs ----
    ft_configs = ["ft_only", "ft_rag"]
    dims = [("medcon_precision", "Precision"), ("medcon_recall", "Recall"), ("medcon_f1", "F1")]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    for ax_idx, (key, dim_name) in enumerate(dims):
        ax = axes[ax_idx]
        x = np.arange(len(ft_configs))
        
        for i, label in enumerate(labels):
            vals = [results[label].get(c, {}).get(key, 0) for c in ft_configs]
            bars = ax.bar(x + i * w, vals, w, label=label, color=colors[i],
                         edgecolor="black", linewidth=0.5)
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.02,
                            f"{val:.3f}", ha="center", va="top", fontsize=9,
                            fontweight="bold", color="white")
        
        ax.set_title(f"MEDCON {dim_name}", fontsize=12, fontweight="bold")
        ax.set_xticks(x + w * (n_models - 1) / 2)
        ax.set_xticklabels([CONFIG_LABELS.get(c, c) for c in ft_configs], fontsize=10)
        ax.set_ylim(0.65, 0.85)
        if ax_idx == 0:
            ax.set_ylabel("Score")
    
    axes[0].legend(fontsize=9)
    fig.suptitle("MEDCON Precision / Recall / F1 for Fine-tuned Configurations",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path / "medcon_prf_ft.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: medcon_prf_ft.png")
    
    # ---- Plot 3: RAG Ablation MEDCON ----
    if rag_configs:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(rag_configs))
        
        for i, label in enumerate(labels):
            vals = [results[label].get(c, {}).get("medcon_f1", 0) for c in rag_configs]
            bars = ax.bar(x + i * w, vals, w, label=label, color=colors[i],
                         edgecolor="black", linewidth=0.5)
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                            f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
        ax.set_ylabel("MEDCON F1", fontsize=12)
        ax.set_title("RAG Ablation: MEDCON F1 (Medical Concept Overlap)",
                     fontsize=14, fontweight="bold")
        ax.set_xticks(x + w * (n_models - 1) / 2)
        ax.set_xticklabels([CONFIG_LABELS.get(c, c) for c in rag_configs], fontsize=9)
        ax.legend(fontsize=9)
        all_vals = [results[l].get(c, {}).get("medcon_f1", 0) 
                   for l in labels for c in rag_configs]
        min_v = min(v for v in all_vals if v > 0) if all_vals else 0.6
        ax.set_ylim(min_v - 0.05, max(all_vals) + 0.05)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / "medcon_rag_ablation.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: medcon_rag_ablation.png")
    
    # ---- Plot 4: Concept Count Comparison ----
    count_configs = ["baseline", "ft_only", "ft_rag", "teacher"]
    present_count = [c for c in count_configs 
                     if any(results[l].get(c, {}).get("avg_generated_concepts", 0) > 0 for l in labels)]
    
    if present_count:
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(present_count))
        
        for i, label in enumerate(labels):
            vals = [results[label].get(c, {}).get("avg_generated_concepts", 0) for c in present_count]
            bars = ax.bar(x + i * w, vals, w, label=label, color=colors[i],
                         edgecolor="black", linewidth=0.5)
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            f"{val:.0f}", ha="center", va="bottom", fontsize=8)
        
        # Reference line
        ref_avg = list(results.values())[0].get("ft_only", {}).get("avg_reference_concepts", 116)
        ax.axhline(y=ref_avg, color="red", linestyle="--", linewidth=1.5, alpha=0.7,
                   label=f"Reference avg ({ref_avg:.0f})")
        
        ax.set_ylabel("Avg UMLS Concepts per Summary", fontsize=12)
        ax.set_title("Medical Concept Density: Generated vs Reference",
                     fontsize=14, fontweight="bold")
        ax.set_xticks(x + w * (n_models - 1) / 2)
        ax.set_xticklabels([CONFIG_LABELS.get(c, c) for c in present_count], fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / "medcon_concept_counts.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: medcon_concept_counts.png")
    
    # ---- Plot 5: Summary Table ----
    fig, ax = plt.subplots(figsize=(18, 3 + n_models * len(present_comp) * 0.3))
    ax.axis("off")
    
    col_labels = ["Model", "Config", "Precision", "Recall", "F1", "Gen CUIs", "Ref CUIs", "Overlap"]
    table_data = []
    
    for label in labels:
        for ci, cfg in enumerate(present_comp):
            d = results[label].get(cfg, {})
            if not d:
                continue
            table_data.append([
                label if ci == 0 else "",
                CONFIG_LABELS.get(cfg, cfg).replace("\n", " "),
                f"{d.get('medcon_precision', 0):.3f}",
                f"{d.get('medcon_recall', 0):.3f}",
                f"{d.get('medcon_f1', 0):.3f}",
                f"{d.get('avg_generated_concepts', 0):.0f}",
                f"{d.get('avg_reference_concepts', 0):.0f}",
                f"{d.get('avg_overlap_concepts', 0):.0f}",
            ])
    
    table = ax.table(cellText=table_data, colLabels=col_labels,
                    loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)
    for j in range(len(col_labels)):
        table[0, j].set_text_props(fontweight="bold")
        table[0, j].set_facecolor("#E8E8E8")
    
    ax.set_title("MEDCON Results: Medical Concept Overlap (UMLS 2025AB + QuickUMLS)",
                 fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(output_path / "medcon_summary_table.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: medcon_summary_table.png")
    
    print(f"\nAll MEDCON plots saved to {output_path}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot MEDCON results")
    parser.add_argument("--results", required=True, help="Path to medcon_results.json")
    parser.add_argument("--output-dir", default="./medcon_plots")
    args = parser.parse_args()
    
    results = load_results(args.results)
    plot_medcon(results, args.output_dir)
