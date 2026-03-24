"""
Logprobs Uncertainty Quantification Analysis

Analyses per-token log probabilities from model generation to quantify
uncertainty and detect potential hallucinations.

Scientific foundations:
    - Kadavath et al. (2022) "Language Models (Mostly) Know What They Know"
      Showed token-level logprobs correlate with factual accuracy.
    - Kuhn et al. (2023) "Semantic Uncertainty: Linguistic Invariances for
      Uncertainty Estimation in Natural Language Generation"
      Proposed using generation probabilities for hallucination detection.
    - Xiong et al. (2024) "Can LLMs Express Their Uncertainty? An Empirical
      Evaluation of Confidence Elicitation in LLMs"
      Comprehensive survey on uncertainty quantification methods for LLMs.
    - Manakul et al. (2023) "SelfCheckGPT: Zero-Resource Black-Box
      Hallucination Detection for Generative Large Language Models"
      Used sampling-based consistency for hallucination detection.

Usage:
    python logprobs_analysis.py \
        --eval-json ./evaluation_results/evaluation_*.json \
        --label "Phi-3.5 (3.8B)" \
        --output-dir ./logprobs_analysis

    # Multiple models
    python logprobs_analysis.py \
        --eval-json eval_phi.json --eval-json eval_llama3b.json \
        --labels "Phi-3.5 (3.8B)" "Llama-3.2 (3B)" \
        --output-dir ./logprobs_analysis

Author: Alireza Rashidi
MSc Project: Trustworthy SLMs for Ambient Clinical Scribing
"""

import json
import argparse
import re
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# Section Parsing (shared with plot_dissertation.py)
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

SECTION_HEADERS = [
    "Chief Complaint", "History of Present Illness", "Past Medical History",
    "Medications", "Allergies", "Examination Findings", "Physical Examination",
    "Assessment", "Plan", "Safety Netting",
]


def find_section_token_ranges(token_texts: List[str]) -> Dict[str, Tuple[int, int]]:
    """
    Map clinical section names to token index ranges in the generated output.
    
    Returns dict: section_key -> (start_token_idx, end_token_idx)
    """
    # Reconstruct text with token positions
    full_text = ""
    token_positions = []  # (start_char, end_char) for each token
    for token in token_texts:
        start = len(full_text)
        full_text += token
        token_positions.append((start, len(full_text)))
    
    # Find section boundaries in the reconstructed text
    section_ranges = {}
    section_starts = []
    
    for header in SECTION_HEADERS:
        pattern = rf"\*?\*?{re.escape(header)}\*?\*?\s*:"
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            section_starts.append((match.start(), header))
    
    # Sort by position
    section_starts.sort(key=lambda x: x[0])
    
    # Convert char positions to token indices
    def char_to_token_idx(char_pos):
        for i, (s, e) in enumerate(token_positions):
            if s <= char_pos < e:
                return i
        return len(token_texts) - 1
    
    for i, (char_start, header) in enumerate(section_starts):
        token_start = char_to_token_idx(char_start)
        if i + 1 < len(section_starts):
            token_end = char_to_token_idx(section_starts[i + 1][0])
        else:
            token_end = len(token_texts)
        
        # Map header to section key
        header_lower = header.lower()
        for key, name in CLINICAL_SECTIONS:
            if name.lower() in header_lower or header_lower in name.lower():
                section_ranges[key] = (token_start, token_end)
                break
    
    return section_ranges


# =============================================================================
# Analysis 1: Per-Sample Confidence Statistics
# =============================================================================

def compute_sample_confidence(logprobs_data: Dict) -> Dict[str, float]:
    """
    Compute confidence statistics for a single sample.
    
    Ref: Kadavath et al. (2022) showed mean logprob correlates with accuracy.
    """
    token_lps = logprobs_data.get("token_logprobs", [])
    if not token_lps:
        return {}
    
    arr = np.array(token_lps)
    
    return {
        "mean_logprob": float(arr.mean()),
        "median_logprob": float(np.median(arr)),
        "std_logprob": float(arr.std()),
        "min_logprob": float(arr.min()),
        "max_logprob": float(arr.max()),
        "p10_logprob": float(np.percentile(arr, 10)),  # 10th percentile (worst tokens)
        "p25_logprob": float(np.percentile(arr, 25)),
        "num_low_conf_tokens": int(np.sum(arr < -3.0)),  # Tokens with logprob < -3
        "frac_low_conf": float(np.mean(arr < -3.0)),
        "perplexity": float(np.exp(-arr.mean())),  # Sequence perplexity
        "num_tokens": len(token_lps),
    }


# =============================================================================
# Analysis 2: Section-Level Confidence
# =============================================================================

def compute_section_confidence(logprobs_data: Dict) -> Dict[str, Dict[str, float]]:
    """
    Compute mean logprob per clinical section.
    
    Sections with lower confidence may indicate hallucination-prone areas.
    Ref: Kuhn et al. (2023) - semantic uncertainty varies across output regions.
    """
    token_lps = logprobs_data.get("token_logprobs", [])
    token_texts = logprobs_data.get("token_texts", [])
    
    if not token_lps or not token_texts:
        return {}
    
    section_ranges = find_section_token_ranges(token_texts)
    
    section_confidence = {}
    for key, (start, end) in section_ranges.items():
        section_lps = token_lps[start:end]
        if section_lps:
            arr = np.array(section_lps)
            section_confidence[key] = {
                "mean_logprob": float(arr.mean()),
                "min_logprob": float(arr.min()),
                "num_tokens": len(section_lps),
                "frac_low_conf": float(np.mean(arr < -3.0)),
                "perplexity": float(np.exp(-arr.mean())),
            }
    
    return section_confidence


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

def run_logprobs_analysis(
    eval_json_paths: List[str],
    labels: List[str],
    output_dir: str,
    config_name: str = "ft_rag",
):
    """Run full logprobs analysis pipeline."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for eval_path, label in zip(eval_json_paths, labels):
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing: {label}")
        logger.info(f"{'='*60}")
        
        with open(eval_path) as f:
            eval_data = json.load(f)
        
        comp = eval_data.get("comparative", {}).get(config_name, {})
        if comp.get("error"):
            logger.warning(f"  {config_name} errored, skipping")
            continue
        
        logprobs_list = comp.get("logprobs_per_sample", [])
        if not logprobs_list or all(lp is None for lp in logprobs_list):
            logger.warning(f"  No logprobs data found. Re-run evaluator with --return-logprobs")
            continue
        
        # Get judge scores for correlation
        judge_per_sample = comp.get("metrics", {}).get("llm_judge_per_sample", [])
        references = eval_data.get("references", [])
        raw_outputs = comp.get("raw_outputs", [])
        
        logger.info(f"  Found {len(logprobs_list)} samples with logprobs")
        
        # ---- Per-sample confidence ----
        sample_stats = []
        for i, lp in enumerate(logprobs_list):
            if lp is None:
                continue
            stats = compute_sample_confidence(lp)
            
            # Add judge scores if available
            if i < len(judge_per_sample):
                js = judge_per_sample[i]
                stats["judge_overall"] = js.get("overall", 0)
                stats["judge_hallucination"] = js.get("hallucination", 0)
                stats["judge_safety"] = js.get("clinical_safety", 0)
                stats["judge_accuracy"] = js.get("clinical_accuracy", 0)
                stats["has_critical_errors"] = len(js.get("critical_errors", [])) > 0
            
            stats["sample_idx"] = i
            sample_stats.append(stats)
        
        # ---- Section-level confidence ----
        section_stats = defaultdict(list)
        for lp in logprobs_list:
            if lp is None:
                continue
            sec_conf = compute_section_confidence(lp)
            for key, conf in sec_conf.items():
                section_stats[key].append(conf)
        
        # Aggregate section stats
        section_summary = {}
        for key, stats_list in section_stats.items():
            if stats_list:
                mean_lps = [s["mean_logprob"] for s in stats_list]
                section_summary[key] = {
                    "mean_logprob": float(np.mean(mean_lps)),
                    "std_logprob": float(np.std(mean_lps)),
                    "mean_perplexity": float(np.mean([s["perplexity"] for s in stats_list])),
                    "mean_frac_low_conf": float(np.mean([s["frac_low_conf"] for s in stats_list])),
                    "num_samples": len(stats_list),
                }
        
        all_results[label] = {
            "sample_stats": sample_stats,
            "section_summary": section_summary,
            "config": config_name,
        }
        
        # Print summary
        if sample_stats:
            mean_lps = [s["mean_logprob"] for s in sample_stats]
            perplexities = [s["perplexity"] for s in sample_stats]
            logger.info(f"  Mean logprob: {np.mean(mean_lps):.4f} (std: {np.std(mean_lps):.4f})")
            logger.info(f"  Mean perplexity: {np.mean(perplexities):.2f}")
            
            # Correlation with hallucination
            halluc_scores = [s.get("judge_hallucination", 0) for s in sample_stats]
            if any(h > 0 for h in halluc_scores):
                corr = np.corrcoef(mean_lps, halluc_scores)[0, 1]
                logger.info(f"  Logprob-Hallucination correlation: {corr:.4f}")
    
    # ---- Save results ----
    results_path = output_path / "logprobs_analysis.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {results_path}")
    
    # ---- Generate plots ----
    _generate_plots(all_results, labels, output_path)
    
    # ---- Generate markdown report ----
    _generate_report(all_results, labels, output_path)
    
    return all_results


# =============================================================================
# Plotting
# =============================================================================

def _generate_plots(all_results: Dict, labels: List[str], output_dir: Path):
    """Generate all logprobs analysis plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return
    
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(labels), 3)))
    
    # ---- Plot 1: Logprob vs Hallucination Score (scatter) ----
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, label in enumerate(labels):
        data = all_results.get(label, {})
        stats = data.get("sample_stats", [])
        if not stats:
            continue
        
        x = [s["mean_logprob"] for s in stats]
        y = [s.get("judge_hallucination", 0) for s in stats]
        
        ax.scatter(x, y, color=colors[i], label=label, alpha=0.7, s=60, edgecolors="black", linewidth=0.5)
        
        # Trend line
        if len(x) >= 3 and any(yv > 0 for yv in y):
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            xs = np.linspace(min(x), max(x), 100)
            ax.plot(xs, p(xs), "--", color=colors[i], alpha=0.5)
            corr = np.corrcoef(x, y)[0, 1]
            ax.annotate(f"r={corr:.3f}", xy=(min(x), max(y) - 0.3 * i),
                       fontsize=9, color=colors[i])
    
    ax.set_xlabel("Mean Log Probability (higher = more confident)", fontsize=11)
    ax.set_ylabel("Judge Hallucination Score (higher = less hallucination)", fontsize=11)
    ax.set_title("Model Confidence vs Hallucination\n(Kadavath et al., 2022; Kuhn et al., 2023)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "logprob_vs_hallucination.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: logprob_vs_hallucination.png")
    
    # ---- Plot 2: Section-Level Confidence ----
    section_keys = [k for k, _ in CLINICAL_SECTIONS]
    section_names = [n for _, n in CLINICAL_SECTIONS]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(section_keys))
    w = 0.8 / len(labels)
    
    for i, label in enumerate(labels):
        data = all_results.get(label, {})
        sec_summary = data.get("section_summary", {})
        vals = [sec_summary.get(k, {}).get("mean_logprob", 0) for k in section_keys]
        bars = ax.bar(x + i * w, vals, w, label=label, color=colors[i])
        for bar, val in zip(bars, vals):
            if val != 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.1,
                        f"{val:.2f}", ha="center", va="top", fontsize=7, rotation=45)
    
    ax.set_xlabel("Clinical Section")
    ax.set_ylabel("Mean Log Probability (less negative = more confident)")
    ax.set_title("Per-Section Model Confidence\n(Lower values indicate higher uncertainty)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x + w * (len(labels) - 1) / 2)
    ax.set_xticklabels(section_names, fontsize=9, rotation=30, ha="right")
    ax.legend(fontsize=8)
    ax.axhline(y=-3.0, color="red", linestyle="--", alpha=0.5, label="Low confidence threshold")
    plt.tight_layout()
    plt.savefig(output_dir / "section_confidence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: section_confidence.png")
    
    # ---- Plot 3: Perplexity Distribution ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, label in enumerate(labels):
        data = all_results.get(label, {})
        stats = data.get("sample_stats", [])
        if not stats:
            continue
        perplexities = [s["perplexity"] for s in stats]
        ax.hist(perplexities, bins=20, alpha=0.5, color=colors[i], label=label, edgecolor="black")
    
    ax.set_xlabel("Perplexity (lower = more confident)")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Distribution of Per-Sample Perplexity", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "perplexity_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: perplexity_distribution.png")
    
    # ---- Plot 4: Confidence vs Judge Overall (scatter) ----
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, label in enumerate(labels):
        data = all_results.get(label, {})
        stats = data.get("sample_stats", [])
        if not stats:
            continue
        
        x_vals = [s["mean_logprob"] for s in stats]
        y_vals = [s.get("judge_overall", 0) for s in stats]
        has_errors = [s.get("has_critical_errors", False) for s in stats]
        
        # Plot non-error samples
        x_ok = [xv for xv, e in zip(x_vals, has_errors) if not e]
        y_ok = [yv for yv, e in zip(y_vals, has_errors) if not e]
        x_err = [xv for xv, e in zip(x_vals, has_errors) if e]
        y_err = [yv for yv, e in zip(y_vals, has_errors) if e]
        
        ax.scatter(x_ok, y_ok, color=colors[i], label=f"{label} (no errors)", 
                   alpha=0.7, s=50, edgecolors="black", linewidth=0.5)
        if x_err:
            ax.scatter(x_err, y_err, color=colors[i], marker="x", s=80,
                      label=f"{label} (critical errors)", linewidths=2)
    
    ax.set_xlabel("Mean Log Probability")
    ax.set_ylabel("Judge Overall Score (/5)")
    ax.set_title("Model Confidence vs Quality Score\n(Samples with critical errors marked with X)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "logprob_vs_quality.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: logprob_vs_quality.png")
    
    # ---- Plot 5: Token-level Confidence Heatmap (first sample only) ----
    for label in labels:
        data = all_results.get(label, {})
        stats = data.get("sample_stats", [])
        if not stats:
            continue
        
        # Get first sample's token logprobs
        eval_path = None
        for ep in eval_json_paths_global:
            with open(ep) as f:
                ed = json.load(f)
            if label in str(ed.get("config", {})):
                eval_path = ep
                break
        
        if not eval_path:
            continue
        
        # We already have the data in all_results, but need raw token data
        # This will be available from the eval JSON
        break  # Only do first model as example
    
    print(f"\nAll logprobs plots saved to {output_dir}/")


# Store eval paths globally for the heatmap plot
eval_json_paths_global = []


# =============================================================================
# Report Generation
# =============================================================================

def _generate_report(all_results: Dict, labels: List[str], output_dir: Path):
    """Generate markdown report for logprobs analysis."""
    lines = [
        "# Uncertainty Quantification Analysis via Log Probabilities",
        "",
        "## Scientific Background",
        "",
        "Token-level log probabilities provide a measure of model confidence",
        "for each generated token. Lower logprobs indicate higher uncertainty,",
        "which has been shown to correlate with factual errors and hallucinations",
        "(Kadavath et al., 2022; Kuhn et al., 2023).",
        "",
        "### Key Metrics",
        "",
        "- **Mean Log Probability**: Average confidence across all tokens.",
        "  More negative values indicate lower overall confidence.",
        "- **Perplexity**: exp(-mean_logprob). Lower is more confident.",
        "  Equivalent to the geometric mean of inverse token probabilities.",
        "- **Fraction Low-Confidence Tokens**: Proportion of tokens with",
        "  logprob < -3.0, indicating uncertain predictions.",
        "- **Per-Section Confidence**: Identifies which clinical sections",
        "  the model is least confident about.",
        "",
    ]
    
    # Per-model summary
    for label in labels:
        data = all_results.get(label, {})
        stats = data.get("sample_stats", [])
        sec = data.get("section_summary", {})
        
        if not stats:
            continue
        
        lines.append(f"## {label}")
        lines.append("")
        
        # Overall statistics
        mean_lps = [s["mean_logprob"] for s in stats]
        perplexities = [s["perplexity"] for s in stats]
        low_conf_fracs = [s["frac_low_conf"] for s in stats]
        
        lines.append("### Overall Confidence")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|---|---|")
        lines.append(f"| Mean logprob | {np.mean(mean_lps):.4f} |")
        lines.append(f"| Std logprob | {np.std(mean_lps):.4f} |")
        lines.append(f"| Mean perplexity | {np.mean(perplexities):.2f} |")
        lines.append(f"| Mean frac low-conf tokens | {np.mean(low_conf_fracs):.4f} |")
        lines.append(f"| Num samples | {len(stats)} |")
        
        # Correlation with judge
        halluc_scores = [s.get("judge_hallucination", 0) for s in stats]
        overall_scores = [s.get("judge_overall", 0) for s in stats]
        
        if any(h > 0 for h in halluc_scores):
            corr_halluc = np.corrcoef(mean_lps, halluc_scores)[0, 1]
            corr_overall = np.corrcoef(mean_lps, overall_scores)[0, 1]
            lines.append(f"| Correlation (logprob ↔ hallucination) | {corr_halluc:.4f} |")
            lines.append(f"| Correlation (logprob ↔ overall quality) | {corr_overall:.4f} |")
        
        lines.append("")
        
        # Section confidence
        if sec:
            lines.append("### Per-Section Confidence")
            lines.append("")
            lines.append("| Section | Mean LogProb | Perplexity | Frac Low-Conf |")
            lines.append("|---|---|---|---|")
            
            for key, name in CLINICAL_SECTIONS:
                if key in sec:
                    s = sec[key]
                    lines.append(
                        f"| {name} | {s['mean_logprob']:.4f} | "
                        f"{s['mean_perplexity']:.2f} | {s['mean_frac_low_conf']:.4f} |"
                    )
            lines.append("")
    
    report_path = output_dir / "logprobs_report.md"
    with open(report_path, "w", encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"Saved: logprobs_report.md")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    
    parser = argparse.ArgumentParser(
        description="Logprobs uncertainty quantification analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single model
    python logprobs_analysis.py \\
        --eval-json eval_phi.json \\
        --labels "Phi-3.5 (3.8B)" \\
        --output-dir ./logprobs_analysis

    # Multiple models
    python logprobs_analysis.py \\
        --eval-json eval_phi.json --eval-json eval_llama3b.json \\
        --labels "Phi-3.5 (3.8B)" "Llama-3.2 (3B)" \\
        --output-dir ./logprobs_analysis
        """,
    )
    
    parser.add_argument("--eval-json", action="append", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--config", default="ft_rag",
                        help="Which config to analyze (default: ft_rag)")
    parser.add_argument("--output-dir", default="./logprobs_analysis")
    
    args = parser.parse_args()
    
    if len(args.eval_json) != len(args.labels):
        parser.error("Number of --eval-json must match --labels")
    
    # Store globally for heatmap plot
    eval_json_paths_global = args.eval_json
    
    results = run_logprobs_analysis(
        eval_json_paths=args.eval_json,
        labels=args.labels,
        output_dir=args.output_dir,
        config_name=args.config,
    )
    
    print(f"\nAnalysis complete! Results in {args.output_dir}")
