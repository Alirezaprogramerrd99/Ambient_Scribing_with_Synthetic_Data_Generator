"""
Logprobs Uncertainty Quantification Analysis

Analyses per-token log probabilities from model generation to quantify
uncertainty and detect potential hallucinations.

Scientific foundations:
    - Kadavath et al. (2022) "Language Models (Mostly) Know What They Know"
    - Kuhn et al. (2023) "Semantic Uncertainty"
    - Xiong et al. (2024) "Can LLMs Express Their Uncertainty?"
    - Manakul et al. (2023) "SelfCheckGPT"

Usage:
    python logprobs_analysis.py \
        --eval-json eval_phi.json \
        --eval-json eval_llama3b.json \
        --eval-json eval_llama1b.json \
        --labels "Phi-3.5 (3.8B)" "Llama-3.2 (3B)" "Llama-3.2 (1B)" \
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
# Section Parsing - Robust version
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

SECTION_HEADER_VARIANTS = {
    "chief_complaint": [
        "Chief Complaint", "chief complaint", "CC",
    ],
    "history_of_present_illness": [
        "History of Present Illness", "history of present illness",
        "HPI", "History of Presenting Illness",
    ],
    "past_medical_history": [
        "Past Medical History", "past medical history", "PMH",
        "Past Medical Hx", "Medical History",
    ],
    "medications": [
        "Medications", "medications", "Current Medications", "Meds",
    ],
    "allergies": [
        "Allergies", "allergies", "Drug Allergies", "Known Allergies",
    ],
    "physical_examination": [
        "Examination Findings", "Physical Examination", "Examination",
        "examination findings", "physical examination", "Physical Exam",
        "Exam Findings", "Clinical Examination",
    ],
    "assessment": [
        "Assessment", "assessment", "Clinical Assessment",
        "Assessment and Plan", "Impression",
    ],
    "plan": [
        "Plan", "plan", "Management Plan", "Treatment Plan",
    ],
    "safety_netting": [
        "Safety Netting", "safety netting", "Safety Net",
        "Safety Netting Advice", "Red Flags",
    ],
}


def find_section_token_ranges(token_texts: List[str]) -> Dict[str, Tuple[int, int]]:
    """
    Map clinical section names to token index ranges in the generated output.
    Handles sub-word tokenization by reconstructing text and using flexible regex.
    """
    full_text = ""
    token_char_starts = []
    for token in token_texts:
        token_char_starts.append(len(full_text))
        full_text += token

    section_starts = []

    for section_key, variants in SECTION_HEADER_VARIANTS.items():
        best_match = None
        for variant in variants:
            words = variant.split()
            flex_pattern = r'\s*'.join(re.escape(w) for w in words)
            pattern = rf'(?:\*\*\s*)?{flex_pattern}(?:\s*\*\*)?[\s]*[:\-]?'
            for match in re.finditer(pattern, full_text, re.IGNORECASE):
                if best_match is None or match.start() < best_match:
                    best_match = match.start()
        if best_match is not None:
            section_starts.append((best_match, section_key))

    section_starts.sort(key=lambda x: x[0])

    if not section_starts:
        return {}

    def char_to_token_idx(char_pos: int) -> int:
        lo, hi = 0, len(token_char_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if token_char_starts[mid] <= char_pos:
                lo = mid
            else:
                hi = mid - 1
        return lo

    section_ranges = {}
    for i, (char_start, section_key) in enumerate(section_starts):
        token_start = char_to_token_idx(char_start)
        if i + 1 < len(section_starts):
            token_end = char_to_token_idx(section_starts[i + 1][0])
        else:
            token_end = len(token_texts)
        if token_end - token_start >= 3:
            section_ranges[section_key] = (token_start, token_end)

    return section_ranges


# =============================================================================
# Analysis Functions
# =============================================================================

def compute_sample_confidence(logprobs_data: Dict) -> Dict[str, float]:
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
        "p10_logprob": float(np.percentile(arr, 10)),
        "p25_logprob": float(np.percentile(arr, 25)),
        "num_low_conf_tokens": int(np.sum(arr < -3.0)),
        "frac_low_conf": float(np.mean(arr < -3.0)),
        "perplexity": float(np.exp(-arr.mean())),
        "num_tokens": len(token_lps),
    }


def compute_section_confidence(logprobs_data: Dict) -> Dict[str, Dict[str, float]]:
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
# Main Pipeline
# =============================================================================

def run_logprobs_analysis(
    eval_json_paths: List[str],
    labels: List[str],
    output_dir: str,
    config_name: str = "ft_rag",
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for eval_path, label in zip(eval_json_paths, labels):
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing: {label}")
        logger.info(f"  File: {eval_path}")
        logger.info(f"{'='*60}")

        with open(eval_path) as f:
            eval_data = json.load(f)

        comp = eval_data.get("comparative", {}).get(config_name, {})
        if comp.get("error"):
            logger.warning(f"  {config_name} errored, skipping")
            continue

        logprobs_list = comp.get("logprobs_per_sample", [])
        if not logprobs_list or all(lp is None for lp in logprobs_list):
            logger.warning(f"  No logprobs data for {label}. Re-run with --return-logprobs")
            continue

        judge_per_sample = comp.get("metrics", {}).get("llm_judge_per_sample", [])
        valid_logprobs = [lp for lp in logprobs_list if lp is not None]
        logger.info(f"  Found {len(valid_logprobs)} samples with logprobs")

        sample_stats = []
        for i, lp in enumerate(logprobs_list):
            if lp is None:
                continue
            stats = compute_sample_confidence(lp)
            if not stats:
                continue
            if i < len(judge_per_sample):
                js = judge_per_sample[i]
                stats["judge_overall"] = js.get("overall", 0)
                stats["judge_hallucination"] = js.get("hallucination", 0)
                stats["judge_safety"] = js.get("clinical_safety", 0)
                stats["judge_accuracy"] = js.get("clinical_accuracy", 0)
                stats["has_critical_errors"] = len(js.get("critical_errors", [])) > 0
            stats["sample_idx"] = i
            sample_stats.append(stats)

        section_stats = defaultdict(list)
        for lp in logprobs_list:
            if lp is None:
                continue
            sec_conf = compute_section_confidence(lp)
            for key, conf in sec_conf.items():
                section_stats[key].append(conf)

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

        if sample_stats:
            mean_lps = [s["mean_logprob"] for s in sample_stats]
            perplexities = [s["perplexity"] for s in sample_stats]
            logger.info(f"  Mean logprob: {np.mean(mean_lps):.4f} (std: {np.std(mean_lps):.4f})")
            logger.info(f"  Mean perplexity: {np.mean(perplexities):.2f}")
            logger.info(f"  Sections found: {list(section_summary.keys())}")
            halluc_scores = [s.get("judge_hallucination", 0) for s in sample_stats]
            if any(h > 0 for h in halluc_scores):
                corr = np.corrcoef(mean_lps, halluc_scores)[0, 1]
                logger.info(f"  Logprob-Hallucination correlation: {corr:.4f}")

    if not all_results:
        logger.error("No valid results to analyze!")
        return {}

    results_path = output_path / "logprobs_analysis.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {results_path}")

    valid_labels = [l for l in labels if l in all_results]
    _generate_plots(all_results, valid_labels, output_path)
    _generate_report(all_results, valid_labels, output_path)
    return all_results


# =============================================================================
# Plotting - Fixed multi-model
# =============================================================================

def _generate_plots(all_results: Dict, labels: List[str], output_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available")
        return

    n_models = len(labels)
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_models, 3)))

    # Plot 1: Logprob vs Hallucination
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, label in enumerate(labels):
        data = all_results.get(label, {})
        stats = data.get("sample_stats", [])
        if not stats:
            continue
        x = [s["mean_logprob"] for s in stats]
        y = [s.get("judge_hallucination", 0) for s in stats]
        ax.scatter(x, y, color=colors[i], label=label, alpha=0.7, s=60,
                   edgecolors="black", linewidth=0.5)
        if len(x) >= 3 and any(yv > 0 for yv in y):
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            xs = np.linspace(min(x), max(x), 100)
            ax.plot(xs, p(xs), "--", color=colors[i], alpha=0.5, linewidth=1.5)
            corr = np.corrcoef(x, y)[0, 1]
            ax.annotate(f"{label}: r={corr:.3f}",
                       xy=(0.02, 0.98 - 0.06 * i), xycoords="axes fraction",
                       fontsize=9, color=colors[i], fontweight="bold")
    ax.set_xlabel("Mean Log Probability (higher = more confident)", fontsize=11)
    ax.set_ylabel("Judge Hallucination Score (higher = less hallucination)", fontsize=11)
    ax.set_title("Model Confidence vs Hallucination\n(Kadavath et al., 2022; Kuhn et al., 2023)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "logprob_vs_hallucination.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: logprob_vs_hallucination.png")

    # Plot 2: Section-Level Confidence
    section_keys = [k for k, _ in CLINICAL_SECTIONS]
    section_names = [n for _, n in CLINICAL_SECTIONS]
    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(section_keys))
    w = 0.8 / max(n_models, 1)
    for i, label in enumerate(labels):
        data = all_results.get(label, {})
        sec_summary = data.get("section_summary", {})
        vals = [sec_summary.get(k, {}).get("mean_logprob", 0) for k in section_keys]
        bars = ax.bar(x + i * w, vals, w, label=label, color=colors[i])
        for bar, val in zip(bars, vals):
            if val != 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        min(val - 0.003, -0.003),
                        f"{val:.2f}", ha="center", va="top", fontsize=6, rotation=45)
    ax.set_xlabel("Clinical Section")
    ax.set_ylabel("Mean Log Probability (less negative = more confident)")
    ax.set_title("Per-Section Model Confidence\n(Lower values indicate higher uncertainty)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x + w * (n_models - 1) / 2)
    ax.set_xticklabels(section_names, fontsize=9, rotation=30, ha="right")
    ax.legend(fontsize=8)
    ax.axhline(y=-3.0, color="red", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "section_confidence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: section_confidence.png")

    # Plot 3: Perplexity Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, label in enumerate(labels):
        data = all_results.get(label, {})
        stats = data.get("sample_stats", [])
        if not stats:
            continue
        perplexities = [s["perplexity"] for s in stats]
        ax.hist(perplexities, bins=15, alpha=0.4, color=colors[i],
                label=label, edgecolor=colors[i], linewidth=1.5)
    ax.set_xlabel("Perplexity (lower = more confident)")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Distribution of Per-Sample Perplexity", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "perplexity_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: perplexity_distribution.png")

    # Plot 4: Confidence vs Judge Overall
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, label in enumerate(labels):
        data = all_results.get(label, {})
        stats = data.get("sample_stats", [])
        if not stats:
            continue
        x_vals = [s["mean_logprob"] for s in stats]
        y_vals = [s.get("judge_overall", 0) for s in stats]
        has_errors = [s.get("has_critical_errors", False) for s in stats]
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
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "logprob_vs_quality.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: logprob_vs_quality.png")

    # Plot 5: Summary table figure
    if n_models >= 2:
        fig, ax = plt.subplots(figsize=(10, 3 + n_models * 0.5))
        ax.axis("off")
        col_labels = ["Model", "Mean LogProb", "Perplexity", "r(logprob,halluc)", "Critical Errors"]
        table_data = []
        for label in labels:
            data = all_results.get(label, {})
            stats = data.get("sample_stats", [])
            if not stats:
                continue
            mean_lps = [s["mean_logprob"] for s in stats]
            perps = [s["perplexity"] for s in stats]
            halluc = [s.get("judge_hallucination", 0) for s in stats]
            n_errors = sum(1 for s in stats if s.get("has_critical_errors", False))
            corr_str = "N/A"
            if any(h > 0 for h in halluc):
                corr_str = f"{np.corrcoef(mean_lps, halluc)[0, 1]:.3f}"
            table_data.append([label, f"{np.mean(mean_lps):.4f}",
                             f"{np.mean(perps):.3f}", corr_str, f"{n_errors}/{len(stats)}"])
        table = ax.table(cellText=table_data, colLabels=col_labels,
                        loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax.set_title("Logprobs Uncertainty Summary Across Models",
                     fontsize=13, fontweight="bold", pad=20)
        plt.tight_layout()
        plt.savefig(output_dir / "logprobs_summary_table.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: logprobs_summary_table.png")

    print(f"\nAll logprobs plots saved to {output_dir}/")


# =============================================================================
# Report
# =============================================================================

def _generate_report(all_results: Dict, labels: List[str], output_dir: Path):
    lines = [
        "# Uncertainty Quantification Analysis via Log Probabilities", "",
        "## Scientific Background", "",
        "Token-level log probabilities provide a measure of model confidence",
        "for each generated token. Lower logprobs indicate higher uncertainty,",
        "which has been shown to correlate with factual errors and hallucinations",
        "(Kadavath et al., 2022; Kuhn et al., 2023).", "",
    ]
    for label in labels:
        data = all_results.get(label, {})
        stats = data.get("sample_stats", [])
        sec = data.get("section_summary", {})
        if not stats:
            continue
        lines.append(f"## {label}")
        lines.append("")
        mean_lps = [s["mean_logprob"] for s in stats]
        perplexities = [s["perplexity"] for s in stats]
        low_conf_fracs = [s["frac_low_conf"] for s in stats]
        lines.extend(["### Overall Confidence", "",
                      "| Metric | Value |", "|---|---|",
                      f"| Mean logprob | {np.mean(mean_lps):.4f} |",
                      f"| Std logprob | {np.std(mean_lps):.4f} |",
                      f"| Mean perplexity | {np.mean(perplexities):.2f} |",
                      f"| Mean frac low-conf tokens | {np.mean(low_conf_fracs):.4f} |",
                      f"| Num samples | {len(stats)} |"])
        halluc_scores = [s.get("judge_hallucination", 0) for s in stats]
        overall_scores = [s.get("judge_overall", 0) for s in stats]
        if any(h > 0 for h in halluc_scores):
            lines.append(f"| Correlation (logprob vs hallucination) | {np.corrcoef(mean_lps, halluc_scores)[0, 1]:.4f} |")
            lines.append(f"| Correlation (logprob vs overall quality) | {np.corrcoef(mean_lps, overall_scores)[0, 1]:.4f} |")
        lines.append("")
        if sec:
            lines.extend(["### Per-Section Confidence", "",
                         "| Section | Mean LogProb | Perplexity | Frac Low-Conf | N Samples |",
                         "|---|---|---|---|---|"])
            for key, name in CLINICAL_SECTIONS:
                if key in sec:
                    s = sec[key]
                    lines.append(f"| {name} | {s['mean_logprob']:.4f} | "
                                f"{s['mean_perplexity']:.2f} | {s['mean_frac_low_conf']:.4f} | "
                                f"{s['num_samples']} |")
            lines.append("")
    report_path = output_dir / "logprobs_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved: logprobs_report.md")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Logprobs UQ analysis")
    parser.add_argument("--eval-json", action="append", required=True,
                        help="Path to evaluation JSON (repeat for multiple models)")
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--config", default="ft_rag")
    parser.add_argument("--output-dir", default="./logprobs_analysis")
    args = parser.parse_args()
    if len(args.eval_json) != len(args.labels):
        parser.error(f"--eval-json count ({len(args.eval_json)}) != --labels count ({len(args.labels)})")
    for ep in args.eval_json:
        if not Path(ep).exists():
            parser.error(f"File not found: {ep}")
    results = run_logprobs_analysis(
        eval_json_paths=args.eval_json, labels=args.labels,
        output_dir=args.output_dir, config_name=args.config,
    )
    print(f"\nAnalysis complete! Results in {args.output_dir}")
