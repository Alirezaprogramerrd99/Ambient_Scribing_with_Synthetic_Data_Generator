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

# Fixed threshold for flagging low-confidence tokens.
# A token logprob below -2.0 nats means the model assigned < exp(-2.0) ≈ 13.5%
# probability to its chosen token — a meaningful signal of genuine uncertainty.
# Using a fixed value (rather than a per-sample percentile) makes the metric
# comparable across samples and models: two samples with identical distributions
# will produce identical frac_low_conf, which is the desired property.
# Ref: Malinin & Gales (2020) use fixed entropy thresholds for uncertainty flagging.
LOW_CONF_THRESHOLD: float = -2.0


def compute_sample_confidence(logprobs_data: Dict) -> Dict[str, float]:
    token_lps = logprobs_data.get("token_logprobs", [])
    if not token_lps:
        return {}
    arr = np.array(token_lps)
    n = len(token_lps)
    
    # Sum of logprobs = sequence log-probability (confidence score)
    # This is log p(s|x) = sum_i log p(s_i | s_<i, x)
    # Higher (closer to 0) = model is more confident in its generation.
    # Kadavath et al. (2022) use this as a confidence score for AUROC.
    # Ref: Kuhn et al. (2023) Section 2
    sum_lp = float(arr.sum())
    
    # Length-normalised sequence log-probability = mean logprob
    # This is (1/N) * sum_i log p(s_i | s_<i, x)
    # Removes length bias: longer sequences have lower sum_lp by construction.
    # Ref: Malinin & Gales (2020), discussed in Kuhn et al. Section 3.3
    mean_lp = float(arr.mean())
    
    # Sequence NLL (negative log-likelihood) = -sum of logprobs
    # This is -log p(s|x). Higher = model is LESS confident = more uncertain.
    # This is the SAME information as sequence log-probability, just negated.
    # Used as an uncertainty score for failure detection (AUROC):
    #   AUROC asks "do hallucinated samples get higher NLL than correct ones?"
    # Note: This is NOT the true predictive entropy H(Y|x) from Kuhn et al.
    # Eq. 1, which requires marginalising over all possible outputs via Monte
    # Carlo sampling. With a single generation, the MC estimate degenerates
    # to the sequence NLL. We use this standard proxy following the baselines
    # in Kuhn et al. (2023) and Kadavath et al. (2022).
    # Ref: Kuhn et al. (2023) — "predictive entropy" baseline (single-sample)
    # Ref: Kadavath et al. (2022) — sequence probability for uncertainty
    # Ref: Xiong et al. (2024) Table 5 — "seq-prob" white-box method
    sequence_nll = -sum_lp
    
    # Length-normalised NLL = -mean logprob
    # Removes length bias from the uncertainty score.
    # Ref: Malinin & Gales (2020) — length-normalised sequence log-probability
    # Ref: Kuhn et al. (2023) Section 3.3 — "normalised entropy" baseline
    # Ref: Xiong et al. (2024) Table 5 — "len-norm-prob" white-box method
    length_norm_nll = -mean_lp
    
    # Perplexity = exp(length_norm_nll) = exp(-mean_logprob)
    # Geometric mean of inverse token probabilities.
    # Perplexity of 1.0 = perfect confidence; higher = more uncertain.
    # Monotonic transform of length_norm_nll, so produces identical AUROC.
    perplexity = float(np.exp(length_norm_nll))
    
    return {
        "mean_logprob": mean_lp,
        "sum_logprob": sum_lp,
        "median_logprob": float(np.median(arr)),
        "std_logprob": float(arr.std()),
        "min_logprob": float(arr.min()),
        "max_logprob": float(arr.max()),
        "p10_logprob": float(np.percentile(arr, 10)),
        "p25_logprob": float(np.percentile(arr, 25)),
        "num_low_conf_tokens": int(np.sum(arr < LOW_CONF_THRESHOLD)),
        "frac_low_conf": float(np.mean(arr < LOW_CONF_THRESHOLD)),
        "low_conf_threshold": LOW_CONF_THRESHOLD,
        "sequence_nll": sequence_nll,
        "length_norm_nll": length_norm_nll,
        "perplexity": perplexity,
        "num_tokens": n,
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
                "frac_low_conf": float(np.mean(arr < LOW_CONF_THRESHOLD)),
                "perplexity": float(np.exp(-arr.mean())),
            }
    return section_confidence


# =============================================================================
# AUROC for Hallucination Detection
# Ref: Kuhn et al. (2023) Section 6: "we evaluate uncertainty by treating
#      uncertainty estimation as the problem of predicting whether to rely
#      on a model generation... The AUROC metric is equivalent to the
#      probability that a randomly chosen correct answer has a higher
#      uncertainty score than a randomly chosen incorrect answer."
# Ref: Xiong et al. (2024) Section 4: use AUROC for failure prediction
# =============================================================================

def compute_auroc(uncertainty_scores: List[float], is_correct: List[bool]) -> float:
    """
    Compute AUROC: can the uncertainty score distinguish correct from incorrect?
    
    Higher AUROC = better uncertainty estimation.
    0.5 = random (uncertainty provides no signal).
    1.0 = perfect (all incorrect samples have higher uncertainty than correct ones).
    
    Args:
        uncertainty_scores: Higher = MORE uncertain (e.g., sequence_nll)
        is_correct: True if sample is "correct" (low hallucination)
    """
    from sklearn.metrics import roc_auc_score
    # AUROC expects: higher score = more likely to be POSITIVE class
    # Our convention: higher uncertainty = more likely INCORRECT
    # So we predict "incorrect" with uncertainty as the score
    # and is_correct=False as the positive class for "failure detection"
    try:
        # Binary labels: 1 = incorrect (hallucinated), 0 = correct
        binary_labels = [0 if c else 1 for c in is_correct]
        if len(set(binary_labels)) < 2:
            return 0.5  # All same class, AUROC undefined
        return float(roc_auc_score(binary_labels, uncertainty_scores))
    except Exception:
        return 0.5


def compute_auroc_simple(scores: List[float], labels: List[int]) -> float:
    """
    Manual AUROC without sklearn dependency.
    Computes the probability that a randomly chosen positive example
    has a higher score than a randomly chosen negative example.
    """
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return 0.5
    concordant = sum(1 for p in pos for n in neg if p > n)
    tied = sum(1 for p in pos for n in neg if p == n)
    total = len(pos) * len(neg)
    return (concordant + 0.5 * tied) / total


# =============================================================================
# Expected Calibration Error (ECE)
# Ref: Guo et al. (2017) "On Calibration of Modern Neural Networks" Formula is drived from this paper.
# Ref: Xiong et al. (2024) Section 4
#
# IMPORTANT CAVEAT: ECE was originally designed for classifiers where the
# predicted probability directly corresponds to P(correct). For LLMs,
# token-level confidence (exp(mean_logprob)) reflects the model's probability
# of its chosen token sequence, NOT the probability that the output is
# factually correct. A model can produce high-probability hallucinations.
# (Kuhn et al., 2023, Section 6: "the language model outputs a likelihood 
# for a given token-sequence, but not for an entire meaning.")
#
# We use ECE here to measure the gap between token-level confidence and
# output quality (as judged by the LLM judge). The resulting ECE values 
# quantify HOW overconfident the models are in their token selections,
# not whether the models produce calibrated correctness probabilities.
# This is an informative diagnostic, not a calibration guarantee.
# =============================================================================

def compute_ece(confidence_scores: List[float], is_correct: List[bool], 
                n_bins: int = 10) -> Dict[str, Any]:
    """
    Compute Expected Calibration Error.
    
    Args:
        confidence_scores: Values in [0, 1] where 1 = fully confident
        is_correct: True if the sample is correct
        n_bins: Number of calibration bins
    
    Returns:
        Dict with ece, per-bin accuracies, confidences, and counts
    """
    confidences = np.array(confidence_scores)
    accuracies = np.array([1.0 if c else 0.0 for c in is_correct])
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []
    
    ece = 0.0
    total = len(confidences)
    
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if i == 0:  # Include 0 in first bin
            mask = (confidences >= lo) & (confidences <= hi)
        
        count = mask.sum()
        if count == 0:
            bin_accs.append(0)
            bin_confs.append((lo + hi) / 2)
            bin_counts.append(0)
            continue
        
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        bin_accs.append(float(bin_acc))
        bin_confs.append(float(bin_conf))
        bin_counts.append(int(count))
        
        ece += (count / total) * abs(bin_acc - bin_conf)
    
    return {
        "ece": float(ece),
        "bin_accuracies": bin_accs,
        "bin_confidences": bin_confs,
        "bin_counts": bin_counts,
        "n_bins": n_bins,
    }


def logprob_to_confidence(mean_logprobs: List[float]) -> List[float]:
    """
    Convert mean logprobs to confidence scores in [0, 1].
    
    Uses the transformation: confidence = exp(mean_logprob)
    Since mean_logprob is in (-inf, 0], exp maps to (0, 1].
    A mean_logprob of 0 = 100% confident, -inf = 0% confident.
    
    This represents the model's average token-level probability — i.e.,
    how probable the model considers its own token choices. This is NOT
    a calibrated probability of factual correctness. A model can assign
    high token probability to hallucinated content.
    
    Ref: Xiong et al. (2024) Table 5 — "seq-prob" and "len-norm-prob"
         use the same transformation as white-box confidence baselines.
    """
    return [float(np.exp(lp)) for lp in mean_logprobs]


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
            
            # ---- AUROC: Can uncertainty detect hallucinations? ----
            # Binarise: hallucination <= 3 = "hallucinated", > 3 = "not hallucinated"
            # Ref: Kuhn et al. (2023) use AUROC as primary evaluation metric
            is_correct = [s.get("judge_hallucination", 0) > 3 for s in sample_stats]
            
            # AUROC using different uncertainty measures
            pred_entropies = [s["sequence_nll"] for s in sample_stats]
            ln_entropies = [s["length_norm_nll"] for s in sample_stats]
            perplexities = [s["perplexity"] for s in sample_stats]
            
            try:
                auroc_pred_ent = compute_auroc(pred_entropies, is_correct)
                auroc_ln_ent = compute_auroc(ln_entropies, is_correct)
                auroc_perplexity = compute_auroc(perplexities, is_correct)
            except ImportError:
                # Fallback without sklearn
                binary_labels = [0 if c else 1 for c in is_correct]
                auroc_pred_ent = compute_auroc_simple(pred_entropies, binary_labels)
                auroc_ln_ent = compute_auroc_simple(ln_entropies, binary_labels)
                auroc_perplexity = compute_auroc_simple(perplexities, binary_labels)
            
            logger.info(f"  AUROC (sequence NLL): {auroc_pred_ent:.4f}")
            logger.info(f"  AUROC (length-norm NLL): {auroc_ln_ent:.4f}")
            logger.info(f"  AUROC (perplexity):          {auroc_perplexity:.4f}")
            
            # ---- ECE: Is the model well-calibrated? ----
            # Convert logprobs to confidence, then measure calibration
            # Ref: Xiong et al. (2024), Guo et al. (2017)
            confidences = logprob_to_confidence(mean_lps)
            ece_result = compute_ece(confidences, is_correct)
            logger.info(f"  ECE: {ece_result['ece']:.4f}")
            
            # Store AUROC and ECE in results
            all_results[label]["auroc"] = {
                "sequence_nll": auroc_pred_ent,
                "length_norm_nll": auroc_ln_ent,
                "perplexity": auroc_perplexity,
                "hallucination_threshold": 3,
                "n_correct": sum(is_correct),
                "n_incorrect": sum(not c for c in is_correct),
            }
            all_results[label]["ece"] = ece_result
            all_results[label]["correlation"] = {
                "logprob_vs_hallucination": float(corr) if any(h > 0 for h in halluc_scores) else None,
            }

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
    all_sec_vals = []
    for i, label in enumerate(labels):
        data = all_results.get(label, {})
        sec_summary = data.get("section_summary", {})
        vals = [sec_summary.get(k, {}).get("mean_logprob", 0) for k in section_keys]
        all_sec_vals.extend([v for v in vals if v != 0])
        bars = ax.bar(x + i * w, vals, w, label=label, color=colors[i])
        for bar, val in zip(bars, vals):
            if val != 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        val - 0.001,
                        f"{val:.3f}", ha="center", va="top", fontsize=6, rotation=45)
    ax.set_xlabel("Clinical Section")
    ax.set_ylabel("Mean Log Probability (less negative = more confident)")
    ax.set_title("Per-Section Model Confidence\n(Lower values indicate higher uncertainty)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x + w * (n_models - 1) / 2)
    ax.set_xticklabels(section_names, fontsize=9, rotation=30, ha="right")
    ax.legend(fontsize=8)
    # Zoom y-axis to actual data range instead of showing -3.0 threshold
    if all_sec_vals:
        min_val = min(all_sec_vals)
        ax.set_ylim(min_val * 1.5, 0.005)
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

    # ---- Plot 6: AUROC Comparison Bar Chart ----
    # Ref: Kuhn et al. (2023) Fig. 1a - compares AUROC across uncertainty methods
    # Ref: Xiong et al. (2024) Table 2 - reports AUROC for failure prediction
    auroc_methods = ["sequence_nll", "length_norm_nll", "perplexity"]
    auroc_labels = ["Sequence\nNLL", "Length-Norm\nNLL", "Perplexity"]
    
    has_auroc = any(all_results.get(l, {}).get("auroc") for l in labels)
    if has_auroc:
        fig, ax = plt.subplots(figsize=(12, 7))
        x = np.arange(len(auroc_methods))
        w = 0.8 / max(n_models, 1)
        
        all_vals = []
        for i, label in enumerate(labels):
            auroc_data = all_results.get(label, {}).get("auroc", {})
            vals = [auroc_data.get(m, 0.5) for m in auroc_methods]
            all_vals.extend(vals)
            bars = ax.bar(x + i * w, vals, w, label=label, color=colors[i],
                         edgecolor="black", linewidth=0.5)
            for bar, val in zip(bars, vals):
                # Place text inside bar if bar is tall, above if short
                if val > 0.55:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.03,
                            f"{val:.3f}", ha="center", va="top", fontsize=9, 
                            fontweight="bold", color="white")
                else:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, linewidth=1.5,
                   label="Random baseline (0.5)")
        ax.set_ylabel("AUROC", fontsize=12)
        ax.set_title("Hallucination Detection: AUROC of Uncertainty Measures\n"
                     "(Kadavath et al., 2022; Kuhn et al., 2023; Xiong et al., 2024)",
                     fontsize=13, fontweight="bold")
        ax.set_xticks(x + w * (n_models - 1) / 2)
        ax.set_xticklabels(auroc_labels, fontsize=10)
        ax.legend(fontsize=9, loc="upper right")
        # Dynamic ylim based on data
        max_val = max(all_vals) if all_vals else 0.8
        ax.set_ylim(0.3, max(max_val + 0.08, 0.95))
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "auroc_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: auroc_comparison.png")
    
    # ---- Plot 7: ECE Reliability Diagram ----
    # Ref: Guo et al. (2017) Fig. 1 - the standard reliability diagram format
    # Ref: Xiong et al. (2024) Fig. 2 - reliability diagrams for LLMs
    # Shows: for each confidence bin, the actual accuracy vs predicted confidence
    has_ece = any(all_results.get(l, {}).get("ece") for l in labels)
    if has_ece:
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=True)
        if n_models == 1:
            axes = [axes]
        
        for i, label in enumerate(labels):
            ax = axes[i]
            ece_data = all_results.get(label, {}).get("ece", {})
            if not ece_data:
                continue
            
            bin_accs = ece_data["bin_accuracies"]
            bin_confs = ece_data["bin_confidences"]
            bin_counts = ece_data["bin_counts"]
            n_bins = ece_data["n_bins"]
            
            bar_width = 1.0 / n_bins
            
            # Plot all bins, show gap between confidence and accuracy
            for j in range(n_bins):
                bin_center = (j + 0.5) / n_bins
                if bin_counts[j] > 0:
                    # Accuracy bar
                    ax.bar(bin_center, bin_accs[j], width=bar_width * 0.9,
                           color=colors[i], alpha=0.7, edgecolor="black", linewidth=0.5)
                    # Gap (overconfidence) shown as red overlay
                    if bin_confs[j] > bin_accs[j]:
                        ax.bar(bin_center, bin_confs[j] - bin_accs[j], width=bar_width * 0.9,
                               bottom=bin_accs[j], color="red", alpha=0.2, edgecolor="red",
                               linewidth=0.5, linestyle="--")
                    # Annotate with count
                    ax.text(bin_center, bin_accs[j] + 0.02, f"n={bin_counts[j]}",
                           ha="center", fontsize=7, color="gray")
            
            # Perfect calibration line
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
            
            ax.set_xlabel("Confidence", fontsize=10)
            if i == 0:
                ax.set_ylabel("Non-hallucination Rate", fontsize=10)
            ax.set_title(f"{label}\nECE = {ece_data['ece']:.4f}", fontsize=11, fontweight="bold")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend(fontsize=8)
            ax.set_aspect("equal")
        
        fig.suptitle("Token-Level Confidence vs Output Quality\n(Guo et al., 2017; Xiong et al., 2024)",
                     fontsize=13, fontweight="bold", y=1.05)
        plt.tight_layout()
        plt.savefig(output_dir / "ece_reliability_diagram.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: ece_reliability_diagram.png")
    
    # ---- Plot 8: Sequence NLL vs Length-Normalised NLL ----
    # Ref: Kuhn et al. (2023) Section 3.3: "longer sequences have lower joint
    #      likelihoods... length-normalising the log-probabilities"
    # Ref: Xiong et al. (2024) Table 5: "seq-prob" vs "len-norm-prob"
    # This plot shows whether normalisation matters for your models
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Sequence NLL distribution
    ax = axes[0]
    for i, label in enumerate(labels):
        stats = all_results.get(label, {}).get("sample_stats", [])
        if not stats:
            continue
        vals = [s["sequence_nll"] for s in stats]
        ax.hist(vals, bins=15, alpha=0.4, color=colors[i], label=label,
                edgecolor=colors[i], linewidth=1.5)
    ax.set_xlabel("Sequence NLL (= -sum of logprobs)")
    ax.set_ylabel("Count")
    ax.set_title("Sequence NLL (Unnormalised)\nHigher = more uncertain", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    
    # Right: Length-normalised entropy distribution
    ax = axes[1]
    for i, label in enumerate(labels):
        stats = all_results.get(label, {}).get("sample_stats", [])
        if not stats:
            continue
        vals = [s["length_norm_nll"] for s in stats]
        ax.hist(vals, bins=15, alpha=0.4, color=colors[i], label=label,
                edgecolor=colors[i], linewidth=1.5)
    ax.set_xlabel("Length-Normalised NLL (= -mean logprob)")
    ax.set_ylabel("Count")
    ax.set_title("Length-Normalised NLL\n(Malinin & Gales, 2020; Xiong et al., 2024)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    
    fig.suptitle("Effect of Length Normalisation on Uncertainty Score Distribution",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "entropy_normalisation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: entropy_normalisation.png")
    
    # ---- Plot 9: Comprehensive UQ Summary Table ----
    if n_models >= 1:
        fig, ax = plt.subplots(figsize=(18, 2.5 + n_models * 0.8))
        ax.axis("off")
        col_labels = ["Model", "Mean\nLogProb", "Perplexity", "Seq.\nNLL",
                      "AUROC\n(Seq.NLL)", "AUROC\n(Len.Norm.)", "AUROC\n(Perplex.)",
                      "ECE", "r(lp,halluc)", "Errors"]
        table_data = []
        for label in labels:
            data = all_results.get(label, {})
            stats = data.get("sample_stats", [])
            auroc = data.get("auroc", {})
            ece = data.get("ece", {})
            corr_data = data.get("correlation", {})
            if not stats:
                continue
            mean_lps = [s["mean_logprob"] for s in stats]
            perps = [s["perplexity"] for s in stats]
            pred_ents = [s["sequence_nll"] for s in stats]
            n_errors = sum(1 for s in stats if s.get("has_critical_errors", False))
            corr_val = corr_data.get("logprob_vs_hallucination")
            corr_str = f"{corr_val:.3f}" if corr_val is not None else "N/A"
            
            table_data.append([
                label,
                f"{np.mean(mean_lps):.4f}",
                f"{np.mean(perps):.3f}",
                f"{np.mean(pred_ents):.1f}",
                f"{auroc.get('sequence_nll', 0.5):.3f}",
                f"{auroc.get('length_norm_nll', 0.5):.3f}",
                f"{auroc.get('perplexity', 0.5):.3f}",
                f"{ece.get('ece', 0):.4f}",
                corr_str,
                f"{n_errors}/{len(stats)}",
            ])
        
        table = ax.table(cellText=table_data, colLabels=col_labels,
                        loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.5)
        for j in range(len(col_labels)):
            table[0, j].set_text_props(fontweight="bold")
            table[0, j].set_facecolor("#E8E8E8")
        
        ax.set_title("Uncertainty Quantification: Comprehensive Summary\n"
                     "(Kadavath et al., 2022; Kuhn et al., 2023; Xiong et al., 2024)",
                     fontsize=13, fontweight="bold", pad=20)
        plt.tight_layout()
        plt.savefig(output_dir / "uq_comprehensive_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: uq_comprehensive_summary.png")

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
        
        # AUROC results
        auroc = data.get("auroc", {})
        if auroc:
            lines.extend([
                "### Hallucination Detection AUROC",
                "",
                "AUROC measures whether uncertainty scores can distinguish hallucinated",
                "from non-hallucinated outputs (Kuhn et al., 2023; Xiong et al., 2024).",
                "AUROC = 0.5 is random; AUROC = 1.0 is perfect detection.",
                "",
                "| Uncertainty Measure | AUROC | Interpretation |",
                "|---|---|---|",
                f"| Sequence NLL (-sum logprobs) | {auroc.get('sequence_nll', 0.5):.4f} | {'Useful' if auroc.get('sequence_nll', 0.5) > 0.6 else 'Near-random'} |",
                f"| Length-Norm NLL (-mean logprob) | {auroc.get('length_norm_nll', 0.5):.4f} | {'Useful' if auroc.get('length_norm_nll', 0.5) > 0.6 else 'Near-random'} |",
                f"| Perplexity (exp(-mean logprob)) | {auroc.get('perplexity', 0.5):.4f} | {'Useful' if auroc.get('perplexity', 0.5) > 0.6 else 'Near-random'} |",
                f"| Hallucination threshold | <= {auroc.get('hallucination_threshold', 3)}/5 |  |",
                f"| N correct / N incorrect | {auroc.get('n_correct', 0)} / {auroc.get('n_incorrect', 0)} |  |",
                "",
            ])
        
        # ECE results
        ece_data = data.get("ece", {})
        if ece_data:
            lines.extend([
                "### Calibration (ECE)",
                "",
                "Expected Calibration Error measures the gap between token-level",
                "confidence (exp(mean_logprob)) and output quality (Guo et al., 2017).",
                "**Caveat:** Token-level confidence reflects how probable the model",
                "considers its chosen tokens, NOT the probability of factual correctness.",
                "A model can produce high-probability hallucinations. ECE here quantifies",
                "HOW overconfident models are, not whether they are calibrated classifiers.",
                "",
                f"| ECE | {ece_data.get('ece', 0):.4f} |",
                "",
            ])
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
