"""
Dissertation Plots & Tables

Reads ONLY from evaluation JSONs (one per model, from evaluator.py) and
an optional MEDCON JSON. Computes all metrics directly from raw_outputs
and references — no post_eval_metrics.py dependency.

Metrics computed:
  - ROUGE-1, ROUGE-2, ROUGE-L (rouge_score or evaluate)
  - BLEU-1..4 (nltk or evaluate)
  - BERTScore P/R/F1 (bert_score, roberta-large)
  - METEOR (evaluate + nltk, replaces BLEURT — Google bucket unavailable)
  - MEDCON (from --medcon-json, QuickUMLS)
  - LLM Judge scores (from eval JSON, pre-computed by evaluator.py)

Outputs:
  - Table A: Automated metrics (all models , configs , overall + per-section)
  - Table B: LLM Judge scores
  - Plots: ROUGE, BLEU, BERTScore, Judge, Radar, Heatmap, Per-Section, Size

Usage:
    python plot_dissertation_v2.py \
        --eval-jsons eval_phi.json eval_llama3b.json eval_llama1b.json \
        --labels "Phi-3.5 (3.8B)" "Llama-3.2 (3B)" "Llama-3.2 (1B)" \
        --medcon-json ./medcon_results.json \
        --output-dir ./dissertation_final

"""

import json
import argparse
import re
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Constants
# =============================================================================

CONFIGS = ["baseline", "rag_only", "ft_only", "ft_rag", "teacher"]

CONFIG_SHORT = {
    "baseline": "Base",
    "rag_only": "Base+RAG",
    "ft_only": "FT",
    "ft_rag": "FT+RAG",
    "teacher": "Teacher",
}

CONFIG_LABELS = {
    "baseline": "Base\n(no FT, no RAG)",
    "rag_only": "Base + RAG",
    "ft_only": "Fine-tuned\n(no RAG)",
    "ft_rag": "Fine-tuned + RAG",
    "teacher": "Teacher\n(GPT-4o-mini)",
}

RAG_METHOD = {
    "baseline": "None",
    "rag_only": "Dense+Rerank+QE",
    "ft_only": "None",
    "ft_rag": "Dense+Rerank+QE",
    "teacher": "None (API)",
}

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

JUDGE_DIMS = [
    ("avg_clinical_accuracy", "Clin.Acc"),
    ("avg_completeness", "Complete"),
    ("avg_hallucination", "Halluc.↑"),
    ("avg_clinical_safety", "Safety"),
    ("avg_coherence", "Coherence"),
    ("avg_conciseness", "Concise"),
    ("avg_overall", "Overall"),
]


# =============================================================================
# Helpers
# =============================================================================

def fmt(v, decimals=3):
    return f"{v:.{decimals}f}" if v > 0 else "\u2014"


def parse_sections(text: str) -> Dict[str, str]:
    sections = {}
    for key, pattern in SECTION_PATTERNS.items():
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            sections[key] = match.group(1).strip()
    return sections


# =============================================================================
# Metric Computation (from raw text pairs)
# =============================================================================

_rouge_scorer = None
_bleu_smooth = None
_bert_scorer_loaded = False


def compute_rouge_pair(ref: str, cand: str) -> Dict[str, float]:
    """ROUGE-1, ROUGE-2, ROUGE-L for a single pair."""
    global _rouge_scorer
    if not ref.strip() or not cand.strip():
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    try:
        if _rouge_scorer is None:
            try:
                import evaluate
                _rouge_scorer = evaluate.load("rouge")
            except Exception:
                from rouge_score import rouge_scorer as rs
                _rouge_scorer = rs.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        if hasattr(_rouge_scorer, 'compute'):
            # HF evaluate
            scores = _rouge_scorer.compute(predictions=[cand], references=[ref])
            return {
                "rouge1": scores.get("rouge1", 0.0),
                "rouge2": scores.get("rouge2", 0.0),
                "rougeL": scores.get("rougeL", 0.0),
            }
        else:
            # rouge_score library
            scores = _rouge_scorer.score(ref, cand)
            return {
                "rouge1": scores['rouge1'].fmeasure,
                "rouge2": scores['rouge2'].fmeasure,
                "rougeL": scores['rougeL'].fmeasure,
            }
    except Exception:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def compute_bleu_pair(ref: str, cand: str) -> Dict[str, float]:
    """BLEU-1..4 for a single pair."""
    if not ref.strip() or not cand.strip():
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smooth = SmoothingFunction().method1
        ref_tokens = ref.lower().split()
        gen_tokens = cand.lower().split()
        if len(ref_tokens) < 4 or len(gen_tokens) < 4:
            return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}
        return {
            "bleu1": sentence_bleu([ref_tokens], gen_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth),
            "bleu2": sentence_bleu([ref_tokens], gen_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth),
            "bleu3": sentence_bleu([ref_tokens], gen_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth),
            "bleu4": sentence_bleu([ref_tokens], gen_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth),
        }
    except ImportError:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}


def compute_rouge_batch(refs: List[str], cands: List[str]) -> Dict[str, float]:
    """ROUGE-1, ROUGE-2, ROUGE-L computed in a single batch pass.

    Uses HF evaluate (mean-aggregated) when available, otherwise falls back
    to rouge_score with per-sample averaging. Output structure is identical
    to what compute_rouge_pair returns so callers are unaffected.
    """
    valid = [(r, c) for r, c in zip(refs, cands) if r.strip() and c.strip()]
    if not valid:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    rs, cs = zip(*valid)
    rs, cs = list(rs), list(cs)

    # ---- Primary: HF evaluate (one .compute call for the whole batch) ----
    try:
        import evaluate
        scorer = evaluate.load("rouge")
        scores = scorer.compute(predictions=cs, references=rs)
        return {
            "rouge1": float(scores.get("rouge1", 0.0)),
            "rouge2": float(scores.get("rouge2", 0.0)),
            "rougeL": float(scores.get("rougeL", 0.0)),
        }
    except Exception:
        pass

    # ---- Fallback: rouge_score per-sample then average ----
    try:
        from rouge_score import rouge_scorer as rouge_lib
        scorer = rouge_lib.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        r1, r2, rl = [], [], []
        for ref, cand in zip(rs, cs):
            s = scorer.score(ref, cand)
            r1.append(s['rouge1'].fmeasure)
            r2.append(s['rouge2'].fmeasure)
            rl.append(s['rougeL'].fmeasure)
        return {
            "rouge1": float(np.mean(r1)),
            "rouge2": float(np.mean(r2)),
            "rougeL": float(np.mean(rl)),
        }
    except ImportError:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def compute_bertscore_batch(refs: List[str], cands: List[str]) -> Dict[str, float]:
    """BERTScore P/R/F1 averaged over batch."""
    try:
        from bert_score import score as bert_score_fn
        valid = [(r, c) for r, c in zip(refs, cands) if r.strip() and c.strip()]
        if not valid:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        rs, cs = zip(*valid)
        P, R, F1 = bert_score_fn(list(cs), list(rs), model_type="roberta-large", verbose=False)
        return {
            "precision": float(P.mean()),
            "recall": float(R.mean()),
            "f1": float(F1.mean()),
            "per_sample_f1": [float(f) for f in F1],
        }
    except (ImportError, Exception) as e:
        print(f"    BERTScore failed: {e}")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def compute_meteor_batch(refs: List[str], cands: List[str]) -> float:
    """METEOR score averaged over batch via HF evaluate.

    BLEURT is not used: Google's checkpoint bucket is no longer publicly
    accessible, so evaluate.load('bleurt') always fails at download time.
    METEOR is a well-established alternative (uses unigram F-mean with
    stemming/synonym matching) and ships cleanly with evaluate + nltk.
    """
    try:
        import nltk
        # METEOR requires punkt tokeniser and wordnet; download silently if absent
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)  # NLTK >= 3.8 renamed punkt → punkt_tab
        nltk.download('wordnet', quiet=True)
        import evaluate
        meteor = evaluate.load("meteor")
        valid = [(r, c) for r, c in zip(refs, cands) if r.strip() and c.strip()]
        if not valid:
            return 0.0
        rs, cs = zip(*valid)
        result = meteor.compute(predictions=list(cs), references=list(rs))
        return float(result["meteor"])
    except Exception as e:
        print(f"    METEOR failed: {e}")
        return 0.0


def compute_all_metrics_for_config(
    references: List[str],
    candidates: List[str],
    judge_data: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Compute all metrics for a (references, candidates) pair."""
    valid = [(r, c) for r, c in zip(references, candidates) if r.strip() and c.strip()]
    if not valid:
        return {"error": "No valid pairs"}
    
    refs, cands = zip(*valid)
    refs, cands = list(refs), list(cands)
    
    # ROUGE (single batch call)
    print("    Computing ROUGE...")
    rouge = compute_rouge_batch(refs, cands)
    
    # BLEU (per-sample then average)
    bleu_scores = [compute_bleu_pair(r, c) for r, c in zip(refs, cands)]
    bleu = {
        f"avg_bleu{i}": np.mean([s[f"bleu{i}"] for s in bleu_scores])
        for i in range(1, 5)
    }
    
    # BERTScore (batch)
    print("    Computing BERTScore...")
    bertscore = compute_bertscore_batch(refs, cands)
    
    # METEOR
    print("    Computing METEOR...")
    meteor_val = compute_meteor_batch(refs, cands)

    return {
        "rouge": rouge,
        "bleu": bleu,
        "bertscore": bertscore,
        "meteor": meteor_val,
        "llm_judge": judge_data or {},
        "num_samples": len(refs),
    }


def compute_per_section_metrics(
    references: List[str],
    candidates: List[str],
) -> Dict[str, Dict[str, float]]:
    """Per-section ROUGE-1, ROUGE-L, BLEU-4, BERTScore."""
    section_scores = {
        key: {"rouge1": [], "rougeL": [], "bleu4": [], "bertscore_refs": [], "bertscore_cands": []}
        for key, _ in CLINICAL_SECTIONS
    }
    
    for ref, cand in zip(references, candidates):
        ref_sections = parse_sections(ref)
        cand_sections = parse_sections(cand)
        for key, _ in CLINICAL_SECTIONS:
            ref_sec = ref_sections.get(key, "")
            cand_sec = cand_sections.get(key, "")
            if ref_sec:
                rouge = compute_rouge_pair(ref_sec, cand_sec)
                section_scores[key]["rouge1"].append(rouge["rouge1"])
                section_scores[key]["rougeL"].append(rouge["rougeL"])
                bleu = compute_bleu_pair(ref_sec, cand_sec)
                section_scores[key]["bleu4"].append(bleu["bleu4"])
                section_scores[key]["bertscore_refs"].append(ref_sec)
                section_scores[key]["bertscore_cands"].append(cand_sec)
    
    # Compute BERTScore per section (batch for efficiency)
    result = {}
    for key, _ in CLINICAL_SECTIONS:
        sd = section_scores[key]
        bs_f1 = 0.0
        if sd["bertscore_refs"]:
            bs = compute_bertscore_batch(sd["bertscore_refs"], sd["bertscore_cands"])
            bs_f1 = bs.get("f1", 0.0)
        
        result[key] = {
            "rouge1": np.mean(sd["rouge1"]) if sd["rouge1"] else 0.0,
            "rougeL": np.mean(sd["rougeL"]) if sd["rougeL"] else 0.0,
            "bleu4": np.mean(sd["bleu4"]) if sd["bleu4"] else 0.0,
            "bertscore": bs_f1,
        }
    
    return result


# =============================================================================
# Data Loading & Processing
# =============================================================================

def load_all_data(
    eval_json_paths: List[str],
    labels: List[str],
    medcon_data: Optional[Dict],
) -> Dict:
    """
    Load eval JSONs and compute all metrics from raw_outputs.
    
    Returns:
        {label: {config: {metrics: {...}, per_section: {...}}}}
    """
    all_data = {}
    
    for eval_path, label in zip(eval_json_paths, labels):
        print(f"\n{'='*60}")
        print(f"Processing: {label}")
        print(f"{'='*60}")
        
        with open(eval_path) as f:
            eval_json = json.load(f)
        
        references = eval_json.get("references", [])
        comparative = eval_json.get("comparative", {})
        
        all_data[label] = {}
        
        for config in CONFIGS:
            comp = comparative.get(config, {})
            if not comp or comp.get("error"):
                continue
            
            candidates = comp.get("raw_outputs", [])
            if not references or not candidates:
                continue
            
            print(f"\n  Config: {config} ({len(candidates)} samples)")
            
            # Get pre-computed judge data from eval JSON
            judge_data = comp.get("metrics", {}).get("llm_judge", {})
            
            # Compute all metrics from raw text
            metrics = compute_all_metrics_for_config(references, candidates, judge_data)
            
            # MEDCON from external JSON
            mc = 0.0
            if medcon_data and label in medcon_data:
                mc = medcon_data[label].get(config, {}).get("medcon_f1", 0.0)
            metrics["medcon_f1"] = mc
            
            # Per-section
            print(f"    Computing per-section metrics...")
            per_section = compute_per_section_metrics(references, candidates)
            
            all_data[label][config] = {
                "metrics": metrics,
                "per_section": per_section,
            }
            
            print(f"    ROUGE-L: {metrics['rouge']['rougeL']:.3f}  "
                  f"BLEU-4: {metrics['bleu']['avg_bleu4']:.3f}  "
                  f"BERTScore: {metrics['bertscore']['f1']:.3f}")
    
    return all_data


# =============================================================================
# Table A: Automated Metrics
# =============================================================================

def generate_table_a(data: Dict, labels: List[str], output_dir: Path):
    lines = [
        "# Table A: Automated Evaluation Metrics",
        "",
        "Computed directly from raw model outputs (100-sample test set).",
        "MEDCON uses QuickUMLS + UMLS 2025AB. METEOR computed via evaluate + nltk (replaces BLEURT).",
        "",
        "| Model | Config | RAG | Scope | ROUGE-1 | ROUGE-L | BLEU-4 | BERTScore | MEDCON-F1 | METEOR |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    
    for label in labels:
        first_label = True
        for config in CONFIGS:
            cd = data.get(label, {}).get(config)
            if not cd:
                continue
            m = cd["metrics"]
            rag = RAG_METHOD.get(config, "\u2014")
            model_col = label if first_label else ""
            first_label = False
            cl = CONFIG_SHORT.get(config, config)
            
            r1 = m["rouge"]["rouge1"]
            rl = m["rouge"]["rougeL"]
            bl = m["bleu"]["avg_bleu4"]
            bs = m["bertscore"]["f1"]
            mc = m.get("medcon_f1", 0.0)
            meteor_v = m.get("meteor", 0.0)

            lines.append(
                f"| {model_col} | {cl} | {rag} | **Overall** | "
                f"{fmt(r1)} | {fmt(rl)} | {fmt(bl)} | {fmt(bs)} | {fmt(mc)} | {fmt(meteor_v)} |"
            )
            
            # Per-section
            ps = cd.get("per_section", {})
            for key, name in CLINICAL_SECTIONS:
                sd = ps.get(key, {})
                sr1 = sd.get("rouge1", 0.0)
                srl = sd.get("rougeL", 0.0)
                sbl = sd.get("bleu4", 0.0)
                sbs = sd.get("bertscore", 0.0)
                if srl > 0:
                    lines.append(
                        f"| | | | {name} | {fmt(sr1)} | {fmt(srl)} | {fmt(sbl)} | {fmt(sbs)} | \u2014 | \u2014 |"
                    )
        
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
    
    path = output_dir / "table_a_automated_metrics.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved: {path}")


# =============================================================================
# Table B: Judge Scores
# =============================================================================

def generate_table_b(data: Dict, labels: List[str], output_dir: Path):
    header = "| Model | Config | RAG | " + " | ".join(d[1] for d in JUDGE_DIMS) + " | Critical Err |"
    sep = "|---|---|---|" + "---|" * len(JUDGE_DIMS) + "---|"
    
    lines = [
        "# Table B: LLM-as-Judge Evaluation (GPT-4o-mini, 1-5 scale)",
        "",
        "Higher is better. Halluc.\u2191 means higher = less hallucination.",
        "",
        header, sep,
    ]
    
    for label in labels:
        first_label = True
        for config in CONFIGS:
            cd = data.get(label, {}).get(config)
            if not cd:
                continue
            judge = cd["metrics"].get("llm_judge", {})
            rag = RAG_METHOD.get(config, "\u2014")
            model_col = label if first_label else ""
            first_label = False
            cl = CONFIG_SHORT.get(config, config)
            
            vals = []
            for key, _ in JUDGE_DIMS:
                v = judge.get(key, 0)
                vals.append(f"{v:.2f}" if v > 0 else "\u2014")
            
            n_err = judge.get("samples_with_critical_errors", 0)
            n_total = judge.get("num_evaluated", cd["metrics"].get("num_samples", 0))
            err_str = f"{int(n_err)}/{int(n_total)}" if n_total > 0 else "\u2014"
            
            lines.append(
                f"| {model_col} | {cl} | {rag} | " + " | ".join(vals) + f" | {err_str} |"
            )
        lines.append(sep)
    
    path = output_dir / "table_b_judge_scores.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved: {path}")


# =============================================================================
# Plots
# =============================================================================

def _get_colors(n):
    return plt.cm.Set2(np.linspace(0, 1, max(n, 3)))


def plot_rouge_variants(data: Dict, labels: List[str], output_dir: Path):
    rouge_types = [("rouge1", "ROUGE-1"), ("rouge2", "ROUGE-2"), ("rougeL", "ROUGE-L")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    colors = _get_colors(len(labels))
    
    for ax_idx, (rk, rn) in enumerate(rouge_types):
        ax = axes[ax_idx]
        x = np.arange(len(CONFIGS))
        w = 0.8 / len(labels)
        for i, label in enumerate(labels):
            vals = [data.get(label, {}).get(c, {}).get("metrics", {}).get("rouge", {}).get(rk, 0) for c in CONFIGS]
            bars = ax.bar(x + i * w, vals, w, label=label, color=colors[i])
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.01,
                            f"{val:.3f}", ha="center", va="bottom", fontsize=6, rotation=45)
        ax.set_title(rn, fontsize=13, fontweight="bold")
        ax.set_xticks(x + w*(len(labels)-1)/2)
        ax.set_xticklabels([CONFIG_SHORT[c] for c in CONFIGS], fontsize=8)
        ax.set_ylim(0, 0.85)
        if ax_idx == 0: ax.set_ylabel("Score")
    axes[0].legend(fontsize=8)
    fig.suptitle("ROUGE Scores Across Models and Configurations", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "rouge_variants.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: rouge_variants.png")



def plot_bleu_variants(data: Dict, labels: List[str], output_dir: Path):
    bleu_keys = [("avg_bleu1", "BLEU-1"), ("avg_bleu2", "BLEU-2"), ("avg_bleu3", "BLEU-3"), ("avg_bleu4", "BLEU-4")]
    cfgs = ["ft_only", "ft_rag"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    colors = _get_colors(len(labels))
    for ax_idx, config in enumerate(cfgs):
        ax = axes[ax_idx]
        x = np.arange(len(bleu_keys))
        w = 0.8 / len(labels)
        for i, label in enumerate(labels):
            vals = [data.get(label, {}).get(config, {}).get("metrics", {}).get("bleu", {}).get(bk, 0) for bk, _ in bleu_keys]
            bars = ax.bar(x + i*w, vals, w, label=label, color=colors[i])
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_title(CONFIG_SHORT[config], fontsize=12, fontweight="bold")
        ax.set_xticks(x + w*(len(labels)-1)/2)
        ax.set_xticklabels([bn for _, bn in bleu_keys], fontsize=9)
        ax.set_ylim(0, 0.85)
        if ax_idx == 0: ax.set_ylabel("Score")
    axes[0].legend(fontsize=8)
    fig.suptitle("BLEU Scores: Fine-tuned Configurations", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "bleu_variants.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: bleu_variants.png")




def plot_bertscore(data: Dict, labels: List[str], output_dir: Path):
    bert_dims = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]
    cfgs = ["baseline", "ft_only", "ft_rag", "teacher"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    colors = _get_colors(len(labels))
    for ax_idx, (bk, bn) in enumerate(bert_dims):
        ax = axes[ax_idx]
        x = np.arange(len(cfgs))
        w = 0.8 / len(labels)
        for i, label in enumerate(labels):
            vals = [data.get(label, {}).get(c, {}).get("metrics", {}).get("bertscore", {}).get(bk, 0) for c in cfgs]
            bars = ax.bar(x + i*w, vals, w, label=label, color=colors[i])
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f"{val:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_title(f"BERTScore {bn}", fontsize=12, fontweight="bold")
        ax.set_xticks(x + w*(len(labels)-1)/2)
        ax.set_xticklabels([CONFIG_SHORT[c] for c in cfgs], fontsize=8)
        ax.set_ylim(0.4, 1.0)
        if ax_idx == 0: ax.set_ylabel("Score")
    axes[0].legend(fontsize=8)
    fig.suptitle("BERTScore (roberta-large)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "bertscore.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: bertscore.png")




def plot_judge_dimensions(data: Dict, labels: List[str], output_dir: Path):
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = _get_colors(len(labels))
    x = np.arange(len(JUDGE_DIMS))
    w = 0.8 / (len(labels) + 1)
    for i, label in enumerate(labels):
        judge = data.get(label, {}).get("ft_rag", {}).get("metrics", {}).get("llm_judge", {})
        vals = [judge.get(dk, 0) for dk, _ in JUDGE_DIMS]
        bars = ax.bar(x + i*w, vals, w, label=label, color=colors[i])
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05, f"{val:.2f}", ha="center", va="bottom", fontsize=7)
    # Teacher
    t_judge = data.get(labels[0], {}).get("teacher", {}).get("metrics", {}).get("llm_judge", {})
    t_vals = [t_judge.get(dk, 0) for dk, _ in JUDGE_DIMS]
    bars = ax.bar(x + len(labels)*w, t_vals, w, label="Teacher", color="lightgray", edgecolor="gray", hatch="//")
    for bar, val in zip(bars, t_vals):
        if val > 0:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05, f"{val:.2f}", ha="center", va="bottom", fontsize=7, color="gray")
    ax.set_ylabel("Score (1-5)")
    ax.set_title("LLM-as-Judge: FT+RAG Config", fontsize=13, fontweight="bold")
    ax.set_xticks(x + w*len(labels)/2)
    ax.set_xticklabels([dn for _, dn in JUDGE_DIMS], fontsize=9)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 5.5)
    plt.tight_layout()
    plt.savefig(output_dir / "judge_dimensions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: judge_dimensions.png")


def _radar_vals(m: Dict) -> List[float]:
    """Extract the 7 normalised (0-1) radar values for a metrics dict.

    Order must match RADAR_CATEGORIES below.
    Judge scores (1-5) are divided by 5 to normalise to [0,1].
    All other metrics are already in [0,1].
    """
    return [
        m.get("rouge", {}).get("rougeL", 0),          # ROUGE-L
        m.get("bleu", {}).get("avg_bleu4", 0),         # BLEU-4
        m.get("bertscore", {}).get("f1", 0),           # BERTScore
        m.get("meteor", 0),                             # METEOR
        m.get("medcon_f1", 0),                          # MEDCON-F1
        m.get("llm_judge", {}).get("avg_overall", 0) / 5.0,       # Judge Overall
        m.get("llm_judge", {}).get("avg_hallucination", 0) / 5.0, # Judge Halluc.
    ]

RADAR_CATEGORIES = [
    "ROUGE-L", "BLEU-4", "BERTScore", "METEOR",
    "MEDCON-F1", "Judge\nOverall", "Judge\nHalluc."
]




def plot_radar(data: Dict, labels: List[str], output_dir: Path):
    n = len(RADAR_CATEGORIES)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist() + [0]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    colors = _get_colors(len(labels))

    for i, label in enumerate(labels):
        m = data.get(label, {}).get("ft_rag", {}).get("metrics", {})
        vals = _radar_vals(m)
        closed = vals + [vals[0]]
        ax.plot(angles, closed, "o-", label=label, color=colors[i], linewidth=2)
        ax.fill(angles, closed, alpha=0.1, color=colors[i])





    # Teacher
    tm = data.get(labels[0], {}).get("teacher", {}).get("metrics", {})
    t_closed = _radar_vals(tm) + [_radar_vals(tm)[0]]
    ax.plot(angles, t_closed, "^--", label="Teacher", color="gray", linewidth=1.5, alpha=0.5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])          # clear default labels; we'll draw manually
    ax.set_ylim(0, 1.0)

    #  Manually place category labels with per-angle padding 
    PAD = 1.18   # fractional radial offset (1.0 = edge of plot, >1 = outside)
    for angle, label in zip(angles[:-1], RADAR_CATEGORIES):
        x = angle * 180 / np.pi           # degrees, used for ha/va logic
        ha = "center"
        va = "center"
        # Adjust horizontal alignment based on which side of the circle
        if x < 10 or x > 350:
            ha = "center"
        elif 10 <= x < 180:
            ha = "left"
        else:
            ha = "right"
        # Adjust vertical alignment for top/bottom labels
        if 80 <= x <= 100:
            va = "bottom"
        elif 260 <= x <= 280:
            va = "top"
        ax.text(
            angle, ax.get_ylim()[1] * PAD,
            label,
            ha=ha, va=va,
            fontsize=10,
            multialignment="center",
        )

    ax.set_title("Multi-Metric Radar: FT+RAG", fontsize=14, fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "radar_multimetric.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: radar_multimetric.png")




def plot_heatmap(data: Dict, labels: List[str], output_dir: Path):
    # All values normalised to [0, 1]. Judge scores (1-5) divided by 5.
    metric_defs = [
        ("ROUGE-1",    lambda m: m.get("rouge", {}).get("rouge1", 0)),
        ("ROUGE-L",    lambda m: m.get("rouge", {}).get("rougeL", 0)),
        ("BLEU-4",     lambda m: m.get("bleu", {}).get("avg_bleu4", 0)),
        ("BERTScore",  lambda m: m.get("bertscore", {}).get("f1", 0)),
        ("METEOR",     lambda m: m.get("meteor", 0)),
        ("MEDCON-F1",  lambda m: m.get("medcon_f1", 0)),
        ("Judge Overall",   lambda m: m.get("llm_judge", {}).get("avg_overall", 0) / 5.0),
        ("Hallucination↑",  lambda m: m.get("llm_judge", {}).get("avg_hallucination", 0) / 5.0),
    ]
    
    
    
    row_labels = []
    matrix = []
    for label in labels:
        for config in CONFIGS:
            cd = data.get(label, {}).get(config)
            if not cd:
                row_labels.append(f"{label}\n{CONFIG_SHORT.get(config, config)}")
                matrix.append([0]*len(metric_defs))
                continue
            m = cd["metrics"]
            row_labels.append(f"{label}\n{CONFIG_SHORT.get(config, config)}")
            matrix.append([fn(m) for _, fn in metric_defs])
    
    matrix = np.array(matrix)
    col_labels = [mn for mn, _ in metric_defs]

    fig, ax = plt.subplots(figsize=(14, max(8, len(row_labels)*0.5)))
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7, color=color)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Score (0-1)")
    ax.set_title("Comprehensive Metrics Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: metrics_heatmap.png")





def plot_per_section_rouge(data: Dict, labels: List[str], output_dir: Path, config="ft_rag"):
    section_keys = [k for k, _ in CLINICAL_SECTIONS]
    section_names = [n for _, n in CLINICAL_SECTIONS]
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = _get_colors(len(labels))
    x = np.arange(len(section_keys))
    w = 0.8 / len(labels)
    for i, label in enumerate(labels):
        ps = data.get(label, {}).get(config, {}).get("per_section", {})
        vals = [ps.get(k, {}).get("rougeL", 0) for k in section_keys]
        bars = ax.bar(x + i*w, vals, w, label=label, color=colors[i])
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=7, rotation=45)
    ax.set_xlabel("Clinical Section")
    ax.set_ylabel("ROUGE-L")
    ax.set_title(f"Per-Section ROUGE-L: {CONFIG_SHORT.get(config, config)}", fontsize=13, fontweight="bold")
    ax.set_xticks(x + w*(len(labels)-1)/2)
    ax.set_xticklabels(section_names, fontsize=9, rotation=30, ha="right")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(output_dir / "per_section_rouge.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: per_section_rouge.png")


def plot_size_vs_performance(data: Dict, labels: List[str], output_dir: Path):
    sizes = []
    for label in labels:
        match = re.search(r'(\d+\.?\d*)\s*B', label)
        sizes.append(float(match.group(1)) if match else 0)
    if not any(sizes):
        return
    
    metric_fns = [
        ("ROUGE-L", lambda m: m.get("rouge", {}).get("rougeL", 0)),
        ("BLEU-4", lambda m: m.get("bleu", {}).get("avg_bleu4", 0)),
        ("Judge Overall", lambda m: m.get("llm_judge", {}).get("avg_overall", 0)),
        ("BERTScore", lambda m: m.get("bertscore", {}).get("f1", 0)),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colors = _get_colors(len(labels))
    for ax_idx, (mn, fn) in enumerate(metric_fns):
        ax = axes[ax_idx//2][ax_idx%2]
        vals = [fn(data.get(l, {}).get("ft_rag", {}).get("metrics", {})) for l in labels]
        for i, (s, v, label) in enumerate(zip(sizes, vals, labels)):
            ax.scatter(s, v, s=150, color=colors[i], zorder=5, edgecolors="black")
            ax.annotate(label, (s, v), textcoords="offset points", xytext=(10, 5), fontsize=8)
        if len(sizes) >= 2:
            z = np.polyfit(sizes, vals, 1)
            xs = np.linspace(min(sizes)*0.8, max(sizes)*1.2, 100)
            ax.plot(xs, np.poly1d(z)(xs), "--", color="gray", alpha=0.5)
        ax.set_xlabel("Model Size (B)")
        ax.set_ylabel(mn)
        ax.set_title(f"Size vs {mn}", fontsize=11, fontweight="bold")
    fig.suptitle("Model Size-Performance Tradeoff (FT+RAG)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "size_vs_performance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: size_vs_performance.png")


def plot_medcon(data: Dict, labels: List[str], output_dir: Path):
    """MEDCON F1 across configs (from medcon_json)."""
    cfgs = ["baseline", "ft_only", "ft_rag", "teacher"]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = _get_colors(len(labels))
    x = np.arange(len(cfgs))
    w = 0.8 / len(labels)
    for i, label in enumerate(labels):
        vals = [data.get(label, {}).get(c, {}).get("metrics", {}).get("medcon_f1", 0) for c in cfgs]
        bars = ax.bar(x + i*w, vals, w, label=label, color=colors[i])
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_title("MEDCON-F1 (Medical Concept Overlap)", fontsize=13, fontweight="bold")
    ax.set_xticks(x + w*(len(labels)-1)/2)
    ax.set_xticklabels([CONFIG_SHORT[c] for c in cfgs], fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "medcon_scores.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: medcon_scores.png")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dissertation Plots & Tables (self-contained)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--eval-jsons", nargs="+", required=True,
                        help="Evaluation JSONs (one per model)")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="Model labels (same order as --eval-jsons)")
    parser.add_argument("--medcon-json", default=None,
                        help="MEDCON results JSON (QuickUMLS)")
    parser.add_argument("--output-dir", default="./dissertation_final")
    
    args = parser.parse_args()
    
    if len(args.eval_jsons) != len(args.labels):
        print("ERROR: --eval-jsons and --labels must have same count")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    medcon_data = None
    if args.medcon_json:
        with open(args.medcon_json) as f:
            medcon_data = json.load(f)
        print(f"Loaded MEDCON: {args.medcon_json}")
    
    # Load and compute all metrics
    data = load_all_data(args.eval_jsons, args.labels, medcon_data)
    
    # Save computed metrics as JSON
    save_data = {}
    for label, configs in data.items():
        save_data[label] = {}
        for config, cd in configs.items():
            m = cd["metrics"]
            save_data[label][config] = {
                "rouge": {k: float(v) for k, v in m["rouge"].items()},
                "bleu": {k: float(v) for k, v in m["bleu"].items()},
                "bertscore": {k: float(v) for k, v in m["bertscore"].items() if k != "per_sample_f1"},
                "meteor": float(m.get("meteor", 0)),
                "medcon_f1": float(m.get("medcon_f1", 0)),
                "llm_judge": m.get("llm_judge", {}),
                "num_samples": m.get("num_samples", 0),
                "per_section": {
                    key: {k: float(v) for k, v in secs.items()}
                    for key, secs in cd.get("per_section", {}).items()
                },
            }
    with open(output_dir / "computed_metrics.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: computed_metrics.json")
    
    # Tables
    print("\n" + "="*50)
    print("GENERATING TABLES")
    print("="*50)
    generate_table_a(data, args.labels, output_dir)
    generate_table_b(data, args.labels, output_dir)
    
    # Plots
    print("\n" + "="*50)
    print("GENERATING PLOTS")
    print("="*50)
    plot_rouge_variants(data, args.labels, output_dir)
    plot_bleu_variants(data, args.labels, output_dir)
    plot_bertscore(data, args.labels, output_dir)
    plot_medcon(data, args.labels, output_dir)
    plot_judge_dimensions(data, args.labels, output_dir)
    plot_radar(data, args.labels, output_dir)
    plot_heatmap(data, args.labels, output_dir)
    plot_per_section_rouge(data, args.labels, output_dir)
    plot_size_vs_performance(data, args.labels, output_dir)
    
    print(f"\nAll outputs saved to {output_dir}/")
