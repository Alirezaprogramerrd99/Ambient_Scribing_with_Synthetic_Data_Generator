"""
RAG Ablation Analysis — Standalone Script

Reads RAG ablation evaluation JSONs (one per model) and produces:
- Table C: RAG Ablation with per-section metrics (markdown)
- RAG ablation comparison plots

Separate from plot_dissertation.py to avoid MEDCON data contamination.
Each model's RAG ablation JSON contains rag_backends with 4 configs:
  dense_only, dense_rerank, dense_rerank_qe, full_medical

Usage:
    python plot_rag_ablation.py \
        --eval-jsons phi_rag.json llama3b_rag.json llama1b_rag.json \
        --labels "Phi-3.5 (3.8B)" "Llama-3.2 (3B)" "Llama-3.2 (1B)" \
        --medcon-json ./medcon_rag_results.json \
        --output-dir ./rag_ablation_tables

Author: Alireza Rashidi
MSc Project: Trustworthy SLMs for Ambient Clinical Scribing
"""

import json
import argparse
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Clinical Section Definitions
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

# RAG config display names
RAG_DISPLAY = {
    "dense_only": "Dense Only",
    "dense_rerank": "Dense+Rerank",
    "dense_rerank_qe": "Dense+Rerank+QE",
    "full_medical": "Full Medical",
    # Legacy keys
    "llama_index": "Dense (LlamaIndex)",
    "manual": "Custom Dense",
    "hybrid": "Full Medical",
}

RAG_CONFIGS_NEW = ["dense_only", "dense_rerank", "dense_rerank_qe", "full_medical"]
RAG_CONFIGS_OLD = ["llama_index", "manual", "hybrid"]


# =============================================================================
# Section Parsing
# =============================================================================

def parse_sections(text: str) -> Dict[str, str]:
    """Parse clinical summary into sections."""
    sections = {}
    for key, pattern in SECTION_PATTERNS.items():
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            sections[key] = match.group(1).strip()
    return sections


# =============================================================================
# Per-Section Metrics
# =============================================================================

def compute_section_metrics(generated: str, reference: str) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-L, BLEU-4, BERTScore for a section pair."""
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
        pass
    
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


def compute_per_section_for_config(
    references: List[str],
    candidates: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute per-section metrics averaged over all samples."""
    section_scores = {
        key: {"rouge1": [], "rougeL": [], "bleu": [], "bertscore": []}
        for key, _ in CLINICAL_SECTIONS
    }
    
    for ref, cand in zip(references, candidates):
        ref_sections = parse_sections(ref)
        cand_sections = parse_sections(cand)
        
        for key, _ in CLINICAL_SECTIONS:
            ref_sec = ref_sections.get(key, "")
            cand_sec = cand_sections.get(key, "")
            if ref_sec:
                metrics = compute_section_metrics(cand_sec, ref_sec)
                for mk, mv in metrics.items():
                    section_scores[key][mk].append(mv)
    
    # Average
    result = {}
    for key, scores_dict in section_scores.items():
        result[key] = {
            mk: (sum(vals) / len(vals) if vals else 0.0)
            for mk, vals in scores_dict.items()
        }
    return result


# =============================================================================
# Formatting Helper
# =============================================================================

def fmt(v, decimals=3):
    """Format value with dashes for missing data."""
    return f"{v:.{decimals}f}" if v > 0 else "\u2014"


# =============================================================================
# Table C: RAG Ablation
# =============================================================================

def generate_table_c(
    eval_json_paths: List[str],
    labels: List[str],
    medcon_data: Optional[Dict],
    output_dir: Path,
):
    """
    Generate Table C: RAG Ablation Study.
    
    Reads rag_backends from each eval JSON, computes per-section metrics,
    includes teacher as reference, uses actual MEDCON when available.
    """
    lines = [
        "# Table C: RAG Ablation Study",
        "",
        "All results use fine-tuned models on 50-sample test set.",
        "Teacher (GPT-4o-mini, no RAG) shown as reference.",
        "Per-section values: ROUGE-L / BERTScore. MEDCON at overall level only.",
        "",
        "| Model | RAG Config | Scope | ROUGE-L | ROUGE-1 | BLEU-4 | BERTScore | MEDCON-F1 | Judge Avg |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    
    all_per_section = {}  # For plots later
    
    for eval_path, label in zip(eval_json_paths, labels):
        print(f"\nProcessing: {label}")
        with open(eval_path) as f:
            eval_data = json.load(f)
        
        references = eval_data.get("references", [])
        rag_backends = eval_data.get("rag_backends", {})
        
        if not references:
            print(f"  WARNING: No references found in {eval_path}")
            continue
        
        # Detect key format
        available_keys = list(rag_backends.keys())
        if any(k in available_keys for k in RAG_CONFIGS_NEW):
            rag_keys = RAG_CONFIGS_NEW
        else:
            rag_keys = RAG_CONFIGS_OLD
        
        first_label = True
        model_sections = {}
        
        for rag_name in rag_keys:
            rag_data = rag_backends.get(rag_name, {})
            if not rag_data or rag_data.get("error"):
                continue
            
            candidates = rag_data.get("raw_outputs", [])
            if not candidates:
                continue
            
            metrics = rag_data.get("metrics", {})
            
            # Overall metrics: try eval JSON first, compute from raw_outputs if missing
            rl = metrics.get("rouge", {}).get("rougeL", 0)
            r1 = metrics.get("rouge", {}).get("rouge1", 0)
            bl = metrics.get("bleu", {}).get("avg_bleu4", 0)
            bs = metrics.get("bertscore", {}).get("f1", 0)
            judge_avg = metrics.get("llm_judge", {}).get("avg_overall", 0)
            
            # If ROUGE/BLEU/BERTScore missing, compute from raw outputs
            if rl == 0 or bl == 0 or bs == 0:
                print(f"  Computing overall metrics from raw_outputs for {rag_name}...")
                overall_rouge1 = []
                overall_rougeL = []
                overall_bleu = []
                overall_bertscore = []
                
                try:
                    from rouge_score import rouge_scorer
                    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
                    for ref, cand in zip(references, candidates):
                        if ref.strip() and cand.strip():
                            scores = scorer.score(ref, cand)
                            overall_rouge1.append(scores['rouge1'].fmeasure)
                            overall_rougeL.append(scores['rougeL'].fmeasure)
                except ImportError:
                    pass
                
                try:
                    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
                    smooth = SmoothingFunction().method1
                    for ref, cand in zip(references, candidates):
                        ref_tokens = ref.lower().split()
                        gen_tokens = cand.lower().split()
                        if len(ref_tokens) >= 4 and len(gen_tokens) >= 4:
                            overall_bleu.append(sentence_bleu(
                                [ref_tokens], gen_tokens,
                                weights=(0.25, 0.25, 0.25, 0.25),
                                smoothing_function=smooth,
                            ))
                except ImportError:
                    pass
                
                try:
                    from bert_score import score as bert_score_fn
                    valid_pairs = [(r, c) for r, c in zip(references, candidates) 
                                   if r.strip() and c.strip()]
                    if valid_pairs:
                        refs_clean, cands_clean = zip(*valid_pairs)
                        P, R, F1 = bert_score_fn(
                            list(cands_clean), list(refs_clean),
                            model_type="roberta-large",
                            verbose=False,
                        )
                        overall_bertscore = [float(f) for f in F1]
                except (ImportError, Exception) as e:
                    print(f"    BERTScore overall failed: {e}")
                
                if overall_rouge1 and rl == 0:
                    r1 = sum(overall_rouge1) / len(overall_rouge1)
                if overall_rougeL and rl == 0:
                    rl = sum(overall_rougeL) / len(overall_rougeL)
                if overall_bleu and bl == 0:
                    bl = sum(overall_bleu) / len(overall_bleu)
                if overall_bertscore and bs == 0:
                    bs = sum(overall_bertscore) / len(overall_bertscore)
            
            # MEDCON: prefer actual QuickUMLS data
            mc = 0.0
            if medcon_data and label in medcon_data:
                mc_key = f"rag_{rag_name}"
                mc = medcon_data[label].get(mc_key, {}).get("medcon_f1", 0.0)
            if mc == 0.0:
                mc = metrics.get("medcon", {}).get("f1", 0)
            
            rag_label = RAG_DISPLAY.get(rag_name, rag_name)
            model_col = label if first_label else ""
            first_label = False
            
            # Overall row
            lines.append(
                f"| {model_col} | {rag_label} | **Overall** | "
                f"{fmt(rl)} | {fmt(r1)} | {fmt(bl)} | {fmt(bs)} | {fmt(mc)} | {fmt(judge_avg, 2)} |"
            )
            
            # Per-section metrics
            print(f"  Computing per-section metrics for {rag_name}...")
            sec_metrics = compute_per_section_for_config(references, candidates)
            model_sections[rag_name] = sec_metrics
            
            for key, name in CLINICAL_SECTIONS:
                sd = sec_metrics.get(key, {})
                sec_rl = sd.get("rougeL", 0.0)
                sec_r1 = sd.get("rouge1", 0.0)
                sec_bl = sd.get("bleu", 0.0)
                sec_bs = sd.get("bertscore", 0.0)
                if any(v > 0 for v in (sec_rl, sec_r1, sec_bl, sec_bs)):
                    lines.append(
                        f"| | | {name} | {fmt(sec_rl)} | {fmt(sec_r1)} | {fmt(sec_bl)} | {fmt(sec_bs)} | \u2014 | \u2014 |"
                    )
        
        # Teacher reference row (from comparative section)
        comp = eval_data.get("comparative", {}).get("teacher", {})
        if comp and not comp.get("error"):
            t_metrics = comp.get("metrics", {})
            t_rl = t_metrics.get("rouge", {}).get("rougeL", 0)
            t_r1 = t_metrics.get("rouge", {}).get("rouge1", 0)
            t_bl = t_metrics.get("bleu", {}).get("avg_bleu4", 0)
            t_bs = t_metrics.get("bertscore", {}).get("f1", 0)
            t_judge = t_metrics.get("llm_judge", {}).get("avg_overall", 0)
            
            # Compute from raw outputs if missing
            t_candidates = comp.get("raw_outputs", [])
            if (t_rl == 0 or t_bl == 0 or t_bs == 0) and t_candidates and references:
                print(f"  Computing overall metrics for teacher from raw_outputs...")
                try:
                    from rouge_score import rouge_scorer
                    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
                    t_r1_list, t_rl_list = [], []
                    for ref, cand in zip(references, t_candidates):
                        if ref.strip() and cand.strip():
                            scores = scorer.score(ref, cand)
                            t_r1_list.append(scores['rouge1'].fmeasure)
                            t_rl_list.append(scores['rougeL'].fmeasure)
                    if t_rl_list and t_rl == 0:
                        t_rl = sum(t_rl_list) / len(t_rl_list)
                        t_r1 = sum(t_r1_list) / len(t_r1_list)
                except ImportError:
                    pass
                
                try:
                    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
                    smooth = SmoothingFunction().method1
                    t_bl_list = []
                    for ref, cand in zip(references, t_candidates):
                        ref_tokens = ref.lower().split()
                        gen_tokens = cand.lower().split()
                        if len(ref_tokens) >= 4 and len(gen_tokens) >= 4:
                            t_bl_list.append(sentence_bleu(
                                [ref_tokens], gen_tokens,
                                weights=(0.25, 0.25, 0.25, 0.25),
                                smoothing_function=smooth,
                            ))
                    if t_bl_list and t_bl == 0:
                        t_bl = sum(t_bl_list) / len(t_bl_list)
                except ImportError:
                    pass
                
                try:
                    from bert_score import score as bert_score_fn
                    valid = [(r, c) for r, c in zip(references, t_candidates) if r.strip() and c.strip()]
                    if valid and t_bs == 0:
                        refs_c, cands_c = zip(*valid)
                        P, R, F1 = bert_score_fn(list(cands_c), list(refs_c), model_type="roberta-large", verbose=False)
                        t_bs = float(F1.mean())
                except (ImportError, Exception):
                    pass
            
            t_mc = 0.0
            if medcon_data and label in medcon_data:
                t_mc = medcon_data[label].get("teacher", {}).get("medcon_f1", 0.0)
            if t_mc == 0.0:
                t_mc = t_metrics.get("medcon", {}).get("f1", 0)
            t_judge = t_metrics.get("llm_judge", {}).get("avg_overall", 0)
            
            lines.append(
                f"| | Teacher (ref.) | **Overall** | "
                f"{fmt(t_rl)} | {fmt(t_r1)} | {fmt(t_bl)} | {fmt(t_bs)} | {fmt(t_mc)} | {fmt(t_judge, 2)} |"
            )
        
        lines.append("|---|---|---|---|---|---|---|---|---|")
        all_per_section[label] = model_sections
    
    # Write table
    table_path = output_dir / "table_c_rag_ablation.md"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSaved: {table_path}")
    
    return all_per_section


# =============================================================================
# RAG Ablation Bar Plot
# =============================================================================

def plot_rag_ablation_comparison(
    eval_json_paths: List[str],
    labels: List[str],
    medcon_data: Optional[Dict],
    output_dir: Path,
):
    """Bar plot comparing RAG configs across models for key metrics."""
    
    # Collect data
    data = {}  # {label: {rag_name: {metric: value}}}
    
    for eval_path, label in zip(eval_json_paths, labels):
        with open(eval_path) as f:
            eval_data = json.load(f)
        
        rag_backends = eval_data.get("rag_backends", {})
        available_keys = list(rag_backends.keys())
        if any(k in available_keys for k in RAG_CONFIGS_NEW):
            rag_keys = RAG_CONFIGS_NEW
        else:
            rag_keys = RAG_CONFIGS_OLD
        
        references = eval_data.get("references", [])
        data[label] = {}
        for rag_name in rag_keys:
            rag_data = rag_backends.get(rag_name, {})
            if not rag_data or rag_data.get("error"):
                continue
            metrics = rag_data.get("metrics", {})
            candidates = rag_data.get("raw_outputs", [])
            
            rl = metrics.get("rouge", {}).get("rougeL", 0)
            bl = metrics.get("bleu", {}).get("avg_bleu4", 0)
            
            # Compute from raw_outputs if missing
            if (rl == 0 or bl == 0) and candidates and references:
                try:
                    from rouge_score import rouge_scorer
                    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
                    rl_list = []
                    for ref, cand in zip(references, candidates):
                        if ref.strip() and cand.strip():
                            scores = scorer.score(ref, cand)
                            rl_list.append(scores['rougeL'].fmeasure)
                    if rl_list and rl == 0:
                        rl = sum(rl_list) / len(rl_list)
                except ImportError:
                    pass
                
                try:
                    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
                    smooth = SmoothingFunction().method1
                    bl_list = []
                    for ref, cand in zip(references, candidates):
                        rt = ref.lower().split()
                        gt = cand.lower().split()
                        if len(rt) >= 4 and len(gt) >= 4:
                            bl_list.append(sentence_bleu([rt], gt, weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth))
                    if bl_list and bl == 0:
                        bl = sum(bl_list) / len(bl_list)
                except ImportError:
                    pass
            
            mc = 0.0
            if medcon_data and label in medcon_data:
                mc = medcon_data[label].get(f"rag_{rag_name}", {}).get("medcon_f1", 0.0)
            if mc == 0.0:
                mc = metrics.get("medcon", {}).get("f1", 0)
            
            data[label][rag_name] = {
                "ROUGE-L": rl,
                "BLEU-4": bl,
                "MEDCON-F1": mc,
                "Judge Avg": metrics.get("llm_judge", {}).get("avg_overall", 0) / 5.0,
            }
    
    if not data:
        print("No RAG ablation data to plot")
        return
    
    # Determine RAG configs present
    all_rag_names = set()
    for label_data in data.values():
        all_rag_names.update(label_data.keys())
    rag_names = [r for r in RAG_CONFIGS_NEW + RAG_CONFIGS_OLD if r in all_rag_names]
    
    metric_names = ["ROUGE-L", "BLEU-4", "MEDCON-F1", "Judge Avg"]
    
    fig, axes = plt.subplots(1, len(metric_names), figsize=(5 * len(metric_names), 6), sharey=False)
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(labels), 3)))
    
    for ax_idx, mn in enumerate(metric_names):
        ax = axes[ax_idx]
        x = np.arange(len(rag_names))
        w = 0.8 / max(len(labels), 1)
        
        for i, label in enumerate(labels):
            vals = [data.get(label, {}).get(rn, {}).get(mn, 0) for rn in rag_names]
            bars = ax.bar(x + i * w, vals, w, label=label, color=colors[i])
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f"{val:.3f}", ha="center", va="bottom", fontsize=6, rotation=45)
        
        ax.set_title(mn, fontsize=12, fontweight="bold")
        ax.set_xlabel("RAG Config")
        ax.set_xticks(x + w * (len(labels) - 1) / 2)
        ax.set_xticklabels([RAG_DISPLAY.get(r, r) for r in rag_names], fontsize=7, rotation=20, ha="right")
        if ax_idx == 0:
            ax.set_ylabel("Score")
    
    axes[0].legend(fontsize=8)
    fig.suptitle("RAG Ablation: Config Comparison Across Models", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "rag_ablation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'rag_ablation_comparison.png'}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAG Ablation Analysis — Standalone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python plot_rag_ablation.py \\
        --eval-jsons phi_rag.json llama3b_rag.json llama1b_rag.json \\
        --labels "Phi-3.5 (3.8B)" "Llama-3.2 (3B)" "Llama-3.2 (1B)" \\
        --medcon-json ./medcon_rag_results.json \\
        --output-dir ./rag_ablation_tables
        """,
    )
    parser.add_argument("--eval-jsons", nargs="+", required=True,
                        help="Evaluation JSONs (one per model, from RAG ablation runs)")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="Model labels (same order as --eval-jsons)")
    parser.add_argument("--medcon-json", default=None,
                        help="Path to MEDCON results JSON (actual QuickUMLS values)")
    parser.add_argument("--output-dir", default="./rag_ablation_results")
    
    args = parser.parse_args()
    
    if len(args.eval_jsons) != len(args.labels):
        print("ERROR: --eval-jsons count must match --labels count")
        exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load MEDCON data
    medcon_data = None
    if args.medcon_json:
        with open(args.medcon_json) as f:
            medcon_data = json.load(f)
        print(f"Loaded MEDCON data from {args.medcon_json}")
    
    print(f"Models: {args.labels}")
    print(f"Output: {output_dir}")
    
    # Generate Table C
    print("\n" + "=" * 60)
    print("GENERATING TABLE C: RAG ABLATION")
    print("=" * 60)
    per_section = generate_table_c(
        args.eval_jsons, args.labels, medcon_data, output_dir,
    )
    
    # Generate comparison plot
    print("\n" + "=" * 60)
    print("GENERATING RAG ABLATION PLOTS")
    print("=" * 60)
    plot_rag_ablation_comparison(
        args.eval_jsons, args.labels, medcon_data, output_dir,
    )
    
    # Save per-section data as JSON for reference
    json_path = output_dir / "rag_ablation_per_section.json"
    # Convert numpy to float for JSON serialization
    serializable = {}
    for label, configs in per_section.items():
        serializable[label] = {}
        for cfg, sections in configs.items():
            serializable[label][cfg] = {
                k: {mk: float(mv) for mk, mv in v.items()}
                for k, v in sections.items()
            }
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved: {json_path}")
    
    print(f"\nAll RAG ablation outputs saved to {output_dir}/")