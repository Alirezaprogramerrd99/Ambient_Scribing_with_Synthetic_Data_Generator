"""
Post-Evaluation Metrics: ROUGE, BERTScore, BLEU, BLEURT, MEDCON

Standalone script that reads evaluation JSON files and computes additional
metrics WITHOUT any Unsloth/model dependency. Uses the HuggingFace `evaluate`
library as primary backend (same as ACI-Bench, MediGen, and other clinical NLP
papers) with fallbacks to standalone packages.

Metrics computed:
    1. ROUGE (1/2/L)     - N-gram overlap (evaluate → rouge-score fallback)
    2. BERTScore (P/R/F1) - Semantic similarity (evaluate → bert-score fallback)
    3. BLEU (1-4)         - Precision-oriented n-grams (evaluate → nltk fallback)
    4. BLEURT             - Learned evaluation metric (evaluate only)
    5. MEDCON (F1)        - Medical concept overlap (QuickUMLS → regex fallback)
    6. Per-config and per-sample breakdowns
    7. Literature comparison table

Dependencies (install in order of priority):
    pip install evaluate                    # HuggingFace evaluate (primary)
    pip install rouge-score bert-score      # Fallbacks
    pip install nltk                        # BLEU fallback
    pip install bleurt@https://github.com/google-research/bleurt/archive/master.zip  # BLEURT
    pip install quickumls                   # MEDCON (optional, requires UMLS data)

Usage:
    # Basic: compute metrics for one eval file
    python post_eval_metrics.py --eval-json ./evaluation_results_phi/evaluation_*.json

    # Compare multiple models
    python post_eval_metrics.py \
        --eval-json ./evaluation_results_phi/eval.json \
        --eval-json ./evaluation_results_qwen/eval.json \
        --eval-json ./evaluation_results_llama3b/eval.json \
        --labels "Phi-3.5-mini" "Qwen2.5-3B" "Llama-3.2-3B" \
        --output-dir ./post_eval_results

Author: Alireza Rashidi
MSc Project: Trustworthy SLMs for Ambient Clinical Scribing
"""

import json
import logging
import argparse
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# Metric Implementations
# =============================================================================

def compute_bleu(candidates: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute BLEU scores using HuggingFace `evaluate` library (preferred)
    with nltk fallback.
    
    The `evaluate` library is the standard used by ACI-Bench, MediGen,
    and most clinical NLP papers, ensuring our results are directly comparable.
    """
    # ---- Primary: HuggingFace evaluate ----
    try:
        import evaluate
        
        bleu_metric = evaluate.load("bleu")
        
        valid_refs, valid_cands = [], []
        for r, c in zip(references, candidates):
            if r.strip() and c.strip():
                valid_refs.append(r)
                valid_cands.append(c)
        
        if not valid_cands:
            return {"error": "No valid pairs"}
        
        results = {}
        for n in [1, 2, 3, 4]:
            score = bleu_metric.compute(
                predictions=valid_cands,
                references=valid_refs,
                max_order=n,
            )
            results[f"avg_bleu{n}"] = score["bleu"]
        
        # Also get the full BLEU-4 with all sub-metrics
        full = bleu_metric.compute(predictions=valid_cands, references=valid_refs)
        results["corpus_bleu4"] = full["bleu"]
        results["precisions"] = full["precisions"]
        results["brevity_penalty"] = full["brevity_penalty"]
        results["num_samples"] = len(valid_cands)
        results["method"] = "evaluate"
        
        logger.info(f"  BLEU computed via `evaluate` library ({len(valid_cands)} samples)")
        return results
    
    except ImportError:
        logger.info("  `evaluate` not available for BLEU, falling back to nltk...")
    except Exception as e:
        logger.warning(f"  `evaluate` BLEU failed ({e}), falling back to nltk...")
    
    # ---- Fallback: nltk ----
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu
        from nltk.tokenize import word_tokenize
        import nltk
        
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
        
        smoother = SmoothingFunction().method1
        
        sentence_scores = {"bleu1": [], "bleu2": [], "bleu3": [], "bleu4": []}
        refs_tokenized, cands_tokenized = [], []
        
        for ref, cand in zip(references, candidates):
            if not ref.strip() or not cand.strip():
                continue
            ref_tokens = word_tokenize(ref.lower())
            cand_tokens = word_tokenize(cand.lower())
            refs_tokenized.append([ref_tokens])
            cands_tokenized.append(cand_tokens)
            
            for n, key in [(1, "bleu1"), (2, "bleu2"), (3, "bleu3"), (4, "bleu4")]:
                weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
                score = sentence_bleu([ref_tokens], cand_tokens, weights=weights,
                                      smoothing_function=smoother)
                sentence_scores[key].append(score)
        
        results = {}
        for key, scores in sentence_scores.items():
            results[f"avg_{key}"] = sum(scores) / len(scores) if scores else 0.0
        if cands_tokenized:
            results["corpus_bleu4"] = corpus_bleu(refs_tokenized, cands_tokenized,
                                                   smoothing_function=smoother)
        results["num_samples"] = len(cands_tokenized)
        results["method"] = "nltk_fallback"
        
        logger.info(f"  BLEU computed via nltk fallback ({len(cands_tokenized)} samples)")
        return results
    
    except ImportError:
        logger.warning("  Neither `evaluate` nor `nltk` available for BLEU")
        return {"error": "No BLEU library available. Install: pip install evaluate  OR  pip install nltk"}


def compute_bertscore(
    candidates: List[str], 
    references: List[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
    batch_size: int = 8,
) -> Dict[str, float]:
    """
    Compute BERTScore using HuggingFace `evaluate` library (preferred)
    with bert-score package fallback.
    
    Does NOT require Unsloth — only needs torch + evaluate (or bert-score).
    """
    valid_pairs = [
        (r, c) for r, c in zip(references, candidates)
        if r.strip() and c.strip()
    ]
    if not valid_pairs:
        return {"error": "No valid pairs"}
    
    refs = [p[0] for p in valid_pairs]
    cands = [p[1] for p in valid_pairs]
    
    # ---- Primary: HuggingFace evaluate ----
    try:
        import evaluate
        
        bertscore_metric = evaluate.load("bertscore")
        logger.info(f"  Computing BERTScore via `evaluate` ({model_type}, {len(cands)} samples)...")
        
        results = bertscore_metric.compute(
            predictions=cands,
            references=refs,
            model_type=model_type,
            batch_size=batch_size,
        )
        
        import numpy as np
        p_arr = np.array(results["precision"])
        r_arr = np.array(results["recall"])
        f1_arr = np.array(results["f1"])
        
        return {
            "precision": float(p_arr.mean()),
            "recall": float(r_arr.mean()),
            "f1": float(f1_arr.mean()),
            "precision_std": float(p_arr.std()),
            "recall_std": float(r_arr.std()),
            "f1_std": float(f1_arr.std()),
            "per_sample_f1": [float(f) for f in f1_arr],
            "model_type": model_type,
            "num_samples": len(cands),
            "method": "evaluate",
        }
    
    except ImportError:
        logger.info("  `evaluate` not available for BERTScore, falling back to bert-score...")
    except Exception as e:
        logger.warning(f"  `evaluate` BERTScore failed ({e}), falling back to bert-score...")
    
    # ---- Fallback: bert-score package ----
    try:
        from bert_score import score as bert_score_fn
        
        logger.info(f"  Computing BERTScore via bert-score ({model_type}, {len(cands)} samples)...")
        P, R, F1 = bert_score_fn(
            cands, refs,
            model_type=model_type,
            verbose=False,
            batch_size=batch_size,
        )
        
        return {
            "precision": P.mean().item(),
            "recall": R.mean().item(),
            "f1": F1.mean().item(),
            "precision_std": P.std().item(),
            "recall_std": R.std().item(),
            "f1_std": F1.std().item(),
            "per_sample_f1": [f.item() for f in F1],
            "model_type": model_type,
            "num_samples": len(cands),
            "method": "bert_score_fallback",
        }
    
    except ImportError:
        logger.warning("  Neither `evaluate` nor `bert-score` available")
        return {"error": "No BERTScore library available. Install: pip install evaluate bert-score"}
    except Exception as e:
        logger.error(f"  BERTScore failed: {e}")
        return {"error": str(e)}


def compute_rouge(candidates: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute ROUGE scores using HuggingFace `evaluate` library (preferred)
    with rouge-score package fallback.
    """
    valid_pairs = [(r, c) for r, c in zip(references, candidates) if r.strip() and c.strip()]
    if not valid_pairs:
        return {"error": "No valid pairs"}
    refs = [p[0] for p in valid_pairs]
    cands = [p[1] for p in valid_pairs]
    
    # ---- Primary: HuggingFace evaluate ----
    try:
        import evaluate
        rouge_metric = evaluate.load("rouge")
        results = rouge_metric.compute(predictions=cands, references=refs)
        logger.info(f"  ROUGE computed via `evaluate` ({len(cands)} samples)")
        return {
            "rouge1": results["rouge1"],
            "rouge2": results["rouge2"],
            "rougeL": results["rougeL"],
            "rougeLsum": results.get("rougeLsum", None),
            "num_samples": len(cands),
            "method": "evaluate",
        }
    except ImportError:
        logger.info("  `evaluate` not available for ROUGE, falling back to rouge-score...")
    except Exception as e:
        logger.warning(f"  `evaluate` ROUGE failed ({e}), falling back...")
    
    # ---- Fallback: rouge-score ----
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        r1, r2, rl = [], [], []
        for ref, cand in zip(refs, cands):
            scores = scorer.score(ref, cand)
            r1.append(scores['rouge1'].fmeasure)
            r2.append(scores['rouge2'].fmeasure)
            rl.append(scores['rougeL'].fmeasure)
        logger.info(f"  ROUGE computed via rouge-score fallback ({len(cands)} samples)")
        return {
            "rouge1": sum(r1) / len(r1),
            "rouge2": sum(r2) / len(r2),
            "rougeL": sum(rl) / len(rl),
            "num_samples": len(cands),
            "method": "rouge_score_fallback",
        }
    except ImportError:
        return {"error": "No ROUGE library. Install: pip install evaluate  OR  pip install rouge-score"}


def compute_bleurt(candidates: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute BLEURT scores using HuggingFace `evaluate` library.
    
    BLEURT is a learned evaluation metric trained on human judgments.
    Used by ACI-Bench as part of their composite evaluation score.
    
    Note: First run downloads the BLEURT checkpoint (~2GB).
    """
    valid_pairs = [(r, c) for r, c in zip(references, candidates) if r.strip() and c.strip()]
    if not valid_pairs:
        return {"error": "No valid pairs"}
    refs = [p[0] for p in valid_pairs]
    cands = [p[1] for p in valid_pairs]
    
    try:
        import evaluate
        bleurt_metric = evaluate.load("bleurt", module_type="metric")
        logger.info(f"  Computing BLEURT ({len(cands)} samples)...")
        results = bleurt_metric.compute(predictions=cands, references=refs)
        scores = results["scores"]
        
        import numpy as np
        arr = np.array(scores)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "per_sample": [float(s) for s in scores],
            "num_samples": len(cands),
            "method": "evaluate",
        }
    except ImportError:
        logger.warning("  `evaluate` not available for BLEURT. Install: pip install evaluate bleurt")
        return {"error": "evaluate or bleurt not installed"}
    except Exception as e:
        logger.warning(f"  BLEURT failed: {e}")
        return {"error": str(e)}


def compute_medcon(
    candidates: List[str],
    references: List[str],
    quickumls_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute MEDCON (Medical Concept F1) using QuickUMLS or regex fallback.
    
    MEDCON extracts medical concepts from both candidate and reference,
    then computes precision, recall, and F1 of concept overlap.
    
    If QuickUMLS is not available, falls back to a regex-based extraction
    of common clinical terms (medications, symptoms, procedures).
    """
    try:
        if quickumls_path:
            from quickumls import QuickUMLS
            matcher = QuickUMLS(quickumls_path, threshold=0.7)
            
            def extract_concepts(text):
                matches = matcher.match(text)
                concepts = set()
                for match_group in matches:
                    for match in match_group:
                        concepts.add(match['term'].lower())
                return concepts
        else:
            # Regex-based fallback for clinical concept extraction
            extract_concepts = _regex_clinical_concepts
        
        precisions, recalls, f1s = [], [], []
        
        for ref, cand in zip(references, candidates):
            if not ref.strip() or not cand.strip():
                continue
            
            ref_concepts = extract_concepts(ref)
            cand_concepts = extract_concepts(cand)
            
            if not ref_concepts and not cand_concepts:
                continue
            
            if not ref_concepts:
                precisions.append(0.0)
                recalls.append(0.0)
                f1s.append(0.0)
                continue
            
            overlap = ref_concepts & cand_concepts
            
            p = len(overlap) / len(cand_concepts) if cand_concepts else 0.0
            r = len(overlap) / len(ref_concepts) if ref_concepts else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)
        
        n = len(f1s) if f1s else 1
        return {
            "precision": sum(precisions) / n,
            "recall": sum(recalls) / n,
            "f1": sum(f1s) / n,
            "num_samples": n,
            "method": "quickumls" if quickumls_path else "regex_fallback",
        }
    
    except ImportError:
        logger.info("  QuickUMLS not available, using regex-based clinical concept extraction")
        # Retry with regex fallback
        return compute_medcon(candidates, references, quickumls_path=None)
    except Exception as e:
        logger.error(f"MEDCON failed: {e}")
        return {"error": str(e)}


def _regex_clinical_concepts(text: str) -> set:
    """
    Extract clinical concepts from text using regex patterns.
    
    This is a fallback when QuickUMLS is not installed. It captures:
    - Medications (common drug names and patterns)
    - Symptoms and conditions
    - Procedures and tests
    - Anatomical terms
    - Clinical measurements
    """
    text_lower = text.lower()
    concepts = set()
    
    # Common medications (non-exhaustive but covers frequent ones)
    medications = [
        r'\b(?:paracetamol|ibuprofen|amoxicillin|metformin|amlodipine|'
        r'atorvastatin|omeprazole|sertraline|salbutamol|prednisolone|'
        r'ramipril|bisoprolol|lansoprazole|naproxen|codeine|'
        r'co-codamol|diazepam|fluoxetine|citalopram|gabapentin|'
        r'tramadol|morphine|insulin|levothyroxine|warfarin|'
        r'aspirin|clopidogrel|lisinopril|losartan|doxycycline|'
        r'azithromycin|ciprofloxacin|flucloxacillin|trimethoprim)\b',
    ]
    
    # Symptoms and conditions
    symptoms = [
        r'\b(?:chest pain|shortness of breath|headache|fever|cough|'
        r'nausea|vomiting|diarrhoea|diarrhea|constipation|fatigue|'
        r'dizziness|palpitations|swelling|oedema|edema|rash|'
        r'hypertension|high blood pressure|diabetes|asthma|copd|'
        r'pneumonia|bronchitis|uti|urinary tract infection|'
        r'anxiety|depression|insomnia|back pain|abdominal pain|'
        r'sore throat|ear pain|joint pain|muscle pain|'
        r'breathlessness|tachycardia|bradycardia|hypotension|'
        r'anaemia|anemia|hypothyroidism|hyperthyroidism)\b',
    ]
    
    # Procedures and tests
    procedures = [
        r'\b(?:ecg|electrocardiogram|x-ray|xray|ct scan|mri|'
        r'blood test|urine test|ultrasound|endoscopy|colonoscopy|'
        r'spirometry|peak flow|biopsy|surgery|referral|'
        r'full blood count|fbc|urea|electrolytes|liver function|'
        r'thyroid function|hba1c|cholesterol|creatinine|'
        r'chest x-ray|cxr|blood pressure|bp|bmi)\b',
    ]
    
    # Anatomical terms
    anatomy = [
        r'\b(?:chest|lungs?|heart|abdomen|liver|kidneys?|'
        r'throat|ears?|eyes?|head|neck|spine|back|'
        r'shoulder|knee|hip|ankle|wrist|elbow)\b',
    ]
    
    # Clinical findings
    findings = [
        r'\b(?:crackles|wheeze|wheezing|murmur|tenderness|'
        r'guarding|rebound|distension|erythema|'
        r'clear lungs|normal heart sounds|soft abdomen|'
        r'reduced air entry|bilateral|unilateral)\b',
    ]
    
    all_patterns = medications + symptoms + procedures + anatomy + findings
    
    for pattern in all_patterns:
        matches = re.findall(pattern, text_lower)
        concepts.update(matches)
    
    return concepts


# =============================================================================
# Evaluation JSON Processing
# =============================================================================

def extract_outputs_from_eval_json(
    eval_path: str,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Extract candidate outputs and references from an evaluation JSON.
    
    The evaluator stores per-sample judge data but not raw outputs.
    We need to re-extract from the test data using the same logic.
    
    Returns:
        Dict mapping config names to {"candidates": [...], "references": [...]}
    """
    with open(eval_path) as f:
        results = json.load(f)
    
    return results


def load_test_data_texts(test_data_path: str, max_samples: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """
    Load test data and extract dialogues + references.
    Mirrors the evaluator's load_test_data and extract functions.
    """
    samples = []
    with open(test_data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    
    if max_samples:
        samples = samples[:max_samples]
    
    dialogues = []
    references = []
    
    for sample in samples:
        text = sample.get("text", "")
        
        # Extract dialogue (user message content)
        dialogue = _extract_between(text, "consultation:\n\n", "\n\nProduce a structured")
        if not dialogue:
            dialogue = _extract_between(text, "consultation:\n\n", "\n\n")
        
        # Extract reference (assistant response)
        # Try Phi format
        ref = _extract_after(text, "<|assistant|>\n")
        if ref and "<|end|>" in ref:
            ref = ref.split("<|end|>")[0]
        
        # Try Qwen format
        if not ref:
            ref = _extract_after(text, "<|im_start|>assistant\n")
            if ref and "<|im_end|>" in ref:
                ref = ref.split("<|im_end|>")[0]
        
        # Try Llama format
        if not ref:
            ref = _extract_after(text, "<|start_header_id|>assistant<|end_header_id|>\n\n")
            if ref and "<|eot_id|>" in ref:
                ref = ref.split("<|eot_id|>")[0]
        
        dialogues.append(dialogue.strip() if dialogue else "")
        references.append(ref.strip() if ref else "")
    
    return dialogues, references


def _extract_between(text: str, start: str, end: str) -> Optional[str]:
    """Extract text between two markers."""
    s = text.find(start)
    if s == -1:
        return None
    s += len(start)
    e = text.find(end, s)
    if e == -1:
        return text[s:]
    return text[s:e]


def _extract_after(text: str, marker: str) -> Optional[str]:
    """Extract text after a marker."""
    idx = text.find(marker)
    if idx == -1:
        return None
    return text[idx + len(marker):]


# =============================================================================
# Literature Comparison Table
# =============================================================================

LITERATURE_BASELINES = [
    {
        "paper": "MediGen (Leong et al., 2024)",
        "model": "LLaMA3-8B",
        "params": "8B",
        "task": "Medical report generation",
        "rouge_l": 0.58,
        "bertscore_f1": 0.72,
        "bleu": None,
        "medcon_f1": None,
        "notes": "Fine-tuned, dialogue-to-report",
    },
    {
        "paper": "LLaMA-Clinic (Wang et al., 2024)",
        "model": "LLaMA2-13B",
        "params": "13B",
        "task": "Clinical note generation",
        "rouge_l": None,
        "bertscore_f1": None,
        "bleu": None,
        "medcon_f1": None,
        "notes": "SFT + DistillDirect RL, expert-rated 4.2/5",
    },
    {
        "paper": "Radiology Reports (2024)",
        "model": "LLaMA3-8B",
        "params": "8B",
        "task": "Radiology conclusion generation",
        "rouge_l": 0.4628,
        "bertscore_f1": 0.8054,
        "bleu": None,
        "medcon_f1": None,
        "notes": "LoRA r=16, RTX 3090",
    },
    {
        "paper": "Discharge Summary DoRA (2026)",
        "model": "Qwen2-7B + DoRA",
        "params": "7B",
        "task": "Discharge summary generation",
        "rouge_l": None,
        "bertscore_f1": None,
        "bleu": None,
        "medcon_f1": None,
        "notes": "DoRA > LoRA > QLoRA across models",
    },
    {
        "paper": "MIMIC-IV Benchmark (2025)",
        "model": "Gemma-3-27B",
        "params": "27B",
        "task": "Clinical note summarization",
        "rouge_l": None,
        "bertscore_f1": None,
        "bleu": None,
        "medcon_f1": None,
        "notes": "Best overall extractive summarization",
    },
    {
        "paper": "German Discharge (2025)",
        "model": "LLaMA3 (prompt eng.)",
        "params": "8B",
        "task": "Discharge summary (German)",
        "rouge_l": 0.25,
        "bertscore_f1": 0.64,
        "bleu": None,
        "medcon_f1": None,
        "notes": "Prompt engineering only, no fine-tuning",
    },
    {
        "paper": "ACI-Bench (Yim et al., 2023)",
        "model": "Various baselines",
        "params": "Various",
        "task": "Visit note generation",
        "rouge_l": None,
        "bertscore_f1": None,
        "bleu": None,
        "medcon_f1": None,
        "notes": "Benchmark dataset, uses ROUGE+BERTScore+MEDCON",
    },
]


def generate_literature_comparison(
    our_results: Dict[str, Dict],
    output_path: Path,
):
    """Generate a markdown table comparing our results with literature."""
    lines = [
        "# Literature Comparison Table",
        "",
        "## Our Results vs Published Work",
        "",
        "| Paper / Model | Params | Task | ROUGE-L | BERTScore-F1 | BLEU-4 | MEDCON-F1 | Notes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    
    # Add our results
    for label, metrics in our_results.items():
        rouge_l = metrics.get("rouge_l", None)
        bert_f1 = metrics.get("bertscore", {}).get("f1", None)
        bleu4 = metrics.get("bleu", {}).get("avg_bleu4", None)
        medcon = metrics.get("medcon", {}).get("f1", None)
        
        def f(v): return f"{v:.4f}" if isinstance(v, float) else "—"
        
        lines.append(
            f"| **{label} (Ours)** | **SLM** | **Clinical scribing** | "
            f"**{f(rouge_l)}** | **{f(bert_f1)}** | **{f(bleu4)}** | **{f(medcon)}** | "
            f"**QLoRA + RAG** |"
        )
    
    # Add literature baselines
    for paper in LITERATURE_BASELINES:
        def f(v): return f"{v:.4f}" if isinstance(v, float) else "—"
        
        lines.append(
            f"| {paper['paper']} | {paper['params']} | {paper['task']} | "
            f"{f(paper['rouge_l'])} | {f(paper['bertscore_f1'])} | "
            f"{f(paper['bleu'])} | {f(paper['medcon_f1'])} | {paper['notes']} |"
        )
    
    lines.extend([
        "",
        "**Notes:**",
        "- Our models are 1B-3.8B parameters, significantly smaller than most literature (7B-27B)",
        "- ROUGE-L and BERTScore are not directly comparable across different datasets and tasks",
        "- Our evaluation uses synthetic dialogue-to-summary data; papers vary in data sources",
        "- MEDCON (regex) is an approximation; QuickUMLS provides more accurate medical concept extraction",
    ])
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    logger.info(f"Literature comparison saved to {output_path}")


# =============================================================================
# Main Pipeline
# =============================================================================

def run_post_eval_metrics(
    eval_json_paths: List[str],
    labels: List[str],
    output_dir: str,
    quickumls_path: Optional[str] = None,
    bertscore_model: str = "microsoft/deberta-xlarge-mnli",
    skip_bertscore: bool = False,
    skip_bleurt: bool = False,
    configs_to_evaluate: List[str] = None,
):
    """
    Run all post-evaluation metrics on evaluation JSONs.
    
    Reads raw_outputs and references directly from the evaluation JSON
    (saved by the updated evaluator) and computes ROUGE, BLEU, BERTScore,
    BLEURT, and MEDCON for each configuration of each model.
    
    Args:
        eval_json_paths: Paths to evaluation JSON files (one per model).
        labels: Human-readable labels for each model.
        output_dir: Where to save results.
        quickumls_path: Path to QuickUMLS installation (optional).
        bertscore_model: BERTScore model to use.
        skip_bertscore: Skip BERTScore computation.
        skip_bleurt: Skip BLEURT computation.
        configs_to_evaluate: Which configs to evaluate (default: all 5).
    """
    if configs_to_evaluate is None:
        configs_to_evaluate = ["baseline", "rag_only", "ft_only", "ft_rag", "teacher"]
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_model_results = {}
    
    for eval_path, label in zip(eval_json_paths, labels):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {label}")
        logger.info(f"  File: {eval_path}")
        logger.info(f"{'='*60}")
        
        with open(eval_path) as f:
            eval_data = json.load(f)
        
        # Get references from the evaluation JSON (saved by updated evaluator)
        references = eval_data.get("references", [])
        if not references:
            logger.error(f"  No references found in {eval_path}.")
            logger.error(f"  Re-run the evaluator with the updated version that saves references.")
            all_model_results[label] = {"error": "No references in eval JSON"}
            continue
        
        model_results = {}
        
        for config_name in configs_to_evaluate:
            comp = eval_data.get("comparative", {}).get(config_name, {})
            
            if comp.get("error"):
                logger.warning(f"  {config_name}: Skipped (error in original eval)")
                model_results[config_name] = {"error": comp["error"]}
                continue
            
            # Get raw generated outputs for this config
            candidates = comp.get("raw_outputs", [])
            if not candidates:
                logger.warning(f"  {config_name}: No raw_outputs found. "
                              f"Re-run evaluator with updated version.")
                # Carry over existing metrics only
                model_results[config_name] = {
                    "rouge_l": comp.get("metrics", {}).get("rouge_l"),
                    "llm_judge": comp.get("metrics", {}).get("llm_judge", {}),
                    "warning": "No raw outputs — only original metrics available",
                }
                continue
            
            logger.info(f"\n  --- {config_name} ({len(candidates)} samples) ---")
            
            # Start with existing metrics from the evaluator
            existing_metrics = comp.get("metrics", {})
            config_results = {
                "llm_judge": existing_metrics.get("llm_judge", {}),
                "clinical_structure": existing_metrics.get("clinical_structure", {}),
                "avg_generation_time": existing_metrics.get("avg_generation_time"),
                "valid_outputs": existing_metrics.get("valid_outputs"),
                "empty_outputs": existing_metrics.get("empty_outputs"),
            }
            
            # Filter empty pairs
            valid_refs = []
            valid_cands = []
            for r, c in zip(references, candidates):
                if r.strip() and c.strip():
                    valid_refs.append(r)
                    valid_cands.append(c)
            
            if not valid_cands:
                logger.warning(f"  {config_name}: No valid output pairs")
                model_results[config_name] = config_results
                continue
            
            logger.info(f"  Valid pairs: {len(valid_cands)}/{len(candidates)}")
            
            # ---- Compute ROUGE ----
            logger.info(f"  Computing ROUGE...")
            config_results["rouge"] = compute_rouge(valid_cands, valid_refs)
            config_results["rouge_l"] = config_results["rouge"].get("rougeL")
            
            # ---- Compute BLEU ----
            logger.info(f"  Computing BLEU...")
            config_results["bleu"] = compute_bleu(valid_cands, valid_refs)
            
            # ---- Compute BERTScore ----
            if not skip_bertscore:
                logger.info(f"  Computing BERTScore...")
                config_results["bertscore"] = compute_bertscore(
                    valid_cands, valid_refs, model_type=bertscore_model
                )
            
            # ---- Compute BLEURT ----
            if not skip_bleurt:
                logger.info(f"  Computing BLEURT...")
                config_results["bleurt"] = compute_bleurt(valid_cands, valid_refs)
            
            # ---- Compute MEDCON ----
            logger.info(f"  Computing MEDCON...")
            config_results["medcon"] = compute_medcon(
                valid_cands, valid_refs, quickumls_path=quickumls_path
            )
            
            model_results[config_name] = config_results
            
            # Print summary
            rl = config_results.get("rouge_l")
            bl = config_results.get("bleu", {}).get("avg_bleu4")
            bs = config_results.get("bertscore", {}).get("f1")
            mc = config_results.get("medcon", {}).get("f1")
            logger.info(f"  Results: ROUGE-L={f'{rl:.4f}' if rl else 'N/A'} | "
                       f"BLEU-4={f'{bl:.4f}' if bl else 'N/A'} | "
                       f"BERTScore={f'{bs:.4f}' if bs else 'N/A'} | "
                       f"MEDCON={f'{mc:.4f}' if mc else 'N/A'}")
        
        # ---- Process RAG backend comparison results ----
        rag_backends = eval_data.get("rag_backends", {})
        if rag_backends:
            logger.info(f"\n  --- RAG Backend Comparison ---")
            rag_results = {}
            
            for backend_name, backend_data in rag_backends.items():
                if backend_data.get("error"):
                    logger.warning(f"  {backend_name}: Skipped (error)")
                    rag_results[backend_name] = {"error": backend_data["error"]}
                    continue
                
                candidates = backend_data.get("raw_outputs", [])
                if not candidates:
                    logger.warning(f"  {backend_name}: No raw_outputs found")
                    rag_results[backend_name] = {
                        "llm_judge": backend_data.get("metrics", {}).get("llm_judge", {}),
                        "warning": "No raw outputs — only original metrics available",
                    }
                    continue
                
                logger.info(f"\n  --- RAG: {backend_name} ({len(candidates)} samples) ---")
                
                existing_metrics = backend_data.get("metrics", {})
                backend_results = {
                    "llm_judge": existing_metrics.get("llm_judge", {}),
                    "avg_generation_time": existing_metrics.get("avg_generation_time"),
                    "avg_rag_score": existing_metrics.get("avg_rag_score"),
                }
                
                valid_refs, valid_cands = [], []
                for r, c in zip(references, candidates):
                    if r.strip() and c.strip():
                        valid_refs.append(r)
                        valid_cands.append(c)
                
                if valid_cands:
                    logger.info(f"  Computing ROUGE...")
                    backend_results["rouge"] = compute_rouge(valid_cands, valid_refs)
                    backend_results["rouge_l"] = backend_results["rouge"].get("rougeL")
                    
                    logger.info(f"  Computing BLEU...")
                    backend_results["bleu"] = compute_bleu(valid_cands, valid_refs)
                    
                    if not skip_bertscore:
                        logger.info(f"  Computing BERTScore...")
                        backend_results["bertscore"] = compute_bertscore(
                            valid_cands, valid_refs, model_type=bertscore_model
                        )
                    
                    if not skip_bleurt:
                        logger.info(f"  Computing BLEURT...")
                        backend_results["bleurt"] = compute_bleurt(valid_cands, valid_refs)
                    
                    logger.info(f"  Computing MEDCON...")
                    backend_results["medcon"] = compute_medcon(
                        valid_cands, valid_refs, quickumls_path=quickumls_path
                    )
                
                rag_results[backend_name] = backend_results
            
            model_results["_rag_backends"] = rag_results
        
        all_model_results[label] = model_results
    
    # ---- Save comprehensive results ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_output = {
        "timestamp": datetime.now().isoformat(),
        "labels": labels,
        "eval_files": eval_json_paths,
        "models": all_model_results,
    }
    
    results_path = output_path / f"post_eval_metrics_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results_output, f, indent=2, default=str)
    logger.info(f"\nResults saved to {results_path}")
    
    # ---- Generate comparison report ----
    report_path = output_path / f"metrics_comparison_{timestamp}.md"
    _generate_comparison_report(all_model_results, labels, report_path)
    
    # ---- Generate literature comparison ----
    # Use ft_rag results (best config) for each model
    lit_results = {}
    for label, model_data in all_model_results.items():
        best = model_data.get("ft_rag", model_data.get("ft_only", {}))
        if not best.get("error"):
            lit_results[label] = best
    
    if lit_results:
        lit_path = output_path / f"literature_comparison_{timestamp}.md"
        generate_literature_comparison(lit_results, lit_path)
    
    # ---- Generate plotting script ----
    plot_path = output_path / "plot_post_eval.py"
    _generate_plot_script(all_model_results, labels, plot_path)
    
    return results_output


def _generate_comparison_report(
    all_results: Dict[str, Dict],
    labels: List[str],
    output_path: Path,
):
    """Generate a comprehensive markdown comparison report."""
    lines = [
        "# Post-Evaluation Metrics Comparison Report",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]
    
    configs = ["baseline", "rag_only", "ft_only", "ft_rag", "teacher"]
    
    for config_name in configs:
        lines.append(f"## Configuration: {config_name}")
        lines.append("")
        
        # Build header
        header = "| Metric |"
        sep = "|---|"
        for label in labels:
            header += f" {label} |"
            sep += "---|"
        lines.append(header)
        lines.append(sep)
        
        # Metrics rows
        metric_keys = [
            ("rouge_l", "ROUGE-L"),
            ("avg_generation_time", "Avg Time (s)"),
        ]
        
        judge_keys = [
            ("avg_overall", "Judge Overall"),
            ("avg_clinical_accuracy", "Accuracy"),
            ("avg_completeness", "Completeness"),
            ("avg_hallucination", "Hallucination"),
            ("avg_clinical_safety", "Safety"),
            ("avg_coherence", "Coherence"),
            ("avg_conciseness", "Conciseness"),
        ]
        
        bertscore_keys = [
            ("f1", "BERTScore-F1"),
            ("precision", "BERTScore-P"),
            ("recall", "BERTScore-R"),
        ]
        
        bleu_keys = [
            ("avg_bleu1", "BLEU-1"),
            ("avg_bleu4", "BLEU-4"),
        ]
        
        medcon_keys = [
            ("f1", "MEDCON-F1"),
            ("precision", "MEDCON-P"),
            ("recall", "MEDCON-R"),
        ]
        
        # Standard metrics
        for key, display in metric_keys:
            row = f"| {display} |"
            for label in labels:
                data = all_results.get(label, {}).get(config_name, {})
                val = data.get(key)
                row += f" {val:.4f} |" if isinstance(val, float) else " — |"
            lines.append(row)
        
        # Judge metrics
        for key, display in judge_keys:
            row = f"| {display} |"
            for label in labels:
                data = all_results.get(label, {}).get(config_name, {})
                judge = data.get("llm_judge", {})
                val = judge.get(key)
                row += f" {val:.2f}/5 |" if isinstance(val, float) else " — |"
            lines.append(row)
        
        # BERTScore
        for key, display in bertscore_keys:
            row = f"| {display} |"
            for label in labels:
                data = all_results.get(label, {}).get(config_name, {})
                bert = data.get("bertscore", {})
                val = bert.get(key)
                row += f" {val:.4f} |" if isinstance(val, float) else " — |"
            lines.append(row)
        
        # BLEU
        for key, display in bleu_keys:
            row = f"| {display} |"
            for label in labels:
                data = all_results.get(label, {}).get(config_name, {})
                bleu = data.get("bleu", {})
                val = bleu.get(key)
                row += f" {val:.4f} |" if isinstance(val, float) else " — |"
            lines.append(row)
        
        # MEDCON
        for key, display in medcon_keys:
            row = f"| {display} |"
            for label in labels:
                data = all_results.get(label, {}).get(config_name, {})
                medcon = data.get("medcon", {})
                val = medcon.get(key)
                row += f" {val:.4f} |" if isinstance(val, float) else " — |"
            lines.append(row)
        
        lines.append("")
    
    # ---- RAG Backend Comparison Section ----
    # Check if any model has RAG backend results
    has_rag = any("_rag_backends" in all_results.get(l, {}) for l in labels)
    if has_rag:
        lines.append("## RAG Backend Comparison")
        lines.append("")
        
        # Collect all backend names across models
        all_backends = set()
        for label in labels:
            rag_data = all_results.get(label, {}).get("_rag_backends", {})
            all_backends.update(rag_data.keys())
        all_backends = sorted(all_backends)
        
        for backend_name in all_backends:
            lines.append(f"### Backend: {backend_name}")
            lines.append("")
            
            header = "| Metric |"
            sep = "|---|"
            for label in labels:
                header += f" {label} |"
                sep += "---|"
            lines.append(header)
            lines.append(sep)
            
            # Standard metrics
            for key, display in metric_keys:
                row = f"| {display} |"
                for label in labels:
                    data = all_results.get(label, {}).get("_rag_backends", {}).get(backend_name, {})
                    val = data.get(key)
                    row += f" {val:.4f} |" if isinstance(val, float) else " — |"
                lines.append(row)
            
            # Judge metrics
            for key, display in judge_keys:
                row = f"| {display} |"
                for label in labels:
                    data = all_results.get(label, {}).get("_rag_backends", {}).get(backend_name, {})
                    judge = data.get("llm_judge", {})
                    val = judge.get(key)
                    row += f" {val:.2f}/5 |" if isinstance(val, float) else " — |"
                lines.append(row)
            
            # BERTScore
            for key, display in bertscore_keys:
                row = f"| {display} |"
                for label in labels:
                    data = all_results.get(label, {}).get("_rag_backends", {}).get(backend_name, {})
                    bert = data.get("bertscore", {})
                    val = bert.get(key)
                    row += f" {val:.4f} |" if isinstance(val, float) else " — |"
                lines.append(row)
            
            # BLEU
            for key, display in bleu_keys:
                row = f"| {display} |"
                for label in labels:
                    data = all_results.get(label, {}).get("_rag_backends", {}).get(backend_name, {})
                    bleu = data.get("bleu", {})
                    val = bleu.get(key)
                    row += f" {val:.4f} |" if isinstance(val, float) else " — |"
                lines.append(row)
            
            # MEDCON
            for key, display in medcon_keys:
                row = f"| {display} |"
                for label in labels:
                    data = all_results.get(label, {}).get("_rag_backends", {}).get(backend_name, {})
                    medcon = data.get("medcon", {})
                    val = medcon.get(key)
                    row += f" {val:.4f} |" if isinstance(val, float) else " — |"
                lines.append(row)
            
            lines.append("")
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    logger.info(f"Comparison report saved to {output_path}")


def _generate_plot_script(
    all_results: Dict[str, Dict],
    labels: List[str],
    output_path: Path,
):
    """Generate a matplotlib plotting script for the results."""
    
    plot_data = {}
    for label, model_data in all_results.items():
        plot_data[label] = {"configs": {}, "rag_backends": {}}
        for config_name, metrics in model_data.items():
            if config_name == "_rag_backends":
                # Store RAG backend data separately
                for backend_name, backend_metrics in metrics.items():
                    if isinstance(backend_metrics, dict) and not backend_metrics.get("error"):
                        plot_data[label]["rag_backends"][backend_name] = {
                            "rouge_l": backend_metrics.get("rouge_l", 0),
                            "judge_overall": backend_metrics.get("llm_judge", {}).get("avg_overall", 0),
                            "bertscore_f1": backend_metrics.get("bertscore", {}).get("f1", 0),
                            "bleu4": backend_metrics.get("bleu", {}).get("avg_bleu4", 0),
                            "medcon_f1": backend_metrics.get("medcon", {}).get("f1", 0),
                        }
            elif isinstance(metrics, dict) and not metrics.get("error"):
                plot_data[label]["configs"][config_name] = {
                    "rouge_l": metrics.get("rouge_l", 0),
                    "judge_overall": metrics.get("llm_judge", {}).get("avg_overall", 0),
                    "judge_halluc": metrics.get("llm_judge", {}).get("avg_hallucination", 0),
                    "judge_safety": metrics.get("llm_judge", {}).get("avg_clinical_safety", 0),
                    "bertscore_f1": metrics.get("bertscore", {}).get("f1", 0),
                    "bleu4": metrics.get("bleu", {}).get("avg_bleu4", 0),
                    "medcon_f1": metrics.get("medcon", {}).get("f1", 0),
                }
    
    script = f'''"""Auto-generated comparison plots."""
import matplotlib.pyplot as plt
import numpy as np
import json

data = {json.dumps(plot_data, indent=2)}
labels = {json.dumps(labels)}
configs = ["baseline", "rag_only", "ft_only", "ft_rag", "teacher"]
config_labels = ["Base", "Base+RAG", "FT", "FT+RAG", "Teacher"]

colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))

# ---- Figure 1: ROUGE-L across configs ----
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(configs))
width = 0.8 / len(labels)

for i, label in enumerate(labels):
    values = [data.get(label, {{}}).get("configs", {{}}).get(c, {{}}).get("rouge_l", 0) for c in configs]
    bars = ax.bar(x + i * width, values, width, label=label, color=colors[i])
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{{val:.3f}}", ha="center", va="bottom", fontsize=7)

ax.set_xlabel("Configuration")
ax.set_ylabel("ROUGE-L")
ax.set_title("ROUGE-L Comparison Across Models and Configurations")
ax.set_xticks(x + width * (len(labels) - 1) / 2)
ax.set_xticklabels(config_labels)
ax.legend()
ax.set_ylim(0, 1.0)
plt.tight_layout()
plt.savefig("comparison_rouge.png", dpi=150)
plt.close()
print("Saved comparison_rouge.png")

# ---- Figure 2: Judge dimensions (ft_rag only) ----
dims = ["judge_overall", "judge_halluc", "judge_safety"]
dim_labels = ["Overall", "Hallucination", "Safety"]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(dims))
width = 0.8 / len(labels)

for i, label in enumerate(labels):
    ft_rag = data.get(label, {{}}).get("configs", {{}}).get("ft_rag", {{}})
    values = [ft_rag.get(d, 0) for d in dims]
    bars = ax.bar(x + i * width, values, width, label=label, color=colors[i])
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f"{{val:.2f}}", ha="center", va="bottom", fontsize=8)

ax.set_xlabel("Dimension")
ax.set_ylabel("Score (1-5)")
ax.set_title("LLM Judge Scores: FT+RAG Configuration")
ax.set_xticks(x + width * (len(labels) - 1) / 2)
ax.set_xticklabels(dim_labels)
ax.legend()
ax.set_ylim(0, 5.5)
plt.tight_layout()
plt.savefig("comparison_judge.png", dpi=150)
plt.close()
print("Saved comparison_judge.png")

# ---- Figure 3: Radar chart (ft_rag) ----
radar_dims = ["judge_overall", "judge_halluc", "judge_safety", "rouge_l", "bertscore_f1", "medcon_f1"]
radar_labels = ["Overall", "Hallucination", "Safety", "ROUGE-L", "BERTScore", "MEDCON"]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
angles = np.linspace(0, 2 * np.pi, len(radar_dims), endpoint=False).tolist()
angles += angles[:1]

for i, label in enumerate(labels):
    ft_rag = data.get(label, {{}}).get("configs", {{}}).get("ft_rag", {{}})
    values = []
    for d in radar_dims:
        v = ft_rag.get(d, 0)
        # Normalise judge scores to 0-1 range (they are 0-5)
        if "judge" in d:
            v = v / 5.0
        values.append(v)
    values += values[:1]
    ax.plot(angles, values, "o-", label=label, color=colors[i], linewidth=2)
    ax.fill(angles, values, alpha=0.1, color=colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels, fontsize=9)
ax.set_ylim(0, 1.0)
ax.set_title("Model Comparison Radar: FT+RAG", y=1.08)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig("comparison_radar.png", dpi=150)
plt.close()
print("Saved comparison_radar.png")

# ---- Figure 4: RAG Backend Comparison (ROUGE-L per backend per model) ----
all_backends = set()
for label in labels:
    all_backends.update(data.get(label, {{}}).get("rag_backends", {{}}).keys())
all_backends = sorted(all_backends)

if all_backends:
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(all_backends))
    width = 0.8 / len(labels)
    
    for i, label in enumerate(labels):
        rag = data.get(label, {{}}).get("rag_backends", {{}})
        values = [rag.get(b, {{}}).get("rouge_l", 0) for b in all_backends]
        bars = ax.bar(x + i * width, values, width, label=label, color=colors[i])
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{{val:.3f}}", ha="center", va="bottom", fontsize=8)
    
    ax.set_xlabel("RAG Backend")
    ax.set_ylabel("ROUGE-L")
    ax.set_title("RAG Backend Comparison Across Models")
    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels(all_backends)
    ax.legend()
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig("comparison_rag_backends.png", dpi=150)
    plt.close()
    print("Saved comparison_rag_backends.png")
else:
    print("No RAG backend data found, skipping RAG backend plot.")

print("\\nAll plots generated!")
'''
    
    with open(output_path, "w") as f:
        f.write(script)
    
    logger.info(f"Plot script saved to {output_path}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    
    parser = argparse.ArgumentParser(
        description="Post-evaluation metrics: BERTScore, MEDCON, BLEU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--eval-json", action="append", required=True,
        help="Path to evaluation JSON (can specify multiple times)",
    )
    parser.add_argument(
        "--labels", nargs="+", required=True,
        help="Labels for each model (same order as --eval-json)",
    )
    parser.add_argument("--output-dir", default="./post_eval_results")
    parser.add_argument("--quickumls-path", default=None,
                        help="Path to QuickUMLS installation (optional)")
    parser.add_argument("--bertscore-model", default="microsoft/deberta-xlarge-mnli")
    parser.add_argument("--skip-bertscore", action="store_true")
    parser.add_argument("--skip-bleurt", action="store_true",
                        help="Skip BLEURT (requires separate install)")
    parser.add_argument(
        "--configs", nargs="+",
        default=["baseline", "rag_only", "ft_only", "ft_rag", "teacher"],
        help="Which evaluation configs to process",
    )
    
    args = parser.parse_args()
    
    if len(args.eval_json) != len(args.labels):
        parser.error("Number of --eval-json files must match number of --labels")
    
    results = run_post_eval_metrics(
        eval_json_paths=args.eval_json,
        labels=args.labels,
        output_dir=args.output_dir,
        quickumls_path=args.quickumls_path,
        bertscore_model=args.bertscore_model,
        skip_bertscore=args.skip_bertscore,
        skip_bleurt=args.skip_bleurt,
        configs_to_evaluate=args.configs,
    )
    
    print(f"\nPost-evaluation metrics complete!")
    print(f"Results: {args.output_dir}")
