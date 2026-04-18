#!/usr/bin/env python3
"""
MEDCON (Medical Concept) Metrics using QuickUMLS

Standalone script for computing MEDCON precision/recall/F1 using proper
UMLS concept matching via QuickUMLS, rather than regex fallback.

Designed to run in WSL Ubuntu with QuickUMLS installed, reading evaluation
JSONs from the Windows filesystem via /mnt/d/.

MEDCON measures the overlap of medical concepts (identified by UMLS CUIs)
between generated and reference clinical summaries.

Usage (from WSL):
    python compute_medcon.py \
        --eval-json /mnt/d/ambient-scribe/final_eval_phi/evaluation_*.json \
        --eval-json /mnt/d/ambient-scribe/final_eval_llama3b/evaluation_*.json \
        --eval-json /mnt/d/ambient-scribe/final_eval_llama1b/evaluation_*.json \
        --labels "Phi-3.5 (3.8B)" "Llama-3.2 (3B)" "Llama-3.2 (1B)" \
        --quickumls-path /mnt/d/quickumls_data \
        --output-dir /mnt/d/ambient-scribe/medcon_results

Author: Alireza Rashidi
MSc Project: Trustworthy SLMs for Ambient Clinical Scribing
"""

import json
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# QuickUMLS-based Concept Extraction
# =============================================================================

class UMLSConceptExtractor:
    """Extract UMLS concepts from clinical text using QuickUMLS."""
    
    def __init__(
        self,
        quickumls_path: str,
        threshold: float = 1.0,
        similarity_name: str = "jaccard",
        window: int = 5,
        accepted_semtypes: Optional[Set[str]] = None,
    ):
        """
        Args:
            quickumls_path: Path to QuickUMLS installation data
            threshold: Similarity threshold (0-1). 1.0 = exact match (ACI-Bench default). Lower = more matches but noisier
            similarity_name: Similarity metric ("jaccard", "dice", "cosine", "overlap")
            window: Maximum number of tokens in a concept mention
            accepted_semtypes: Set of UMLS semantic types to accept.
                If None, uses clinically relevant types for clinical scribing.
        """
        from quickumls import QuickUMLS
        
        # gathered from UMLS Semantic Types and relevance to clinical documentation: https://uts.nlm.nih.gov/uts/umls/semantic-network/
        
        if accepted_semtypes is None:
            accepted_semtypes = {
                # --- DISO: Disorders (all) ---
                "T047",  # Disease or Syndrome
                "T184",  # Sign or Symptom
                "T033",  # Finding
                "T037",  # Injury or Poisoning
                "T019",  # Congenital Abnormality
                "T046",  # Pathologic Function
                "T191",  # Neoplastic Process
                "T048",  # Mental or Behavioral Dysfunction
                "T190",  # Anatomical Abnormality
                "T020",  # Acquired Abnormality
                "T049",  # Cell or Molecular Dysfunction
                "T050",  # Experimental Model of Disease
                # --- CHEM: Chemicals & Drugs (all) ---
                "T121",  # Pharmacologic Substance
                "T200",  # Clinical Drug
                "T116",  # Amino Acid, Peptide, or Protein (biologics, biomarkers)
                "T195",  # Antibiotic
                "T123",  # Biologically Active Substance
                "T122",  # Biomedical or Dental Material
                "T103",  # Chemical
                "T120",  # Chemical Viewed Functionally
                "T104",  # Chemical Viewed Structurally
                "T196",  # Element, Ion, or Isotope
                "T126",  # Enzyme
                "T131",  # Hazardous or Poisonous Substance
                "T125",  # Hormone
                "T129",  # Immunologic Factor
                "T130",  # Indicator, Reagent, or Diagnostic Aid
                "T197",  # Inorganic Chemical
                "T114",  # Nucleic Acid, Nucleoside, or Nucleotide
                "T109",  # Organic Chemical
                "T192",  # Receptor
                "T127",  # Vitamin
                # --- ANAT: Anatomy (all) ---
                "T023",  # Body Part, Organ, or Organ Component
                "T029",  # Body Location or Region
                "T017",  # Anatomical Structure
                "T030",  # Body Space or Junction
                "T031",  # Body Substance
                "T022",  # Body System
                "T025",  # Cell
                "T026",  # Cell Component
                "T018",  # Embryonic Structure
                "T021",  # Fully Formed Anatomical Structure
                "T024",  # Tissue
                # --- PHYS: Physiology (all) ---
                "T201",  # Clinical Attribute
                "T039",  # Physiologic Function
                "T043",  # Cell Function
                "T045",  # Genetic Function
                "T041",  # Mental Process
                "T044",  # Molecular Function
                "T032",  # Organism Attribute
                "T040",  # Organism Function
                "T042",  # Organ or Tissue Function
                # --- PHEN: Phenomena (all) ---
                "T038",  # Biologic Function
                "T034",  # Laboratory or Test Result
                "T067",  # Phenomenon or Process
                "T068",  # Human-caused Phenomenon or Process
                "T069",  # Environmental Effect of Humans
                "T070",  # Natural Phenomenon or Process
                # --- DEVI: Devices (all) ---
                "T074",  # Medical Device
                "T203",  # Drug Delivery Device
                "T075",  # Research Device
                # --- GENE: Genes & Molecular Sequences (all) ---
                "T028",  # Gene or Genome
                "T085",  # Molecular Sequence
                "T086",  # Nucleotide Sequence
                "T087",  # Amino Acid Sequence
                "T088",  # Carbohydrate Sequence
                # --- Additional (not in ACI-Bench exclusion list; kept intentionally) ---
                "T060",  # Diagnostic Procedure  [PROC group — kept for clinical scribing]
                "T061",  # Therapeutic or Preventive Procedure  [PROC]
                "T058",  # Health Care Activity  [PROC]
                "T059",  # Laboratory Procedure  [PROC]
                "T170",  # Intellectual Product (clinical scales: GCS, PHQ-9)  [CONC]
                "T055",  # Individual Behavior (smoking, alcohol)  [ACTI]
            }
        
        self.accepted_semtypes = accepted_semtypes
        
        logger.info(f"Loading QuickUMLS from {quickumls_path}...")
        start = time.time()
        self.matcher = QuickUMLS(
            quickumls_path,
            threshold=threshold,
            similarity_name=similarity_name,
            window=window,
        )
        logger.info(f"QuickUMLS loaded in {time.time() - start:.1f}s")
    
    def extract_cuis(self, text: str) -> Set[str]:
        """
        Extract unique UMLS CUIs from text.
        
        Returns set of CUI strings (e.g., {"C0011849", "C0020538"})
        """
        if not text or not text.strip():
            return set()
        
        cuis = set()
        try:
            matches = self.matcher.match(text, ignore_syntax=True)
            for match_group in matches:
                for candidate in match_group:
                    # Filter by semantic type
                    sem_types = candidate.get("semtypes", set())
                    if sem_types & self.accepted_semtypes:
                        cuis.add(candidate["cui"])
        except Exception as e:
            logger.warning(f"QuickUMLS matching failed: {e}")
        
        return cuis
    
    def extract_concepts_detailed(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract concepts with full details (for debugging/analysis).
        
        Returns list of dicts with cui, term, semtypes, similarity, span.
        """
        if not text or not text.strip():
            return []
        
        concepts = []
        try:
            matches = self.matcher.match(text, ignore_syntax=True)
            for match_group in matches:
                for candidate in match_group:
                    sem_types = candidate.get("semtypes", set())
                    if sem_types & self.accepted_semtypes:
                        concepts.append({
                            "cui": candidate["cui"],
                            "term": candidate["term"],
                            "semtypes": list(sem_types),
                            "similarity": candidate["similarity"],
                            "start": candidate.get("start", -1),
                            "end": candidate.get("end", -1),
                        })
        except Exception as e:
            logger.warning(f"QuickUMLS matching failed: {e}")
        
        return concepts


# =============================================================================
# MEDCON Computation
# =============================================================================

def compute_medcon(
    reference_cuis: Set[str],
    generated_cuis: Set[str],
) -> Dict[str, float]:
    """
    Compute MEDCON (Medical Concept) precision, recall, F1.
    
    MEDCON-P = |CUI_gen ∩ CUI_ref| / |CUI_gen|  (of generated concepts, how many are in reference)
    MEDCON-R = |CUI_gen ∩ CUI_ref| / |CUI_ref|  (of reference concepts, how many are generated)
    MEDCON-F1 = harmonic mean of P and R
    """
    if not generated_cuis and not reference_cuis:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0,
                "n_generated": 0, "n_reference": 0, "n_overlap": 0}
    
    if not generated_cuis:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "n_generated": 0, "n_reference": len(reference_cuis), "n_overlap": 0}
    
    if not reference_cuis:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "n_generated": len(generated_cuis), "n_reference": 0, "n_overlap": 0}
    
    overlap = reference_cuis & generated_cuis
    
    precision = len(overlap) / len(generated_cuis)
    recall = len(overlap) / len(reference_cuis)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_generated": len(generated_cuis),
        "n_reference": len(reference_cuis),
        "n_overlap": len(overlap),
    }


# =============================================================================
# Main Pipeline
# =============================================================================

def run_medcon_analysis(
    eval_json_paths: List[str],
    labels: List[str],
    quickumls_path: str,
    output_dir: str,
    configs_to_score: List[str] = None,
    threshold: float = 1.0,
):
    """Run MEDCON analysis on evaluation JSONs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Default configs to score
    if configs_to_score is None:
        configs_to_score = ["ft_only", "ft_rag", "baseline", "rag_only", "teacher"]
    
    # Initialize extractor
    extractor = UMLSConceptExtractor(quickumls_path, threshold=threshold)
    
    all_results = {}
    
    for eval_path, label in zip(eval_json_paths, labels):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {label}")
        logger.info(f"  File: {eval_path}")
        logger.info(f"{'='*60}")
        
        with open(eval_path) as f:
            eval_data = json.load(f)
        
        references = eval_data.get("references", [])
        if not references:
            logger.warning(f"  No references found, skipping")
            continue
        
        # Extract CUIs from all references (done once)
        logger.info(f"  Extracting CUIs from {len(references)} references...")
        ref_cuis_list = []
        for i, ref in enumerate(references):
            cuis = extractor.extract_cuis(ref)
            ref_cuis_list.append(cuis)
            if (i + 1) % 10 == 0:
                logger.info(f"    {i+1}/{len(references)} references processed")
        
        avg_ref_concepts = sum(len(c) for c in ref_cuis_list) / len(ref_cuis_list)
        logger.info(f"  Avg concepts per reference: {avg_ref_concepts:.1f}")
        
        model_results = {}
        
        # Score each config
        comparative = eval_data.get("comparative", {})
        for config_name in configs_to_score:
            config_data = comparative.get(config_name, {})
            if config_data.get("error") or not config_data:
                continue
            
            raw_outputs = config_data.get("raw_outputs", [])
            if not raw_outputs:
                continue

            if len(raw_outputs) != len(ref_cuis_list):
                logger.warning(
                    f"  [{config_name}] raw_outputs length ({len(raw_outputs)}) != "
                    f"references length ({len(ref_cuis_list)}). "
                    f"Only {min(len(raw_outputs), len(ref_cuis_list))} samples will be scored."
                )
            
            logger.info(f"\n  Config: {config_name} ({len(raw_outputs)} samples)")
            
            # Extract CUIs from generated outputs
            sample_scores = []
            for i, (gen, ref_cuis) in enumerate(zip(raw_outputs, ref_cuis_list)):
                gen_cuis = extractor.extract_cuis(gen)
                score = compute_medcon(ref_cuis, gen_cuis)
                sample_scores.append(score)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"    {i+1}/{len(raw_outputs)} samples scored")
            
            # Aggregate
            if sample_scores:
                avg_p = sum(s["precision"] for s in sample_scores) / len(sample_scores)
                avg_r = sum(s["recall"] for s in sample_scores) / len(sample_scores)
                avg_f1 = sum(s["f1"] for s in sample_scores) / len(sample_scores)
                avg_gen = sum(s["n_generated"] for s in sample_scores) / len(sample_scores)
                avg_ref = sum(s["n_reference"] for s in sample_scores) / len(sample_scores)
                avg_overlap = sum(s["n_overlap"] for s in sample_scores) / len(sample_scores)
                
                model_results[config_name] = {
                    "medcon_precision": avg_p,
                    "medcon_recall": avg_r,
                    "medcon_f1": avg_f1,
                    "avg_generated_concepts": avg_gen,
                    "avg_reference_concepts": avg_ref,
                    "avg_overlap_concepts": avg_overlap,
                    "per_sample": sample_scores,
                    "num_samples": len(sample_scores),
                }
                
                logger.info(f"    MEDCON P={avg_p:.4f} R={avg_r:.4f} F1={avg_f1:.4f}")
                logger.info(f"    Avg concepts: gen={avg_gen:.1f} ref={avg_ref:.1f} overlap={avg_overlap:.1f}")
        
        # Also score RAG ablation configs if present
        rag_backends = eval_data.get("rag_backends", {})
        for rag_name, rag_data in rag_backends.items():
            if rag_data.get("error") or not rag_data:
                continue
            raw_outputs = rag_data.get("raw_outputs", [])
            if not raw_outputs:
                continue

            if len(raw_outputs) != len(ref_cuis_list):
                logger.warning(
                    f"  [rag_{rag_name}] raw_outputs length ({len(raw_outputs)}) != "
                    f"references length ({len(ref_cuis_list)}). "
                    f"Only {min(len(raw_outputs), len(ref_cuis_list))} samples will be scored."
                )
            
            logger.info(f"\n  RAG config: {rag_name} ({len(raw_outputs)} samples)")
            
            sample_scores = []
            for i, (gen, ref_cuis) in enumerate(zip(raw_outputs, ref_cuis_list)):
                gen_cuis = extractor.extract_cuis(gen)
                score = compute_medcon(ref_cuis, gen_cuis)
                sample_scores.append(score)
            
            if sample_scores:
                avg_p = sum(s["precision"] for s in sample_scores) / len(sample_scores)
                avg_r = sum(s["recall"] for s in sample_scores) / len(sample_scores)
                avg_f1 = sum(s["f1"] for s in sample_scores) / len(sample_scores)
                avg_gen = sum(s["n_generated"] for s in sample_scores) / len(sample_scores)
                avg_ref = sum(s["n_reference"] for s in sample_scores) / len(sample_scores)
                avg_overlap = sum(s["n_overlap"] for s in sample_scores) / len(sample_scores)
                
                model_results[f"rag_{rag_name}"] = {
                    "medcon_precision": avg_p,
                    "medcon_recall": avg_r,
                    "medcon_f1": avg_f1,
                    "avg_generated_concepts": avg_gen,
                    "avg_reference_concepts": avg_ref,
                    "avg_overlap_concepts": avg_overlap,
                    "num_samples": len(sample_scores),
                    "per_sample": sample_scores,
                }
                logger.info(f"    MEDCON P={avg_p:.4f} R={avg_r:.4f} F1={avg_f1:.4f}")
                logger.info(f"    Avg concepts: gen={avg_gen:.1f} ref={avg_ref:.1f} overlap={avg_overlap:.1f}")
        
        all_results[label] = model_results
    
    # Save results
    # Remove per_sample for the summary (too large)
    summary = {}
    for label, configs in all_results.items():
        summary[label] = {}
        for cfg, data in configs.items():
            summary[label][cfg] = {k: v for k, v in data.items() if k != "per_sample"}
    
    summary_path = output_path / "medcon_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSummary saved to {summary_path}")
    
    # Save full results (with per-sample)
    full_path = output_path / "medcon_results_full.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Full results saved to {full_path}")
    
    # Generate markdown report
    _generate_report(summary, labels, output_path)
    
    return all_results


def _generate_report(summary: Dict, labels: List[str], output_dir: Path):
    """Generate markdown comparison report."""
    lines = [
        "# MEDCON (Medical Concept Overlap) Results",
        "",
        "Computed using QuickUMLS with UMLS 2025AB.",
        "MEDCON measures overlap of UMLS Concept Unique Identifiers (CUIs)",
        "between generated and reference clinical summaries.",
        "",
    ]
    
    for label in labels:
        configs = summary.get(label, {})
        if not configs:
            continue
        
        lines.append(f"## {label}")
        lines.append("")
        lines.append("| Config | Precision | Recall | F1 | Avg Gen CUIs | Avg Ref CUIs | N |")
        lines.append("|---|---|---|---|---|---|---|")
        
        for cfg_name, data in configs.items():
            lines.append(
                f"| {cfg_name} | {data['medcon_precision']:.4f} | "
                f"{data['medcon_recall']:.4f} | {data['medcon_f1']:.4f} | "
                f"{data.get('avg_generated_concepts', 0):.1f} | "
                f"{data.get('avg_reference_concepts', 0):.1f} | "
                f"{data['num_samples']} |"
            )
        lines.append("")
    
    report_path = output_dir / "medcon_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Report saved to {report_path}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    
    parser = argparse.ArgumentParser(
        description="Compute MEDCON metrics using QuickUMLS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example (from WSL):
    python compute_medcon.py \\
        --eval-json /mnt/d/ambient-scribe/final_eval_phi/evaluation_*.json \\
        --eval-json /mnt/d/ambient-scribe/final_eval_llama3b/evaluation_*.json \\
        --labels "Phi-3.5 (3.8B)" "Llama-3.2 (3B)" \\
        --quickumls-path /mnt/d/quickumls_data \\
        --output-dir /mnt/d/ambient-scribe/medcon_results
        """,
    )
    
    parser.add_argument("--eval-json", action="append", required=True,
                        help="Path to evaluation JSON (repeat for multiple)")
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--quickumls-path", required=True,
                        help="Path to QuickUMLS installation data")
    parser.add_argument("--output-dir", default="./medcon_results")
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="QuickUMLS similarity threshold (default 1.0, exact match, matching ACI-Bench)")
    parser.add_argument("--configs", nargs="+",
                        default=["ft_only", "ft_rag", "baseline", "rag_only", "teacher"],
                        help="Which configs to score")
    
    args = parser.parse_args()
    
    if len(args.eval_json) != len(args.labels):
        parser.error(f"--eval-json count ({len(args.eval_json)}) != --labels ({len(args.labels)})")
    
    for ep in args.eval_json:
        if not Path(ep).exists():
            parser.error(f"File not found: {ep}")
    
    results = run_medcon_analysis(
        eval_json_paths=args.eval_json,
        labels=args.labels,
        quickumls_path=args.quickumls_path,
        output_dir=args.output_dir,
        configs_to_score=args.configs,
        threshold=args.threshold,
    )
    
    print(f"\nMEDCON analysis complete! Results in {args.output_dir}")