"""
Student Model Evaluator with LLM-as-a-Judge

Comprehensive evaluation framework for the fine-tuned student model:
1. Automated metrics (ROUGE, BERTScore, clinical accuracy)
2. LLM-as-a-Judge evaluation (GPT-4o-mini judges clinical quality)
3. Comparative experiments (5-way: baseline, RAG-only, FT-only, FT+RAG, teacher)
4. RAG backend comparison (LlamaIndex vs Manual vs Hybrid)
5. Efficiency profiling (latency, VRAM, throughput)

Author: Alireza Rashidi
MSc Project: Trustworthy SLMs for Ambient Clinical Scribing
"""
import patch_torch  # Must be first — patches torch.int1-int7 for Windows

import os
# CRITICAL: Unsloth/patch_torch sets HF_HUB_ENABLE_HF_TRANSFER=1 at import time,
# which breaks model downloads (including the RAG embedding model) when hf_transfer
# is buggy or incompatible. Force it off immediately after import.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()  # Load OPENAI_API_KEY from .env file

logger = logging.getLogger(__name__)



# RAG Ablation Config Registry  (shared by Experiment 1 and Experiment 2)

# Single source of truth for all RAG ablation override dicts.
# Used by both _run_comparative_experiment (Experiment 1) and
# _run_rag_backend_comparison (Experiment 2) so the definitions
# never drift out of sync.
_RAG_ABLATION_CONFIGS: dict = {
    "dense_only": {
        "use_reranker": False,
        "use_query_expansion": False,
        "use_clinical_filtering": False,
    },
    "dense_rerank": {
        "use_reranker": True,
        "use_query_expansion": False,
        "use_clinical_filtering": False,
    },
    "dense_rerank_qe": {
        "use_reranker": True,
        "use_query_expansion": True,
        "use_clinical_filtering": False,
    },
    "full_medical": {
        "use_reranker": True,
        "use_query_expansion": True,
        "use_clinical_filtering": True,
    },
}


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EvaluationConfig:
    """Configuration for student model evaluation."""
    
    # Test data
    test_data_path: str = "./data/training_data/test.jsonl"
    
    # Models to evaluate
    student_model: str = "clinical-scribe"
    base_model: str = "phi3.5:3.8b-mini-instruct-q4_K_M"  # Unmodified Phi-3.5 in Ollama
    teacher_model: str = "gpt-4o-mini"
    teacher_provider: str = "openai"
    
    # Model paths for native PyTorch inference (used by inference_fixed.py)
    ft_model_path: str = "./checkpoints/phi35_clinical_scribe/hf_merged"
    base_model_hf: str = "unsloth/Phi-3.5-mini-instruct"  # HuggingFace ID for base model
    
    # Generation parameters (for temperature/sampling experiments)
    temperature: float = 0.3
    top_p: float = 0.9
    
    # Logprobs for uncertainty quantification (Kadavath et al., 2022)
    return_logprobs: bool = False
    
    # LLM-as-a-Judge
    judge_model: str = "gpt-4o-mini"
    judge_provider: str = "openai"
    enable_llm_judge: bool = True
    
    # Metrics
    compute_rouge: bool = True
    compute_bertscore: bool = True  # Enabled for dissertation comparison
    compute_clinical_accuracy: bool = True
    
    # RAG ablation configs to compare (Experiment 2: ablation study table)
    # Options: dense_only, dense_rerank, dense_rerank_qe, full_medical
    # Legacy names (llama_index, hybrid) are auto-mapped
    rag_backends: List[str] = field(default_factory=lambda: [
        "dense_only", "dense_rerank", "dense_rerank_qe", "full_medical"
    ])
    
    # RAG config to use for ft_rag and rag_only in Experiment 1 (five-way comparison).
    # This is independent of rag_backends (Experiment 2).
    # Options: dense_only, dense_rerank, dense_rerank_qe, full_medical
    comparative_rag_config: str = "dense_rerank"
    
    # Which experiment configs to run (default: all 5)
    # Options: baseline, rag_only, ft_only, ft_rag, teacher
    configs_to_run: List[str] = field(default_factory=lambda: [
        "baseline", "rag_only", "ft_only", "ft_rag", "teacher"
    ])
    
    # Whether to run RAG backend comparison experiment
    run_rag_comparison: bool = True
    
    # Output
    output_dir: str = "./evaluation_results"
    
    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    
    # Limits
    max_samples: Optional[int] = None  # None = use all test samples


# =============================================================================
# LLM-as-a-Judge
# =============================================================================

# Evaluation rubric for the LLM judge
JUDGE_SYSTEM_PROMPT = """You are an expert clinical evaluator assessing the quality of AI-generated clinical summaries from doctor-patient consultations.

You will be given:
1. The original doctor-patient dialogue
2. A reference clinical summary (ground truth from the teacher model)
3. A candidate clinical summary (generated by the model being evaluated)

Evaluate the candidate summary on these dimensions. Score each from 1 (worst) to 5 (best):

**Clinical Accuracy (1-5):** Does the summary correctly capture diagnoses, symptoms, and clinical findings from the dialogue? Are there factual errors?

**Completeness (1-5):** Are all important clinical elements present? (Chief complaint, HPI, medications, allergies, assessment, plan, safety netting)

**Hallucination (1-5):** Does the summary contain information NOT present in the dialogue? Score 5 if no hallucinations, 1 if significant fabrications.

**Clinical Safety (1-5):** Could this summary lead to patient harm if used in practice? Score 5 if safe, 1 if contains dangerous errors (wrong medications, missed red flags).

**Coherence (1-5):** Is the summary well-structured, readable, and logically organised?

**Conciseness (1-5):** Is the summary appropriately concise without unnecessary verbosity?

Respond ONLY with valid JSON in this exact format (no markdown, no explanation):
{
    "clinical_accuracy": <int 1-5>,
    "completeness": <int 1-5>,
    "hallucination": <int 1-5>,
    "clinical_safety": <int 1-5>,
    "coherence": <int 1-5>,
    "conciseness": <int 1-5>,
    "overall": <float 1-5>,
    "critical_errors": ["list of any dangerous clinical errors found"],
    "strengths": ["list of strengths"],
    "weaknesses": ["list of weaknesses"]
}"""

JUDGE_USER_TEMPLATE = """## Original Dialogue
{dialogue}

## Reference Summary (Ground Truth)
{reference}

## Candidate Summary (Model Output)
{candidate}

Evaluate the candidate summary against the reference. Return ONLY the JSON evaluation."""


class LLMJudge:
    """
    LLM-as-a-Judge evaluator for clinical summary quality.
    
    Uses GPT-4o-mini (or another capable model) to score generated
    summaries against ground-truth references on clinical dimensions.
    
    Example:
        judge = LLMJudge(model="gpt-4o-mini", provider="openai")
        score = judge.evaluate_single(dialogue, reference, candidate)
        print(f"Clinical accuracy: {score['clinical_accuracy']}/5")
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
        api_key: Optional[str] = None,
        temperature: float = 0.1,  # Low temp for consistent scoring
        max_retries: int = 2,
    ):
        self.model = model
        self.provider = provider
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "API key for LLM A Judge is required. Set OPENAI_API_KEY environment " \
                "variable or pass api_key parameter."
            )

        self.temperature = temperature
        self.max_retries = max_retries
        
        
        self._client = None
        self._init_client()
    
    def _init_client(self):
        """Initialise the API client."""
        if self.provider == "openai":
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("Install openai: pip install openai")
        elif self.provider == "anthropic":
            try:
                import anthropic
                self._client = anthropic.Anthropic()
            except ImportError:
                raise ImportError("Install anthropic: pip install anthropic")
        elif self.provider == "ollama":
            import httpx
            self._client = httpx.Client(timeout=120.0)
    
    
    def evaluate_single(
        self,
        dialogue: str,
        reference: str,
        candidate: str,
    ) -> Dict[str, Any]:
        """
        Evaluate a single candidate summary against a reference.
        
        Args:
            dialogue: Original doctor-patient dialogue.
            reference: Ground truth summary (from teacher model).
            candidate: Summary generated by the model being evaluated.
        
        Returns:
            Dictionary with scores and feedback.
        """
        user_message = JUDGE_USER_TEMPLATE.format(
            dialogue=dialogue,
            reference=reference,
            candidate=candidate,
        )
        
        for attempt in range(self.max_retries + 1):
            try:
                response_text = self._call_llm(user_message)
                
                # Parse JSON response
                # Clean up potential markdown wrapping
                cleaned = response_text.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[1]
                    cleaned = cleaned.rsplit("```", 1)[0]
                
                scores = json.loads(cleaned)
                
                # Validate required fields
                required = [
                    "clinical_accuracy", "completeness", "hallucination",
                    "clinical_safety", "coherence", "conciseness"
                ]
                for key in required:
                    if key not in scores:
                        scores[key] = 3  # Default middle score
                    scores[key] = max(1, min(5, int(scores[key])))
                
                # Compute overall if not provided
                if "overall" not in scores:
                    # Weighted average (safety and accuracy matter most)
                    weights = {
                        "clinical_accuracy": 0.25,
                        "completeness": 0.15,
                        "hallucination": 0.20,
                        "clinical_safety": 0.25,
                        "coherence": 0.08,
                        "conciseness": 0.07,
                    }
                    scores["overall"] = sum(
                        scores[k] * w for k, w in weights.items()
                    )
                
                return scores
                
            except (json.JSONDecodeError, KeyError) as e:
                if attempt < self.max_retries:
                    logger.warning(f"Judge parse error (attempt {attempt + 1}): {e}")
                    continue
                else:
                    logger.error(f"Judge failed after {self.max_retries + 1} attempts")
                    return self._default_scores()
            except Exception as e:
                logger.error(f"Judge API error: {e}")
                return self._default_scores()
    
    def evaluate_batch(
        self,
        dialogues: List[str],
        references: List[str],
        candidates: List[str],
    ) -> List[Dict[str, Any]]:
        """Evaluate a batch of summaries."""
        results = []
        for i, (d, r, c) in enumerate(zip(dialogues, references, candidates)):
            logger.info(f"  Judging {i + 1}/{len(dialogues)}...")
            score = self.evaluate_single(d, r, c)
            results.append(score)
        return results
    
    
    
    def _call_llm(self, user_message: str) -> str:
        """Call the LLM judge."""
        if self.provider == "openai":
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=self.temperature,
                max_tokens=1000,
            )
            return response.choices[0].message.content
        
        elif self.provider == "anthropic":
            response = self._client.messages.create(
                model=self.model,
                max_tokens=1000,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
                temperature=self.temperature,
            )
            return response.content[0].text
        
        
        
        elif self.provider == "ollama":
            response = self._client.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    "stream": False,
                    "options": {"temperature": self.temperature},
                },
            )
            return response.json()["message"]["content"]
    
    
    
    def _default_scores(self) -> Dict[str, Any]:
        """Return default scores when judge fails."""
        return {
            "clinical_accuracy": 0,
            "completeness": 0,
            "hallucination": 0,
            "clinical_safety": 0,
            "coherence": 0,
            "conciseness": 0,
            "overall": 0,
            "critical_errors": ["Judge evaluation failed"],
            "strengths": [],
            "weaknesses": ["Could not evaluate"],
            "judge_error": True,
        }


# =============================================================================
# Test Data Loader
# =============================================================================

def load_test_data(path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load test data from JSONL file."""
    samples = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    if max_samples:
        samples = samples[:max_samples]
    
    return samples


def extract_dialogue_from_chatml(chatml_text: str) -> str:
    """
    Extract the dialogue text from a chat-template formatted string.
    
    Supports all template formats:
        - Phi-3.5:  <|user|>\n...<|end|>
        - Qwen2.5:  <|im_start|>user\n...<|im_end|>
        - Llama 3:  <|start_header_id|>user<|end_header_id|>\n\n...<|eot_id|>
    """
    import re
    
    # Try Phi format: <|user|>\n...<|end|>
    user_match = re.search(r"<\|user\|>\n(.*?)(?:<\|end\|>)", chatml_text, re.DOTALL)
    
    # Try Qwen format: <|im_start|>user\n...<|im_end|>
    if not user_match:
        user_match = re.search(r"<\|im_start\|>user\n(.*?)(?:<\|im_end\|>)", chatml_text, re.DOTALL)
    
    # Try Llama format: <|start_header_id|>user<|end_header_id|>\n\n...<|eot_id|>
    if not user_match:
        user_match = re.search(
            r"<\|start_header_id\|>user<\|end_header_id\|>\n\n(.*?)(?:<\|eot_id\|>)",
            chatml_text, re.DOTALL
        )
    
    if not user_match:
        return ""
    
    user_content = user_match.group(1)
    
    # Remove RAG context prefix if present
    if "Summarise the following clinical consultation:" in user_content:
        dialogue_part = user_content.split("Summarise the following clinical consultation:")[-1]
    elif "Summarize the following clinical consultation:" in user_content:
        dialogue_part = user_content.split("Summarize the following clinical consultation:")[-1]
    else:
        dialogue_part = user_content
    
    # Remove the summary instruction suffix
    if "Produce a structured clinical summary" in dialogue_part:
        dialogue_part = dialogue_part.split("Produce a structured clinical summary")[0]
    
    return dialogue_part.strip()


def extract_reference_from_chatml(chatml_text: str) -> str:
    """
    Extract the reference summary (assistant response) from a chat-template string.
    
    Supports all template formats:
        - Phi-3.5:  <|assistant|>\n...<|end|>
        - Qwen2.5:  <|im_start|>assistant\n...<|im_end|>
        - Llama 3:  <|start_header_id|>assistant<|end_header_id|>\n\n...<|eot_id|>
    """
    import re
    
    # Try Phi format
    assistant_match = re.search(
        r"<\|assistant\|>\n(.*?)(?:<\|end\|>|$)", 
        chatml_text, re.DOTALL
    )
    
    # Try Qwen format
    if not assistant_match:
        assistant_match = re.search(
            r"<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|$)",
            chatml_text, re.DOTALL
        )
    
    # Try Llama format
    if not assistant_match:
        assistant_match = re.search(
            r"<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)(?:<\|eot_id\|>|$)",
            chatml_text, re.DOTALL
        )
    
    if assistant_match:
        return assistant_match.group(1).strip()
    return ""


# Main Evaluator

class StudentEvaluator:
    """
    Comprehensive evaluator for the student model.
    
    Runs:
    1. Five-way comparative experiment (baseline, RAG-only, FT-only, FT+RAG, teacher)
    2. RAG backend comparison
    3. Automated metrics (ROUGE, BERTScore)
    4. LLM-as-a-Judge scoring
    5. Efficiency profiling
    
    Example:
        config = EvaluationConfig(test_data_path="./data/training_data/test.jsonl")
        evaluator = StudentEvaluator(config)
        results = evaluator.run_full_evaluation()
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.judge = None
        
        if config.enable_llm_judge:
            self.judge = LLMJudge(
                model=config.judge_model,
                provider=config.judge_provider,
            )
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """
        Run the complete evaluation pipeline.
        
        Returns:
            Comprehensive results dictionary.
        """
        logger.info("=" * 60)
        logger.info("Starting Full Student Model Evaluation")
        logger.info("=" * 60)
        
        # Load test data
        test_data = load_test_data(
            self.config.test_data_path, 
            self.config.max_samples
        )
        logger.info(f"Loaded {len(test_data)} test samples")
        
        # Extract dialogues and references
        dialogues = [extract_dialogue_from_chatml(s["text"]) for s in test_data]
        references = [extract_reference_from_chatml(s["text"]) for s in test_data]
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "num_test_samples": len(test_data),
            "config": {
                "student_model": self.config.student_model,
                "base_model": self.config.base_model,
                "ft_model_path": self.config.ft_model_path,
                "base_model_hf": self.config.base_model_hf,
                "teacher_model": self.config.teacher_model,
                "judge_model": self.config.judge_model,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
            },
            "references": references,
            "dialogues": dialogues,
        }
        
        # ---- Experiment 1: Five-way comparison ----
        logger.info(f"\n--- Experiment 1: Comparative ({', '.join(self.config.configs_to_run)}) ---")
        try:
            results["comparative"] = self._run_comparative_experiment(
                dialogues, references
            )
        except Exception as e:
            logger.error(f"Experiment 1 (comparative) failed: {e}")
            results["comparative"] = {"error": str(e)}
        
        # ---- Experiment 2: RAG backend comparison ----
        if self.config.run_rag_comparison:
            logger.info("\n--- Experiment 2: RAG Backend Comparison ---")
            try:
                results["rag_backends"] = self._run_rag_backend_comparison(
                    dialogues, references
                )
            except Exception as e:
                logger.error(f"Experiment 2 (RAG backends) failed: {e}")
                results["rag_backends"] = {"error": str(e)}
        else:
            logger.info("\n--- Experiment 2: RAG Backend Comparison (SKIPPED) ---")
            results["rag_backends"] = {}
        
        # Save results — ALWAYS, even if experiments partially failed
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"\nResults saved to {results_path}")
        except Exception as e:
            logger.error(f"Failed to save results JSON: {e}")
        
        # Generate report
        try:
            report_path = self._generate_report(results, output_dir)
            results["report_path"] = str(report_path)
            logger.info(f"Report saved to {report_path}")
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
        
        return results
    
    # -------------------------------------------------------------------------
    # Experiment 1: Five-Way Comparison
    # -------------------------------------------------------------------------
    
    def _run_comparative_experiment(
        self,
        dialogues: List[str],
        references: List[str],
    ) -> Dict[str, Dict]:
        """
        Run all 5 configurations and compare.
        
        Configurations:
            1. baseline:    Base model (no fine-tuning, no RAG)
            2. rag_only:    Base model + RAG
            3. ft_only:     Fine-tuned model (no RAG)
            4. ft_rag:      Fine-tuned model + RAG  (expected best)
            5. teacher:     GPT-4o-mini + RAG       (upper bound)
        """
        from .inference_fixed import ClinicalScribeInference, InferenceConfig
        
        all_results = {}
        
        # Derive a short model name for descriptions and logs
        _model_name = Path(self.config.base_model_hf).name  # e.g. "Phi-3.5-mini-instruct"
        _short_name = _model_name.replace("-instruct", "").replace("-Instruct", "")
        
        _run = self.config.configs_to_run  # Shorthand
        
        # ---- Step 1: Load fine-tuned model (needed for ft_only and ft_rag) ----
        scribe = None
        if "ft_only" in _run or "ft_rag" in _run:
            logger.info(f"\nLoading fine-tuned model: {self.config.ft_model_path}")
            ft_model_path = self.config.ft_model_path
            
            try:
                ft_config_no_rag = InferenceConfig(
                    model_path=ft_model_path,
                    use_rag=False,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    return_logprobs=self.config.return_logprobs,
                )
                scribe = ClinicalScribeInference(ft_config_no_rag)
            except Exception as e:
                logger.error(f"Failed to load fine-tuned model: {e}")
                if "ft_only" in _run:
                    all_results["ft_only"] = {"error": str(e)}
                if "ft_rag" in _run:
                    all_results["ft_rag"] = {"error": str(e)}
        
        # ---- Step 2: FT-only (no RAG) ----
        if "ft_only" in _run and scribe:
            logger.info(f"\nEvaluating: Fine-tuned {_short_name} (no RAG)")
            try:
                outputs = scribe.batch_inference(dialogues, use_rag=False)
                candidates = [o.get("raw_output", "") for o in outputs]
                gen_times = [o.get("generation_time", 0) for o in outputs]
                
                metrics = self._compute_metrics(references, candidates)
                metrics["avg_generation_time"] = (
                    sum(gen_times) / len(gen_times) if gen_times else 0
                )
                
                if self.judge:
                    logger.info("  Running LLM judge on ft_only...")
                    judge_scores = self.judge.evaluate_batch(
                        dialogues, references, candidates
                    )
                    metrics["llm_judge"] = self._aggregate_judge_scores(judge_scores)
                    metrics["llm_judge_per_sample"] = judge_scores
                
                all_results["ft_only"] = {
                    "description": f"Fine-tuned {_short_name} (no RAG)",
                    "metrics": metrics,
                    "num_samples": len(candidates),
                    "raw_outputs": candidates,
                    "logprobs_per_sample": [o.get("logprobs") for o in outputs] if self.config.return_logprobs else [],
                }
            except Exception as e:
                logger.error(f"  ft_only failed: {e}")
                all_results["ft_only"] = {"error": str(e)}
        
        # ---- Step 3: FT+RAG (add RAG to the same model) ----
        if "ft_rag" in _run and scribe:
            logger.info(f"\nEvaluating: Fine-tuned {_short_name} + RAG ({self.config.comparative_rag_config})")
            try:
                # Initialise RAG on the existing model instance using the
                # configured RAG config (controlled by --comparative-rag-config).
                scribe.config.use_rag = True
                scribe.config.rag_backend = "llama_index"
                scribe._retriever = None  # Clear any cached retriever
                scribe._init_rag(rag_overrides=_RAG_ABLATION_CONFIGS[self.config.comparative_rag_config])
                
                outputs = scribe.batch_inference(dialogues, use_rag=True)
                candidates = [o.get("raw_output", "") for o in outputs]
                gen_times = [o.get("generation_time", 0) for o in outputs]
                
                metrics = self._compute_metrics(references, candidates)
                metrics["avg_generation_time"] = (
                    sum(gen_times) / len(gen_times) if gen_times else 0
                )
                
                if self.judge:
                    logger.info("  Running LLM judge on ft_rag...")
                    judge_scores = self.judge.evaluate_batch(
                        dialogues, references, candidates
                    )
                    metrics["llm_judge"] = self._aggregate_judge_scores(judge_scores)
                    metrics["llm_judge_per_sample"] = judge_scores
                
                all_results["ft_rag"] = {
                    "description": f"Fine-tuned {_short_name} + RAG",
                    "metrics": metrics,
                    "num_samples": len(candidates),
                    "raw_outputs": candidates,
                    "logprobs_per_sample": [o.get("logprobs") for o in outputs] if self.config.return_logprobs else [],
                }
            except Exception as e:
                logger.error(f"  ft_rag failed: {e}")
                all_results["ft_rag"] = {"error": str(e)}
        
        # ---- Step 4: Free the fine-tuned model ----
        if scribe is not None:
            del scribe
        import torch, gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # ---- Step 5: Baseline (un-fine-tuned, no RAG) ----
        if "baseline" in _run or "rag_only" in _run:
            logger.info(f"\nEvaluating: Base model (no fine-tuning, no RAG)")
            try:
                base_config = InferenceConfig(
                    model_path=self.config.base_model_hf,
                    use_rag=False,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    return_logprobs=self.config.return_logprobs,
                )
                base_scribe = ClinicalScribeInference(base_config)
                
                if "baseline" in _run:
                    outputs = base_scribe.batch_inference(dialogues, use_rag=False)
                    candidates = [o.get("raw_output", "") for o in outputs]
                    gen_times = [o.get("generation_time", 0) for o in outputs]
                    
                    metrics = self._compute_metrics(references, candidates)
                    metrics["avg_generation_time"] = (
                        sum(gen_times) / len(gen_times) if gen_times else 0
                    )
                    
                    if self.judge:
                        logger.info("  Running LLM judge on baseline...")
                        judge_scores = self.judge.evaluate_batch(
                            dialogues, references, candidates
                        )
                        metrics["llm_judge"] = self._aggregate_judge_scores(judge_scores)
                        metrics["llm_judge_per_sample"] = judge_scores
                    
                    all_results["baseline"] = {
                        "description": f"Base {_short_name} (no fine-tuning, no RAG)",
                        "metrics": metrics,
                        "num_samples": len(candidates),
                        "raw_outputs": candidates,
                        "logprobs_per_sample": [o.get("logprobs") for o in outputs] if self.config.return_logprobs else [],
                    }
                
                if "rag_only" in _run:
                    # ---- Step 6: RAG-only (same base model + RAG) ----
                    logger.info(f"\nEvaluating: Base {_short_name} + RAG, no fine-tuning ({self.config.comparative_rag_config})")
                    base_scribe.config.use_rag = True
                    base_scribe.config.rag_backend = "llama_index"
                    base_scribe._retriever = None  # Clear any cached retriever
                    base_scribe._init_rag(rag_overrides=_RAG_ABLATION_CONFIGS[self.config.comparative_rag_config])
                    
                    outputs = base_scribe.batch_inference(dialogues, use_rag=True)
                    candidates = [o.get("raw_output", "") for o in outputs]
                    gen_times = [o.get("generation_time", 0) for o in outputs]
                    
                    metrics = self._compute_metrics(references, candidates)
                    metrics["avg_generation_time"] = (
                        sum(gen_times) / len(gen_times) if gen_times else 0
                    )
                    
                    if self.judge:
                        logger.info("  Running LLM judge on rag_only...")
                        judge_scores = self.judge.evaluate_batch(
                            dialogues, references, candidates
                        )
                        metrics["llm_judge"] = self._aggregate_judge_scores(judge_scores)
                        metrics["llm_judge_per_sample"] = judge_scores
                    
                    all_results["rag_only"] = {
                        "description": f"Base {_short_name} + RAG (no fine-tuning)",
                        "metrics": metrics,
                        "num_samples": len(candidates),
                        "raw_outputs": candidates,
                        "logprobs_per_sample": [o.get("logprobs") for o in outputs] if self.config.return_logprobs else [],
                    }
                
                del base_scribe
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                logger.error(f"  Baseline evaluation failed: {e}")
                if "baseline" in _run:
                    all_results["baseline"] = {"error": str(e)}
                if "rag_only" in _run:
                    all_results["rag_only"] = {"error": str(e)}
        
        # Teacher evaluation (via API, not Ollama)
        if "teacher" in _run:
            logger.info(f"\nEvaluating: Teacher model ({self.config.teacher_model})")
            teacher_results = self._evaluate_teacher(dialogues, references)
            if teacher_results:
                all_results["teacher"] = teacher_results
        
        return all_results
    
    def _evaluate_teacher(
        self,
        dialogues: List[str],
        references: List[str],
    ) -> Optional[Dict]:
        """Evaluate teacher model (GPT-4o-mini via API)."""
        try:
            if self.config.teacher_provider == "openai":
                import openai
                client = openai.OpenAI()
            else:
                logger.warning("Teacher evaluation only supports OpenAI provider currently")
                return None
            
            from .inference_fixed import SYSTEM_PROMPT, SUMMARY_INSTRUCTION
            
            candidates = []
            gen_times = []
            
            for i, dialogue in enumerate(dialogues):
                logger.info(f"  Teacher generating {i + 1}/{len(dialogues)}...")
                start = time.time()
                
                prompt = (
                    f"Summarise the following clinical consultation:\n\n"
                    f"{dialogue}\n\n{SUMMARY_INSTRUCTION}"
                )
                
                try:
                    response = client.chat.completions.create(
                        model=self.config.teacher_model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.3,
                        max_tokens=2048,
                    )
                    output = response.choices[0].message.content
                    candidates.append(output)
                    gen_times.append(time.time() - start)
                except Exception as e:
                    logger.warning(f"  Teacher failed on sample {i}: {e}")
                    candidates.append("")
                    gen_times.append(0)
            
            metrics = self._compute_metrics(references, candidates)
            metrics["avg_generation_time"] = (
                sum(gen_times) / len(gen_times) if gen_times else 0
            )
            
            if self.judge:
                logger.info("  Running LLM judge on teacher...")
                judge_scores = self.judge.evaluate_batch(
                    dialogues, references, candidates
                )
                metrics["llm_judge"] = self._aggregate_judge_scores(judge_scores)
                metrics["llm_judge_per_sample"] = judge_scores
            
            return {
                "description": f"Teacher ({self.config.teacher_model} + RAG)",
                "metrics": metrics,
                "num_samples": len(candidates),
                "raw_outputs": candidates,
                "logprobs_per_sample": [],  # Teacher (API) has no logprobs
            }
            
        except Exception as e:
            logger.error(f"Teacher evaluation failed: {e}")
            return {"error": str(e)}
    
    # -------------------------------------------------------------------------
    # Experiment 2: RAG Backend Comparison
    # -------------------------------------------------------------------------
    
    def _run_rag_backend_comparison(
        self,
        dialogues: List[str],
        references: List[str],
    ) -> Dict[str, Dict]:
        """
        RAG Ablation Study: Compare RAG configurations progressively.
        
        Tests the contribution of each RAG component:
        - dense_only: BGE embeddings + ChromaDB (baseline retrieval)
        - dense_rerank: + Cross-encoder reranking (Nogueira & Cho, 2019)
        - dense_rerank_qe: + Medical query expansion
        - full_medical: + Clinical relevance filtering
        
        CRITICAL: We load the model ONCE and swap only the RAG retriever 
        for each config. Reloading the model causes CUDA memory corruption 
        on Windows (illegal memory access after the first unload/reload cycle).
        """
        from .inference_fixed import ClinicalScribeInference, InferenceConfig
        
        all_results = {}
        
        # Use the module-level registry — same definitions as Experiment 1.
        rag_ablation_configs = _RAG_ABLATION_CONFIGS
        
        # Filter to only requested backends
        configs_to_run = {}
        for name in self.config.rag_backends:
            if name in rag_ablation_configs:
                configs_to_run[name] = rag_ablation_configs[name]
            else:
                # Legacy support: map old names to new
                legacy_map = {
                    "llama_index": "dense_only",
                    "hybrid": "full_medical",
                }
                mapped = legacy_map.get(name)
                if mapped and mapped in rag_ablation_configs:
                    configs_to_run[mapped] = rag_ablation_configs[mapped]
                else:
                    logger.warning(f"Unknown RAG config: {name}, skipping")
        
        if not configs_to_run:
            logger.warning("No valid RAG configs to run")
            return {}
        
        # Load model once with RAG disabled — we'll init RAG manually per config
        logger.info("\nLoading model for RAG ablation study...")
        try:
            inf_config = InferenceConfig(
                model_path=self.config.ft_model_path,
                use_rag=False,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                return_logprobs=self.config.return_logprobs,
            )
            scribe = ClinicalScribeInference(inf_config)
        except Exception as e:
            logger.error(f"Failed to load model for RAG ablation: {e}")
            return {name: {"error": str(e)} for name in configs_to_run}
        
        for config_name, rag_overrides in configs_to_run.items():
            logger.info(f"\nRAG Ablation: {config_name} ({rag_overrides})")
            
            
            try:
                # Swap the retriever without reloading the model
                scribe.config.use_rag = True
                scribe.config.rag_backend = "llama_index"
                scribe._retriever = None  # Clear old retriever
                scribe._init_rag(rag_overrides=rag_overrides)
                
                outputs = scribe.batch_inference(dialogues, use_rag=True)
                candidates = [o.get("raw_output", "") for o in outputs]
                gen_times = [o.get("generation_time", 0) for o in outputs]
                rag_scores = [
                    sum(o.get("rag_scores", [])) / len(o["rag_scores"])
                    if o.get("rag_scores") else 0
                    for o in outputs
                ]
                
                metrics = self._compute_metrics(references, candidates)
                metrics["avg_generation_time"] = (
                    sum(gen_times) / len(gen_times) if gen_times else 0
                )
                metrics["avg_rag_score"] = (
                    sum(rag_scores) / len(rag_scores) if rag_scores else 0
                )
                metrics["rag_config"] = rag_overrides
                
                if self.judge:
                    logger.info(f"  Running LLM judge on {config_name}...")
                    judge_scores = self.judge.evaluate_batch(
                        dialogues, references, candidates
                    )
                    metrics["llm_judge"] = self._aggregate_judge_scores(judge_scores)
                    metrics["llm_judge_per_sample"] = judge_scores
                
                all_results[config_name] = {
                    "metrics": metrics,
                    "num_samples": len(candidates),
                    "raw_outputs": candidates,
                    "logprobs_per_sample": [o.get("logprobs") for o in outputs] if self.config.return_logprobs else [],
                }
                
            except Exception as e:
                logger.error(f"  Failed: {e}")
                import traceback
                traceback.print_exc()
                all_results[config_name] = {"error": str(e)}
        
        # Clean up after all configs are done
        try:
            del scribe
            import torch, gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass
        
        return all_results
    
    # -------------------------------------------------------------------------
    # Metrics Computation
    # -------------------------------------------------------------------------
    
    def _compute_metrics(
        self,
        references: List[str],
        candidates: List[str],
    ) -> Dict[str, Any]:
        """Compute automated metrics."""
        metrics = {}
        
        # Filter out empty candidates
        valid_pairs = [
            (r, c) for r, c in zip(references, candidates)
            if r.strip() and c.strip()
        ]
        
        if not valid_pairs:
            return {"error": "No valid output pairs", "empty_outputs": len(candidates)}
        
        refs = [p[0] for p in valid_pairs]
        cands = [p[1] for p in valid_pairs]
        
        metrics["valid_outputs"] = len(valid_pairs)
        metrics["empty_outputs"] = len(candidates) - len(valid_pairs)
        
        # ROUGE
        if self.config.compute_rouge:
            try:
                from src.evaluation.metrics import compute_rouge
                rouge_result = compute_rouge(cands, refs)
                metrics["rouge_l"] = rouge_result.value
                metrics["rouge_details"] = rouge_result.details
            except ImportError:
                try:
                    from rouge_score import rouge_scorer
                    scorer = rouge_scorer.RougeScorer(
                        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
                    )
                    r1, r2, rl = [], [], []
                    for ref, cand in zip(refs, cands):
                        scores = scorer.score(ref, cand)
                        r1.append(scores['rouge1'].fmeasure)
                        r2.append(scores['rouge2'].fmeasure)
                        rl.append(scores['rougeL'].fmeasure)
                    
                    metrics["rouge_l"] = sum(rl) / len(rl)
                    metrics["rouge_details"] = {
                        "rouge1": sum(r1) / len(r1),
                        "rouge2": sum(r2) / len(r2),
                        "rougeL": sum(rl) / len(rl),
                        "num_samples": len(rl),
                    }
                except ImportError:
                    logger.warning("rouge-score not available")
        
        # BERTScore
        if self.config.compute_bertscore:
            try:
                from bert_score import score as bert_score
                # P, R, F1 = bert_score(cands, refs, lang="en", verbose=False)
                
                P, R, F1 = bert_score(cands, refs, model_type="microsoft/deberta-xlarge-mnli", verbose=False)
                
                metrics["bertscore_f1"] = F1.mean().item()
                metrics["bertscore_precision"] = P.mean().item()
                metrics["bertscore_recall"] = R.mean().item()
            except ImportError:
                logger.warning("bert-score not available")
        
        # Clinical structure check
        if self.config.compute_clinical_accuracy:
            metrics["clinical_structure"] = self._check_clinical_structure(cands)
        
        return metrics
    
    def _check_clinical_structure(self, candidates: List[str]) -> Dict[str, float]:
        """Check whether generated summaries contain required clinical sections."""
        required_sections = [
            "chief complaint", "history of present illness", "assessment", "plan"
        ]
        optional_sections = [
            "medications", "allergies", "examination", "safety netting",
            "past medical history"
        ]
        
        required_rates = {s: 0 for s in required_sections}
        optional_rates = {s: 0 for s in optional_sections}
        
        for cand in candidates:
            cand_lower = cand.lower()
            for section in required_sections:
                if section in cand_lower:
                    required_rates[section] += 1
            for section in optional_sections:
                if section in cand_lower:
                    optional_rates[section] += 1
        
        n = len(candidates) if candidates else 1
        required_rates = {k: v / n for k, v in required_rates.items()}
        optional_rates = {k: v / n for k, v in optional_rates.items()}
        
        avg_required = sum(required_rates.values()) / len(required_rates)
        avg_optional = sum(optional_rates.values()) / len(optional_rates)
        
        return {
            "required_section_coverage": avg_required,
            "optional_section_coverage": avg_optional,
            "per_section": {**required_rates, **optional_rates},
        }
    
    def _aggregate_judge_scores(
        self, 
        scores: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Aggregate per-sample LLM judge scores."""
        valid = [s for s in scores if not s.get("judge_error")]
        if not valid:
            return {"error": "All judge evaluations failed"}
        
        dimensions = [
            "clinical_accuracy", "completeness", "hallucination",
            "clinical_safety", "coherence", "conciseness", "overall"
        ]
        
        aggregated = {}
        for dim in dimensions:
            values = [s.get(dim, 0) for s in valid if isinstance(s.get(dim), (int, float))]
            if values:
                aggregated[f"avg_{dim}"] = sum(values) / len(values)
                aggregated[f"min_{dim}"] = min(values)
                aggregated[f"max_{dim}"] = max(values)
        
        aggregated["num_evaluated"] = len(valid)
        aggregated["num_failed"] = len(scores) - len(valid)
        
        # Count critical errors
        all_errors = []
        for s in valid:
            all_errors.extend(s.get("critical_errors", []))
        aggregated["total_critical_errors"] = len(all_errors)
        aggregated["samples_with_critical_errors"] = sum(
            1 for s in valid if s.get("critical_errors")
        )
        
        return aggregated
    
    # -------------------------------------------------------------------------
    # Report Generation
    # -------------------------------------------------------------------------
    
    def _generate_report(self, results: Dict, output_dir: Path) -> Path:
        """Generate a markdown evaluation report."""
        lines = [
            "# Student Model Evaluation Report",
            f"**Date:** {results['timestamp']}",
            f"**Test Samples:** {results['num_test_samples']}",
            f"**Student Model:** {self.config.student_model}",
            f"**Judge Model:** {self.config.judge_model}",
            "",
        ]
        
        # Comparative Results Table
        comp = results.get("comparative", {})
        if comp:
            lines.append("## 1. Comparative Experiment Results")
            lines.append("")
            
            # Build comparison table
            configs = [k for k in comp if not comp[k].get("error")]
            if configs:
                header = "| Configuration | ROUGE-L | BERTScore-F1 | Avg Time (s) |"
                separator = "|---|---|---|---|"
                
                # Add judge columns if available
                has_judge = any(
                    "llm_judge" in comp[c].get("metrics", {}) 
                    for c in configs
                )
                if has_judge:
                    header += " Judge Overall | Judge Safety | Judge Halluc |"
                    separator += "---|---|---|"
                
                lines.append(header)
                lines.append(separator)
                
                for c in ["baseline", "rag_only", "ft_only", "ft_rag", "teacher"]:
                    if c not in comp or comp[c].get("error"):
                        continue
                    
                    m = comp[c].get("metrics", {})
                    desc = comp[c].get("description", c)
                    rouge = m.get("rouge_l", "N/A")
                    gen_time = m.get("avg_generation_time", "N/A")
                    
                    rouge_str = f"{rouge:.3f}" if isinstance(rouge, float) else str(rouge)
                    time_str = f"{gen_time:.1f}" if isinstance(gen_time, float) else str(gen_time)
                    bert_f1 = m.get("bertscore_f1", "N/A")
                    bert_str = f"{bert_f1:.3f}" if isinstance(bert_f1, float) else str(bert_f1)
                    
                    row = f"| {desc} | {rouge_str} | {bert_str} | {time_str} |"
                    
                    if has_judge:
                        judge = m.get("llm_judge", {})
                        overall = judge.get("avg_overall", "N/A")
                        safety = judge.get("avg_clinical_safety", "N/A")
                        halluc = judge.get("avg_hallucination", "N/A")
                        
                        overall_str = f"{overall:.2f}" if isinstance(overall, float) else str(overall)
                        safety_str = f"{safety:.2f}" if isinstance(safety, float) else str(safety)
                        halluc_str = f"{halluc:.2f}" if isinstance(halluc, float) else str(halluc)
                        
                        row += f" {overall_str} | {safety_str} | {halluc_str} |"
                    
                    lines.append(row)
                
                lines.append("")
            
            # Note errors
            errors = [k for k in comp if comp[k].get("error")]
            if errors:
                lines.append("**Skipped configurations:**")
                for e in errors:
                    lines.append(f"- {e}: {comp[e]['error']}")
                lines.append("")
        
        # RAG Backend Results
        rag = results.get("rag_backends", {})
        if rag:
            lines.append("## 2. RAG Backend Comparison")
            lines.append("")
            
            backends = [k for k in rag if not rag[k].get("error")]
            if backends:
                header = "| Backend | ROUGE-L | Avg RAG Score | Avg Time (s) |"
                if any("llm_judge" in rag[b].get("metrics", {}) for b in backends):
                    header += " Judge Overall |"
                
                lines.append(header)
                lines.append("|---|---|---|---|" + ("---|" if "Judge" in header else ""))
                
                for b in backends:
                    m = rag[b].get("metrics", {})
                    rouge = m.get("rouge_l", "N/A")
                    rag_score = m.get("avg_rag_score", "N/A")
                    gen_time = m.get("avg_generation_time", "N/A")
                    
                    rouge_str = f"{rouge:.3f}" if isinstance(rouge, float) else str(rouge)
                    rag_str = f"{rag_score:.3f}" if isinstance(rag_score, float) else str(rag_score)
                    time_str = f"{gen_time:.1f}" if isinstance(gen_time, float) else str(gen_time)
                    
                    row = f"| {b} | {rouge_str} | {rag_str} | {time_str} |"
                    
                    judge = m.get("llm_judge", {})
                    if judge and "avg_overall" in judge:
                        overall = judge["avg_overall"]
                        overall_str = f"{overall:.2f}" if isinstance(overall, float) else str(overall)
                        row += f" {overall_str} |"
                    
                    lines.append(row)
                
                lines.append("")
        
        # Write report
        report_content = "\n".join(lines)
        report_path = output_dir / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_path, "w") as f:
            f.write(report_content)
        
        return report_path


# CLI Entry Point

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned student model")
    parser.add_argument("--test-data", default="./data/training_data/test.jsonl")
    parser.add_argument("--student-model", default="clinical-scribe")
    parser.add_argument("--base-model", default="phi3.5:3.8b-mini-instruct-q4_K_M")
    parser.add_argument("--ft-model-path", default="./checkpoints/phi35_clinical_scribe/hf_merged",
                        help="Path to fine-tuned merged model directory")
    parser.add_argument("--base-model-hf", default="unsloth/Phi-3.5-mini-instruct",
                        help="HuggingFace model ID for the un-fine-tuned base model")
    parser.add_argument("--output-dir", default="./evaluation_results")
    parser.add_argument("--no-judge", action="store_true", help="Disable LLM judge")
    parser.add_argument("--no-bertscore", action="store_true", help="Disable BERTScore (faster)")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Generation temperature (0=greedy, >0=sampling)")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--configs", nargs="+",
                        default=["baseline", "rag_only", "ft_only", "ft_rag", "teacher"],
                        help="Which configs to run (e.g. --configs ft_rag teacher)")
    parser.add_argument("--skip-rag-comparison", action="store_true",
                        help="Skip RAG backend comparison experiment")
    parser.add_argument("--rag-configs", nargs="+",
                        default=["dense_only", "dense_rerank", "dense_rerank_qe", "full_medical"],
                        help="RAG ablation configs to compare in Experiment 2. "
                             "Options: dense_only, dense_rerank, dense_rerank_qe, full_medical")
    parser.add_argument("--comparative-rag-config", default="dense_rerank",
                        choices=list(_RAG_ABLATION_CONFIGS.keys()),
                        help="RAG config to use for ft_rag and rag_only in the five-way "
                             "comparison (Experiment 1). Does not affect Experiment 2. "
                             "Options: dense_only, dense_rerank, dense_rerank_qe, full_medical "
                             "(default: dense_rerank)")
    parser.add_argument("--return-logprobs", action="store_true",
                        help="Compute per-token logprobs for uncertainty quantification")
    
    args = parser.parse_args()
    
    config = EvaluationConfig(
        test_data_path=args.test_data,
        student_model=args.student_model,
        base_model=args.base_model,
        ft_model_path=args.ft_model_path,
        base_model_hf=args.base_model_hf,
        output_dir=args.output_dir,
        enable_llm_judge=not args.no_judge,
        compute_bertscore=not args.no_bertscore,
        judge_model=args.judge_model,
        max_samples=args.max_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        configs_to_run=args.configs,
        run_rag_comparison=not args.skip_rag_comparison,
        rag_backends=args.rag_configs,
        comparative_rag_config=args.comparative_rag_config,
        return_logprobs=args.return_logprobs,
    )
    
    evaluator = StudentEvaluator(config)
    results = evaluator.run_full_evaluation()
    
    print(f"\n✓ Evaluation complete!")
    print(f"  Results: {config.output_dir}")