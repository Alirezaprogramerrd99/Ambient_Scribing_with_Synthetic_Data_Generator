"""
RAG-Augmented Inference with Fine-Tuned Student Model (Native PyTorch)

Fixes applied:
1. Load from hf_merged (full merged model), not /final (LoRA adapters only)
2. Proper EOS token handling with explicit stop sequences
3. RAG context truncation to prevent context window overflow
4. Manual ChatML formatting to match exact training format

Author: Alireza Rashidi
MSc Project: Trustworthy SLMs for Ambient Clinical Scribing
"""
import patch_torch

import os
# ---------------------------------------------------------
# WINDOWS COMPATIBILITY PATCHES
# ---------------------------------------------------------
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # Prevent broken hf_transfer from blocking downloads
import torch._inductor.config

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import torch
from unsloth import FastLanguageModel

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class InferenceConfig:
    """Configuration for native student model inference."""

    # CRITICAL: Point to the MERGED model, not the LoRA adapter checkpoint.
    # The exporter merges adapters into: ./checkpoints/phi35_clinical_scribe/hf_merged
    # The /final directory only contains adapter weights, which Unsloth may
    # fail to apply when load_in_4bit=False, resulting in an untuned model.
    model_path: str = "./checkpoints/phi35_clinical_scribe/hf_merged"

    # Generation
    temperature: float = 0.3
    max_tokens: int = 2048 
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    num_ctx: int = 4096

    # RAG
    use_rag: bool = True
    rag_backend: str = "llama_index"
    knowledge_base_path: str = "./medical_knowledge/sample"
    rag_persist_dir: str = "./data/llama_index_chroma_db"
    rag_top_k: int = 5
    rag_max_context_chars: int = 3000  # Limit RAG context to prevent overflow
    
    # Logprobs for uncertainty quantification (Kadavath et al., 2022)
    return_logprobs: bool = False


# =============================================================================
# Prompts (must match training format EXACTLY)
# =============================================================================

SYSTEM_PROMPT = (
    "You are a clinical documentation assistant. Given a doctor-patient "
    "conversation and relevant clinical guidelines, produce a structured "
    "clinical summary in the specified format. Follow these rules strictly:\n"
    "1. Only include information explicitly stated in the conversation.\n"
    "2. Do NOT fabricate or infer patient demographics (age, gender, occupation) "
    "unless the patient or doctor explicitly mentions them.\n"
    "3. Do NOT invent symptoms, medications, diagnoses, or examination findings.\n"
    "4. If information for a section is not available in the conversation, "
    "write 'Not discussed' or 'Not mentioned' for that section.\n"
    "5. Be accurate and concise. Use clinical terminology appropriately."
)

SUMMARY_INSTRUCTION = (
    "Produce a structured clinical summary with the following sections:\n"
    "- Chief Complaint\n"
    "- History of Present Illness\n"
    "- Past Medical History\n"
    "- Medications\n"
    "- Allergies\n"
    "- Examination Findings\n"
    "- Assessment\n"
    "- Plan\n"
    "- Safety Netting"
)


# =============================================================================
# Summary Parser
# =============================================================================

def parse_structured_summary(text: str) -> Dict[str, str]:
    """Parse the structured summary output into a dictionary."""
    sections = {}
    section_names = [
        ("History of Present Illness", "history_of_present_illness"),
        ("Past Medical History", "past_medical_history"),
        ("Chief Complaint", "chief_complaint"),
        ("Examination Findings", "physical_examination"),
        ("Physical Examination", "physical_examination"),
        ("Social History", "social_history"),
        ("Family History", "family_history"),
        ("Safety Netting", "safety_netting"),
        ("Medications", "medications"),
        ("Allergies", "allergies"),
        ("Assessment", "assessment"),
        ("Plan", "plan"),
    ]

    pattern_parts = "|".join(re.escape(name) for name, _ in section_names)
    pattern = rf"\*?\*?({pattern_parts})\*?\*?\s*:\s*"
    parts = re.split(pattern, text, flags=re.IGNORECASE)

    if len(parts) >= 3:
        for i in range(1, len(parts) - 1, 2):
            section_label = parts[i].strip()
            content = parts[i + 1].strip()
            for name, key in section_names:
                if section_label.lower() == name.lower():
                    sections[key] = content
                    break
    return sections


def _clean_output(text: str) -> str:
    """
    Clean model output by removing everything after the actual summary ends.
    
    The model sometimes continues generating after the summary (hallucinating
    new consultations, repeating prompts, etc.). This function cuts at the
    first sign of repetition or prompt leakage.
    """
    # Cut at any repeated prompt fragments
    stop_markers = [
        "Summarise the following",
        "summarise the following",
        "Produce a structured clinical summary",
        "You are a clinical documentation assistant",
        "<|user|>",
        "<|system|>",
        "<|end|>",
        "<|endoftext|>",
        "<|im_start|>",          # Qwen2.5 tokens
        "<|im_end|>",            # Qwen2.5 tokens
        "<|eot_id|>",            # Llama 3 tokens
        "<|start_header_id|>",   # Llama 3 tokens
        "<|end_of_text|>",       # Llama 3 tokens
        "Relevant clinical guidelines:",  # RAG context leaking into output
        "\nDoctor:",  # New dialogue starting = model is hallucinating
    ]

    cleaned = text
    for marker in stop_markers:
        idx = cleaned.find(marker)
        if idx > 50:  # Only cut if we have some real content first
            cleaned = cleaned[:idx].rstrip()
            break

    return cleaned.strip()


# =============================================================================
# Main Inference Class
# =============================================================================

class ClinicalScribeInference:
    """
    Native PyTorch inference with a fine-tuned (merged) clinical scribe model.
    
    IMPORTANT: model_path must point to the MERGED model directory
    (hf_merged), not the LoRA adapter directory (final).
    """

    def __init__(self, config: InferenceConfig):
        self.config = config
        self._retriever = None
        self._last_logprobs = None  # Stores logprobs from last generation

        logger.info(f"Loading model from: {config.model_path}")

        # Determine if this is a local path or a HuggingFace model ID
        model_path = Path(config.model_path)
        is_local = model_path.exists()
        
        if is_local:
            # Verify local model directory
            # Check we're loading the merged model, not just adapters
            has_adapter_config = (model_path / "adapter_config.json").exists()
            has_model_safetensors = (
                (model_path / "model.safetensors").exists()
                or (model_path / "model.safetensors.index.json").exists()
            )

            if has_adapter_config and not has_model_safetensors:
                logger.warning(
                    "=" * 60 + "\n"
                    "WARNING: This looks like a LoRA adapter directory, not a merged model!\n"
                    "The model may run WITHOUT fine-tuning applied.\n"
                    "Use the hf_merged path instead (e.g. ./checkpoints/<model>/hf_merged)\n"
                    + "=" * 60
                )
        else:
            # Treat as HuggingFace model ID (e.g. "unsloth/Phi-3.5-mini-instruct")
            logger.info(f"  Path not found locally, treating as HuggingFace model ID: {config.model_path}")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(config.model_path),
            max_seq_length=config.num_ctx,
            dtype=torch.bfloat16,
            load_in_4bit=False,
        )

        # Enable Unsloth's 2x faster inference
        FastLanguageModel.for_inference(self.model)

        # Identify all EOS/stop token IDs for this model
        self._stop_token_ids = self._get_stop_token_ids()
        logger.info(f"Stop token IDs: {self._stop_token_ids}")
        logger.info("Model loaded successfully!")

        if config.use_rag:
            self._init_rag()

    def _get_stop_token_ids(self) -> List[int]:
        """Get all token IDs that should trigger generation stop."""
        # Phi-3.5 uses: <|end|>, <|endoftext|>, <|user|>
        # Qwen2.5 uses: <|im_end|>, <|endoftext|>, <|im_start|>
        # Llama 3 uses: <|eot_id|>, <|end_of_text|>, <|start_header_id|>
        stop_tokens = [
            "<|end|>", "<|endoftext|>", "<|user|>",        # Phi-3.5
            "<|im_end|>", "<|im_start|>",                   # Qwen2.5
            "<|eot_id|>", "<|end_of_text|>",                # Llama 3
            "<|start_header_id|>",                           # Llama 3
        ]
        stop_ids = []

        for token_str in stop_tokens:
            token_id = self.tokenizer.convert_tokens_to_ids(token_str)
            # convert_tokens_to_ids returns unk_token_id if not found
            if token_id != self.tokenizer.unk_token_id:
                stop_ids.append(token_id)

        # Always include EOS token
        if self.tokenizer.eos_token_id and self.tokenizer.eos_token_id not in stop_ids:
            stop_ids.append(self.tokenizer.eos_token_id)

        return stop_ids

    def _init_rag(self, rag_overrides: Optional[Dict[str, Any]] = None):
        """
        Initialise RAG retriever.
        
        Args:
            rag_overrides: Optional dict to override RAG settings for ablation.
                Keys: use_reranker, use_query_expansion, 
                use_clinical_filtering, similarity_top_k
        """
        try:
            from src.knowledge_base import RAGFactory, RAGConfig, RAGBackend

            backend_map = {
                "llama_index": RAGBackend.LLAMA_INDEX,
                "hybrid": RAGBackend.HYBRID,
            }
            
            overrides = rag_overrides or {}
            
            # If query expansion or clinical filtering requested, use HYBRID
            use_qe = overrides.get("use_query_expansion", False)
            use_cf = overrides.get("use_clinical_filtering", False)
            
            if use_qe or use_cf:
                backend = RAGBackend.HYBRID
            else:
                backend = backend_map.get(self.config.rag_backend, RAGBackend.LLAMA_INDEX)

            rag_config = RAGConfig(
                backend=backend,
                vector_store="chroma",
                persist_dir=self.config.rag_persist_dir,
                embedding_model="BAAI/bge-base-en-v1.5",
                chunk_size=512,
                chunk_overlap=50,
                similarity_top_k=overrides.get("similarity_top_k", self.config.rag_top_k),
                # Default: reranker ON (dense_rerank), justified by RAG ablation study
                # showing cross-encoder reranking improves ROUGE-L by up to 18%
                use_reranker=overrides.get("use_reranker", True),
                use_query_expansion=use_qe,
                use_clinical_filtering=use_cf,
            )

            factory = RAGFactory(rag_config)
            self._retriever = factory.get_retriever()
            
            desc_parts = [self.config.rag_backend]
            if overrides.get("use_reranker", True):
                desc_parts.append("reranker")
            if use_qe:
                desc_parts.append("query_expansion")
            if use_cf:
                desc_parts.append("clinical_filter")
            logger.info(f"RAG retriever initialised ({' + '.join(desc_parts)})")

        except Exception as e:
            logger.warning(f"RAG initialisation failed: {e}. Proceeding without RAG.")
            self.config.use_rag = False

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def generate_summary(self, dialogue: str) -> Dict[str, Any]:
        """Generate clinical summary with optional RAG context."""
        start_time = time.time()

        result = {
            "summary": {},
            "raw_output": "",
            "generation_time": 0.0,
            "rag_sources": [],
            "rag_scores": [],
            "rag_context": None,
        }

        # RAG retrieval
        rag_context = None
        if self.config.use_rag and self._retriever:
            rag_result = self._retrieve_context(dialogue)
            rag_context = rag_result.get("context")
            result["rag_sources"] = rag_result.get("sources", [])
            result["rag_scores"] = rag_result.get("scores", [])
            result["rag_context"] = rag_context

        # Build prompt and generate
        prompt = self._build_prompt(dialogue, rag_context)
        raw_output = self._generate(prompt)

        # Clean and parse
        cleaned = _clean_output(raw_output)
        result["raw_output"] = cleaned
        result["summary"] = parse_structured_summary(cleaned)
        result["generation_time"] = time.time() - start_time
        
        # Attach logprobs if computed
        if self._last_logprobs:
            result["logprobs"] = self._last_logprobs
        
        return result

    def generate_summary_no_rag(self, dialogue: str) -> Dict[str, Any]:
        """Generate summary without RAG (for ablation comparison)."""
        start_time = time.time()

        prompt = self._build_prompt(dialogue, rag_context=None)
        raw_output = self._generate(prompt)

        cleaned = _clean_output(raw_output)
        result = {
            "summary": parse_structured_summary(cleaned),
            "raw_output": cleaned,
            "generation_time": time.time() - start_time,
            "rag_sources": [],
            "rag_scores": [],
            "rag_context": None,
        }
        
        if self._last_logprobs:
            result["logprobs"] = self._last_logprobs
        
        return result
  

    def batch_inference(
        self, dialogues: List[str], use_rag: bool = True
    ) -> List[Dict[str, Any]]:
        """Run inference on a batch of dialogues."""
        results = []
        for i, dialogue in enumerate(dialogues):
            logger.info(f"Processing {i + 1}/{len(dialogues)}...")
            try:
                if use_rag:
                    result = self.generate_summary(dialogue)
                else:
                    result = self.generate_summary_no_rag(dialogue)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed on sample {i + 1}: {e}")
                results.append({
                    "summary": {},
                    "raw_output": "",
                    "generation_time": 0,
                    "error": str(e),
                })
        return results

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    def _retrieve_context(self, dialogue: str) -> Dict[str, Any]:
        """Retrieve relevant clinical guidelines, with length truncation."""
        query = self._extract_query_from_dialogue(dialogue)

        try:
            response = self._retriever.retrieve(query, top_k=self.config.rag_top_k)

            context_parts = []
            sources = []
            scores = []
            total_chars = 0

            # Detect if results come from a cross-encoder reranker
            # (scores outside 0-1 range) vs cosine similarity (0-1 range).
            # For reranked results, skip score thresholding and trust top-k.
            is_reranked = any(
                s.score is not None and (s.score < 0 or s.score > 1.0)
                for s in response.results
            )
            min_score = -float("inf") if is_reranked else 0.3

            for r in response.results:
                if r.score < min_score:
                    continue

                # Enforce context length limit to prevent overflow
                if total_chars + len(r.text) > self.config.rag_max_context_chars:
                    # Add truncated remainder if we have room
                    remaining = self.config.rag_max_context_chars - total_chars
                    if remaining > 100:
                        context_parts.append(r.text[:remaining] + "...")
                        sources.append(r.source_file)
                        scores.append(r.score)
                    break

                context_parts.append(r.text)
                sources.append(r.source_file)
                scores.append(r.score)
                total_chars += len(r.text)

            return {
                "context": "\n\n---\n\n".join(context_parts) if context_parts else None,
                "sources": sources,
                "scores": scores,
            }

        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")
            return {"context": None, "sources": [], "scores": []}

    def _extract_query_from_dialogue(self, dialogue: str) -> str:
        """Extract key clinical terms from dialogue for RAG query."""
        lines = dialogue.strip().split("\n")
        patient_lines = [
            line.split(":", 1)[1].strip()
            for line in lines
            if line.lower().startswith("patient:") and ":" in line
        ][:3]

        doctor_lines = [
            line.split(":", 1)[1].strip()
            for line in lines
            if line.lower().startswith("doctor:") and ":" in line
            and any(kw in line.lower() for kw in [
                "think", "suspect", "diagnos", "assess", "condition",
                "test", "refer", "prescrib", "recommend"
            ])
        ][:2]

        query_parts = patient_lines + doctor_lines
        return " ".join(query_parts)[:500] if query_parts else dialogue[:300]

    def _build_prompt(self, dialogue: str, rag_context: Optional[str] = None) -> str:
        """Build the user message (content only, not the ChatML wrapper)."""
        parts = []
        if rag_context:
            parts.append(f"Relevant clinical guidelines:\n{rag_context}\n")
        parts.append(f"Summarise the following clinical consultation:\n\n{dialogue}")
        parts.append(f"\n\n{SUMMARY_INSTRUCTION}")
        return "\n".join(parts)

    def _generate(self, prompt: str) -> str:
        """
        Generate using the tokenizer's chat template (matches training format).
        
        Uses explicit eos_token_id list to ensure the model stops after
        producing the summary, preventing infinite continuation.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        # Tokenize with the chat template — this produces the exact same
        # <|system|>...<|end|><|user|>...<|end|><|assistant|> format
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to("cuda")

        input_length = inputs["input_ids"].shape[1]

        # Check context usage
        if input_length > self.config.num_ctx - 100:
            logger.warning(
                f"Input length ({input_length} tokens) near context limit "
                f"({self.config.num_ctx}). Output may be degraded."
            )

        with torch.no_grad():
            # Use greedy decoding when temperature is 0, sampling otherwise
            # Ref: Woo et al. (2025) tested temp 0 vs 1; Renze & Guven (2024)
            # found no significant difference in problem-solving for temp 0-1.
            use_sampling = self.config.temperature > 0
            
            gen_kwargs = dict(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                use_cache=True,
                do_sample=use_sampling,
                eos_token_id=self._stop_token_ids,
                pad_token_id=self.tokenizer.eos_token_id,
                output_scores=self.config.return_logprobs,
                return_dict_in_generate=self.config.return_logprobs,
            )
            
            if use_sampling:
                gen_kwargs["temperature"] = self.config.temperature
                gen_kwargs["top_p"] = self.config.top_p
            
            outputs = self.model.generate(**gen_kwargs)

        # Extract generated tokens and text
        if self.config.return_logprobs:
            # outputs is a GenerateDecoderOnlyOutput with .sequences and .scores
            output_tokens = outputs.sequences[0][input_length:]
            response = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
            
            # Compute per-token log probabilities
            # outputs.scores is a tuple of (num_generated_tokens,) tensors of shape (batch, vocab)
            token_logprobs = []
            token_texts = []
            for i, score_tensor in enumerate(outputs.scores):
                # Apply log_softmax to get log probabilities
                log_probs = torch.nn.functional.log_softmax(score_tensor[0], dim=-1)
                # Get the logprob of the actually generated token
                generated_token_id = output_tokens[i].item()
                token_lp = log_probs[generated_token_id].item()
                token_logprobs.append(token_lp)
                token_texts.append(self.tokenizer.decode([generated_token_id]))
            
            self._last_logprobs = {
                "token_logprobs": token_logprobs,
                "token_texts": token_texts,
                "mean_logprob": sum(token_logprobs) / len(token_logprobs) if token_logprobs else 0,
                "min_logprob": min(token_logprobs) if token_logprobs else 0,
                "num_tokens": len(token_logprobs),
            }
        else:
            output_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
            self._last_logprobs = None
        
        return response


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(
        description="Run native inference with fine-tuned clinical scribe"
    )
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG")
    parser.add_argument(
        "--backend", default="llama_index",
        choices=["llama_index", "manual", "hybrid"],
    )
    parser.add_argument(
        "--model-path",
        default="./checkpoints/phi35_clinical_scribe/hf_merged",
        help="Path to merged model directory",
    )

    args = parser.parse_args()

    config = InferenceConfig(
        model_path=args.model_path,
        use_rag=not args.no_rag,
        rag_backend=args.backend,
    )
    scribe = ClinicalScribeInference(config)

    dialogue_text = (
        "Doctor: Good morning. What brings you in today?\n"
        "Patient: I've been having chest pain for about three days.\n"
        "Doctor: Can you describe the pain? Where exactly is it?\n"
        "Patient: It's in the centre of my chest, like a pressure.\n"
        "Doctor: Does it get worse with activity or stress?\n"
        "Patient: Yes, it's worse when I climb stairs.\n"
        "Doctor: Any shortness of breath, nausea, or sweating?\n"
        "Patient: A little breathless, but no nausea.\n"
        "Doctor: Do you have any medical conditions?\n"
        "Patient: I have high blood pressure. I take amlodipine.\n"
        "Doctor: Any family history of heart problems?\n"
        "Patient: My father had a heart attack at 55.\n"
        "Doctor: I'd like to do an ECG and some blood tests."
    )

    print(f"\nGenerating summary...")
    print(f"RAG: {'enabled (' + args.backend + ')' if not args.no_rag else 'disabled'}\n")

    result = scribe.generate_summary(dialogue_text)

    print("=" * 60)
    print("Generated Summary")
    print("=" * 60)
    print(result["raw_output"])
    print(f"\n--- Generation time: {result['generation_time']:.1f}s ---")

    if result["rag_sources"]:
        print(f"--- RAG sources: {result['rag_sources']} ---")
        print(f"--- RAG scores: {[f'{s:.3f}' for s in result['rag_scores']]} ---")