"""
RAG-Augmented Inference with Fine-Tuned Student Model (Native PyTorch)
"""
import patch_torch

import os
# ---------------------------------------------------------
# WINDOWS COMPATIBILITY PATCHES
# Bypasses unstable Triton/Dynamo compilers on Windows
# ---------------------------------------------------------
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"  
os.environ["TORCH_COMPILE_DISABLE"] = "1"    
os.environ["TORCHDYNAMO_DISABLE"] = "1"      
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
    
    # Point directly to our traiing checkpoint..
    model_path: str = "./checkpoints/phi35_clinical_scribe/final"
 
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


SYSTEM_PROMPT = (
    "You are a clinical documentation assistant. Given a doctor-patient "
    "conversation and relevant clinical guidelines, produce a structured "
    "clinical summary in the specified format. Be accurate and concise. "
    "Only include information explicitly stated in the conversation. "
    "Do not fabricate symptoms, medications, or findings."
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
# Summary Parser (Unchanged)
# =============================================================================

def parse_structured_summary(text: str) -> Dict[str, str]:
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

# =============================================================================
# Main Inference Class
# =============================================================================

class ClinicalScribeInference:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self._retriever = None
        
        logger.info("Loading Native Model into VRAM via Unsloth...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_path,
            max_seq_length=self.config.num_ctx,
            dtype=torch.bfloat16,   # Force native 16-bit brain power.
            load_in_4bit=False, # for Disabing the buggy Windows 4-bit dequabtizer
        )
        # Enable 2x faster native inference
        # FastLanguageModel.for_inference(self.model)


        logger.info("Model loaded successfully!")
        
        if config.use_rag:
            logger.info("HERE FOR USING RAG!!!!")
            self._init_rag()
    
    def _init_rag(self):
        try:
            from src.knowledge_base import RAGFactory, RAGConfig, RAGBackend
            backend_map = {"llama_index": RAGBackend.LLAMA_INDEX, "manual": RAGBackend.MANUAL, "hybrid": RAGBackend.HYBRID}
            backend = backend_map.get(self.config.rag_backend, RAGBackend.LLAMA_INDEX)
            
            rag_config = RAGConfig(
                backend=backend,
                vector_store="chroma",
                persist_dir=self.config.rag_persist_dir,
                embedding_model="BAAI/bge-base-en-v1.5",
                chunk_size=512,
                chunk_overlap=50,
                similarity_top_k=self.config.rag_top_k,
            )
            
            factory = RAGFactory(rag_config)
            self._retriever = factory.get_retriever()
            logger.info(f"RAG retriever initialised ({self.config.rag_backend} backend)")
        except Exception as e:
            logger.warning(f"RAG initialization failed: {e}. Proceeding without RAG.")
            self.config.use_rag = False

    def generate_summary(self, dialogue: str) -> Dict[str, Any]:
        start_time = time.time()
        
        result = {"summary": {}, "raw_output": "", "generation_time": 0.0, "rag_sources": [], "rag_scores": [], "rag_context": None}
        
        rag_context = None
        if self.config.use_rag and self._retriever:
            rag_result = self._retrieve_context(dialogue)
            rag_context = rag_result.get("context")
            result["rag_sources"] = rag_result.get("sources", [])
            result["rag_scores"] = rag_result.get("scores", [])
        
        prompt = self._build_prompt(dialogue, rag_context)
        raw_output = self._generate(prompt)
        
        result["raw_output"] = raw_output
        result["summary"] = parse_structured_summary(raw_output)
        result["generation_time"] = time.time() - start_time
        return result

    def _retrieve_context(self, dialogue: str) -> Dict[str, Any]:
        query = dialogue[:300] 
        try:
            response = self._retriever.retrieve(query, top_k=self.config.rag_top_k)
            context_parts, sources, scores = [], [], []
            
            for r in response.results:
                if r.score >= 0.3:
                    context_parts.append(r.text)
                    sources.append(r.source_file)
                    scores.append(r.score)
            
            return {
                "context": "\n\n---\n\n".join(context_parts) if context_parts else None,
                "sources": sources,
                "scores": scores,
            }
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")
            return {"context": None, "sources": [], "scores": []}

    def _build_prompt(self, dialogue: str, rag_context: Optional[str] = None) -> str:
        parts = []
        if rag_context:
            parts.append(f"Relevant clinical guidelines:\n{rag_context}\n")
        parts.append(f"Summarise the following clinical consultation:\n\n{dialogue}")
        parts.append(f"\n\n{SUMMARY_INSTRUCTION}")
        return "\n".join(parts)

    def _generate(self, prompt: str) -> str:
        """Native PyTorch generation using the SAFE Chat Template."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        # This securely tokenizes the text while PRESERVING special control tokens
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        ).to("cuda")
 
        # Generate summary (Greedy Decoding)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_tokens,
            use_cache=True,
            do_sample=False,  # Keep creativity off for factual clinical data
            pad_token_id=self.tokenizer.eos_token_id,
        )
 
        # Decode only the newly generated tokens
        output_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
        return response


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    
    parser = argparse.ArgumentParser(description="Run native inference with fine-tuned clinical scribe")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG")
    parser.add_argument("--backend", default="llama_index", choices=["llama_index", "manual", "hybrid"])
    
    args = parser.parse_args()
    
    config = InferenceConfig(use_rag=not args.no_rag, rag_backend=args.backend)
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