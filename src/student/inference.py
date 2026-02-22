"""
RAG-Augmented Inference with Fine-Tuned Student Model

Connects the fine-tuned Phi-3.5-mini (via Ollama) with the existing
RAG retriever stack for clinical summary generation.

Reuses:
    - src.knowledge_base.RAGFactory / RAGConfig / RAGBackend
    - Ollama API (same as teacher pipeline's OllamaClient)

Author: Alireza Rashidi
MSc Project: Trustworthy SLMs for Ambient Clinical Scribing
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class InferenceConfig:
    """Configuration for student model inference."""
    
    # Ollama
    model_name: str = "clinical-scribe"
    ollama_base_url: str = "http://localhost:11434"
    timeout: float = 120.0
    
    # Generation
    temperature: float = 0.3
    max_tokens: int = 2048
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    num_ctx: int = 4096
    
    # RAG
    use_rag: bool = True
    rag_backend: str = "llama_index"  # "llama_index", "manual", "hybrid"
    knowledge_base_path: str = "./medical_knowledge/sample"
    rag_persist_dir: str = "./data/llama_index_chroma_db"
    rag_top_k: int = 5
    rag_score_threshold: float = 0.3


# =============================================================================
# System Prompt
# =============================================================================

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
# Summary Parser
# =============================================================================

def parse_structured_summary(text: str) -> Dict[str, str]:
    """
    Parse the structured summary output from the model into a dictionary.
    
    Handles both **Section:** and Section: formats.
    """
    sections = {}
    
    # Define section patterns (order matters - longest match first)
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
    
    # Build regex pattern
    pattern_parts = "|".join(
        re.escape(name) for name, _ in section_names
    )
    pattern = rf"\*?\*?({pattern_parts})\*?\*?\s*:\s*"
    
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    
    # parts alternates: [preamble, section_name, content, section_name, content, ...]
    if len(parts) >= 3:
        for i in range(1, len(parts) - 1, 2):
            section_label = parts[i].strip()
            content = parts[i + 1].strip()
            
            # Map to field key
            for name, key in section_names:
                if section_label.lower() == name.lower():
                    sections[key] = content
                    break
    
    # Handle SOAP if present
    soap_match = re.search(r"SOAP\s*Note\s*:", text, re.IGNORECASE)
    if soap_match:
        soap_text = text[soap_match.end():]
        soap = {}
        for letter, field_name in [("S", "subjective"), ("O", "objective"), ("A", "assessment"), ("P", "plan")]:
            match = re.search(rf"\b{letter}\s*:\s*(.+?)(?=\b[SOAP]\s*:|$)", soap_text, re.DOTALL)
            if match:
                soap[field_name] = match.group(1).strip()
        if soap:
            sections["soap"] = soap
    
    return sections


# =============================================================================
# Main Inference Class
# =============================================================================

class ClinicalScribeInference:
    """
    RAG-augmented inference pipeline for the fine-tuned student model.
    
    Connects:
        - Fine-tuned Phi-3.5-mini via Ollama (local inference)
        - Existing RAG retriever stack (LlamaIndex / Manual / Hybrid)
    
    Example:
        config = InferenceConfig(model_name="clinical-scribe", use_rag=True)
        scribe = ClinicalScribeInference(config)
        result = scribe.generate_summary(dialogue_text)
        print(result["summary"]["chief_complaint"])
    """
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self._client = httpx.Client(timeout=config.timeout)
        self._retriever = None
        
        if config.use_rag:
            self._init_rag()
    
    def _init_rag(self):
        """Initialise the RAG retriever using the existing knowledge base stack."""
        try:
            from src.knowledge_base import RAGFactory, RAGConfig, RAGBackend
            
            backend_map = {
                "llama_index": RAGBackend.LLAMA_INDEX,
                "manual": RAGBackend.MANUAL,
                "hybrid": RAGBackend.HYBRID,
            }
            
            backend = backend_map.get(self.config.rag_backend, RAGBackend.LLAMA_INDEX)
            
            rag_config = RAGConfig(
                backend=backend,
                vector_store="chroma",
                persist_dir=self.config.rag_persist_dir,
                embedding_model="BAAI/bge-base-en-v1.5",
                chunk_size=512,
                chunk_overlap=50,
                similarity_top_k=self.config.rag_top_k,
                # score_threshold=self.config.rag_score_threshold,
            )
            
            factory = RAGFactory(rag_config)
            self._retriever = factory.get_retriever()
            logger.info(f"RAG retriever initialised ({self.config.rag_backend} backend)")
            
        except ImportError:
            logger.warning(
                "Could not import RAG modules. "
                "Ensure src.knowledge_base is available. "
                "Proceeding without RAG."
            )
            self.config.use_rag = False
    
    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    
    def generate_summary(self, dialogue: str) -> Dict[str, Any]:
        """
        Generate a clinical summary from dialogue text.
        
        Args:
            dialogue: Raw dialogue text (e.g., "Doctor: ... Patient: ...")
        
        Returns:
            Dictionary with keys:
                - summary: Parsed summary dict
                - raw_output: Raw model output text
                - generation_time: Time in seconds
                - rag_sources: List of source files used (if RAG enabled)
                - rag_scores: Retrieval scores (if RAG enabled)
                - rag_context: Retrieved context text (if RAG enabled)
        """
        start_time = time.time()
        
        result = {
            "summary": {},
            "raw_output": "",
            "generation_time": 0.0,
            "rag_sources": [],
            "rag_scores": [],
            "rag_context": None,
        }
        
        # Step 1: RAG retrieval
        rag_context = None
        if self.config.use_rag and self._retriever:
            rag_result = self._retrieve_context(dialogue)
            rag_context = rag_result.get("context")
            result["rag_sources"] = rag_result.get("sources", [])
            result["rag_scores"] = rag_result.get("scores", [])
            result["rag_context"] = rag_context
        
        # Step 2: Build prompt
        prompt = self._build_prompt(dialogue, rag_context)
        
        # Step 3: Generate via Ollama
        raw_output = self._generate(prompt)
        result["raw_output"] = raw_output
        
        # Step 4: Parse output
        result["summary"] = parse_structured_summary(raw_output)
        
        result["generation_time"] = time.time() - start_time
        return result
    
    def generate_summary_no_rag(self, dialogue: str) -> Dict[str, Any]:
        """Generate summary without RAG (for ablation comparison)."""
        start_time = time.time()
        
        prompt = self._build_prompt(dialogue, rag_context=None)
        raw_output = self._generate(prompt)
        
        return {
            "summary": parse_structured_summary(raw_output),
            "raw_output": raw_output,
            "generation_time": time.time() - start_time,
            "rag_sources": [],
            "rag_scores": [],
            "rag_context": None,
        }
    
    def batch_inference(
        self, 
        dialogues: List[str], 
        use_rag: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run inference on a batch of dialogues.
        
        Args:
            dialogues: List of dialogue texts.
            use_rag: Whether to use RAG for this batch.
        
        Returns:
            List of result dictionaries.
        """
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
    
    def is_model_available(self) -> bool:
        """Check if the Ollama model is available."""
        try:
            response = self._client.get(f"{self.config.ollama_base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any(
                    self.config.model_name in m.get("name", "")
                    for m in models
                )
        except Exception:
            pass
        return False
    
    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------
    
    def _retrieve_context(self, dialogue: str) -> Dict[str, Any]:
        """Retrieve relevant clinical guidelines for the dialogue."""
        # Extract key clinical terms for the query
        query = self._extract_query_from_dialogue(dialogue)
        
        try:
            response = self._retriever.retrieve(query, top_k=self.config.rag_top_k)
            
            context_parts = []
            sources = []
            scores = []
            
            for r in response.results:
                if r.score >= self.config.rag_score_threshold:
                    context_parts.append(r.text)
                    sources.append(r.source_file)
                    scores.append(r.score)
            
            context = "\n\n---\n\n".join(context_parts) if context_parts else None
            
            return {
                "context": context,
                "sources": sources,
                "scores": scores,
            }
            
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")
            return {"context": None, "sources": [], "scores": []}
    
    def _extract_query_from_dialogue(self, dialogue: str) -> str:
        """Extract key clinical terms from dialogue for RAG query."""
        # Take the first few patient statements (usually contain chief complaint)
        lines = dialogue.strip().split("\n")
        patient_lines = [
            line.split(":", 1)[1].strip()
            for line in lines
            if line.lower().startswith("patient:")
        ][:3]
        
        # Also grab doctor's assessment-related lines
        doctor_lines = [
            line.split(":", 1)[1].strip()
            for line in lines
            if line.lower().startswith("doctor:")
            and any(kw in line.lower() for kw in [
                "think", "suspect", "diagnos", "assess", "condition",
                "test", "refer", "prescrib", "recommend"
            ])
        ][:2]
        
        query_parts = patient_lines + doctor_lines
        return " ".join(query_parts)[:500] if query_parts else dialogue[:300]
    
    def _build_prompt(self, dialogue: str, rag_context: Optional[str] = None) -> str:
        """Build the user prompt for the model."""
        parts = []
        
        if rag_context:
            parts.append(f"Relevant clinical guidelines:\n{rag_context}\n")
        
        parts.append(f"Summarise the following clinical consultation:\n\n{dialogue}")
        parts.append(f"\n\n{SUMMARY_INSTRUCTION}")
        
        return "\n".join(parts)
    
    def _generate(self, prompt: str) -> str:
        """Call Ollama API for generation."""
        try:
            response = self._client.post(
                f"{self.config.ollama_base_url}/api/generate",
                json={
                    "model": self.config.model_name,
                    "prompt": prompt,
                    "system": SYSTEM_PROMPT,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens,
                        "top_p": self.config.top_p,
                        "repeat_penalty": self.config.repeat_penalty,
                        "num_ctx": self.config.num_ctx,
                    },
                },
                timeout=self.config.timeout,
            )
            
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
            
        except httpx.TimeoutException:
            logger.error("Ollama generation timed out")
            return ""
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return ""


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned clinical scribe")
    parser.add_argument("--model", default="clinical-scribe", help="Ollama model name")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG")
    parser.add_argument("--backend", default="llama_index", choices=["llama_index", "manual", "hybrid"])
    parser.add_argument("--dialogue", type=str, help="Dialogue text (or path to file)")
    
    args = parser.parse_args()
    
    config = InferenceConfig(
        model_name=args.model,
        use_rag=not args.no_rag,
        rag_backend=args.backend,
    )
    
    scribe = ClinicalScribeInference(config)
    
    # Check model availability
    if not scribe.is_model_available():
        print(f"⚠ Model '{args.model}' not found in Ollama.")
        print(f"  Run: ollama create {args.model} -f Modelfile")
        exit(1)
    
    # Use provided dialogue or demo
    if args.dialogue:
        if Path(args.dialogue).exists():
            with open(args.dialogue) as f:
                dialogue_text = f.read()
        else:
            dialogue_text = args.dialogue
    else:
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
    
    print(f"\nGenerating summary with model '{args.model}'...")
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
