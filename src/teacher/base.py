"""
Abstract Base Teacher Model

Defines the interface for teacher models that generate synthetic
clinical dialogue-summary pairs for knowledge distillation.
"""

import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.models import (
    DialogueTurn,
    ClinicalSummary,
    SOAPNote,
    SyntheticSample,
    ScenarioMetadata,
    GenerationMetadata,
    RAGMetadata,
    DifficultyMetadata,
    DifficultyLevel,
    Speaker,
    ClinicalSpecialty,
    ValidationResult,
    ValidationStatus,
)
from src.knowledge_base import (
    BaseRetriever,
    RetrievalResponse,
    MedicalQueryProcessor,
)
from src.config import get_clinical_prompts, ClinicalPrompts
from src.utils import (
    RetryContext,
    retry_with_exponential_backoff,
    MaxRetriesExceededError,
    logger as default_logger,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class GenerationConfig:
    """Configuration for generation"""
    
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.9
    top_k: int = 40
    
    # RAG settings
    use_rag: bool = True
    rag_top_k: int = 5
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Output settings
    include_soap: bool = True
    include_difficulty: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "use_rag": self.use_rag,
            "rag_top_k": self.rag_top_k,
            "max_retries": self.max_retries,
        }


@dataclass
class GenerationResult:
    """Result of a single generation attempt"""
    
    success: bool
    sample: Optional[SyntheticSample] = None
    error: Optional[str] = None
    raw_response: Optional[str] = None
    attempts: int = 1
    generation_time_seconds: float = 0.0
    
    @property
    def is_valid(self) -> bool:
        return self.success and self.sample is not None


@dataclass  
class BatchGenerationResult:
    """Result of batch generation"""
    
    samples: List[SyntheticSample] = field(default_factory=list)
    failed_scenarios: List[Tuple[str, str]] = field(default_factory=list)  # (scenario, error)
    
    total_attempted: int = 0
    total_succeeded: int = 0
    total_failed: int = 0
    total_time_seconds: float = 0.0
    
    @property
    def success_rate(self) -> float:
        if self.total_attempted == 0:
            return 0.0
        return self.total_succeeded / self.total_attempted
    
    @property
    def samples_per_minute(self) -> float:
        if self.total_time_seconds == 0:
            return 0.0
        return (self.total_succeeded / self.total_time_seconds) * 60
    
    def add_success(self, sample: SyntheticSample):
        self.samples.append(sample)
        self.total_attempted += 1
        self.total_succeeded += 1
    
    def add_failure(self, scenario: str, error: str):
        self.failed_scenarios.append((scenario, error))
        self.total_attempted += 1
        self.total_failed += 1
    
    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "total_attempted": self.total_attempted,
            "total_succeeded": self.total_succeeded,
            "total_failed": self.total_failed,
            "success_rate": self.success_rate,
            "total_time_seconds": self.total_time_seconds,
            "samples_per_minute": self.samples_per_minute,
        }


# =============================================================================
# Abstract Base Teacher
# =============================================================================

class BaseTeacher(ABC):
    """
    Abstract base class for teacher models
    
    Defines the interface for generating synthetic clinical data.
    Subclasses implement specific LLM backends (Ollama, OpenAI, Anthropic).
    
    The teacher model:
    1. Takes a clinical scenario as input
    2. Optionally retrieves relevant guidelines via RAG
    3. Generates a realistic doctor-patient dialogue
    4. Creates a comprehensive clinical summary
    5. Validates the output structure
    6. Returns a SyntheticSample for training
    """
    
    def __init__(
        self,
        model_name: str,
        retriever: Optional[BaseRetriever] = None,
        config: Optional[GenerationConfig] = None,
        prompts: Optional[ClinicalPrompts] = None,
    ):
        """
        Initialize teacher model
        
        Args:
            model_name: Name of the LLM model
            retriever: RAG retriever for clinical guidelines
            config: Generation configuration
            prompts: Prompt templates
        """
        self.model_name = model_name
        self.retriever = retriever
        self.config = config or GenerationConfig()
        self.prompts = prompts or get_clinical_prompts()
        self.query_processor = MedicalQueryProcessor()
        
        # Statistics
        self._total_generated = 0
        self._total_failed = 0
        
        logger.info(f"Teacher initialized: {self.model_name}")
        if self.retriever:
            logger.info("RAG retrieval enabled")
    
    @property
    @abstractmethod
    def provider(self) -> str:
        """Return the provider name (e.g., 'ollama', 'openai')"""
        pass
    
    @abstractmethod
    def _call_llm(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Call the LLM and return raw response
        
        Args:
            prompt: The prompt to send
            temperature: Sampling temperature (uses config default if None)
            max_tokens: Max tokens to generate
            
        Returns:
            Raw string response from LLM
        """
        pass
    
    @abstractmethod
    def _check_connection(self) -> bool:
        """Check if the LLM is accessible"""
        pass
    
    # =========================================================================
    # RAG Methods
    # =========================================================================
    
    def retrieve_guidelines(
        self,
        scenario: str,
        top_k: Optional[int] = None,
        use_llm_expansion: bool = False,
    ) -> Tuple[str, RAGMetadata]:
        """
        Retrieve relevant clinical guidelines for a scenario
        
        Args:
            scenario: Clinical scenario description
            top_k: Number of results to retrieve
            use_llm_expansion: Whether to use LLM for query expansion
            
        Returns:
            Tuple of (context_string, RAGMetadata)
        """
        if not self.retriever:
            return "", RAGMetadata(rag_enabled=False)
        
        top_k = top_k or self.config.rag_top_k
        
        # Expand query with medical knowledge
        if use_llm_expansion and hasattr(self.query_processor, 'expand_query_with_llm'):
            # Set up LLM client for query processor if not done
            if not self.query_processor._llm_client:
                self.query_processor.set_llm_client(self)
                self.query_processor.prompt_manager = self.prompts._manager if hasattr(self.prompts, '_manager') else None
            expanded_query = self.query_processor.expand_query_with_llm(scenario)
        else:
            expanded_query = self.query_processor.expand_query(scenario)
        
        # Retrieve
        response: RetrievalResponse = self.retriever.retrieve(
            expanded_query,
            top_k=top_k,
        )
        
        # Build context string
        context = response.get_context(max_results=top_k)
        
        # Create metadata
        rag_meta = RAGMetadata(
            rag_enabled=True,
            num_sources_retrieved=response.num_results,
            sources=response.get_sources(),
            retrieval_scores=[r.score for r in response.results],
            context_used=context[:1000] if context else None,  # Truncate for storage
        )
        
        logger.debug(f"Retrieved {response.num_results} sources for scenario")
        return context, rag_meta
    
    # =========================================================================
    # Difficulty Assessment
    # =========================================================================
    
    def assess_difficulty(
        self,
        sample: SyntheticSample,
    ) -> DifficultyMetadata:
        """
        Use LLM to assess the difficulty/complexity of a generated sample
        
        Based on Woo et al. (2025) methodology for difficulty scoring.
        
        Args:
            sample: The generated SyntheticSample to assess
            
        Returns:
            DifficultyMetadata with score and contributing factors
        """
        try:
            # Build the difficulty assessment prompt
            # Get scenario from sample metadata
            scenario_text = sample.scenario.scenario_text if sample.scenario else "Unknown scenario"
            
            prompt = self.prompts.difficulty_assessment(
                scenario=scenario_text,
                dialogue=sample.dialogue_text,
                summary=sample.summary.model_dump_json() if hasattr(sample.summary, 'model_dump_json') else str(sample.summary),
            )
            
            # Call LLM with low temperature for consistency
            raw_response = self._call_llm(
                prompt,
                temperature=0.3,  # Low temperature for consistent scoring
                max_tokens=500,
            )
            
            # Parse the difficulty assessment response
            difficulty_meta = self._parse_difficulty_response(raw_response)
            
            logger.debug(f"Assessed difficulty: {difficulty_meta.difficulty_level.value} "
                        f"(score: {difficulty_meta.difficulty_score})")
            
            return difficulty_meta
            
        except Exception as e:
            logger.warning(f"Difficulty assessment failed: {e}. Using heuristic fallback.")
            return self._heuristic_difficulty_assessment(sample)
    
    def _parse_difficulty_response(self, response: str) -> DifficultyMetadata:
        """
        Parse LLM response for difficulty assessment
        
        Args:
            response: Raw LLM response
            
        Returns:
            DifficultyMetadata
        """
        import re
        
        response = response.strip()
        
        # Try to extract JSON
        try:
            json_str = self._extract_json(response)
            data = json.loads(json_str)
            
            # Extract score (1-10)
            score = data.get("difficulty_score", data.get("score", 5))
            if isinstance(score, str):
                score = int(re.search(r'\d+', score).group())
            score = max(1, min(10, score))  # Clamp to 1-10
            
            # Extract factors
            factors = data.get("factors", data.get("complexity_factors", []))
            if isinstance(factors, str):
                factors = [f.strip() for f in factors.split(",")]
            
            # Extract rationale
            rationale = data.get("rationale", data.get("reasoning", ""))
            
            return DifficultyMetadata.from_score(
                score=score,
                factors=factors,
                rationale=rationale,
            )
            
        except (json.JSONDecodeError, ValueError, AttributeError):
            # Fallback: try to extract score from text
            score_match = re.search(r'(?:score|difficulty)[:\s]*(\d+)', response.lower())
            if score_match:
                score = int(score_match.group(1))
                score = max(1, min(10, score))
                return DifficultyMetadata.from_score(score)
            
            # Default to medium difficulty
            return DifficultyMetadata.from_score(5)
    
    def _heuristic_difficulty_assessment(
        self,
        sample: SyntheticSample,
    ) -> DifficultyMetadata:
        """
        Fallback heuristic-based difficulty assessment
        
        Used when LLM-based assessment fails.
        
        Args:
            sample: SyntheticSample to assess
            
        Returns:
            DifficultyMetadata
        """
        score = 5  # Start at medium
        factors = []
        
        # Factor 1: Dialogue length (more turns = potentially more complex)
        num_turns = len(sample.dialogue)
        if num_turns > 20:
            score += 1
            factors.append("extended_dialogue")
        elif num_turns < 8:
            score -= 1
        
        # Factor 2: Summary complexity (HPI length)
        hpi_length = len(sample.summary.history_of_present_illness.split())
        if hpi_length > 150:
            score += 1
            factors.append("detailed_history")
        
        # Factor 3: Multiple conditions in assessment
        assessment = sample.summary.assessment.lower()
        if " and " in assessment or "differential" in assessment:
            score += 1
            factors.append("multiple_differentials")
        
        # Factor 4: Complex plan
        plan = sample.summary.plan.lower()
        plan_indicators = ["refer", "specialist", "urgent", "monitor", "follow-up"]
        plan_complexity = sum(1 for ind in plan_indicators if ind in plan)
        if plan_complexity >= 3:
            score += 1
            factors.append("comprehensive_plan")
        
        # Factor 5: Medications mentioned
        if sample.summary.medications:
            med_count = sample.summary.medications.count(",") + 1
            if med_count >= 3:
                score += 1
                factors.append("polypharmacy")
        
        # Clamp score
        score = max(1, min(10, score))
        
        return DifficultyMetadata.from_score(
            score=score,
            factors=factors,
            rationale="Heuristic assessment based on dialogue and summary features",
        )
    
    # =========================================================================
    # Generation Methods
    # =========================================================================
    
    def generate(
        self,
        scenario: str,
        use_rag: Optional[bool] = None,
        use_llm_query_expansion: bool = False,
        use_llm_difficulty_assessment: bool = True,
    ) -> GenerationResult:
        """
        Generate a synthetic sample from a scenario
        
        Args:
            scenario: Clinical scenario description
            use_rag: Override RAG setting from config
            use_llm_query_expansion: Use LLM to expand RAG queries
            use_llm_difficulty_assessment: Use LLM to assess sample difficulty
            
        Returns:
            GenerationResult with sample or error
        """
        start_time = time.time()
        use_rag = use_rag if use_rag is not None else self.config.use_rag
        
        # Get RAG context if enabled
        if use_rag and self.retriever:
            guidelines_context, rag_meta = self.retrieve_guidelines(
                scenario,
                use_llm_expansion=use_llm_query_expansion,
            )
        else:
            guidelines_context = "No specific guidelines retrieved."
            rag_meta = RAGMetadata(rag_enabled=False)
        
        # Build prompt
        prompt = self.prompts.dialogue_generation(
            scenario=scenario,
            guidelines_context=guidelines_context,
        )
        
        # Generate with retries
        with RetryContext(
            max_attempts=self.config.max_retries,
            min_wait_seconds=self.config.retry_delay,
        ) as retry_ctx:
            
            while retry_ctx.should_retry():
                try:
                    # Call LLM
                    raw_response = self._call_llm(
                        prompt,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                    )
                    
                    # Parse response
                    sample = self._parse_response(
                        raw_response=raw_response,
                        scenario=scenario,
                        rag_meta=rag_meta,
                    )
                    
                    retry_ctx.success(sample)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parse error: {e}")
                    retry_ctx.failed(e)
                    
                except ValueError as e:
                    logger.warning(f"Validation error: {e}")
                    retry_ctx.failed(e)
                    
                except Exception as e:
                    logger.error(f"Generation error: {e}")
                    retry_ctx.failed(e)
        
        generation_time = time.time() - start_time
        
        if retry_ctx.succeeded:
            self._total_generated += 1
            sample = retry_ctx.result
            sample.generation.generation_time_seconds = generation_time
            
            # Perform LLM-based difficulty assessment if enabled
            if use_llm_difficulty_assessment and self.config.include_difficulty:
                # Always use LLM assessment to get detailed factors and rationale
                # This overwrites any heuristic assessment from metadata parsing
                try:
                    sample.difficulty = self.assess_difficulty(sample)
                except Exception as e:
                    logger.warning(f"LLM difficulty assessment failed: {e}")
                    # Keep existing difficulty or use heuristic fallback
                    if sample.difficulty is None:
                        sample.difficulty = self._heuristic_difficulty_assessment(sample)
            
            return GenerationResult(
                success=True,
                sample=sample,
                raw_response=raw_response,
                attempts=retry_ctx.attempts,
                generation_time_seconds=generation_time,
            )
        else:
            self._total_failed += 1
            return GenerationResult(
                success=False,
                error=str(retry_ctx.last_exception),
                attempts=retry_ctx.attempts,
                generation_time_seconds=generation_time,
            )
    
    def generate_batch(
        self,
        scenarios: List[str],
        use_rag: Optional[bool] = None,
        save_interval: int = 10,
        output_path: Optional[Path] = None,
        progress_callback: Optional[callable] = None,
    ) -> BatchGenerationResult:
        """
        Generate samples for multiple scenarios
        
        Args:
            scenarios: List of scenario descriptions
            use_rag: Override RAG setting
            save_interval: Save progress every N samples
            output_path: Path to save intermediate results
            progress_callback: Called with (current, total) after each sample
            
        Returns:
            BatchGenerationResult with all samples
        """
        batch_result = BatchGenerationResult()
        start_time = time.time()
        
        logger.info(f"Starting batch generation: {len(scenarios)} scenarios")
        
        for i, scenario in enumerate(scenarios):
            # Generate
            result = self.generate(scenario, use_rag=use_rag)
            
            if result.success:
                batch_result.add_success(result.sample)
            else:
                batch_result.add_failure(scenario, result.error)
            
            # Progress callback
            if progress_callback:
                progress_callback(i + 1, len(scenarios))
            
            # Save intermediate results
            if output_path and (i + 1) % save_interval == 0:
                self._save_intermediate(batch_result, output_path)
                logger.info(f"Progress: {i + 1}/{len(scenarios)} "
                           f"(success rate: {batch_result.success_rate:.1%})")
        
        batch_result.total_time_seconds = time.time() - start_time
        
        # Final save
        if output_path:
            self._save_intermediate(batch_result, output_path)
        
        logger.info(f"Batch complete: {batch_result.total_succeeded}/{batch_result.total_attempted} "
                   f"({batch_result.success_rate:.1%})")
        
        return batch_result
    
    # =========================================================================
    # Parsing Methods
    # =========================================================================
    
    def _parse_response(
        self,
        raw_response: str,
        scenario: str,
        rag_meta: RAGMetadata,
    ) -> SyntheticSample:
        """
        Parse LLM response into SyntheticSample
        
        Args:
            raw_response: Raw JSON string from LLM
            scenario: Original scenario
            rag_meta: RAG metadata
            
        Returns:
            Validated SyntheticSample
        """
        # Clean response
        json_str = self._extract_json(raw_response)
        
        # Parse JSON
        data = json.loads(json_str)
        
        # Validate required fields
        if "dialogue" not in data:
            raise ValueError("Missing 'dialogue' field in response")
        if "summary" not in data:
            raise ValueError("Missing 'summary' field in response")
        
        # Parse dialogue
        dialogue_turns = self._parse_dialogue(data["dialogue"])
        
        # Parse summary
        clinical_summary = self._parse_summary(data["summary"])
        
        # Parse metadata
        metadata = data.get("metadata", {})
        
        # Create generation metadata
        gen_meta = GenerationMetadata(
            model_name=self.model_name,
            model_provider=self.provider,
            temperature=self.config.temperature,
            timestamp=datetime.now(),
        )
        
        # Create scenario metadata
        specialty = ClinicalSpecialty.GENERAL_PRACTICE
        if "specialty" in metadata:
            try:
                specialty = ClinicalSpecialty(metadata["specialty"])
            except ValueError:
                pass
        
        scenario_meta = ScenarioMetadata(
            scenario_text=scenario,
            specialty=specialty,
        )
        
        # Create difficulty metadata if present
        difficulty_meta = None
        if self.config.include_difficulty and "complexity" in metadata:
            complexity = metadata["complexity"]
            score = {"low": 3, "medium": 5, "high": 8}.get(complexity, 5)
            difficulty_meta = DifficultyMetadata.from_score(
                score,
                factors=metadata.get("key_clinical_features", []),
            )
        
        # Generate unique ID
        sample_id = f"{self.provider}_{uuid.uuid4().hex[:12]}"
        
        # Create sample
        sample = SyntheticSample(
            id=sample_id,
            dialogue=dialogue_turns,
            summary=clinical_summary,
            scenario=scenario_meta,
            generation=gen_meta,
            rag=rag_meta,
            difficulty=difficulty_meta,
            raw_response=raw_response[:2000],  # Truncate for storage
        )
        
        return sample
    
    def _extract_json(self, response: str) -> str:
        """Extract JSON from LLM response"""
        response = response.strip()
        
        # Remove markdown code blocks
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]
        
        if response.endswith("```"):
            response = response[:-3]
        
        response = response.strip()
        
        # Find JSON object boundaries
        start_idx = response.find("{")
        end_idx = response.rfind("}") + 1
        
        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON object found in response")
        
        return response[start_idx:end_idx]
    
    def _parse_dialogue(self, dialogue_data: List[Dict]) -> List[DialogueTurn]:
        """Parse dialogue list into DialogueTurn objects"""
        turns = []
        
        for i, turn in enumerate(dialogue_data):
            speaker_str = turn.get("speaker", "Doctor")
            text = turn.get("text", "")
            
            # Map speaker string to enum
            if speaker_str.lower() in ["doctor", "dr", "physician"]:
                speaker = Speaker.DOCTOR
            elif speaker_str.lower() in ["patient", "pt"]:
                speaker = Speaker.PATIENT
            else:
                speaker = Speaker.DOCTOR if i % 2 == 0 else Speaker.PATIENT
            
            turns.append(DialogueTurn(
                speaker=speaker,
                text=text,
                turn_number=i,
            ))
        
        return turns
    
    def _parse_summary(self, summary_data: Dict) -> ClinicalSummary:
        """Parse summary dict into ClinicalSummary"""
        
        # Handle SOAP note if present, or generate from other fields
        soap = None
        if "soap" in summary_data and summary_data["soap"]:
            soap_data = summary_data["soap"]
            soap = SOAPNote(
                S=soap_data.get("S", soap_data.get("subjective", "")),
                O=soap_data.get("O", soap_data.get("objective", "")),
                A=soap_data.get("A", soap_data.get("assessment", "")),
                P=soap_data.get("P", soap_data.get("plan", "")),
            )
        else:
            # Auto-generate SOAP from existing fields
            soap = self._generate_soap_from_summary(summary_data)
        
        return ClinicalSummary(
            chief_complaint=summary_data.get("chief_complaint", ""),
            history_of_present_illness=summary_data.get(
                "history_of_present_illness",
                summary_data.get("hpi", "")
            ),
            past_medical_history=summary_data.get("past_medical_history"),
            medications=summary_data.get("medications"),
            allergies=summary_data.get("allergies", "NKDA"),
            social_history=summary_data.get("social_history"),
            family_history=summary_data.get("family_history"),
            physical_examination=summary_data.get("physical_examination"),
            assessment=summary_data.get("assessment", ""),
            plan=summary_data.get("plan", ""),
            safety_netting=summary_data.get("safety_netting"),
            soap=soap,
        )
    
    def _generate_soap_from_summary(self, summary_data: Dict) -> SOAPNote:
        """
        Generate SOAP note from existing summary fields
        
        This is a fallback when the LLM doesn't provide SOAP explicitly.
        
        Args:
            summary_data: Dictionary with summary fields
            
        Returns:
            SOAPNote generated from available data
        """
        # Subjective: Combine chief complaint, HPI, PMH, meds, allergies, social/family history
        subjective_parts = []
        
        if summary_data.get("chief_complaint"):
            subjective_parts.append(f"Chief Complaint: {summary_data['chief_complaint']}")
        
        hpi = summary_data.get("history_of_present_illness") or summary_data.get("hpi", "")
        if hpi:
            subjective_parts.append(f"HPI: {hpi}")
        
        if summary_data.get("past_medical_history"):
            subjective_parts.append(f"PMH: {summary_data['past_medical_history']}")
        
        if summary_data.get("medications"):
            subjective_parts.append(f"Medications: {summary_data['medications']}")
        
        if summary_data.get("allergies"):
            subjective_parts.append(f"Allergies: {summary_data['allergies']}")
        
        if summary_data.get("social_history"):
            subjective_parts.append(f"Social History: {summary_data['social_history']}")
        
        if summary_data.get("family_history"):
            subjective_parts.append(f"Family History: {summary_data['family_history']}")
        
        subjective = " | ".join(subjective_parts) if subjective_parts else "No subjective data recorded."
        
        # Objective: Physical examination and vital signs
        objective = summary_data.get("physical_examination", "No examination findings recorded.")
        
        # Assessment: Working diagnosis
        assessment = summary_data.get("assessment", "Assessment pending.")
        
        # Plan: Treatment plan and safety netting
        plan_parts = []
        if summary_data.get("plan"):
            plan_parts.append(summary_data["plan"])
        if summary_data.get("safety_netting"):
            plan_parts.append(f"Safety Netting: {summary_data['safety_netting']}")
        
        plan = " ".join(plan_parts) if plan_parts else "Plan to be determined."
        
        return SOAPNote(
            S=subjective,
            O=objective,
            A=assessment,
            P=plan,
        )
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def _save_intermediate(
        self,
        batch_result: BatchGenerationResult,
        output_path: Path,
    ):
        """Save intermediate batch results to disk"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save samples as JSONL
        samples_path = output_path.with_suffix(".jsonl")
        with open(samples_path, "w") as f:
            for sample in batch_result.samples:
                f.write(sample.model_dump_json() + "\n")
        
        # Save summary
        summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
        with open(summary_path, "w") as f:
            json.dump(batch_result.to_summary_dict(), f, indent=2)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics"""
        total = self._total_generated + self._total_failed
        return {
            "model_name": self.model_name,
            "provider": self.provider,
            "total_generated": self._total_generated,
            "total_failed": self._total_failed,
            "success_rate": self._total_generated / total if total > 0 else 0,
            "rag_enabled": self.retriever is not None,
        }


# =============================================================================
# Utility Functions
# =============================================================================

def create_generation_config(
    temperature: float = 0.7,
    use_rag: bool = True,
    max_retries: int = 3,
    **kwargs,
) -> GenerationConfig:
    """Create a GenerationConfig with common defaults"""
    return GenerationConfig(
        temperature=temperature,
        use_rag=use_rag,
        max_retries=max_retries,
        **kwargs,
    )


if __name__ == "__main__":
    print("Base Teacher Module")
    print("=" * 60)
    print()
    print("This module provides the abstract base class for teacher models.")
    print("Subclasses implement specific LLM backends:")
    print("  - OllamaTeacher: Local Ollama models")
    print("  - OpenAITeacher: OpenAI API (GPT-4, etc.)")
    print("  - AnthropicTeacher: Anthropic API (Claude)")
    print()
    print("Key classes:")
    print("  - BaseTeacher: Abstract base with common functionality")
    print("  - GenerationConfig: Configuration for generation")
    print("  - GenerationResult: Result of a single generation")
    print("  - BatchGenerationResult: Result of batch generation")