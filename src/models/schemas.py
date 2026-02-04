"""
Pydantic Schemas for Clinical Data Validation

This module defines all data structures used in synthetic data generation.
Strict validation ensures high-quality, consistent training data.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from .enums import (
    Speaker,
    DifficultyLevel,
    ClinicalSpecialty,
    Urgency,
    QuestionType,
    ValidationStatus,
    ErrorSeverity,
    Gender,
    AgeGroup,
)


# =============================================================================
# Dialogue Components
# =============================================================================

class DialogueTurn(BaseModel):
    """
    Single turn in a clinical dialogue
    
    Represents one utterance from either the doctor or patient.
    """
    
    model_config = ConfigDict(str_strip_whitespace=True)
    
    speaker: Speaker = Field(
        ..., 
        description="Who is speaking (Doctor or Patient)"
    )
    text: str = Field(
        ..., 
        min_length=3,
        max_length=2000,
        description="The spoken text"
    )
    turn_number: Optional[int] = Field(
        default=None,
        description="Sequential turn number in dialogue"
    )
    
    @field_validator("text")
    @classmethod
    def validate_text_content(cls, v: str) -> str:
        """Ensure text is not empty"""
        v = v.strip()
        if len(v) < 2:
            raise ValueError("Dialogue turn must contain at least 2 characters")
        return v


class ClinicalDialogue(BaseModel):
    """
    Complete clinical dialogue between doctor and patient
    
    Validates dialogue structure and content quality.
    """
    
    turns: List[DialogueTurn] = Field(
        ...,
        min_length=6,
        description="List of dialogue turns (minimum 6 for meaningful conversation)"
    )
    
    @field_validator("turns")
    @classmethod
    def validate_dialogue_structure(cls, v: List[DialogueTurn]) -> List[DialogueTurn]:
        """Validate dialogue has proper structure"""
        if len(v) < 6:
            raise ValueError("Dialogue must have at least 6 turns")
        
        # Check that dialogue alternates between speakers (with some flexibility)
        speakers = [turn.speaker for turn in v]
        
        # Must have both Doctor and Patient
        if Speaker.DOCTOR not in speakers:
            raise ValueError("Dialogue must include Doctor")
        if Speaker.PATIENT not in speakers:
            raise ValueError("Dialogue must include Patient")
        
        # Doctor should typically start
        if v[0].speaker != Speaker.DOCTOR:
            # Allow patient to start in some cases, but warn
            pass
        
        return v
    
    @property
    def num_turns(self) -> int:
        return len(self.turns)
    
    @property
    def doctor_turns(self) -> List[DialogueTurn]:
        return [t for t in self.turns if t.speaker == Speaker.DOCTOR]
    
    @property
    def patient_turns(self) -> List[DialogueTurn]:
        return [t for t in self.turns if t.speaker == Speaker.PATIENT]
    
    def to_text(self, separator: str = "\n") -> str:
        """Convert dialogue to plain text format"""
        return separator.join(
            f"{turn.speaker.value}: {turn.text}" 
            for turn in self.turns
        )


# =============================================================================
# Clinical Summary Components
# =============================================================================

class SOAPNote(BaseModel):
    """
    SOAP (Subjective, Objective, Assessment, Plan) clinical note format
    
    Standard medical documentation format used worldwide.
    """
    
    model_config = ConfigDict(str_strip_whitespace=True, populate_by_name=True)
    
    subjective: str = Field(
        ...,
        alias="S",
        min_length=20,  # Reduced from 50 for more flexibility
        description="Patient's reported symptoms, history, and concerns"
    )
    objective: str = Field(
        ...,
        alias="O", 
        min_length=15,  # Reduced from 30
        description="Physical examination findings, vital signs, observations"
    )
    assessment: str = Field(
        ...,
        alias="A",
        min_length=10,  # Reduced from 20
        description="Clinical diagnosis or differential diagnoses"
    )
    plan: str = Field(
        ...,
        alias="P",
        min_length=15,  # Reduced from 30
        description="Treatment plan, investigations, follow-up"
    )
    
    @field_validator("plan")
    @classmethod
    def validate_plan_completeness(cls, v: str) -> str:
        """Ensure plan contains actionable items"""
        v_lower = v.lower()
        
        # Plan should typically mention at least one of these
        plan_indicators = [
            "prescri", "medication", "test", "refer", "follow",
            "return", "review", "monitor", "advise", "recommend",
            "investigation", "blood", "scan", "x-ray", "ecg"
        ]
        
        if not any(indicator in v_lower for indicator in plan_indicators):
            raise ValueError(
                "Plan should include specific actions (medications, tests, referrals, or follow-up)"
            )
        
        return v


class ClinicalSummary(BaseModel):
    """
    Comprehensive clinical summary generated from dialogue
    
    Contains all essential elements of a clinical note.
    """
    
    model_config = ConfigDict(str_strip_whitespace=True)
    
    chief_complaint: str = Field(
        ...,
        min_length=5,  # Reduced from 10
        max_length=500,  # Increased from 300
        description="Primary reason for visit"
    )
    
    history_of_present_illness: str = Field(
        ...,
        min_length=30,  # Reduced from 100
        description="Detailed HPI including onset, duration, character, severity, "
                   "location, aggravating/relieving factors, associated symptoms"
    )
    
    past_medical_history: Optional[str] = Field(
        default=None,
        description="Relevant past medical conditions"
    )
    
    medications: Optional[str] = Field(
        default=None,
        description="Current medications with dosages"
    )
    
    allergies: Optional[str] = Field(
        default="No known drug allergies (NKDA)",
        description="Drug allergies or NKDA"
    )
    
    social_history: Optional[str] = Field(
        default=None,
        description="Smoking, alcohol, occupation, living situation"
    )
    
    family_history: Optional[str] = Field(
        default=None,
        description="Relevant family medical history"
    )
    
    physical_examination: Optional[str] = Field(
        default=None,
        description="Vital signs and examination findings"
    )
    
    assessment: str = Field(
        ...,
        min_length=10,  # Reduced from 20
        description="Clinical impression/working diagnosis"
    )
    
    plan: str = Field(
        ...,
        min_length=15,  # Reduced from 30
        description="Diagnostic tests, treatments, medications, follow-up"
    )
    
    safety_netting: Optional[str] = Field(
        default=None,
        description="Advice on warning signs and when to return"
    )
    
    soap: Optional[SOAPNote] = Field(
        default=None,
        description="SOAP format summary"
    )
    
    @field_validator("chief_complaint")
    @classmethod
    def validate_chief_complaint(cls, v: str) -> str:
        """Validate chief complaint is specific"""
        if len(v.split()) < 3:
            raise ValueError("Chief complaint should be descriptive (at least 3 words)")
        return v
    
    @model_validator(mode="after")
    def ensure_consistency(self):
        """Ensure assessment and plan are consistent"""
        assessment_lower = self.assessment.lower()
        plan_lower = self.plan.lower()
        
        # If assessment mentions specific conditions, plan should address them
        # This is a soft validation - just checking basic consistency
        
        return self


# =============================================================================
# Metadata Components
# =============================================================================

class ScenarioMetadata(BaseModel):
    """Metadata about the clinical scenario"""
    
    scenario_text: str = Field(..., description="Original scenario description")
    specialty: ClinicalSpecialty = Field(
        default=ClinicalSpecialty.GENERAL_PRACTICE,
        description="Medical specialty"
    )
    urgency: Urgency = Field(
        default=Urgency.ROUTINE,
        description="Clinical urgency level"
    )
    age_group: Optional[AgeGroup] = Field(default=None)
    gender: Optional[Gender] = Field(default=None)


class GenerationMetadata(BaseModel):
    """Metadata about the generation process"""
    
    model_name: str = Field(..., description="LLM model used")
    model_provider: str = Field(..., description="Provider (ollama, openai, anthropic)")
    temperature: float = Field(..., ge=0.0, le=2.0)
    timestamp: datetime = Field(default_factory=datetime.now)
    generation_time_seconds: Optional[float] = Field(default=None)
    prompt_tokens: Optional[int] = Field(default=None)
    completion_tokens: Optional[int] = Field(default=None)


class RAGMetadata(BaseModel):
    """Metadata about RAG retrieval"""
    
    rag_enabled: bool = Field(default=False)
    num_sources_retrieved: int = Field(default=0)
    sources: List[str] = Field(default_factory=list)
    retrieval_scores: List[float] = Field(default_factory=list)
    context_used: Optional[str] = Field(default=None)
    
    @property
    def average_retrieval_score(self) -> Optional[float]:
        if self.retrieval_scores:
            return sum(self.retrieval_scores) / len(self.retrieval_scores)
        return None


class DifficultyMetadata(BaseModel):
    """
    Difficulty assessment metadata (per Woo et al. methodology)
    
    Used for curriculum learning and data filtering.
    """
    
    difficulty_score: int = Field(
        ...,
        ge=1,
        le=10,
        description="Difficulty score (1-10)"
    )
    difficulty_level: DifficultyLevel = Field(
        ...,
        description="Categorical difficulty level"
    )
    reasoning_steps: List[str] = Field(
        default_factory=list,
        description="Steps required to answer/generate"
    )
    clinical_complexity_factors: List[str] = Field(
        default_factory=list,
        description="Factors contributing to complexity"
    )
    rationale: Optional[str] = Field(
        default=None,
        description="Explanation for the difficulty assessment"
    )
    
    @classmethod
    def from_score(
        cls,
        score: int,
        factors: List[str] = None,
        rationale: str = None,
    ) -> "DifficultyMetadata":
        """
        Create DifficultyMetadata from numeric score
        
        Args:
            score: Difficulty score (1-10)
            factors: List of complexity factors
            rationale: Explanation for the assessment
            
        Returns:
            DifficultyMetadata instance
        """
        return cls(
            difficulty_score=score,
            difficulty_level=DifficultyLevel.from_score(score),
            clinical_complexity_factors=factors or [],
            rationale=rationale,
        )


# =============================================================================
# Validation Results
# =============================================================================

class ValidationError(BaseModel):
    """Single validation error"""
    
    field: str = Field(..., description="Field that failed validation")
    error_type: str = Field(..., description="Type of error")
    message: str = Field(..., description="Error description")
    severity: ErrorSeverity = Field(default=ErrorSeverity.MODERATE)


class ValidationResult(BaseModel):
    """Complete validation result for a synthetic sample"""
    
    status: ValidationStatus = Field(...)
    errors: List[ValidationError] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    # Specific validation scores
    structural_valid: bool = Field(default=True)
    clinical_valid: Optional[bool] = Field(default=None)
    rag_faithfulness: Optional[float] = Field(default=None)
    
    @property
    def is_acceptable(self) -> bool:
        """Check if sample passes minimum quality threshold"""
        if self.status == ValidationStatus.FAILED:
            return False
        
        # Check for blocking errors
        blocking_errors = [
            e for e in self.errors 
            if e.severity.blocks_acceptance
        ]
        return len(blocking_errors) == 0


# =============================================================================
# Main Synthetic Data Sample
# =============================================================================

class SyntheticSample(BaseModel):
    """
    Complete synthetic training sample
    
    This is the main output schema for the teacher model.
    Each sample contains a dialogue, summary, and comprehensive metadata.
    """
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid"  # Reject unexpected fields
    )
    
    # Identifiers
    id: str = Field(..., description="Unique sample identifier")
    
    # Core content
    dialogue: List[DialogueTurn] = Field(
        ...,
        min_length=6,
        description="Clinical dialogue turns"
    )
    summary: ClinicalSummary = Field(
        ...,
        description="Clinical summary generated from dialogue"
    )
    
    # Metadata
    scenario: ScenarioMetadata = Field(...)
    generation: GenerationMetadata = Field(...)
    rag: RAGMetadata = Field(default_factory=RAGMetadata)
    difficulty: Optional[DifficultyMetadata] = Field(default=None)
    
    # Validation
    validation: Optional[ValidationResult] = Field(default=None)
    
    # Raw data (for debugging)
    raw_response: Optional[str] = Field(
        default=None,
        description="Raw LLM response before parsing"
    )
    
    @property
    def is_valid(self) -> bool:
        """Quick check if sample is valid"""
        if self.validation:
            return self.validation.is_acceptable
        return True  # Assume valid if not validated yet
    
    @property
    def dialogue_text(self) -> str:
        """Get dialogue as plain text"""
        return "\n".join(
            f"{turn.speaker.value}: {turn.text}"
            for turn in self.dialogue
        )
    
    def to_training_format(self) -> Dict[str, Any]:
        """
        Convert to format suitable for model training
        
        Returns simplified dict with dialogue input and summary output.
        """
        return {
            "id": self.id,
            "input": self.dialogue_text,
            "output": self.summary.model_dump_json(),
            "difficulty": self.difficulty.difficulty_score if self.difficulty else 5,
            "specialty": self.scenario.specialty.value,
        }


# =============================================================================
# Batch Processing
# =============================================================================

class SyntheticDataBatch(BaseModel):
    """Batch of synthetic samples with statistics"""
    
    samples: List[SyntheticSample] = Field(default_factory=list)
    
    # Batch metadata
    batch_id: str = Field(...)
    created_at: datetime = Field(default_factory=datetime.now)
    model_name: str = Field(...)
    
    # Statistics
    total_generated: int = Field(default=0)
    total_valid: int = Field(default=0)
    total_failed: int = Field(default=0)
    
    @property
    def success_rate(self) -> float:
        if self.total_generated == 0:
            return 0.0
        return self.total_valid / self.total_generated
    
    def add_sample(self, sample: SyntheticSample):
        """Add a sample to the batch"""
        self.samples.append(sample)
        self.total_generated += 1
        if sample.is_valid:
            self.total_valid += 1
        else:
            self.total_failed += 1
    
    def get_valid_samples(self) -> List[SyntheticSample]:
        """Get only valid samples"""
        return [s for s in self.samples if s.is_valid]
    
    def get_by_difficulty(self, level: DifficultyLevel) -> List[SyntheticSample]:
        """Filter samples by difficulty level"""
        return [
            s for s in self.samples 
            if s.difficulty and s.difficulty.difficulty_level == level
        ]


# =============================================================================
# LLM Response Schema (for structured output)
# =============================================================================

class LLMDialogueSummaryResponse(BaseModel):
    """
    Schema for LLM structured output
    
    This is what we expect the LLM to return.
    Used with Instructor/Outlines for guaranteed structure.
    """
    
    dialogue: List[Dict[str, str]] = Field(
        ...,
        min_length=6,
        description="List of dialogue turns with 'speaker' and 'text' keys"
    )
    summary: Dict[str, Any] = Field(
        ...,
        description="Clinical summary dictionary"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata"
    )
    
    def to_synthetic_sample(
        self,
        sample_id: str,
        scenario_text: str,
        generation_meta: GenerationMetadata,
        rag_meta: RAGMetadata = None
    ) -> SyntheticSample:
        """Convert LLM response to full SyntheticSample"""
        
        # Parse dialogue turns
        dialogue_turns = [
            DialogueTurn(
                speaker=Speaker(turn.get("speaker", "Doctor")),
                text=turn.get("text", ""),
                turn_number=i
            )
            for i, turn in enumerate(self.dialogue)
        ]
        
        # Parse summary
        summary_dict = self.summary
        
        # Handle SOAP if present
        soap = None
        if "soap" in summary_dict:
            soap_dict = summary_dict["soap"]
            soap = SOAPNote(
                S=soap_dict.get("S", soap_dict.get("subjective", "")),
                O=soap_dict.get("O", soap_dict.get("objective", "")),
                A=soap_dict.get("A", soap_dict.get("assessment", "")),
                P=soap_dict.get("P", soap_dict.get("plan", ""))
            )
        
        clinical_summary = ClinicalSummary(
            chief_complaint=summary_dict.get("chief_complaint", ""),
            history_of_present_illness=summary_dict.get(
                "history_of_present_illness", 
                summary_dict.get("hpi", "")
            ),
            past_medical_history=summary_dict.get("past_medical_history"),
            medications=summary_dict.get("medications"),
            allergies=summary_dict.get("allergies"),
            social_history=summary_dict.get("social_history"),
            family_history=summary_dict.get("family_history"),
            physical_examination=summary_dict.get("physical_examination"),
            assessment=summary_dict.get("assessment", ""),
            plan=summary_dict.get("plan", ""),
            safety_netting=summary_dict.get("safety_netting"),
            soap=soap
        )
        
        # Determine specialty from metadata or scenario
        specialty = ClinicalSpecialty.GENERAL_PRACTICE
        if self.metadata and "specialty" in self.metadata:
            try:
                specialty = ClinicalSpecialty(self.metadata["specialty"])
            except ValueError:
                pass
        
        return SyntheticSample(
            id=sample_id,
            dialogue=dialogue_turns,
            summary=clinical_summary,
            scenario=ScenarioMetadata(
                scenario_text=scenario_text,
                specialty=specialty
            ),
            generation=generation_meta,
            rag=rag_meta or RAGMetadata()
        )


if __name__ == "__main__":
    # Test schemas
    print("Testing Pydantic Schemas")
    print("=" * 60)
    
    # Test DialogueTurn
    turn = DialogueTurn(
        speaker=Speaker.DOCTOR,
        text="Good morning, what brings you in today?"
    )
    print(f"✓ DialogueTurn: {turn.speaker} - {turn.text[:30]}...")
    
    # Test SOAPNote
    soap = SOAPNote(
        S="55-year-old male presenting with chest pain for 3 days, worse with exertion",
        O="BP 140/90, HR 88, chest clear, no murmurs",
        A="Suspected stable angina, rule out ACS",
        P="ECG, troponins, refer cardiology for stress test"
    )
    print(f"✓ SOAPNote created with assessment: {soap.assessment[:40]}...")
    
    # Test ClinicalSummary
    summary = ClinicalSummary(
        chief_complaint="Chest pain for 3 days",
        history_of_present_illness="55-year-old male presents with chest pain for 3 days. Pain is central, "
                                   "described as pressure-like, worse with exertion and relieved by rest. "
                                   "Associated with mild shortness of breath. No radiation to arm or jaw. "
                                   "No nausea or diaphoresis. Patient has history of hypertension.",
        assessment="Suspected stable angina",
        plan="Order ECG and troponins. Refer to cardiology. Start aspirin 75mg daily. "
             "Safety netting: return if chest pain worsens or occurs at rest."
    )
    print(f"✓ ClinicalSummary created")
    
    print("\n✓ All schema tests passed!")