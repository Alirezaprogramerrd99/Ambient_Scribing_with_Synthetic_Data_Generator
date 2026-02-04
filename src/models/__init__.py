"""
Models Package

Contains Pydantic schemas and enumerations for clinical data validation.
"""

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
    SymptomDuration,
    RAGRetrievalMode,
    OutputFormat,
    RED_FLAG_SYMPTOMS,
    COMMON_COMORBIDITIES,
)

from .schemas import (
    # Dialogue
    DialogueTurn,
    ClinicalDialogue,
    # Summary
    SOAPNote,
    ClinicalSummary,
    # Metadata
    ScenarioMetadata,
    GenerationMetadata,
    RAGMetadata,
    DifficultyMetadata,
    # Validation
    ValidationError,
    ValidationResult,
    # Main
    SyntheticSample,
    SyntheticDataBatch,
    LLMDialogueSummaryResponse,
)

__all__ = [
    # Enums
    "Speaker",
    "DifficultyLevel", 
    "ClinicalSpecialty",
    "Urgency",
    "QuestionType",
    "ValidationStatus",
    "ErrorSeverity",
    "Gender",
    "AgeGroup",
    "SymptomDuration",
    "RAGRetrievalMode",
    "OutputFormat",
    "RED_FLAG_SYMPTOMS",
    "COMMON_COMORBIDITIES",
    # Schemas
    "DialogueTurn",
    "ClinicalDialogue",
    "SOAPNote",
    "ClinicalSummary",
    "ScenarioMetadata",
    "GenerationMetadata",
    "RAGMetadata",
    "DifficultyMetadata",
    "ValidationError",
    "ValidationResult",
    "SyntheticSample",
    "SyntheticDataBatch",
    "LLMDialogueSummaryResponse",
]