"""
Clinical Enumerations

Type-safe enumerations for clinical concepts used throughout the project.
These ensure consistency and prevent typos in clinical terminology.

Author: Alireza Rashidi
MSc Project: Trustworthy SLMs for Ambient Clinical Scribing
"""

from enum import Enum, auto
from typing import List


class Speaker(str, Enum):
    """Speaker roles in clinical dialogue"""
    # other fields are for further categorization if needed
    DOCTOR = "Doctor"
    PATIENT = "Patient"
    NURSE = "Nurse"
    FAMILY_MEMBER = "Family Member"
    
    @classmethod
    def clinical_participants(cls) -> List["Speaker"]:
        """Return main clinical dialogue participants"""
        return [cls.DOCTOR, cls.PATIENT]


class DifficultyLevel(str, Enum):
    """Complexity level for synthetic data (per Woo et al. methodology)"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    
    @classmethod
    def from_score(cls, score: int) -> "DifficultyLevel":
        """Convert numeric score (1-10) to difficulty level"""
        if score <= 3:
            return cls.LOW
        elif score <= 6:
            return cls.MEDIUM
        else:
            return cls.HIGH


class ClinicalSpecialty(str, Enum):
    """Medical specialties for categorization"""
    GENERAL_PRACTICE = "General Practice"
    CARDIOLOGY = "Cardiology"
    RESPIRATORY = "Respiratory"
    GASTROENTEROLOGY = "Gastroenterology"
    NEUROLOGY = "Neurology"
    ENDOCRINOLOGY = "Endocrinology"
    MUSCULOSKELETAL = "Musculoskeletal"
    DERMATOLOGY = "Dermatology"
    PSYCHIATRY = "Psychiatry"
    INFECTIOUS_DISEASE = "Infectious Disease"
    UROLOGY = "Urology"
    GYNECOLOGY = "Gynecology"
    PEDIATRICS = "Pediatrics"
    GERIATRICS = "Geriatrics"
    EMERGENCY = "Emergency Medicine"
    ONCOLOGY = "Oncology"
    NEPHROLOGY = "Nephrology"
    RHEUMATOLOGY = "Rheumatology"
    OPHTHALMOLOGY = "Ophthalmology"
    ENT = "ENT"
    
    @classmethod
    def primary_care_relevant(cls) -> List["ClinicalSpecialty"]:
        """Return specialties commonly seen in primary care"""
        return [
            cls.GENERAL_PRACTICE,
            cls.CARDIOLOGY,
            cls.RESPIRATORY,
            cls.GASTROENTEROLOGY,
            cls.MUSCULOSKELETAL,
            cls.DERMATOLOGY,
            cls.PSYCHIATRY,
            cls.ENDOCRINOLOGY,
        ]


class Urgency(str, Enum):
    """Clinical urgency levels"""
    ROUTINE = "routine"
    URGENT = "urgent"
    EMERGENCY = "emergency"
    
    @property
    def requires_immediate_action(self) -> bool:
        return self in [Urgency.URGENT, Urgency.EMERGENCY]


class QuestionType(str, Enum): # I put this as well for later comparison with QA pairs for writing a paper.
    """
    Types of clinical questions (per Woo et al. methodology)
    
    Used for categorizing synthetic Q&A pairs during training data generation.
    """
    BOOLEAN = "boolean"           # Yes/No answers
    NUMERIC = "numeric"           # Numerical values (lab results, vitals)
    CATEGORICAL = "categorical"   # Multiple choice categories
    FREE_TEXT = "free_text"       # Open-ended responses
    NA_BOOLEAN = "na_boolean"     # Boolean where answer cannot be determined
    NA_NUMERIC = "na_numeric"     # Numeric where value not in notes
    
    @property
    def is_answerable(self) -> bool:
        """Check if question type expects a definitive answer"""
        return self not in [QuestionType.NA_BOOLEAN, QuestionType.NA_NUMERIC]


class ValidationStatus(str, Enum):
    """Status of validation checks"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


class ErrorSeverity(str, Enum):
    """
    Severity levels for clinical errors in generated content
    
    Based on clinical safety assessment frameworks.
    """
    MINOR = "minor"           # Grammatical, formatting issues
    MODERATE = "moderate"     # Missing non-critical information
    MAJOR = "major"           # Incorrect clinical information
    CRITICAL = "critical"     # Potentially harmful errors (wrong medication, dose)
    
    @property
    def blocks_acceptance(self) -> bool:
        """Whether this error severity should reject the sample"""
        return self in [ErrorSeverity.MAJOR, ErrorSeverity.CRITICAL]


class Gender(str, Enum):
    """Patient gender for scenario generation"""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    NOT_SPECIFIED = "not_specified"


class AgeGroup(str, Enum):
    """Age groups for scenario generation"""
    PEDIATRIC = "pediatric"       # 0-17
    YOUNG_ADULT = "young_adult"   # 18-35
    MIDDLE_AGED = "middle_aged"   # 36-55
    OLDER_ADULT = "older_adult"   # 56-75
    ELDERLY = "elderly"           # 76+
    
    @classmethod
    def from_age(cls, age: int) -> "AgeGroup":
        """Convert numeric age to age group"""
        if age < 18:
            return cls.PEDIATRIC
        elif age <= 35:
            return cls.YOUNG_ADULT
        elif age <= 55:
            return cls.MIDDLE_AGED
        elif age <= 75:
            return cls.OLDER_ADULT
        else:
            return cls.ELDERLY
    
    def typical_age_range(self) -> tuple:
        """Return typical age range for this group"""
        ranges = {
            AgeGroup.PEDIATRIC: (0, 17),
            AgeGroup.YOUNG_ADULT: (18, 35),
            AgeGroup.MIDDLE_AGED: (36, 55),
            AgeGroup.OLDER_ADULT: (56, 75),
            AgeGroup.ELDERLY: (76, 95),
        }
        return ranges[self]


class SymptomDuration(str, Enum):
    """Duration categories for symptoms"""
    ACUTE = "acute"           # < 24 hours
    SUBACUTE = "subacute"     # 1-7 days
    SHORT_TERM = "short_term" # 1-4 weeks
    CHRONIC = "chronic"       # > 4 weeks
    
    @classmethod
    def from_description(cls, duration_str: str) -> "SymptomDuration":
        """Parse duration string to category"""
        duration_lower = duration_str.lower()
        
        if any(x in duration_lower for x in ["hour", "today", "just"]):
            return cls.ACUTE
        elif any(x in duration_lower for x in ["day", "days"]) and not "week" in duration_lower:
            days = 1  # Default
            try:
                # Try to extract number of days
                import re
                match = re.search(r'(\d+)\s*day', duration_lower)
                if match:
                    days = int(match.group(1))
            except:
                pass
            return cls.ACUTE if days < 1 else cls.SUBACUTE
        elif any(x in duration_lower for x in ["week", "weeks"]):
            return cls.SHORT_TERM
        elif any(x in duration_lower for x in ["month", "year"]):
            return cls.CHRONIC
        else:
            return cls.SUBACUTE  # Default


class RAGRetrievalMode(str, Enum):
    """RAG retrieval strategies"""
    DENSE = "dense"               # Semantic similarity only
    SPARSE = "sparse"             # BM25/keyword only  
    HYBRID = "hybrid"             # Combined dense + sparse
    RERANKED = "reranked"         # Hybrid with cross-encoder reranking


class OutputFormat(str, Enum):
    """Output formats for synthetic data"""
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"


# Red flag symptoms that require immediate attention
RED_FLAG_SYMPTOMS = [
    "chest pain",
    "shortness of breath",
    "sudden severe headache",
    "slurred speech",
    "facial drooping",
    "arm weakness",
    "loss of consciousness",
    "severe abdominal pain",
    "vomiting blood",
    "blood in stool",
    "suicidal thoughts",
    "severe allergic reaction",
    "difficulty breathing",
    "seizure",
    "severe bleeding",
]

# Common comorbidities for scenario generation
COMMON_COMORBIDITIES = [
    "hypertension",
    "type 2 diabetes",
    "hyperlipidemia",
    "obesity",
    "asthma",
    "COPD",
    "coronary artery disease",
    "heart failure",
    "atrial fibrillation",
    "chronic kidney disease",
    "depression",
    "anxiety",
    "osteoarthritis",
    "hypothyroidism",
]


if __name__ == "__main__":
    # Test enums
    print("Testing Clinical Enums")
    print("=" * 40)
    
    # Test DifficultyLevel
    print(f"\nDifficulty from score 2: {DifficultyLevel.from_score(2)}")
    print(f"Difficulty from score 5: {DifficultyLevel.from_score(5)}")
    print(f"Difficulty from score 8: {DifficultyLevel.from_score(8)}")
    
    # Test AgeGroup
    print(f"\nAge 25 -> {AgeGroup.from_age(25)}")
    print(f"Age 65 -> {AgeGroup.from_age(65)}")
    print(f"Age 82 -> {AgeGroup.from_age(82)}")
    
    # Test SymptomDuration
    print(f"\n'2 hours' -> {SymptomDuration.from_description('2 hours')}")
    print(f"'3 days' -> {SymptomDuration.from_description('3 days')}")
    print(f"'2 weeks' -> {SymptomDuration.from_description('2 weeks')}")
    print(f"'3 months' -> {SymptomDuration.from_description('3 months')}")
    
    # Test error severity
    print(f"\nMINOR blocks acceptance: {ErrorSeverity.MINOR.blocks_acceptance}")
    print(f"CRITICAL blocks acceptance: {ErrorSeverity.CRITICAL.blocks_acceptance}")