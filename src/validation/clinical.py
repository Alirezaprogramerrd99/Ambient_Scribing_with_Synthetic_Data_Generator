"""
Clinical Validation for Synthetic Data

Validates clinical content accuracy and safety using NLP techniques.
Supports scispaCy and MedCAT for medical entity extraction.

Author: Alireza Rashidi
MSc Project: Trustworthy SLMs for Ambient Clinical Scribing
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum

from src.models import (
    SyntheticSample,
    ClinicalSummary,
    ValidationResult,
    ValidationError,
    ValidationStatus,
    ErrorSeverity,
    RED_FLAG_SYMPTOMS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Clinical Entity Types
# =============================================================================

class ClinicalEntityType(str, Enum):
    """Types of clinical entities"""
    SYMPTOM = "symptom"
    DISEASE = "disease"
    MEDICATION = "medication"
    PROCEDURE = "procedure"
    ANATOMY = "anatomy"
    LAB_TEST = "lab_test"
    LAB_VALUE = "lab_value"
    DOSAGE = "dosage"
    FREQUENCY = "frequency"
    DURATION = "duration"


@dataclass
class ClinicalEntity:
    """Extracted clinical entity"""
    
    text: str
    entity_type: ClinicalEntityType
    start_char: int = 0
    end_char: int = 0
    confidence: float = 1.0
    cui: Optional[str] = None  # UMLS Concept Unique Identifier
    snomed_code: Optional[str] = None
    
    def __str__(self) -> str:
        return f"{self.entity_type.value}: {self.text}"


@dataclass
class ClinicalValidationResult:
    """Result of clinical validation"""
    
    is_valid: bool = True
    entities_found: List[ClinicalEntity] = field(default_factory=list)
    potential_hallucinations: List[str] = field(default_factory=list)
    medication_issues: List[str] = field(default_factory=list)
    dosage_issues: List[str] = field(default_factory=list)
    red_flag_mentions: List[str] = field(default_factory=list)
    missing_safety_concerns: List[str] = field(default_factory=list)
    
    def to_validation_errors(self) -> List[ValidationError]:
        """Convert to ValidationError list"""
        errors = []
        
        for hallucination in self.potential_hallucinations:
            errors.append(ValidationError(
                field="clinical_content",
                error_type="potential_hallucination",
                message=hallucination,
                severity=ErrorSeverity.MODERATE,  # Changed from MAJOR - these are just potential issues
            ))
        
        for med_issue in self.medication_issues:
            errors.append(ValidationError(
                field="medications",
                error_type="medication_issue",
                message=med_issue,
                severity=ErrorSeverity.CRITICAL,
            ))
        
        for dosage_issue in self.dosage_issues:
            errors.append(ValidationError(
                field="dosage",
                error_type="dosage_issue",
                message=dosage_issue,
                severity=ErrorSeverity.CRITICAL,
            ))
        
        for safety_concern in self.missing_safety_concerns:
            errors.append(ValidationError(
                field="safety",
                error_type="missing_safety_concern",
                message=safety_concern,
                severity=ErrorSeverity.MAJOR,
            ))
        
        return errors


# =============================================================================
# Simple Clinical Validator (Rule-Based)
# =============================================================================

class SimpleClinicalValidator:
    """
    Rule-based clinical validator
    
    Performs basic clinical validation without requiring external NLP models.
    Good for quick validation and fallback when scispaCy/MedCAT not available.
    """
    
    # Common medications with typical dosage ranges
    MEDICATION_DOSAGES = {
        "aspirin": {"min": 75, "max": 900, "unit": "mg"},
        "paracetamol": {"min": 250, "max": 1000, "unit": "mg"},
        "acetaminophen": {"min": 250, "max": 1000, "unit": "mg"},
        "ibuprofen": {"min": 200, "max": 800, "unit": "mg"},
        "metformin": {"min": 250, "max": 1000, "unit": "mg"},
        "amlodipine": {"min": 2.5, "max": 10, "unit": "mg"},
        "lisinopril": {"min": 2.5, "max": 40, "unit": "mg"},
        "ramipril": {"min": 1.25, "max": 10, "unit": "mg"},
        "atorvastatin": {"min": 10, "max": 80, "unit": "mg"},
        "simvastatin": {"min": 10, "max": 80, "unit": "mg"},
        "omeprazole": {"min": 10, "max": 40, "unit": "mg"},
        "lansoprazole": {"min": 15, "max": 30, "unit": "mg"},
        "amoxicillin": {"min": 250, "max": 1000, "unit": "mg"},
        "doxycycline": {"min": 50, "max": 200, "unit": "mg"},
        "prednisolone": {"min": 1, "max": 60, "unit": "mg"},
        "salbutamol": {"min": 100, "max": 200, "unit": "mcg"},
        "levothyroxine": {"min": 25, "max": 200, "unit": "mcg"},
    }
    
    # Symptoms that should trigger safety netting
    SAFETY_CRITICAL_SYMPTOMS = [
        "chest pain", "shortness of breath", "severe headache",
        "loss of consciousness", "syncope", "collapse",
        "stroke", "tia", "weakness", "numbness",
        "suicidal", "self-harm", "overdose",
        "severe abdominal pain", "vomiting blood", "blood in stool",
    ]
    
    # Potentially dangerous medication combinations (simplified)
    DANGEROUS_COMBINATIONS = [
        ({"warfarin"}, {"aspirin", "ibuprofen", "naproxen"}),
        ({"methotrexate"}, {"trimethoprim", "nsaid"}),
        ({"ssri"}, {"maoi"}),
        ({"ace inhibitor", "lisinopril", "ramipril"}, {"potassium"}),
    ]
    
    def __init__(self):
        self.red_flags = [r.lower() for r in RED_FLAG_SYMPTOMS]
    
    def validate(self, sample: SyntheticSample) -> ClinicalValidationResult:
        """
        Validate clinical content of a sample
        
        Args:
            sample: SyntheticSample to validate
            
        Returns:
            ClinicalValidationResult
        """
        result = ClinicalValidationResult()
        
        # Extract text for analysis
        dialogue_text = " ".join(turn.text for turn in sample.dialogue).lower()
        summary = sample.summary
        
        # Check medications and dosages
        self._validate_medications(summary, result)
        
        # Check for red flags and safety netting
        self._validate_safety(dialogue_text, summary, result)
        
        # Check for potential hallucinations
        self._check_hallucinations(summary, result)
        
        # Determine overall validity
        result.is_valid = (
            len(result.medication_issues) == 0 and
            len(result.dosage_issues) == 0 and
            len(result.potential_hallucinations) == 0
        )
        
        return result
    
    def _validate_medications(
        self,
        summary: ClinicalSummary,
        result: ClinicalValidationResult,
    ):
        """Validate medications and dosages"""
        
        medications_text = summary.medications or ""
        plan_text = summary.plan or ""
        combined_text = f"{medications_text} {plan_text}".lower()
        
        # Extract medication mentions with dosages
        # Pattern: medication name followed by number and unit
        dosage_pattern = r'(\w+)\s+(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units?)'
        
        matches = re.findall(dosage_pattern, combined_text)
        
        for med_name, dosage_str, unit in matches:
            dosage = float(dosage_str)
            
            # Check if medication is in our database
            if med_name in self.MEDICATION_DOSAGES:
                expected = self.MEDICATION_DOSAGES[med_name]
                
                # Check unit match
                if unit != expected["unit"]:
                    result.dosage_issues.append(
                        f"{med_name}: unusual unit '{unit}' (expected {expected['unit']})"
                    )
                
                # Check dosage range
                if dosage < expected["min"] * 0.5:  # Allow some flexibility
                    result.dosage_issues.append(
                        f"{med_name} {dosage}{unit}: dosage unusually low "
                        f"(typical range: {expected['min']}-{expected['max']}{expected['unit']})"
                    )
                elif dosage > expected["max"] * 1.5:
                    result.dosage_issues.append(
                        f"{med_name} {dosage}{unit}: dosage unusually high "
                        f"(typical range: {expected['min']}-{expected['max']}{expected['unit']})"
                    )
            
            # Record as entity
            result.entities_found.append(ClinicalEntity(
                text=f"{med_name} {dosage}{unit}",
                entity_type=ClinicalEntityType.MEDICATION,
                confidence=0.8,
            ))
    
    def _validate_safety(
        self,
        dialogue_text: str,
        summary: ClinicalSummary,
        result: ClinicalValidationResult,
    ):
        """Validate safety concerns are addressed"""
        
        # Find red flags mentioned in dialogue
        mentioned_red_flags = []
        for red_flag in self.red_flags:
            if red_flag in dialogue_text:
                mentioned_red_flags.append(red_flag)
                result.red_flag_mentions.append(red_flag)
        
        # Check safety netting for red flags
        safety_text = (summary.safety_netting or "").lower()
        plan_text = (summary.plan or "").lower()
        combined_safety = f"{safety_text} {plan_text}"
        
        # Check if safety concerns are addressed
        for red_flag in mentioned_red_flags:
            # Check if there's appropriate safety netting
            safety_keywords = ["return", "seek", "emergency", "urgent", "worse", "immediately"]
            has_safety_advice = any(kw in combined_safety for kw in safety_keywords)
            
            if not has_safety_advice:
                result.missing_safety_concerns.append(
                    f"Red flag '{red_flag}' mentioned but safety netting may be inadequate"
                )
    
    def _check_hallucinations(
        self,
        summary: ClinicalSummary,
        result: ClinicalValidationResult,
    ):
        """Check for potential hallucinations"""
        
        # Check for obviously incorrect medical statements
        assessment_lower = (summary.assessment or "").lower()
        plan_lower = (summary.plan or "").lower()
        
        # Pattern checks for potentially hallucinated content
        suspicious_patterns = [
            # Contradictory statements
            (r"no .* and .* present", "Potentially contradictory statement detected"),
            # Very specific false statistics
            (r"\d{2,}% of patients", "Very specific statistic that may be hallucinated"),
            # Made-up medication names (very long or unusual)
            (r'\b[a-z]{15,}(?:mab|nib|zole|pril|sartan)\b', "Unusually long medication name"),
        ]
        
        for pattern, message in suspicious_patterns:
            if re.search(pattern, assessment_lower) or re.search(pattern, plan_lower):
                result.potential_hallucinations.append(message)
        
        # Check for internally inconsistent information
        # (This is a simplified check - could be more sophisticated)
        
        # Check if assessment mentions a condition but plan doesn't address it
        # NOTE: We're lenient here because differential diagnoses don't all need
        # to be explicitly addressed - tests/investigations can rule them out
        conditions_mentioned = self._extract_conditions(assessment_lower)
        plan_addresses = self._extract_conditions(plan_lower)
        
        # Keywords that indicate the plan is addressing differentials indirectly
        investigation_keywords = [
            "test", "x-ray", "xray", "scan", "ct", "mri", "ultrasound",
            "blood", "ecg", "ekg", "monitor", "investigate", "rule out",
            "exclude", "check", "assess", "evaluate", "refer", "follow"
        ]
        plan_has_investigation = any(kw in plan_lower for kw in investigation_keywords)
        
        for condition in conditions_mentioned:
            if condition not in plan_addresses and len(condition) > 5:
                # Check if it's at least mentioned in plan OR if plan has investigations
                if condition not in plan_lower and not plan_has_investigation:
                    result.potential_hallucinations.append(
                        f"Assessment mentions '{condition}' but plan may not address it"
                    )
    
    def _extract_conditions(self, text: str) -> Set[str]:
        """Extract condition/diagnosis mentions from text"""
        # Simple extraction based on common patterns
        conditions = set()
        
        # Common condition suffixes
        patterns = [
            r'\b\w+itis\b',      # inflammation
            r'\b\w+osis\b',      # condition
            r'\b\w+emia\b',      # blood condition
            r'\b\w+pathy\b',     # disease
            r'\b\w+algia\b',     # pain
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            conditions.update(matches)
        
        # Common specific conditions
        common_conditions = [
            "diabetes", "hypertension", "asthma", "copd", "angina",
            "infection", "fracture", "sprain", "arthritis",
        ]
        
        for condition in common_conditions:
            if condition in text:
                conditions.add(condition)
        
        return conditions


# =============================================================================
# Advanced Clinical Validator (scispaCy/MedCAT)
# =============================================================================

class AdvancedClinicalValidator:
    """
    Advanced clinical validator using scispaCy and/or MedCAT
    
    Provides more accurate entity extraction and validation using
    trained medical NLP models.
    
    Note: Requires additional installation:
        pip install scispacy
        pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_lg-0.5.3.tar.gz
    """
    
    def __init__(
        self,
        use_scispacy: bool = True,
        use_medcat: bool = False,
        scispacy_model: str = "en_core_sci_lg",
        medcat_model_path: Optional[str] = None,
    ):
        """
        Initialize advanced validator
        
        Args:
            use_scispacy: Whether to use scispaCy
            use_medcat: Whether to use MedCAT
            scispacy_model: scispaCy model name
            medcat_model_path: Path to MedCAT model pack
        """
        self.use_scispacy = use_scispacy
        self.use_medcat = use_medcat
        
        self._nlp = None
        self._medcat = None
        
        # Lazy load models
        if use_scispacy:
            self._load_scispacy(scispacy_model)
        
        if use_medcat and medcat_model_path:
            self._load_medcat(medcat_model_path)
        
        # Fall back to simple validator
        self.simple_validator = SimpleClinicalValidator()
    
    def _load_scispacy(self, model_name: str):
        """Load scispaCy model"""
        try:
            import spacy
            self._nlp = spacy.load(model_name)
            logger.info(f"Loaded scispaCy model: {model_name}")
        except OSError:
            logger.warning(
                f"scispaCy model '{model_name}' not found. "
                f"Install with: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/{model_name}-0.5.3.tar.gz"
            )
            self.use_scispacy = False
        except ImportError:
            logger.warning("scispaCy not installed. Install with: pip install scispacy")
            self.use_scispacy = False
    
    def _load_medcat(self, model_path: str):
        """Load MedCAT model"""
        try:
            from medcat.cat import CAT
            self._medcat = CAT.load_model_pack(model_path)
            logger.info(f"Loaded MedCAT model from: {model_path}")
        except ImportError:
            logger.warning("MedCAT not installed. Install with: pip install medcat")
            self.use_medcat = False
        except Exception as e:
            logger.warning(f"Failed to load MedCAT model: {e}")
            self.use_medcat = False
    
    def validate(self, sample: SyntheticSample) -> ClinicalValidationResult:
        """
        Validate clinical content using advanced NLP
        
        Args:
            sample: SyntheticSample to validate
            
        Returns:
            ClinicalValidationResult
        """
        # Start with simple validation
        result = self.simple_validator.validate(sample)
        
        # Enhance with scispaCy if available
        if self.use_scispacy and self._nlp:
            self._enhance_with_scispacy(sample, result)
        
        # Enhance with MedCAT if available
        if self.use_medcat and self._medcat:
            self._enhance_with_medcat(sample, result)
        
        return result
    
    def _enhance_with_scispacy(
        self,
        sample: SyntheticSample,
        result: ClinicalValidationResult,
    ):
        """Enhance validation with scispaCy entities"""
        
        # Process summary text
        summary_text = " ".join(filter(None, [
            sample.summary.chief_complaint,
            sample.summary.history_of_present_illness,
            sample.summary.assessment,
            sample.summary.plan,
        ]))
        
        doc = self._nlp(summary_text)
        
        # Extract entities
        for ent in doc.ents:
            entity_type = self._map_spacy_label(ent.label_)
            if entity_type:
                result.entities_found.append(ClinicalEntity(
                    text=ent.text,
                    entity_type=entity_type,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    confidence=0.9,
                ))
    
    def _enhance_with_medcat(
        self,
        sample: SyntheticSample,
        result: ClinicalValidationResult,
    ):
        """Enhance validation with MedCAT entities"""
        
        summary_text = " ".join(filter(None, [
            sample.summary.chief_complaint,
            sample.summary.history_of_present_illness,
            sample.summary.assessment,
            sample.summary.plan,
        ]))
        
        # Get MedCAT entities
        entities = self._medcat.get_entities(summary_text)
        
        for ent_id, ent_data in entities.get("entities", {}).items():
            result.entities_found.append(ClinicalEntity(
                text=ent_data.get("source_value", ""),
                entity_type=ClinicalEntityType.DISEASE,  # MedCAT focuses on conditions
                start_char=ent_data.get("start", 0),
                end_char=ent_data.get("end", 0),
                confidence=ent_data.get("acc", 0.0),
                cui=ent_data.get("cui"),
                snomed_code=ent_data.get("snomed_ct"),
            ))
    
    def _map_spacy_label(self, label: str) -> Optional[ClinicalEntityType]:
        """Map spaCy entity labels to our types"""
        mapping = {
            "DISEASE": ClinicalEntityType.DISEASE,
            "SYMPTOM": ClinicalEntityType.SYMPTOM,
            "MEDICATION": ClinicalEntityType.MEDICATION,
            "CHEMICAL": ClinicalEntityType.MEDICATION,
            "PROCEDURE": ClinicalEntityType.PROCEDURE,
            "ANATOMY": ClinicalEntityType.ANATOMY,
            "ORGAN": ClinicalEntityType.ANATOMY,
        }
        return mapping.get(label.upper())
    
    def extract_entities(self, text: str) -> List[ClinicalEntity]:
        """
        Extract clinical entities from text
        
        Args:
            text: Text to process
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        if self.use_scispacy and self._nlp:
            doc = self._nlp(text)
            for ent in doc.ents:
                entity_type = self._map_spacy_label(ent.label_)
                if entity_type:
                    entities.append(ClinicalEntity(
                        text=ent.text,
                        entity_type=entity_type,
                        start_char=ent.start_char,
                        end_char=ent.end_char,
                    ))
        
        return entities


# =============================================================================
# Factory Function
# =============================================================================

def create_clinical_validator(
    use_advanced: bool = False,
    **kwargs,
) -> SimpleClinicalValidator:
    """
    Create clinical validator
    
    Args:
        use_advanced: Whether to use advanced NLP models
        **kwargs: Additional arguments for advanced validator
        
    Returns:
        Clinical validator instance
    """
    if use_advanced:
        return AdvancedClinicalValidator(**kwargs)
    return SimpleClinicalValidator()


if __name__ == "__main__":
    print("Clinical Validation Module")
    print("=" * 60)
    
    # Test simple validator
    validator = SimpleClinicalValidator()
    
    print("\nMedication dosage checks:")
    for med, info in list(validator.MEDICATION_DOSAGES.items())[:5]:
        print(f"  {med}: {info['min']}-{info['max']} {info['unit']}")
    
    print(f"\nRed flag symptoms tracked: {len(validator.red_flags)}")
    print(f"  Examples: {validator.red_flags[:3]}")
    
    print("\n✓ Clinical validation module ready!")