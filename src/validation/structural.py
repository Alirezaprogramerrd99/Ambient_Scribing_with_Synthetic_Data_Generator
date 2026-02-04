"""
Structural Validation for Synthetic Clinical Data

Validates the structure and completeness of generated dialogue-summary pairs.
Uses Pydantic models and custom rules to ensure data quality.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError as PydanticValidationError

from src.models import (
    DialogueTurn,
    ClinicalSummary,
    SOAPNote,
    SyntheticSample,
    ValidationResult,
    ValidationError,
    ValidationStatus,
    ErrorSeverity,
    Speaker,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Validation Rules Configuration
# =============================================================================

@dataclass
class ValidationRules:
    """Configurable validation rules"""
    
    # Dialogue rules
    min_dialogue_turns: int = 6
    max_dialogue_turns: int = 50
    min_words_per_turn: int = 3
    max_words_per_turn: int = 500
    require_doctor_start: bool = False  # Flexible - patient can start
    require_both_speakers: bool = True
    max_consecutive_same_speaker: int = 3
    
    # Summary rules
    min_chief_complaint_words: int = 3
    min_hpi_words: int = 20
    min_assessment_words: int = 5
    min_plan_words: int = 10
    require_safety_netting: bool = False
    
    # Content rules
    require_soap: bool = False
    check_clinical_terminology: bool = True
    check_plan_actionability: bool = True
    
    # Consistency rules
    check_dialogue_summary_consistency: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_dialogue_turns": self.min_dialogue_turns,
            "max_dialogue_turns": self.max_dialogue_turns,
            "min_words_per_turn": self.min_words_per_turn,
            "require_both_speakers": self.require_both_speakers,
            "min_hpi_words": self.min_hpi_words,
            "check_clinical_terminology": self.check_clinical_terminology,
        }


# =============================================================================
# Structural Validator
# =============================================================================

class StructuralValidator:
    """
    Validates structural integrity of synthetic samples
    
    Checks:
    - Dialogue structure and turn validity
    - Summary completeness and field lengths
    - SOAP note structure (if present)
    - Basic content quality
    
    Example:
        validator = StructuralValidator()
        result = validator.validate(sample)
        
        if result.is_acceptable:
            print("Sample passed validation")
        else:
            for error in result.errors:
                print(f"{error.severity}: {error.message}")
    """
    
    def __init__(self, rules: Optional[ValidationRules] = None):
        """
        Initialize validator
        
        Args:
            rules: Validation rules (uses defaults if not provided)
        """
        self.rules = rules or ValidationRules()
    
    def validate(self, sample: SyntheticSample) -> ValidationResult:
        """
        Validate a synthetic sample
        
        Args:
            sample: SyntheticSample to validate
            
        Returns:
            ValidationResult with status and errors
        """
        errors: List[ValidationError] = []
        warnings: List[str] = []
        
        # Validate dialogue
        dialogue_errors, dialogue_warnings = self._validate_dialogue(sample.dialogue)
        errors.extend(dialogue_errors)
        warnings.extend(dialogue_warnings)
        
        # Validate summary
        summary_errors, summary_warnings = self._validate_summary(sample.summary)
        errors.extend(summary_errors)
        warnings.extend(summary_warnings)
        
        # Validate SOAP if present
        if sample.summary.soap:
            soap_errors, soap_warnings = self._validate_soap(sample.summary.soap)
            errors.extend(soap_errors)
            warnings.extend(soap_warnings)
        
        # Check consistency between dialogue and summary
        if self.rules.check_dialogue_summary_consistency:
            consistency_errors = self._check_consistency(sample)
            errors.extend(consistency_errors)
        
        # Determine overall status
        has_critical = any(e.severity == ErrorSeverity.CRITICAL for e in errors)
        has_major = any(e.severity == ErrorSeverity.MAJOR for e in errors)
        
        if has_critical:
            status = ValidationStatus.FAILED
        elif has_major:
            status = ValidationStatus.WARNING
        elif errors:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.PASSED
        
        return ValidationResult(
            status=status,
            errors=errors,
            warnings=warnings,
            structural_valid=not has_critical and not has_major,
        )
    
    def _validate_dialogue(
        self,
        dialogue: List[DialogueTurn],
    ) -> Tuple[List[ValidationError], List[str]]:
        """Validate dialogue structure"""
        errors = []
        warnings = []
        
        # Check turn count
        num_turns = len(dialogue)
        
        if num_turns < self.rules.min_dialogue_turns:
            errors.append(ValidationError(
                field="dialogue",
                error_type="insufficient_turns",
                message=f"Dialogue has {num_turns} turns, minimum is {self.rules.min_dialogue_turns}",
                severity=ErrorSeverity.MAJOR,
            ))
        
        if num_turns > self.rules.max_dialogue_turns:
            warnings.append(
                f"Dialogue has {num_turns} turns, which exceeds typical maximum of {self.rules.max_dialogue_turns}"
            )
        
        # Check speakers
        speakers = [turn.speaker for turn in dialogue]
        unique_speakers = set(speakers)
        
        if self.rules.require_both_speakers:
            if Speaker.DOCTOR not in unique_speakers:
                errors.append(ValidationError(
                    field="dialogue",
                    error_type="missing_speaker",
                    message="Dialogue is missing Doctor speaker",
                    severity=ErrorSeverity.CRITICAL,
                ))
            
            if Speaker.PATIENT not in unique_speakers:
                errors.append(ValidationError(
                    field="dialogue",
                    error_type="missing_speaker",
                    message="Dialogue is missing Patient speaker",
                    severity=ErrorSeverity.CRITICAL,
                ))
        
        # Check for excessive consecutive same speaker
        consecutive_count = 1
        for i in range(1, len(speakers)):
            if speakers[i] == speakers[i - 1]:
                consecutive_count += 1
                if consecutive_count > self.rules.max_consecutive_same_speaker:
                    warnings.append(
                        f"Turn {i}: {consecutive_count} consecutive turns by {speakers[i].value}"
                    )
            else:
                consecutive_count = 1
        
        # Validate individual turns
        for i, turn in enumerate(dialogue):
            turn_errors = self._validate_turn(turn, i)
            errors.extend(turn_errors)
        
        return errors, warnings
    
    def _validate_turn(
        self,
        turn: DialogueTurn,
        turn_index: int,
    ) -> List[ValidationError]:
        """Validate a single dialogue turn"""
        errors = []
        
        # Check text content
        text = turn.text.strip()
        word_count = len(text.split())
        
        if word_count < self.rules.min_words_per_turn:
            errors.append(ValidationError(
                field=f"dialogue[{turn_index}]",
                error_type="insufficient_content",
                message=f"Turn {turn_index} has only {word_count} words (minimum: {self.rules.min_words_per_turn})",
                severity=ErrorSeverity.MODERATE,
            ))
        
        if word_count > self.rules.max_words_per_turn:
            errors.append(ValidationError(
                field=f"dialogue[{turn_index}]",
                error_type="excessive_content",
                message=f"Turn {turn_index} has {word_count} words (maximum: {self.rules.max_words_per_turn})",
                severity=ErrorSeverity.MINOR,
            ))
        
        # Check for empty or placeholder text
        if not text or text.lower() in ["...", "n/a", "none", ""]:
            errors.append(ValidationError(
                field=f"dialogue[{turn_index}]",
                error_type="empty_content",
                message=f"Turn {turn_index} has empty or placeholder content",
                severity=ErrorSeverity.MAJOR,
            ))
        
        return errors
    
    def _validate_summary(
        self,
        summary: ClinicalSummary,
    ) -> Tuple[List[ValidationError], List[str]]:
        """Validate clinical summary"""
        errors = []
        warnings = []
        
        # Required fields validation
        required_checks = [
            ("chief_complaint", summary.chief_complaint, self.rules.min_chief_complaint_words, ErrorSeverity.MAJOR),
            ("history_of_present_illness", summary.history_of_present_illness, self.rules.min_hpi_words, ErrorSeverity.MAJOR),
            ("assessment", summary.assessment, self.rules.min_assessment_words, ErrorSeverity.MAJOR),
            ("plan", summary.plan, self.rules.min_plan_words, ErrorSeverity.MAJOR),
        ]
        
        for field_name, value, min_words, severity in required_checks:
            if not value:
                errors.append(ValidationError(
                    field=f"summary.{field_name}",
                    error_type="missing_field",
                    message=f"Missing required field: {field_name}",
                    severity=severity,
                ))
            else:
                word_count = len(value.split())
                if word_count < min_words:
                    errors.append(ValidationError(
                        field=f"summary.{field_name}",
                        error_type="insufficient_content",
                        message=f"{field_name} has {word_count} words (minimum: {min_words})",
                        severity=ErrorSeverity.MODERATE,
                    ))
        
        # Check safety netting if required
        if self.rules.require_safety_netting and not summary.safety_netting:
            errors.append(ValidationError(
                field="summary.safety_netting",
                error_type="missing_field",
                message="Safety netting advice is required but missing",
                severity=ErrorSeverity.MODERATE,
            ))
        
        # Check plan actionability
        if self.rules.check_plan_actionability and summary.plan:
            if not self._is_plan_actionable(summary.plan):
                errors.append(ValidationError(
                    field="summary.plan",
                    error_type="non_actionable_plan",
                    message="Plan should contain specific actionable items (medications, tests, referrals, follow-up)",
                    severity=ErrorSeverity.MODERATE,
                ))
        
        # Check clinical terminology
        if self.rules.check_clinical_terminology:
            terminology_issues = self._check_clinical_terminology(summary)
            warnings.extend(terminology_issues)
        
        return errors, warnings
    
    def _validate_soap(
        self,
        soap: SOAPNote,
    ) -> Tuple[List[ValidationError], List[str]]:
        """Validate SOAP note structure"""
        errors = []
        warnings = []
        
        # Check each SOAP component
        components = [
            ("S (Subjective)", soap.subjective, 10),
            ("O (Objective)", soap.objective, 5),
            ("A (Assessment)", soap.assessment, 3),
            ("P (Plan)", soap.plan, 5),
        ]
        
        for name, value, min_words in components:
            if not value:
                errors.append(ValidationError(
                    field=f"soap.{name[0]}",
                    error_type="missing_soap_component",
                    message=f"SOAP component {name} is missing",
                    severity=ErrorSeverity.MODERATE,
                ))
            elif len(value.split()) < min_words:
                warnings.append(f"SOAP component {name} may be too brief")
        
        return errors, warnings
    
    def _is_plan_actionable(self, plan: str) -> bool:
        """Check if plan contains actionable items"""
        plan_lower = plan.lower()
        
        # Keywords indicating actionable items
        action_keywords = [
            # Medications
            "prescribe", "start", "continue", "stop", "increase", "decrease",
            "mg", "tablet", "capsule", "daily", "twice", "medication",
            # Investigations
            "blood test", "x-ray", "scan", "ecg", "mri", "ct", "ultrasound",
            "investigate", "test", "check", "measure", "monitor",
            # Referrals
            "refer", "referral", "specialist", "consultant",
            # Follow-up
            "follow-up", "follow up", "review", "return", "appointment",
            "week", "month", "days",
            # Advice
            "advise", "recommend", "encourage", "lifestyle",
        ]
        
        return any(keyword in plan_lower for keyword in action_keywords)
    
    def _check_clinical_terminology(self, summary: ClinicalSummary) -> List[str]:
        """Check for appropriate clinical terminology usage"""
        warnings = []
        
        # Combine all text for analysis
        all_text = " ".join(filter(None, [
            summary.chief_complaint,
            summary.history_of_present_illness,
            summary.assessment,
            summary.plan,
        ])).lower()
        
        # Check for overly casual language
        casual_terms = {
            "tummy": "abdomen",
            "belly": "abdomen",
            "heart attack": "myocardial infarction (or 'heart attack' in patient terms)",
            "stroke": "cerebrovascular accident (or 'stroke' in patient terms)",
        }
        
        for casual, formal in casual_terms.items():
            if casual in all_text:
                warnings.append(
                    f"Consider using '{formal}' instead of '{casual}' in clinical documentation"
                )
        
        return warnings
    
    def _check_consistency(self, sample: SyntheticSample) -> List[ValidationError]:
        """Check consistency between dialogue and summary"""
        errors = []
        
        # Extract key information from dialogue
        dialogue_text = " ".join(turn.text.lower() for turn in sample.dialogue)
        
        # Check if chief complaint is mentioned in dialogue
        if sample.summary.chief_complaint:
            # Extract key words from chief complaint
            cc_words = set(sample.summary.chief_complaint.lower().split())
            cc_words -= {"the", "a", "an", "with", "for", "of", "and", "or", "in", "on"}
            
            # Check if at least some key words appear in dialogue
            matches = sum(1 for word in cc_words if word in dialogue_text and len(word) > 3)
            
            if matches < len(cc_words) * 0.3:  # Less than 30% match
                errors.append(ValidationError(
                    field="consistency",
                    error_type="chief_complaint_mismatch",
                    message="Chief complaint may not be adequately discussed in dialogue",
                    severity=ErrorSeverity.MINOR,
                ))
        
        return errors


# =============================================================================
# Batch Validation
# =============================================================================

@dataclass
class BatchValidationResult:
    """Result of validating multiple samples"""
    
    total_samples: int = 0
    passed: int = 0
    failed: int = 0
    warnings: int = 0
    
    error_counts: Dict[str, int] = field(default_factory=dict)
    common_errors: List[str] = field(default_factory=list)
    
    @property
    def pass_rate(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.passed / self.total_samples
    
    def add_result(self, result: ValidationResult):
        self.total_samples += 1
        
        if result.status == ValidationStatus.PASSED:
            self.passed += 1
        elif result.status == ValidationStatus.FAILED:
            self.failed += 1
        else:
            self.warnings += 1
        
        # Count error types
        for error in result.errors:
            error_key = f"{error.error_type}:{error.severity.value}"
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
    
    def compute_common_errors(self, min_count: int = 2):
        """Identify most common errors"""
        sorted_errors = sorted(
            self.error_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        self.common_errors = [
            f"{error_type}: {count} occurrences"
            for error_type, count in sorted_errors
            if count >= min_count
        ]


def validate_batch(
    samples: List[SyntheticSample],
    rules: Optional[ValidationRules] = None,
) -> Tuple[BatchValidationResult, List[ValidationResult]]:
    """
    Validate a batch of samples
    
    Args:
        samples: List of samples to validate
        rules: Validation rules
        
    Returns:
        Tuple of (BatchValidationResult, list of individual results)
    """
    validator = StructuralValidator(rules)
    batch_result = BatchValidationResult()
    individual_results = []
    
    for sample in samples:
        result = validator.validate(sample)
        batch_result.add_result(result)
        individual_results.append(result)
    
    batch_result.compute_common_errors()
    
    return batch_result, individual_results


# =============================================================================
# Quick Validation Functions
# =============================================================================

def is_valid_dialogue(dialogue: List[DialogueTurn], min_turns: int = 6) -> bool:
    """Quick check if dialogue is valid"""
    if len(dialogue) < min_turns:
        return False
    
    speakers = {turn.speaker for turn in dialogue}
    return Speaker.DOCTOR in speakers and Speaker.PATIENT in speakers


def is_valid_summary(summary: ClinicalSummary) -> bool:
    """Quick check if summary is valid"""
    required = [
        summary.chief_complaint,
        summary.history_of_present_illness,
        summary.assessment,
        summary.plan,
    ]
    return all(field and len(field.split()) >= 3 for field in required)


def quick_validate(sample: SyntheticSample) -> bool:
    """Quick validation check (less thorough but faster)"""
    return (
        is_valid_dialogue(sample.dialogue) and
        is_valid_summary(sample.summary)
    )


if __name__ == "__main__":
    print("Structural Validation Module")
    print("=" * 60)
    
    # Test with example data
    from src.models import Speaker
    
    # Create test dialogue
    test_dialogue = [
        DialogueTurn(speaker=Speaker.DOCTOR, text="Good morning, what brings you in today?"),
        DialogueTurn(speaker=Speaker.PATIENT, text="I've been having chest pain for the last three days."),
        DialogueTurn(speaker=Speaker.DOCTOR, text="I'm sorry to hear that. Can you describe the pain for me?"),
        DialogueTurn(speaker=Speaker.PATIENT, text="It's a pressure-like sensation in the center of my chest."),
        DialogueTurn(speaker=Speaker.DOCTOR, text="Does anything make it better or worse?"),
        DialogueTurn(speaker=Speaker.PATIENT, text="It gets worse when I walk upstairs and better when I rest."),
    ]
    
    # Create test summary
    test_summary = ClinicalSummary(
        chief_complaint="Chest pain for 3 days",
        history_of_present_illness=(
            "55-year-old male presents with central chest pain for 3 days. "
            "Pain is pressure-like, worse with exertion, relieved by rest. "
            "No radiation to arm or jaw. No associated shortness of breath, "
            "nausea, or diaphoresis."
        ),
        assessment="Suspected stable angina, differential includes ACS, MSK pain",
        plan="Order ECG and troponins. Start aspirin 75mg. Refer to cardiology for stress test. "
             "Follow-up in 1 week. Return if chest pain worsens or occurs at rest.",
    )
    
    # Validate
    validator = StructuralValidator()
    
    print("\nTest Dialogue Validation:")
    print(f"  Valid dialogue: {is_valid_dialogue(test_dialogue)}")
    print(f"  Valid summary: {is_valid_summary(test_summary)}")
    
    print("\n✓ Structural validation module ready!")