"""
Validation Module

Comprehensive validation for synthetic clinical data quality.

Three levels of validation:
1. Structural - Pydantic-based structure and completeness checks
2. Clinical - Medical content accuracy and safety validation
3. RAG Metrics - Quality of RAG retrieval and response generation

Example:
    from src.validation import (
        StructuralValidator,
        SimpleClinicalValidator,
        SimpleRAGEvaluator,
        validate_sample,
    )
    
    # Full validation
    result = validate_sample(sample)
    
    if result.is_acceptable:
        print("Sample passed all validations")
    else:
        for error in result.errors:
            print(f"{error.severity}: {error.message}")
"""

# Structural validation
from .structural import (
    ValidationRules,
    StructuralValidator,
    BatchValidationResult,
    validate_batch,
    is_valid_dialogue,
    is_valid_summary,
    quick_validate,
)

# Clinical validation
from .clinical import (
    ClinicalEntityType,
    ClinicalEntity,
    ClinicalValidationResult,
    SimpleClinicalValidator,
    AdvancedClinicalValidator,
    create_clinical_validator,
)

# RAG metrics
from .rag_metrics import (
    RAGMetricType,
    RAGMetricResult,
    RAGEvaluationResult,
    SimpleRAGEvaluator,
    RAGASEvaluator,
    create_rag_evaluator,
    quick_rag_score,
    evaluate_retrieval_quality,
)

# Re-export from models for convenience
from src.models import (
    ValidationResult,
    ValidationError,
    ValidationStatus,
    ErrorSeverity,
)


# =============================================================================
# Unified Validation
# =============================================================================

def validate_sample(
    sample,
    structural_rules=None,
    clinical_validation: bool = True,
    rag_validation: bool = True,
    use_advanced_clinical: bool = False,
    use_ragas: bool = False,
) -> ValidationResult:
    """
    Comprehensive validation of a synthetic sample
    
    Combines structural, clinical, and RAG validation.
    
    Args:
        sample: SyntheticSample to validate
        structural_rules: Optional custom validation rules
        clinical_validation: Enable clinical validation
        rag_validation: Enable RAG metrics validation
        use_advanced_clinical: Use scispaCy/MedCAT (if available)
        use_ragas: Use RAGAS framework (if available)
        
    Returns:
        Combined ValidationResult
    """
    all_errors = []
    all_warnings = []
    
    # Structural validation (always run)
    structural_validator = StructuralValidator(structural_rules)
    structural_result = structural_validator.validate(sample)
    all_errors.extend(structural_result.errors)
    all_warnings.extend(structural_result.warnings)
    
    # Clinical validation
    clinical_valid = None
    if clinical_validation:
        clinical_validator = create_clinical_validator(use_advanced=use_advanced_clinical)
        clinical_result = clinical_validator.validate(sample)
        clinical_errors = clinical_result.to_validation_errors()
        all_errors.extend(clinical_errors)
        clinical_valid = clinical_result.is_valid
    
    # RAG validation
    rag_faithfulness = None
    if rag_validation and sample.rag.rag_enabled:
        rag_evaluator = create_rag_evaluator(use_ragas=use_ragas)
        rag_result = rag_evaluator.evaluate(sample)
        rag_faithfulness = rag_result.overall_score
        
        # Add warnings for low RAG scores
        if rag_result.overall_score < 0.5:
            all_warnings.append(
                f"Low RAG quality score: {rag_result.overall_score:.2f}"
            )
    
    # Determine overall status
    has_critical = any(e.severity == ErrorSeverity.CRITICAL for e in all_errors)
    has_major = any(e.severity == ErrorSeverity.MAJOR for e in all_errors)
    
    if has_critical:
        status = ValidationStatus.FAILED
    elif has_major:
        status = ValidationStatus.WARNING
    elif all_errors:
        status = ValidationStatus.WARNING
    else:
        status = ValidationStatus.PASSED
    
    return ValidationResult(
        status=status,
        errors=all_errors,
        warnings=all_warnings,
        structural_valid=structural_result.structural_valid,
        clinical_valid=clinical_valid,
        rag_faithfulness=rag_faithfulness,
    )


def validate_batch_comprehensive(
    samples,
    **kwargs,
):
    """
    Validate a batch of samples comprehensively
    
    Args:
        samples: List of SyntheticSample objects
        **kwargs: Arguments passed to validate_sample
        
    Returns:
        Tuple of (summary_stats, individual_results)
    """
    results = []
    passed = 0
    failed = 0
    warnings = 0
    
    for sample in samples:
        result = validate_sample(sample, **kwargs)
        results.append(result)
        
        if result.status == ValidationStatus.PASSED:
            passed += 1
        elif result.status == ValidationStatus.FAILED:
            failed += 1
        else:
            warnings += 1
    
    summary = {
        "total": len(samples),
        "passed": passed,
        "failed": failed,
        "warnings": warnings,
        "pass_rate": passed / len(samples) if samples else 0.0,
    }
    
    return summary, results


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Structural
    "ValidationRules",
    "StructuralValidator",
    "BatchValidationResult",
    "validate_batch",
    "is_valid_dialogue",
    "is_valid_summary",
    "quick_validate",
    # Clinical
    "ClinicalEntityType",
    "ClinicalEntity",
    "ClinicalValidationResult",
    "SimpleClinicalValidator",
    "AdvancedClinicalValidator",
    "create_clinical_validator",
    # RAG Metrics
    "RAGMetricType",
    "RAGMetricResult",
    "RAGEvaluationResult",
    "SimpleRAGEvaluator",
    "RAGASEvaluator",
    "create_rag_evaluator",
    "quick_rag_score",
    "evaluate_retrieval_quality",
    # Common types
    "ValidationResult",
    "ValidationError",
    "ValidationStatus",
    "ErrorSeverity",
    # Unified validation
    "validate_sample",
    "validate_batch_comprehensive",
]