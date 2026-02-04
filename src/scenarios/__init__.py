"""
Scenarios Module

Generates diverse clinical scenarios for synthetic data generation.

Features:
- Demographic diversity (age, gender, occupation)
- Clinical diversity (symptoms, specialties, severity)
- Complexity control (low, medium, high)
- Reproducible generation with seeds
- Pre-defined high-quality scenarios

Example:
    from src.scenarios import ScenarioGenerator, generate_scenarios
    
    # Quick generation
    scenarios = generate_scenarios(count=100, seed=42)
    
    # Detailed control
    generator = ScenarioGenerator(seed=42)
    scenario = generator.generate(
        specialty=ClinicalSpecialty.CARDIOLOGY,
        complexity=DifficultyLevel.HIGH,
    )
    print(scenario.to_text())
"""

from .generator import (
    # Data classes
    PatientDemographics,
    ClinicalPresentation,
    MedicalHistory,
    ClinicalScenario,
    # Templates
    ScenarioTemplates,
    # Generator
    ScenarioGenerator,
    # Predefined scenarios
    PredefinedScenarios,
    # Factory functions
    create_scenario_generator,
    generate_scenarios,
    # I/O
    save_scenarios,
    load_scenarios,
)

__all__ = [
    # Data classes
    "PatientDemographics",
    "ClinicalPresentation",
    "MedicalHistory",
    "ClinicalScenario",
    # Templates
    "ScenarioTemplates",
    # Generator
    "ScenarioGenerator",
    # Predefined
    "PredefinedScenarios",
    # Functions
    "create_scenario_generator",
    "generate_scenarios",
    "save_scenarios",
    "load_scenarios",
]