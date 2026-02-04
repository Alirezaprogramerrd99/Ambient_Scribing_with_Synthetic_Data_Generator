"""
Clinical Scenario Generator

Generates diverse clinical scenarios for synthetic data generation.
Supports controlled diversity across demographics, symptoms, specialties, and complexity.

Author: Alireza Rashidi
MSc Project: Trustworthy SLMs for Ambient Clinical Scribing
"""

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from enum import Enum

from src.models import (
    ClinicalSpecialty,
    AgeGroup,
    Gender,
    DifficultyLevel,
    Urgency,
    RED_FLAG_SYMPTOMS,
    COMMON_COMORBIDITIES,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Scenario Components
# =============================================================================

@dataclass
class PatientDemographics:
    """Patient demographic information"""
    
    age: int
    age_group: AgeGroup
    gender: Gender
    occupation: Optional[str] = None
    
    def to_description(self) -> str:
        """Generate natural language description"""
        gender_str = {
            Gender.MALE: "male",
            Gender.FEMALE: "female",
            Gender.OTHER: "person",
            Gender.NOT_SPECIFIED: "patient",
        }.get(self.gender, "patient")
        
        desc = f"{self.age}-year-old {gender_str}"
        if self.occupation:
            desc += f" ({self.occupation})"
        
        return desc


@dataclass
class ClinicalPresentation:
    """Clinical presentation details"""
    
    chief_complaint: str
    duration: str
    severity: str
    associated_symptoms: List[str] = field(default_factory=list)
    aggravating_factors: List[str] = field(default_factory=list)
    relieving_factors: List[str] = field(default_factory=list)
    red_flags: List[str] = field(default_factory=list)
    
    def to_description(self) -> str:
        """Generate natural language description"""
        desc = f"{self.chief_complaint} for {self.duration}"
        
        if self.severity:
            desc += f", {self.severity} severity"
        
        if self.associated_symptoms:
            desc += f", associated with {', '.join(self.associated_symptoms)}"
        
        if self.aggravating_factors:
            desc += f", worse with {', '.join(self.aggravating_factors)}"
        
        if self.relieving_factors:
            desc += f", better with {', '.join(self.relieving_factors)}"
        
        return desc


@dataclass
class MedicalHistory:
    """Patient medical history"""
    
    past_conditions: List[str] = field(default_factory=list)
    current_medications: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    family_history: List[str] = field(default_factory=list)
    social_history: Dict[str, str] = field(default_factory=dict)
    
    def to_description(self) -> str:
        """Generate natural language description"""
        parts = []
        
        if self.past_conditions:
            parts.append(f"history of {', '.join(self.past_conditions)}")
        
        if self.current_medications:
            parts.append(f"on {', '.join(self.current_medications)}")
        
        if self.allergies:
            parts.append(f"allergic to {', '.join(self.allergies)}")
        
        return ", ".join(parts) if parts else "no significant past history"


@dataclass
class ClinicalScenario:
    """Complete clinical scenario"""
    
    id: str
    demographics: PatientDemographics
    presentation: ClinicalPresentation
    history: MedicalHistory
    specialty: ClinicalSpecialty
    complexity: DifficultyLevel
    urgency: Urgency
    
    # Generation metadata
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    
    def to_text(self) -> str:
        """Generate complete scenario text"""
        parts = [
            self.demographics.to_description(),
            "presenting with",
            self.presentation.to_description(),
        ]
        
        history_desc = self.history.to_description()
        if history_desc and history_desc != "no significant past history":
            parts.append(f"with {history_desc}")
        
        return " ".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "text": self.to_text(),
            "demographics": {
                "age": self.demographics.age,
                "age_group": self.demographics.age_group.value,
                "gender": self.demographics.gender.value,
                "occupation": self.demographics.occupation,
            },
            "presentation": {
                "chief_complaint": self.presentation.chief_complaint,
                "duration": self.presentation.duration,
                "severity": self.presentation.severity,
                "associated_symptoms": self.presentation.associated_symptoms,
                "red_flags": self.presentation.red_flags,
            },
            "history": {
                "past_conditions": self.history.past_conditions,
                "medications": self.history.current_medications,
                "allergies": self.history.allergies,
            },
            "specialty": self.specialty.value,
            "complexity": self.complexity.value,
            "urgency": self.urgency.value,
            "tags": self.tags,
        }


# =============================================================================
# Scenario Templates
# =============================================================================

class ScenarioTemplates:
    """Pre-defined scenario templates organized by specialty"""
    
    # Chief complaints by specialty
    CHIEF_COMPLAINTS = {
        ClinicalSpecialty.CARDIOLOGY: [
            ("chest pain", ["shortness of breath", "sweating", "nausea", "arm pain"]),
            ("palpitations", ["dizziness", "shortness of breath", "chest discomfort"]),
            ("shortness of breath on exertion", ["ankle swelling", "fatigue", "orthopnea"]),
            ("leg swelling", ["shortness of breath", "weight gain", "fatigue"]),
        ],
        ClinicalSpecialty.RESPIRATORY: [
            ("cough", ["sputum production", "shortness of breath", "fever", "wheeze"]),
            ("shortness of breath", ["cough", "wheeze", "chest tightness"]),
            ("wheeze", ["cough", "shortness of breath", "chest tightness"]),
            ("haemoptysis", ["cough", "weight loss", "chest pain"]),
        ],
        ClinicalSpecialty.GASTROENTEROLOGY: [
            ("abdominal pain", ["nausea", "vomiting", "diarrhoea", "constipation"]),
            ("heartburn", ["regurgitation", "difficulty swallowing", "chest pain"]),
            ("diarrhoea", ["abdominal cramps", "blood in stool", "weight loss"]),
            ("constipation", ["abdominal bloating", "straining", "incomplete evacuation"]),
            ("rectal bleeding", ["abdominal pain", "change in bowel habit", "weight loss"]),
        ],
        ClinicalSpecialty.NEUROLOGY: [
            ("headache", ["nausea", "visual disturbance", "neck stiffness", "photophobia"]),
            ("dizziness", ["nausea", "hearing loss", "tinnitus", "unsteadiness"]),
            ("weakness", ["numbness", "tingling", "difficulty walking"]),
            ("memory problems", ["confusion", "word-finding difficulty", "personality change"]),
        ],
        ClinicalSpecialty.MUSCULOSKELETAL: [
            ("back pain", ["leg pain", "numbness", "weakness", "stiffness"]),
            ("knee pain", ["swelling", "stiffness", "instability", "locking"]),
            ("shoulder pain", ["stiffness", "weakness", "difficulty reaching"]),
            ("joint pain", ["swelling", "stiffness", "redness", "warmth"]),
        ],
        ClinicalSpecialty.ENDOCRINOLOGY: [
            ("fatigue", ["weight change", "mood changes", "cold intolerance"]),
            ("increased thirst", ["frequent urination", "weight loss", "blurred vision"]),
            ("weight gain", ["fatigue", "constipation", "dry skin"]),
            ("tremor", ["weight loss", "palpitations", "heat intolerance"]),
        ],
        ClinicalSpecialty.PSYCHIATRY: [
            ("low mood", ["poor sleep", "loss of interest", "poor concentration"]),
            ("anxiety", ["palpitations", "sweating", "difficulty sleeping"]),
            ("sleep problems", ["fatigue", "poor concentration", "irritability"]),
        ],
        ClinicalSpecialty.DERMATOLOGY: [
            ("skin rash", ["itching", "pain", "spreading"]),
            ("itchy skin", ["rash", "dry skin", "sleep disturbance"]),
            ("skin lesion", ["change in size", "change in colour", "bleeding"]),
        ],
        ClinicalSpecialty.INFECTIOUS_DISEASE: [
            ("fever", ["chills", "sweats", "fatigue", "body aches"]),
            ("sore throat", ["fever", "difficulty swallowing", "swollen glands"]),
        ],
        ClinicalSpecialty.UROLOGY: [
            ("urinary frequency", ["urgency", "burning", "blood in urine"]),
            ("difficulty urinating", ["weak stream", "incomplete emptying", "nocturia"]),
        ],
        ClinicalSpecialty.GENERAL_PRACTICE: [
            ("tiredness", ["weight change", "mood changes", "sleep problems"]),
            ("feeling unwell", ["fever", "aches", "fatigue"]),
        ],
    }
    
    # Duration options
    DURATIONS = {
        "acute": ["few hours", "since yesterday", "since this morning", "1 day", "2 days"],
        "subacute": ["3 days", "5 days", "1 week", "10 days"],
        "chronic": ["2 weeks", "3 weeks", "1 month", "2 months", "3 months", "6 months", "1 year"],
    }
    
    # Severity options
    SEVERITIES = ["mild", "moderate", "severe", "gradually worsening", "intermittent"]
    
    # Aggravating/relieving factors by complaint type
    MODIFYING_FACTORS = {
        "pain": {
            "aggravating": ["movement", "exertion", "eating", "lying down", "stress"],
            "relieving": ["rest", "painkillers", "heat", "cold", "position change"],
        },
        "breathing": {
            "aggravating": ["exertion", "lying flat", "cold air", "allergens"],
            "relieving": ["rest", "sitting upright", "inhalers", "fresh air"],
        },
        "gi": {
            "aggravating": ["eating", "spicy food", "alcohol", "stress", "fatty food"],
            "relieving": ["antacids", "not eating", "bland diet"],
        },
    }
    
    # Occupations
    OCCUPATIONS = [
        "office worker", "teacher", "nurse", "builder", "driver",
        "retail worker", "retired", "student", "self-employed",
        "factory worker", "healthcare assistant", "engineer",
    ]
    
    # Social history options
    SMOKING_STATUS = ["never smoked", "ex-smoker", "current smoker (10/day)", "current smoker (20/day)"]
    ALCOHOL_STATUS = ["non-drinker", "occasional", "moderate (14 units/week)", "heavy drinker"]


# =============================================================================
# Scenario Generator
# =============================================================================

class ScenarioGenerator:
    """
    Generates diverse clinical scenarios for synthetic data generation
    
    Features:
    - Demographic diversity (age, gender)
    - Clinical diversity (symptoms, specialties)
    - Complexity control
    - Reproducible generation with seeds
    
    Example:
        generator = ScenarioGenerator(seed=42)
        
        # Generate single scenario
        scenario = generator.generate()
        print(scenario.to_text())
        
        # Generate diverse batch
        scenarios = generator.generate_batch(
            count=100,
            specialties=[ClinicalSpecialty.CARDIOLOGY, ClinicalSpecialty.RESPIRATORY],
        )
    """
    
    def __init__(
        self,
        seed: Optional[int] = None,
        templates: Optional[ScenarioTemplates] = None,
    ):
        """
        Initialize generator
        
        Args:
            seed: Random seed for reproducibility
            templates: Custom templates (uses defaults if not provided)
        """
        self.templates = templates or ScenarioTemplates()
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
        
        self._generated_count = 0
        self._used_combinations: Set[str] = set()
    
    def generate(
        self,
        specialty: Optional[ClinicalSpecialty] = None,
        age_group: Optional[AgeGroup] = None,
        gender: Optional[Gender] = None,
        complexity: Optional[DifficultyLevel] = None,
        include_comorbidities: bool = True,
        include_red_flags: bool = False,
    ) -> ClinicalScenario:
        """
        Generate a single clinical scenario
        
        Args:
            specialty: Specific specialty (random if not provided)
            age_group: Specific age group (random if not provided)
            gender: Specific gender (random if not provided)
            complexity: Desired complexity level
            include_comorbidities: Whether to include past medical history
            include_red_flags: Whether to include red flag symptoms
            
        Returns:
            Generated ClinicalScenario
        """
        # Generate scenario ID
        self._generated_count += 1
        scenario_id = f"scenario_{self._generated_count:05d}"
        
        # Select specialty
        if specialty is None:
            specialty = random.choice(list(self.templates.CHIEF_COMPLAINTS.keys()))
        
        # Generate demographics
        demographics = self._generate_demographics(age_group, gender)
        
        # Generate presentation
        presentation = self._generate_presentation(
            specialty=specialty,
            include_red_flags=include_red_flags,
        )
        
        # Generate history
        history = self._generate_history(
            demographics=demographics,
            include_comorbidities=include_comorbidities,
        )
        
        # Determine complexity
        if complexity is None:
            complexity = self._estimate_complexity(
                presentation=presentation,
                history=history,
            )
        
        # Determine urgency
        urgency = self._determine_urgency(presentation)
        
        # Create scenario
        scenario = ClinicalScenario(
            id=scenario_id,
            demographics=demographics,
            presentation=presentation,
            history=history,
            specialty=specialty,
            complexity=complexity,
            urgency=urgency,
            tags=self._generate_tags(specialty, complexity),
        )
        
        return scenario
    
    def generate_batch(
        self,
        count: int,
        specialties: Optional[List[ClinicalSpecialty]] = None,
        complexity_distribution: Optional[Dict[DifficultyLevel, float]] = None,
        ensure_diversity: bool = True,
    ) -> List[ClinicalScenario]:
        """
        Generate a batch of diverse scenarios
        
        Args:
            count: Number of scenarios to generate
            specialties: Specialties to include (all if not specified)
            complexity_distribution: Distribution of complexity levels
            ensure_diversity: Ensure variety in demographics and presentations
            
        Returns:
            List of generated scenarios
        """
        scenarios = []
        
        # Default specialties
        if specialties is None:
            specialties = list(self.templates.CHIEF_COMPLAINTS.keys())
        
        # Default complexity distribution (30% low, 50% medium, 20% high)
        if complexity_distribution is None:
            complexity_distribution = {
                DifficultyLevel.LOW: 0.3,
                DifficultyLevel.MEDIUM: 0.5,
                DifficultyLevel.HIGH: 0.2,
            }
        
        # Generate complexity assignments
        complexity_assignments = self._distribute_complexity(count, complexity_distribution)
        
        # Generate scenarios with diversity
        specialty_index = 0
        age_groups = list(AgeGroup)
        genders = [Gender.MALE, Gender.FEMALE]
        
        for i in range(count):
            # Rotate through specialties for diversity
            if ensure_diversity:
                specialty = specialties[specialty_index % len(specialties)]
                specialty_index += 1
            else:
                specialty = random.choice(specialties)
            
            # Vary demographics
            age_group = age_groups[i % len(age_groups)] if ensure_diversity else None
            gender = genders[i % len(genders)] if ensure_diversity else None
            
            # Get complexity
            complexity = complexity_assignments[i]
            
            # Include red flags for high complexity
            include_red_flags = complexity == DifficultyLevel.HIGH and random.random() > 0.5
            
            scenario = self.generate(
                specialty=specialty,
                age_group=age_group,
                gender=gender,
                complexity=complexity,
                include_comorbidities=complexity != DifficultyLevel.LOW,
                include_red_flags=include_red_flags,
            )
            
            scenarios.append(scenario)
        
        logger.info(f"Generated {len(scenarios)} scenarios across {len(specialties)} specialties")
        return scenarios
    
    def _generate_demographics(
        self,
        age_group: Optional[AgeGroup],
        gender: Optional[Gender],
    ) -> PatientDemographics:
        """Generate patient demographics"""
        
        # Select age group
        if age_group is None:
            age_group = random.choice(list(AgeGroup))
        
        # Generate age within group
        age_range = age_group.typical_age_range()
        age = random.randint(age_range[0], age_range[1])
        
        # Select gender
        if gender is None:
            gender = random.choice([Gender.MALE, Gender.FEMALE])
        
        # Maybe add occupation
        occupation = None
        if age >= 18 and random.random() > 0.7:
            occupation = random.choice(self.templates.OCCUPATIONS)
        
        return PatientDemographics(
            age=age,
            age_group=age_group,
            gender=gender,
            occupation=occupation,
        )
    
    def _generate_presentation(
        self,
        specialty: ClinicalSpecialty,
        include_red_flags: bool,
    ) -> ClinicalPresentation:
        """Generate clinical presentation"""
        
        # Get complaints for specialty
        complaints = self.templates.CHIEF_COMPLAINTS.get(
            specialty,
            self.templates.CHIEF_COMPLAINTS[ClinicalSpecialty.GENERAL_PRACTICE]
        )
        
        # Select chief complaint and associated symptoms
        complaint, associated = random.choice(complaints)
        
        # Select some associated symptoms
        num_associated = random.randint(1, min(3, len(associated)))
        selected_associated = random.sample(associated, num_associated)
        
        # Select duration
        duration_type = random.choice(["acute", "subacute", "chronic"])
        duration = random.choice(self.templates.DURATIONS[duration_type])
        
        # Select severity
        severity = random.choice(self.templates.SEVERITIES)
        
        # Generate modifying factors
        aggravating = []
        relieving = []
        
        # factor_type = "pain" if "pain" in complaint else "breathing" if "breath" in complaint else "gi"
        
        if any(x in complaint for x in ["pain", "discomfort"]):
            factor_type = "pain"
        elif any(x in complaint for x in ["breath", "cough", "wheeze"]): # <--- Expanded check
            factor_type = "breathing"
        else:
            factor_type = "gi"
            
        
        factors = self.templates.MODIFYING_FACTORS.get(
            factor_type,
            self.templates.MODIFYING_FACTORS["pain"] # Default fallback
        )
        
        if random.random() > 0.5:
            aggravating = random.sample(factors["aggravating"], random.randint(1, 2))
        if random.random() > 0.5:
            relieving = random.sample(factors["relieving"], random.randint(1, 2))
        
        # Maybe add red flags
        red_flags = []
        if include_red_flags:
            relevant_red_flags = [rf for rf in RED_FLAG_SYMPTOMS if any(
                word in rf for word in complaint.split()
            )]
            if relevant_red_flags:
                red_flags = random.sample(relevant_red_flags, min(2, len(relevant_red_flags)))
        
        return ClinicalPresentation(
            chief_complaint=complaint,
            duration=duration,
            severity=severity,
            associated_symptoms=selected_associated,
            aggravating_factors=aggravating,
            relieving_factors=relieving,
            red_flags=red_flags,
        )
    
    def _generate_history(
        self,
        demographics: PatientDemographics,
        include_comorbidities: bool,
    ) -> MedicalHistory:
        """Generate medical history"""
        
        past_conditions = []
        medications = []
        allergies = []
        family_history = []
        social_history = {}
        
        # Add comorbidities based on age
        if include_comorbidities:
            # Older patients more likely to have comorbidities
            num_conditions = 0
            if demographics.age > 40:
                num_conditions = random.randint(0, 2)
            if demographics.age > 60:
                num_conditions = random.randint(1, 3)
            
            if num_conditions > 0:
                past_conditions = random.sample(
                    COMMON_COMORBIDITIES,
                    min(num_conditions, len(COMMON_COMORBIDITIES))
                )
                
                # Add medications for conditions
                condition_medications = {
                    "hypertension": ["ramipril 5mg", "amlodipine 5mg"],
                    "type 2 diabetes": ["metformin 500mg", "gliclazide 80mg"],
                    "hyperlipidemia": ["atorvastatin 20mg", "simvastatin 40mg"],
                    "asthma": ["salbutamol inhaler", "beclometasone inhaler"],
                    "depression": ["sertraline 50mg", "citalopram 20mg"],
                }
                
                for condition in past_conditions:
                    if condition in condition_medications:
                        medications.append(random.choice(condition_medications[condition]))
        
        # Add allergies (10% chance)
        if random.random() > 0.9:
            allergies = random.sample(["penicillin", "aspirin", "ibuprofen", "codeine"], 1)
        
        # Add social history
        social_history["smoking"] = random.choice(self.templates.SMOKING_STATUS)
        social_history["alcohol"] = random.choice(self.templates.ALCOHOL_STATUS)
        
        return MedicalHistory(
            past_conditions=past_conditions,
            current_medications=medications,
            allergies=allergies,
            family_history=family_history,
            social_history=social_history,
        )
    
    def _estimate_complexity(
        self,
        presentation: ClinicalPresentation,
        history: MedicalHistory,
    ) -> DifficultyLevel:
        """Estimate scenario complexity"""
        
        complexity_score = 0
        
        # Multiple associated symptoms increase complexity
        complexity_score += len(presentation.associated_symptoms) * 0.5
        
        # Red flags increase complexity
        complexity_score += len(presentation.red_flags) * 2
        
        # Comorbidities increase complexity
        complexity_score += len(history.past_conditions) * 1
        
        # Medications increase complexity
        complexity_score += len(history.current_medications) * 0.5
        
        if complexity_score < 2:
            return DifficultyLevel.LOW
        elif complexity_score < 5:
            return DifficultyLevel.MEDIUM
        else:
            return DifficultyLevel.HIGH
    
    def _determine_urgency(self, presentation: ClinicalPresentation) -> Urgency:
        """Determine clinical urgency"""
        
        if presentation.red_flags:
            return Urgency.URGENT
        
        if presentation.severity == "severe":
            return Urgency.URGENT
        
        if "acute" in presentation.duration or "hour" in presentation.duration:
            return Urgency.URGENT
        
        return Urgency.ROUTINE
    
    def _distribute_complexity(
        self,
        count: int,
        distribution: Dict[DifficultyLevel, float],
    ) -> List[DifficultyLevel]:
        """Distribute complexity levels according to distribution"""
        
        assignments = []
        
        for level, proportion in distribution.items():
            num = int(count * proportion)
            assignments.extend([level] * num)
        
        # Fill remaining with medium
        while len(assignments) < count:
            assignments.append(DifficultyLevel.MEDIUM)
        
        random.shuffle(assignments)
        return assignments
    
    def _generate_tags(
        self,
        specialty: ClinicalSpecialty,
        complexity: DifficultyLevel,
    ) -> List[str]:
        """Generate tags for the scenario"""
        return [
            specialty.value.lower().replace(" ", "_"),
            complexity.value,
        ]


# =============================================================================
# Predefined Scenario Sets
# =============================================================================

class PredefinedScenarios:
    """Pre-written high-quality scenarios for specific use cases"""
    
    CARDIOLOGY_SCENARIOS = [
        "55-year-old male smoker presenting with central chest pain for 2 hours, "
        "radiating to left arm, associated with sweating and nausea. "
        "History of hypertension and hyperlipidemia.",
        
        "68-year-old female with palpitations and dizziness for 3 days. "
        "History of atrial fibrillation, currently on warfarin.",
        
        "72-year-old male with progressive shortness of breath and ankle swelling "
        "over 2 weeks. Known heart failure patient on ramipril and furosemide.",
    ]
    
    RESPIRATORY_SCENARIOS = [
        "45-year-old asthmatic presenting with worsening wheeze and cough for 5 days, "
        "not responding to usual reliever inhaler. No fever.",
        
        "62-year-old ex-smoker with productive cough and shortness of breath for 1 week. "
        "History of COPD with 2 exacerbations in the past year.",
        
        "28-year-old with dry cough and mild shortness of breath for 10 days "
        "following a viral illness. No significant past history.",
    ]
    
    GENERAL_SCENARIOS = [
        "35-year-old female office worker with fatigue and difficulty concentrating "
        "for 3 months. Sleep poorly, feels low mood.",
        
        "50-year-old male with intermittent headaches for 2 weeks, "
        "worse in the mornings, associated with visual disturbance.",
        
        "42-year-old with epigastric pain and heartburn for 1 month, "
        "worse after meals, particularly spicy food.",
    ]
    
    @classmethod
    def get_all(cls) -> List[str]:
        """Get all predefined scenarios"""
        return (
            cls.CARDIOLOGY_SCENARIOS +
            cls.RESPIRATORY_SCENARIOS +
            cls.GENERAL_SCENARIOS
        )


# =============================================================================
# Factory Functions
# =============================================================================

def create_scenario_generator(
    seed: Optional[int] = None,
) -> ScenarioGenerator:
    """Create a scenario generator"""
    return ScenarioGenerator(seed=seed)


def generate_scenarios(
    count: int,
    specialties: Optional[List[str]] = None,
    complexity: Optional[str] = None,
    seed: Optional[int] = None,
) -> List[str]:
    """
    Quick function to generate scenario texts
    
    Args:
        count: Number of scenarios
        specialties: List of specialty names
        complexity: Complexity level ("low", "medium", "high")
        seed: Random seed
        
    Returns:
        List of scenario text strings
    """
    generator = ScenarioGenerator(seed=seed)
    
    # Convert specialty strings to enums
    specialty_enums = None
    if specialties:
        specialty_enums = []
        for s in specialties:
            try:
                specialty_enums.append(ClinicalSpecialty(s))
            except ValueError:
                logger.warning(f"Unknown specialty: {s}")
    
    scenarios = generator.generate_batch(
        count=count,
        specialties=specialty_enums,
    )
    
    return [s.to_text() for s in scenarios]


def save_scenarios(
    scenarios: List[ClinicalScenario],
    output_path: Union[str, Path],
    format: str = "jsonl",
):
    """
    Save scenarios to file
    
    Args:
        scenarios: List of scenarios
        output_path: Output file path
        format: "jsonl", "json", or "txt"
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "jsonl":
        with open(output_path, "w") as f:
            for scenario in scenarios:
                f.write(json.dumps(scenario.to_dict()) + "\n")
    
    elif format == "json":
        with open(output_path, "w") as f:
            json.dump([s.to_dict() for s in scenarios], f, indent=2)
    
    elif format == "txt":
        with open(output_path, "w") as f:
            for scenario in scenarios:
                f.write(scenario.to_text() + "\n\n")
    
    logger.info(f"Saved {len(scenarios)} scenarios to {output_path}")


def load_scenarios(
    input_path: Union[str, Path],
) -> List[str]:
    """
    Load scenarios from file
    
    Args:
        input_path: Input file path (jsonl, json, or txt)
        
    Returns:
        List of scenario texts
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Scenario file not found: {input_path}")
    
    scenarios = []
    
    if input_path.suffix == ".jsonl":
        with open(input_path) as f:
            for line in f:
                data = json.loads(line)
                scenarios.append(data.get("text", ""))
    
    elif input_path.suffix == ".json":
        with open(input_path) as f:
            data = json.load(f)
            for item in data:
                scenarios.append(item.get("text", "") if isinstance(item, dict) else item)
    
    elif input_path.suffix == ".txt":
        with open(input_path) as f:
            content = f.read()
            scenarios = [s.strip() for s in content.split("\n\n") if s.strip()]
    
    return scenarios


if __name__ == "__main__":
    print("Scenario Generator Module")
    print("=" * 60)
    
    # Create generator
    generator = ScenarioGenerator(seed=42)
    
    # Generate single scenario
    scenario = generator.generate(specialty=ClinicalSpecialty.CARDIOLOGY)
    print(f"\nSingle scenario:")
    print(f"  {scenario.to_text()}")
    print(f"  Specialty: {scenario.specialty.value}")
    print(f"  Complexity: {scenario.complexity.value}")
    
    # Generate batch
    print(f"\nGenerating batch of 5 scenarios...")
    scenarios = generator.generate_batch(count=5)
    
    for i, s in enumerate(scenarios, 1):
        print(f"\n{i}. [{s.specialty.value}] {s.to_text()[:80]}...")
    
    print("\n✓ Scenario generator ready!")