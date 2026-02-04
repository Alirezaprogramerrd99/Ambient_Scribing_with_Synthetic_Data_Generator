"""
Prompt Template Manager

Loads and manages prompt templates for the teacher model.
Supports variable substitution and template validation.

Author: Alireza Rashidi
MSc Project: Trustworthy SLMs for Ambient Clinical Scribing
"""

from pathlib import Path
from typing import Dict, Optional, Any
from functools import lru_cache
import re


class PromptTemplate:
    """
    A prompt template with variable substitution
    
    Example:
        template = PromptTemplate("Hello {name}, you are {age} years old.")
        prompt = template.format(name="Alice", age=30)
    """
    
    def __init__(self, template: str, name: str = "unnamed"):
        self.template = template
        self.name = name
        self._variables = self._extract_variables()
    
    def _extract_variables(self) -> set:
        """Extract variable names from template"""
        # Match {variable_name} patterns
        pattern = r'\{(\w+)\}'
        return set(re.findall(pattern, self.template))
    
    @property
    def variables(self) -> set:
        """Get set of variable names in template"""
        return self._variables
    
    def format(self, **kwargs) -> str:
        """
        Format template with provided variables
        
        Args:
            **kwargs: Variable values
            
        Returns:
            Formatted prompt string
        """
        # Check for missing variables
        missing = self._variables - set(kwargs.keys())
        if missing:
            # Use empty string for missing optional variables
            for var in missing:
                kwargs[var] = ""
        
        return self.template.format(**kwargs)
    
    def format_safe(self, **kwargs) -> str:
        """
        Format template, leaving unfilled variables as-is
        
        Useful for partial formatting.
        """
        result = self.template
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result
    
    def __str__(self) -> str:
        return f"PromptTemplate(name='{self.name}', variables={self._variables})"


class PromptManager:
    """
    Manager for loading and accessing prompt templates
    
    Example:
        manager = PromptManager()
        prompt = manager.get_prompt(
            "dialogue_generation",
            scenario="55-year-old male with chest pain",
            guidelines_context="NICE guidelines for chest pain..."
        )
    """
    
    def __init__(self, prompts_dir: Optional[Path] = None):
        """
        Initialize prompt manager
        
        Args:
            prompts_dir: Directory containing prompt template files
        """
        if prompts_dir is None:
            # Default to the prompts directory relative to this file
            prompts_dir = Path(__file__).parent / "prompts"
        
        self.prompts_dir = Path(prompts_dir)
        self._templates: Dict[str, PromptTemplate] = {}
        
        # Load all templates
        self._load_templates()
    
    def _load_templates(self):
        """Load all prompt templates from the prompts directory"""
        if not self.prompts_dir.exists():
            raise FileNotFoundError(f"Prompts directory not found: {self.prompts_dir}")
        
        for filepath in self.prompts_dir.glob("*.txt"):
            name = filepath.stem  # filename without extension
            with open(filepath, 'r', encoding='utf-8') as f:
                template_text = f.read()
            
            self._templates[name] = PromptTemplate(template_text, name=name)
    
    def get_template(self, name: str) -> PromptTemplate:
        """
        Get a prompt template by name
        
        Args:
            name: Template name (filename without extension)
            
        Returns:
            PromptTemplate instance
        """
        if name not in self._templates:
            raise KeyError(f"Template '{name}' not found. Available: {list(self._templates.keys())}")
        
        return self._templates[name]
    
    def get_prompt(self, name: str, **kwargs) -> str:
        """
        Get a formatted prompt
        
        Args:
            name: Template name
            **kwargs: Variables to substitute
            
        Returns:
            Formatted prompt string
        """
        template = self.get_template(name)
        return template.format(**kwargs)
    
    def list_templates(self) -> Dict[str, set]:
        """
        List all available templates and their variables
        
        Returns:
            Dict mapping template names to their variable sets
        """
        return {name: t.variables for name, t in self._templates.items()}
    
    def reload(self):
        """Reload all templates from disk"""
        self._templates.clear()
        self._load_templates()


# =============================================================================
# Pre-built Prompts for Common Use Cases
# =============================================================================

class ClinicalPrompts:
    """
    Pre-built clinical prompts for the teacher model
    
    Provides easy access to commonly used prompts with sensible defaults.
    """
    
    def __init__(self, prompts_dir: Optional[Path] = None):
        self.manager = PromptManager(prompts_dir)
        # Expose manager as _manager for components that need it
        self._manager = self.manager
    
    def dialogue_generation(
        self,
        scenario: str,
        guidelines_context: str = "No specific guidelines retrieved.",
    ) -> str:
        """
        Generate prompt for dialogue and summary generation
        
        Args:
            scenario: Clinical scenario description
            guidelines_context: Retrieved clinical guidelines
            
        Returns:
            Formatted prompt
        """
        return self.manager.get_prompt(
            "dialogue_generation",
            scenario=scenario,
            guidelines_context=guidelines_context,
        )
    
    def difficulty_assessment(
        self,
        scenario: str,
        dialogue: str,
        summary: str,
    ) -> str:
        """
        Generate prompt for difficulty assessment
        
        Args:
            scenario: Original scenario
            dialogue: Generated dialogue text
            summary: Generated summary text
            
        Returns:
            Formatted prompt
        """
        return self.manager.get_prompt(
            "difficulty_assessment",
            scenario=scenario,
            dialogue=dialogue,
            summary=summary,
        )
    
    def query_expansion(self, scenario: str) -> str:
        """
        Generate prompt for RAG query expansion
        
        Args:
            scenario: Clinical scenario
            
        Returns:
            Formatted prompt
        """
        return self.manager.get_prompt(
            "query_expansion",
            scenario=scenario,
        )


# =============================================================================
# Singleton Access
# =============================================================================

@lru_cache(maxsize=1)
def get_prompt_manager(prompts_dir: Optional[str] = None) -> PromptManager:
    """
    Get cached prompt manager instance
    
    Args:
        prompts_dir: Optional custom prompts directory
        
    Returns:
        PromptManager instance
    """
    path = Path(prompts_dir) if prompts_dir else None
    return PromptManager(path)


@lru_cache(maxsize=1)
def get_clinical_prompts(prompts_dir: Optional[str] = None) -> ClinicalPrompts:
    """
    Get cached clinical prompts instance
    
    Args:
        prompts_dir: Optional custom prompts directory
        
    Returns:
        ClinicalPrompts instance
    """
    path = Path(prompts_dir) if prompts_dir else None
    return ClinicalPrompts(path)


if __name__ == "__main__":
    # Test prompt manager
    print("Testing Prompt Manager")
    print("=" * 60)
    
    # Initialize
    manager = PromptManager()
    
    # List available templates
    print("\nAvailable templates:")
    for name, variables in manager.list_templates().items():
        print(f"  - {name}: {variables}")
    
    # Test dialogue generation prompt
    print("\n" + "-" * 60)
    print("Testing dialogue_generation prompt:")
    
    prompt = manager.get_prompt(
        "dialogue_generation",
        scenario="55-year-old male with chest pain for 3 days",
        guidelines_context="NICE recommends ECG within 10 minutes for suspected ACS."
    )
    
    print(f"\nPrompt length: {len(prompt)} characters")
    print(f"First 200 chars:\n{prompt[:200]}...")
    
    # Test clinical prompts helper
    print("\n" + "-" * 60)
    print("Testing ClinicalPrompts helper:")
    
    clinical = ClinicalPrompts()
    prompt = clinical.dialogue_generation(
        scenario="45-year-old female with palpitations",
        guidelines_context="Consider 12-lead ECG and Holter monitoring."
    )
    
    print(f"Generated prompt length: {len(prompt)} characters")
    
    print("\n✓ All prompt manager tests passed!")