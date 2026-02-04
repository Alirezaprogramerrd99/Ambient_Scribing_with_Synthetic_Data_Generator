"""
Teacher Module

Teacher models for generating synthetic clinical dialogue-summary pairs.
Supports multiple backends: Ollama (local), OpenAI, Anthropic.

Example:
    from src.teacher import create_teacher, OllamaTeacher
    from src.knowledge_base import create_rag_system
    
    # Setup RAG
    retriever = create_rag_system(
        backend="manual",
        documents_dir="./medical_knowledge",
    )
    
    # Create teacher (Ollama - local)
    teacher = create_teacher(
        provider="ollama",
        model_name="llama3.1:8b",
        retriever=retriever,
    )
    
    # Generate sample
    result = teacher.generate("55-year-old male with chest pain for 3 days")
    
    if result.success:
        print(result.sample.dialogue_text)
        print(result.sample.summary.assessment)
"""

# Base classes and data structures
from .base import (
    BaseTeacher,
    GenerationConfig,
    GenerationResult,
    BatchGenerationResult,
    create_generation_config,
)

# Ollama (local) implementation
from .ollama_teacher import (
    OllamaTeacher,
    OllamaClient,
    create_ollama_teacher,
    RECOMMENDED_MODELS,
    get_recommended_model,
)

# API implementations (OpenAI, Anthropic)
from .api_teacher import (
    OpenAITeacher,
    AnthropicTeacher,
    LiteLLMTeacher,
    create_openai_teacher,
    create_anthropic_teacher,
    create_api_teacher,
)


# =============================================================================
# Unified Factory
# =============================================================================

def create_teacher(
    provider: str = "ollama",
    model_name: str = None,
    retriever = None,
    temperature: float = 0.7,
    use_rag: bool = True,
    **kwargs,
) -> BaseTeacher:
    """
    Create a teacher model with the specified backend
    
    Args:
        provider: "ollama", "openai", "anthropic", or "litellm"
        model_name: Model name (uses provider default if None)
        retriever: RAG retriever for clinical guidelines
        temperature: Generation temperature
        use_rag: Enable RAG retrieval
        **kwargs: Additional provider-specific options
        
    Returns:
        Configured teacher instance
        
    Examples:
        # Local Ollama
        teacher = create_teacher("ollama", "llama3.1:8b")
        
        # OpenAI
        teacher = create_teacher("openai", "gpt-4o", api_key="sk-...")
        
        # Anthropic
        teacher = create_teacher("anthropic", "claude-sonnet-4-20250514")
    """
    config = GenerationConfig(
        temperature=temperature,
        use_rag=use_rag,
        **{k: v for k, v in kwargs.items() if hasattr(GenerationConfig, k)},
    )
    
    if provider.lower() == "ollama":
        model_name = model_name or "llama3.1:8b"
        return OllamaTeacher(
            model_name=model_name,
            retriever=retriever,
            config=config,
            base_url=kwargs.get("base_url", "http://localhost:11434"),
            use_json_mode=kwargs.get("use_json_mode", True),
        )
    
    elif provider.lower() == "openai":
        model_name = model_name or "gpt-4o"
        return OpenAITeacher(
            model_name=model_name,
            api_key=kwargs.get("api_key"),
            retriever=retriever,
            config=config,
            use_json_mode=kwargs.get("use_json_mode", True),
        )
    
    elif provider.lower() == "anthropic":
        model_name = model_name or "claude-sonnet-4-20250514"
        return AnthropicTeacher(
            model_name=model_name,
            api_key=kwargs.get("api_key"),
            retriever=retriever,
            config=config,
        )
    
    elif provider.lower() == "litellm":
        model_name = model_name or "gpt-4o"
        return LiteLLMTeacher(
            model_name=model_name,
            retriever=retriever,
            config=config,
            fallback_models=kwargs.get("fallback_models"),
        )
    
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Supported: ollama, openai, anthropic, litellm"
        )


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Base
    "BaseTeacher",
    "GenerationConfig",
    "GenerationResult",
    "BatchGenerationResult",
    "create_generation_config",
    # Ollama
    "OllamaTeacher",
    "OllamaClient",
    "create_ollama_teacher",
    "RECOMMENDED_MODELS",
    "get_recommended_model",
    # API
    "OpenAITeacher",
    "AnthropicTeacher",
    "LiteLLMTeacher",
    "create_openai_teacher",
    "create_anthropic_teacher",
    "create_api_teacher",
    # Factory
    "create_teacher",
]