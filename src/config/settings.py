"""
Configuration Management using Pydantic Settings

This module provides centralized, type-safe configuration for the entire project.
Settings are loaded from environment variables and .env file.

Author: Alireza Rashidi
MSc Project: Trustworthy SLMs for Ambient Clinical Scribing
"""

from pathlib import Path
from typing import Literal, Optional
from functools import lru_cache

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class OllamaSettings(BaseSettings):
    """Ollama (Local LLM) Configuration"""
    
    model_config = SettingsConfigDict(env_prefix="OLLAMA_")
    
    base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API endpoint"
    )
    model: str = Field(
        default="llama3.1:8b",
        description="Ollama model name"
    )
    request_timeout: float = Field(
        default=300.0,
        description="Request timeout in seconds"
    )
    num_ctx: int = Field(
        default=4096,
        description="Context window size"
    )
    num_predict: int = Field(
        default=2048,
        description="Maximum tokens to generate"
    )


class OpenAISettings(BaseSettings):
    """OpenAI Configuration"""
    
    model_config = SettingsConfigDict(env_prefix="OPENAI_")
    
    api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key"
    )
    model: str = Field(
        default="gpt-4o",
        description="OpenAI model name"
    )
    org_id: Optional[str] = Field(
        default=None,
        description="OpenAI organization ID"
    )
    
    @property
    def is_configured(self) -> bool:
        return self.api_key is not None


class AnthropicSettings(BaseSettings):
    """Anthropic (Claude) Configuration"""
    
    model_config = SettingsConfigDict(env_prefix="ANTHROPIC_")
    
    api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key"
    )
    model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Anthropic model name"
    )
    
    @property
    def is_configured(self) -> bool:
        return self.api_key is not None


class QdrantSettings(BaseSettings):
    """Qdrant Vector Database Configuration"""
    
    model_config = SettingsConfigDict(env_prefix="QDRANT_")
    
    host: str = Field(default="localhost")
    port: int = Field(default=6333)
    collection: str = Field(default="medical_knowledge")
    api_key: Optional[str] = Field(default=None)
    
    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


class ChromaSettings(BaseSettings):
    """ChromaDB Configuration"""
    
    model_config = SettingsConfigDict(env_prefix="CHROMA_")
    
    persist_dir: Path = Field(default=Path("./chroma_db"))
    collection_name: str = Field(default="medical_knowledge")


class EmbeddingSettings(BaseSettings):
    """Embedding Model Configuration"""
    
    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")
    
    model: str = Field(
        default="BAAI/bge-base-en-v1.5",
        description="HuggingFace embedding model name"
    )
    # Clinical alternatives:
    # - "pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb"
    # - "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    
    trust_remote_code: bool = Field(default=True)
    device: str = Field(default="cpu")  # or "cuda"


class KnowledgeBaseSettings(BaseSettings):
    """Knowledge Base Configuration"""
    
    model_config = SettingsConfigDict(env_prefix="")
    
    knowledge_base_path: Path = Field(
        default=Path("./medical_knowledge"),
        description="Root path for medical knowledge documents"
    )
    nice_guidelines_path: Path = Field(
        default=Path("./medical_knowledge/nice_guidelines"),
        description="Path to NICE guidelines"
    )
    
    # Chunking configuration
    chunk_size: int = Field(default=512, description="Chunk size in tokens")
    chunk_overlap: int = Field(default=50, description="Overlap between chunks")
    
    @field_validator("knowledge_base_path", "nice_guidelines_path", mode="before")
    @classmethod
    def convert_to_path(cls, v):
        return Path(v) if isinstance(v, str) else v


class GenerationSettings(BaseSettings):
    """Synthetic Data Generation Configuration"""
    
    model_config = SettingsConfigDict(env_prefix="DEFAULT_")
    
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    max_tokens: int = Field(
        default=2048,
        description="Maximum tokens to generate"
    )
    top_k_retrieval: int = Field(
        default=5,
        description="Number of documents to retrieve for RAG"
    )
    
    # Batch processing
    batch_size: int = Field(default=10)
    save_interval: int = Field(default=10)
    max_retries: int = Field(default=3)


class ValidationSettings(BaseSettings):
    """Validation Configuration"""
    
    model_config = SettingsConfigDict(env_prefix="")
    
    enable_clinical_validation: bool = Field(
        default=True,
        description="Enable clinical entity validation with scispaCy/MedCAT"
    )
    enable_rag_metrics: bool = Field(
        default=True,
        description="Enable RAGAS metrics for RAG quality"
    )
    min_dialogue_turns: int = Field(
        default=6,
        description="Minimum number of dialogue turns required"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum retries for failed generations"
    )


class ExperimentTrackingSettings(BaseSettings):
    """MLflow and W&B Configuration"""
    
    model_config = SettingsConfigDict(env_prefix="")
    
    mlflow_tracking_uri: str = Field(default="./mlruns")
    mlflow_experiment_name: str = Field(default="ambient-scribe-teacher")
    
    wandb_project: str = Field(default="ambient-scribe-teacher")
    wandb_entity: Optional[str] = Field(default=None)
    
    enable_mlflow: bool = Field(default=True)
    enable_wandb: bool = Field(default=False)


class PathSettings(BaseSettings):
    """Output Paths Configuration"""
    
    model_config = SettingsConfigDict(env_prefix="")
    
    synthetic_output_dir: Path = Field(default=Path("./data/synthetic_output"))
    processed_data_dir: Path = Field(default=Path("./data/processed"))
    raw_data_dir: Path = Field(default=Path("./data/raw"))
    
    @field_validator("synthetic_output_dir", "processed_data_dir", "raw_data_dir", mode="before")
    @classmethod
    def convert_to_path(cls, v):
        return Path(v) if isinstance(v, str) else v
    
    def ensure_dirs_exist(self):
        """Create output directories if they don't exist"""
        for path in [self.synthetic_output_dir, self.processed_data_dir, self.raw_data_dir]:
            path.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """
    Main Settings Class - Aggregates all configuration sections
    
    Usage:
        from src.config.settings import get_settings
        settings = get_settings()
        print(settings.ollama.model)
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Sub-settings (composed)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    chroma: ChromaSettings = Field(default_factory=ChromaSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    knowledge_base: KnowledgeBaseSettings = Field(default_factory=KnowledgeBaseSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    validation: ValidationSettings = Field(default_factory=ValidationSettings)
    tracking: ExperimentTrackingSettings = Field(default_factory=ExperimentTrackingSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    
    # Global settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    
    # Default vector store to use
    vector_store: Literal["qdrant", "chroma"] = Field(default="chroma")
    
    # Default LLM provider
    default_provider: Literal["ollama", "openai", "anthropic"] = Field(default="ollama")
    
    @model_validator(mode="after")
    def validate_api_keys_for_provider(self):
        """Ensure API keys are set for selected cloud providers"""
        if self.default_provider == "openai" and not self.openai.is_configured:
            raise ValueError("OpenAI API key required when using OpenAI as default provider")
        if self.default_provider == "anthropic" and not self.anthropic.is_configured:
            raise ValueError("Anthropic API key required when using Anthropic as default provider")
        return self
    
    def get_llm_config(self) -> dict:
        """Get configuration for the default LLM provider"""
        if self.default_provider == "ollama":
            return {
                "provider": "ollama",
                "model": self.ollama.model,
                "base_url": self.ollama.base_url,
                "temperature": self.generation.temperature,
                "max_tokens": self.generation.max_tokens,
            }
        elif self.default_provider == "openai":
            return {
                "provider": "openai",
                "model": self.openai.model,
                "api_key": self.openai.api_key,
                "temperature": self.generation.temperature,
                "max_tokens": self.generation.max_tokens,
            }
        elif self.default_provider == "anthropic":
            return {
                "provider": "anthropic",
                "model": self.anthropic.model,
                "api_key": self.anthropic.api_key,
                "temperature": self.generation.temperature,
                "max_tokens": self.generation.max_tokens,
            }
    
    def print_config(self):
        """Print current configuration (hiding sensitive values)"""
        print("=" * 60)
        print("AMBIENT SCRIBE TEACHER - CONFIGURATION")
        print("=" * 60)
        print(f"Default Provider: {self.default_provider}")
        print(f"Vector Store: {self.vector_store}")
        print(f"Log Level: {self.log_level}")
        print()
        print("LLM Settings:")
        print(f"  Ollama Model: {self.ollama.model}")
        print(f"  Ollama URL: {self.ollama.base_url}")
        print(f"  OpenAI Configured: {self.openai.is_configured}")
        print(f"  Anthropic Configured: {self.anthropic.is_configured}")
        print()
        print("Generation Settings:")
        print(f"  Temperature: {self.generation.temperature}")
        print(f"  Max Tokens: {self.generation.max_tokens}")
        print(f"  Top-K Retrieval: {self.generation.top_k_retrieval}")
        print()
        print("Paths:")
        print(f"  Knowledge Base: {self.knowledge_base.knowledge_base_path}")
        print(f"  Output Dir: {self.paths.synthetic_output_dir}")
        print("=" * 60)


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance
    
    Uses lru_cache to ensure settings are only loaded once.
    
    Returns:
        Settings: Configured settings instance
    """
    return Settings()


# Convenience function for quick access
def get_llm_settings():
    """Get LLM configuration for the default provider"""
    return get_settings().get_llm_config()


if __name__ == "__main__":
    # Test settings loading
    settings = get_settings()
    settings.print_config()