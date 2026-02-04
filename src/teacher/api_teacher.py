"""
API-Based Teacher Models

Teacher model implementations using cloud APIs (OpenAI, Anthropic).
Useful for high-quality synthetic data generation and comparison.

"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from .base import BaseTeacher, GenerationConfig, GenerationResult
from src.knowledge_base import BaseRetriever
from src.config import ClinicalPrompts
from src.utils import (
    retry_with_exponential_backoff,
    RateLimitError,
    MaxRetriesExceededError,
)

logger = logging.getLogger(__name__)


# =============================================================================
# OpenAI Teacher
# =============================================================================

class OpenAITeacher(BaseTeacher):
    """
    Teacher model using OpenAI API
    
    Supports GPT-4, GPT-4 Turbo, GPT-4o, etc.
    Excellent for high-quality synthetic data generation.
    
    Example:
        teacher = OpenAITeacher(
            model_name="gpt-4o",
            api_key="sk-...",
            retriever=rag_retriever,
        )
        
        result = teacher.generate("55-year-old male with chest pain")
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        retriever: Optional[BaseRetriever] = None,
        config: Optional[GenerationConfig] = None,
        prompts: Optional[ClinicalPrompts] = None,
        organization: Optional[str] = None,
        use_json_mode: bool = True,
    ):
        """
        Initialize OpenAI teacher
        
        Args:
            model_name: OpenAI model name (e.g., "gpt-4o", "gpt-4-turbo")
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            retriever: RAG retriever
            config: Generation configuration
            prompts: Prompt templates
            organization: OpenAI organization ID
            use_json_mode: Use JSON response format
        """
        super().__init__(
            model_name=model_name,
            retriever=retriever,
            config=config,
            prompts=prompts,
        )
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.organization = organization or os.getenv("OPENAI_ORG_ID")
        self.use_json_mode = use_json_mode
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self._client = None
        self._initialize_client()
    
    @property
    def provider(self) -> str:
        return "openai"
    
    def _initialize_client(self):
        """Initialize OpenAI client"""
        try:
            from openai import OpenAI
            
            self._client = OpenAI(
                api_key=self.api_key,
                organization=self.organization,
            )
            logger.info(f"OpenAI client initialized: {self.model_name}")
            
        except ImportError:
            raise ImportError(
                "OpenAI package required. Install with: pip install openai"
            )
    
    def _check_connection(self) -> bool:
        """Check if OpenAI API is accessible"""
        try:
            # Simple API call to verify connection
            self._client.models.list()
            return True
        except Exception as e:
            logger.error(f"OpenAI connection check failed: {e}")
            return False
    
    @retry_with_exponential_backoff(
        max_attempts=3,
        min_wait_seconds=1.0,
        max_wait_seconds=60.0,
    )
    def _call_llm(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Call OpenAI API
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            
        Returns:
            Generated text
        """
        temperature = temperature or self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens
        
        # Build messages
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert medical documentation specialist. "
                    "Generate realistic clinical dialogues and comprehensive summaries. "
                    "Always respond with valid JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        
        # API call parameters
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        # Add JSON mode if supported
        if self.use_json_mode:
            params["response_format"] = {"type": "json_object"}
        
        try:
            response = self._client.chat.completions.create(**params)
            return response.choices[0].message.content
            
        except Exception as e:
            error_str = str(e).lower()
            if "rate" in error_str or "limit" in error_str:
                raise RateLimitError(f"OpenAI rate limit: {e}")
            raise
    
    def estimate_cost(self, num_samples: int, avg_tokens: int = 3000) -> Dict[str, float]:
        """
        Estimate API cost for generating samples
        
        Args:
            num_samples: Number of samples to generate
            avg_tokens: Average tokens per sample (input + output)
            
        Returns:
            Cost estimate dictionary
        """
        # Pricing per 1M tokens (approximate, check OpenAI for current pricing)
        pricing = {
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},
            "gpt-4": {"input": 30.00, "output": 60.00},
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        }
        
        model_pricing = pricing.get(
            self.model_name,
            pricing["gpt-4o"]  # Default
        )
        
        total_tokens = num_samples * avg_tokens
        input_tokens = total_tokens * 0.6  # Rough estimate
        output_tokens = total_tokens * 0.4
        
        input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
        
        return {
            "estimated_total_tokens": total_tokens,
            "estimated_input_tokens": input_tokens,
            "estimated_output_tokens": output_tokens,
            "estimated_cost_usd": input_cost + output_cost,
            "model": self.model_name,
        }


# =============================================================================
# Anthropic Teacher
# =============================================================================

class AnthropicTeacher(BaseTeacher):
    """
    Teacher model using Anthropic API (Claude)
    
    Supports Claude 3.5 Sonnet, Claude 3 Opus, etc.
    Excellent for nuanced clinical content generation.
    
    Example:
        teacher = AnthropicTeacher(
            model_name="claude-sonnet-4-20250514",
            api_key="sk-ant-...",
            retriever=rag_retriever,
        )
        
        result = teacher.generate("55-year-old male with chest pain")
    """
    
    def __init__(
        self,
        model_name: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        retriever: Optional[BaseRetriever] = None,
        config: Optional[GenerationConfig] = None,
        prompts: Optional[ClinicalPrompts] = None,
    ):
        """
        Initialize Anthropic teacher
        
        Args:
            model_name: Anthropic model name
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            retriever: RAG retriever
            config: Generation configuration
            prompts: Prompt templates
        """
        super().__init__(
            model_name=model_name,
            retriever=retriever,
            config=config,
            prompts=prompts,
        )
        
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self._client = None
        self._initialize_client()
    
    @property
    def provider(self) -> str:
        return "anthropic"
    
    def _initialize_client(self):
        """Initialize Anthropic client"""
        try:
            from anthropic import Anthropic
            
            self._client = Anthropic(api_key=self.api_key)
            logger.info(f"Anthropic client initialized: {self.model_name}")
            
        except ImportError:
            raise ImportError(
                "Anthropic package required. Install with: pip install anthropic"
            )
    
    def _check_connection(self) -> bool:
        """Check if Anthropic API is accessible"""
        try:
            # Simple API call to verify
            self._client.messages.create(
                model=self.model_name,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return True
        except Exception as e:
            logger.error(f"Anthropic connection check failed: {e}")
            return False
    
    @retry_with_exponential_backoff(
        max_attempts=3,
        min_wait_seconds=1.0,
        max_wait_seconds=60.0,
    )
    def _call_llm(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Call Anthropic API
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            
        Returns:
            Generated text
        """
        temperature = temperature or self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens
        
        # System prompt
        system = (
            "You are an expert medical documentation specialist. "
            "Generate realistic clinical dialogues and comprehensive summaries. "
            "Always respond with valid JSON only, no additional text or explanation."
        )
        
        try:
            response = self._client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            
            # Extract text from response
            return response.content[0].text
            
        except Exception as e:
            error_str = str(e).lower()
            if "rate" in error_str or "limit" in error_str:
                raise RateLimitError(f"Anthropic rate limit: {e}")
            raise
    
    def estimate_cost(self, num_samples: int, avg_tokens: int = 3000) -> Dict[str, float]:
        """
        Estimate API cost for generating samples
        
        Args:
            num_samples: Number of samples to generate
            avg_tokens: Average tokens per sample
            
        Returns:
            Cost estimate dictionary
        """
        # Pricing per 1M tokens (approximate)
        pricing = {
            "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
            "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
            "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
            "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
            "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
        }
        
        model_pricing = pricing.get(
            self.model_name,
            pricing["claude-sonnet-4-20250514"]
        )
        
        total_tokens = num_samples * avg_tokens
        input_tokens = total_tokens * 0.6
        output_tokens = total_tokens * 0.4
        
        input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
        
        return {
            "estimated_total_tokens": total_tokens,
            "estimated_input_tokens": input_tokens,
            "estimated_output_tokens": output_tokens,
            "estimated_cost_usd": input_cost + output_cost,
            "model": self.model_name,
        }


# =============================================================================
# LiteLLM Teacher (Unified Interface)
# =============================================================================

class LiteLLMTeacher(BaseTeacher):
    """
    Teacher model using LiteLLM for unified API access
    
    LiteLLM provides a single interface to multiple providers:
    - OpenAI, Anthropic, Cohere, Azure, etc.
    - Easy switching between models
    - Automatic fallback support
    
    Example:
        teacher = LiteLLMTeacher(
            model_name="gpt-4o",  # or "claude-3-5-sonnet", etc.
            retriever=rag_retriever,
        )
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        retriever: Optional[BaseRetriever] = None,
        config: Optional[GenerationConfig] = None,
        prompts: Optional[ClinicalPrompts] = None,
        fallback_models: Optional[List[str]] = None,
    ):
        """
        Initialize LiteLLM teacher
        
        Args:
            model_name: Model name in LiteLLM format
            retriever: RAG retriever
            config: Generation configuration
            prompts: Prompt templates
            fallback_models: List of fallback models
        """
        super().__init__(
            model_name=model_name,
            retriever=retriever,
            config=config,
            prompts=prompts,
        )
        
        self.fallback_models = fallback_models or []
        self._verify_litellm()
    
    @property
    def provider(self) -> str:
        # Determine provider from model name
        if "gpt" in self.model_name.lower():
            return "openai"
        elif "claude" in self.model_name.lower():
            return "anthropic"
        elif "ollama" in self.model_name.lower():
            return "ollama"
        else:
            return "litellm"
    
    def _verify_litellm(self):
        """Verify LiteLLM is installed"""
        try:
            import litellm
            self._litellm = litellm
            logger.info(f"LiteLLM initialized: {self.model_name}")
        except ImportError:
            raise ImportError(
                "LiteLLM required. Install with: pip install litellm"
            )
    
    def _check_connection(self) -> bool:
        """Check API connection"""
        try:
            self._litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
            )
            return True
        except Exception as e:
            logger.error(f"LiteLLM connection check failed: {e}")
            return False
    
    @retry_with_exponential_backoff(
        max_attempts=3,
        min_wait_seconds=1.0,
        max_wait_seconds=60.0,
    )
    def _call_llm(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Call LLM via LiteLLM
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            
        Returns:
            Generated text
        """
        temperature = temperature or self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert medical documentation specialist. "
                    "Generate realistic clinical dialogues and comprehensive summaries. "
                    "Always respond with valid JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        
        try:
            response = self._litellm.completion(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            # Try fallback models
            for fallback in self.fallback_models:
                try:
                    logger.warning(f"Primary model failed, trying fallback: {fallback}")
                    response = self._litellm.completion(
                        model=fallback,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return response.choices[0].message.content
                except:
                    continue
            
            raise


# =============================================================================
# Factory Functions
# =============================================================================

def create_openai_teacher(
    model_name: str = "gpt-4o",
    api_key: Optional[str] = None,
    retriever: Optional[BaseRetriever] = None,
    temperature: float = 0.7,
    **kwargs,
) -> OpenAITeacher:
    """Create OpenAI teacher with common defaults"""
    config = GenerationConfig(temperature=temperature, **kwargs)
    return OpenAITeacher(
        model_name=model_name,
        api_key=api_key,
        retriever=retriever,
        config=config,
    )


def create_anthropic_teacher(
    model_name: str = "claude-sonnet-4-20250514",
    api_key: Optional[str] = None,
    retriever: Optional[BaseRetriever] = None,
    temperature: float = 0.7,
    **kwargs,
) -> AnthropicTeacher:
    """Create Anthropic teacher with common defaults"""
    config = GenerationConfig(temperature=temperature, **kwargs)
    return AnthropicTeacher(
        model_name=model_name,
        api_key=api_key,
        retriever=retriever,
        config=config,
    )


def create_api_teacher(
    provider: str,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    retriever: Optional[BaseRetriever] = None,
    **kwargs,
) -> BaseTeacher:
    """
    Factory to create API-based teacher
    
    Args:
        provider: "openai", "anthropic", or "litellm"
        model_name: Model name (uses provider default if None)
        api_key: API key
        retriever: RAG retriever
        **kwargs: Additional configuration
        
    Returns:
        Configured teacher instance
    """
    if provider.lower() == "openai":
        model_name = model_name or "gpt-4o"
        return create_openai_teacher(model_name, api_key, retriever, **kwargs)
    
    elif provider.lower() == "anthropic":
        model_name = model_name or "claude-sonnet-4-20250514"
        return create_anthropic_teacher(model_name, api_key, retriever, **kwargs)
    
    elif provider.lower() == "litellm":
        model_name = model_name or "gpt-4o"
        config = GenerationConfig(**kwargs)
        return LiteLLMTeacher(model_name, retriever, config)
    
    else:
        raise ValueError(f"Unknown provider: {provider}")


if __name__ == "__main__":
    print("API Teacher Module")
    print("=" * 60)
    print()
    print("Available API Teachers:")
    print("  - OpenAITeacher: GPT-4, GPT-4o, etc.")
    print("  - AnthropicTeacher: Claude 3.5, Claude 3, etc.")
    print("  - LiteLLMTeacher: Unified interface for all providers")
    print()
    print("Usage:")
    print("  from src.teacher import create_api_teacher")
    print("  teacher = create_api_teacher('openai', model_name='gpt-4o')")
    print("  result = teacher.generate('55-year-old with chest pain')")