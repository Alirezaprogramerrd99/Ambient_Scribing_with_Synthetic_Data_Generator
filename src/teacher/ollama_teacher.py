"""
Ollama Teacher Model

Teacher model implementation using Ollama for local LLM inference.
Supports various open-source models like Llama 3.1, Mistral, etc.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

import httpx

from .base import BaseTeacher, GenerationConfig, GenerationResult
from src.knowledge_base import BaseRetriever
from src.config import ClinicalPrompts
from src.utils import RateLimitError, ConnectionError

logger = logging.getLogger(__name__)


# =============================================================================
# Ollama Client
# =============================================================================

class OllamaClient:
    """
    Simple Ollama API client
    
    Provides low-level access to Ollama's generate and chat endpoints.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 300.0,
    ):
        """
        Initialize Ollama client
        
        Args:
            base_url: Ollama server URL
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)
    
    def is_available(self) -> bool:
        """Check if Ollama server is available"""
        try:
            response = self._client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False
    
    def list_models(self) -> List[str]:
        """List available models"""
        try:
            response = self._client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def model_exists(self, model_name: str) -> bool:
        """Check if a model is available"""
        models = self.list_models()
        # Handle both "llama3.1:8b" and "llama3.1" formats
        return any(
            model_name in m or m.startswith(model_name.split(":")[0])
            for m in models
        )
    
    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 0.9,
        top_k: int = 40,
        stream: bool = False,
        format: Optional[str] = None,  # "json" for JSON mode
    ) -> str:
        """
        Generate completion using Ollama
        
        Args:
            model: Model name
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            top_p: Top-p sampling
            top_k: Top-k sampling
            stream: Whether to stream response
            format: Output format ("json" for JSON mode)
            
        Returns:
            Generated text
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": top_p,
                "top_k": top_k,
            },
        }
        
        if format:
            payload["format"] = format
        
        try:
            response = self._client.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("response", "")
            
        except httpx.TimeoutException:
            raise TimeoutError(f"Ollama request timed out after {self.timeout}s")
        except httpx.ConnectError:
            raise ConnectionError(f"Cannot connect to Ollama at {self.base_url}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError("Ollama rate limit exceeded")
            raise
    
    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        format: Optional[str] = None,
    ) -> str:
        """
        Chat completion using Ollama
        
        Args:
            model: Model name
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Max tokens
            format: Output format
            
        Returns:
            Assistant's response
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        
        if format:
            payload["format"] = format
        
        try:
            response = self._client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("message", {}).get("content", "")
            
        except httpx.TimeoutException:
            raise TimeoutError(f"Ollama request timed out after {self.timeout}s")
        except httpx.ConnectError:
            raise ConnectionError(f"Cannot connect to Ollama at {self.base_url}")
    
    def close(self):
        """Close the HTTP client"""
        self._client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# =============================================================================
# Ollama Teacher
# =============================================================================

class OllamaTeacher(BaseTeacher):
    """
    Teacher model using Ollama for local inference
    
    Supports models like:
    - llama3.1:8b, llama3.1:70b
    - mistral:7b, mixtral:8x7b
    - phi3:mini, phi3:medium
    - codellama:13b
    - medllama2:7b (medical fine-tuned)
    
    Example:
        teacher = OllamaTeacher(
            model_name="llama3.1:8b",
            retriever=rag_retriever,
        )
        
        result = teacher.generate("55-year-old male with chest pain")
        if result.success:
            print(result.sample.dialogue_text)
    """
    
    def __init__(
        self,
        model_name: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        retriever: Optional[BaseRetriever] = None,
        config: Optional[GenerationConfig] = None,
        prompts: Optional[ClinicalPrompts] = None,
        timeout: float = 300.0,
        use_json_mode: bool = True,
    ):
        """
        Initialize Ollama teacher
        
        Args:
            model_name: Ollama model name (e.g., "llama3.1:8b")
            base_url: Ollama server URL
            retriever: RAG retriever for guidelines
            config: Generation configuration
            prompts: Prompt templates
            timeout: Request timeout
            use_json_mode: Use Ollama's JSON mode for structured output
        """
        super().__init__(
            model_name=model_name,
            retriever=retriever,
            config=config,
            prompts=prompts,
        )
        
        self.base_url = base_url
        self.timeout = timeout
        self.use_json_mode = use_json_mode
        
        # Create client
        self.client = OllamaClient(base_url=base_url, timeout=timeout)
        
        # Verify connection and model
        self._verify_setup()
    
    @property
    def provider(self) -> str:
        return "ollama"
    
    def _verify_setup(self):
        """Verify Ollama is accessible and model is available"""
        
        if not self.client.is_available():
            logger.warning(
                f"Ollama server not available at {self.base_url}. "
                "Make sure Ollama is running: `ollama serve`"
            )
            return
        
        if not self.client.model_exists(self.model_name):
            available = self.client.list_models()
            logger.warning(
                f"Model '{self.model_name}' not found. "
                f"Available models: {available}. "
                f"Pull with: `ollama pull {self.model_name}`"
            )
    
    def _check_connection(self) -> bool:
        """Check if Ollama is accessible"""
        return self.client.is_available()
    
    # NOTE: No @retry decorator here — retries are handled by
    # BaseTeacher.generate() via RetryContext to avoid compounding
    # (decorator retries × context retries = excessive attempts).
    def _call_llm(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Call Ollama to generate response
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Raw response string
        """
        temperature = temperature or self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens
        
        # Use JSON mode if enabled
        format_mode = "json" if self.use_json_mode else None
        
        response = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            format=format_mode,
        )
        
        return response
    
    def generate_with_system_prompt(
        self,
        scenario: str,
        system_prompt: Optional[str] = None,
        use_rag: Optional[bool] = None,
    ) -> GenerationResult:
        """
        Generate using chat format with system prompt
        
        Some models perform better with explicit system prompts.
        
        Args:
            scenario: Clinical scenario
            system_prompt: Custom system prompt
            use_rag: Override RAG setting
            
        Returns:
            GenerationResult
        """
        import time
        from src.models import RAGMetadata
        start_time = time.time()
        use_rag = use_rag if use_rag is not None else self.config.use_rag
        
        # Get RAG context
        if use_rag and self.retriever:
            guidelines_context, rag_meta = self.retrieve_guidelines(scenario)
        else:
            guidelines_context = "No specific guidelines retrieved."
            rag_meta = RAGMetadata(rag_enabled=False)
        
        # Default system prompt
        if system_prompt is None:
            system_prompt = (
                "You are an expert medical documentation specialist. "
                "Generate realistic clinical dialogues and comprehensive summaries. "
                "Always respond with valid JSON."
            )
        
        # Build user prompt
        user_prompt = self.prompts.dialogue_generation(
            scenario=scenario,
            guidelines_context=guidelines_context,
        )
        
        # Chat messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        # Generate with retries (matching BaseTeacher.generate() behaviour)
        from src.utils import RetryContext
        with RetryContext(
            max_attempts=self.config.max_retries,
            min_wait_seconds=self.config.retry_delay,
        ) as retry_ctx:
            while retry_ctx.should_retry():
                try:
                    raw_response = self.client.chat(
                        model=self.model_name,
                        messages=messages,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        format="json" if self.use_json_mode else None,
                    )
                    
                    sample = self._parse_response(
                        raw_response=raw_response,
                        scenario=scenario,
                        rag_meta=rag_meta,
                    )
                    retry_ctx.success(sample)
                    
                except Exception as e:
                    logger.warning(f"Chat generation attempt failed: {e}")
                    retry_ctx.failed(e)
        
        generation_time = time.time() - start_time
        
        if retry_ctx.succeeded:
            sample = retry_ctx.result
            sample.generation.generation_time_seconds = generation_time
            self._total_generated += 1
            
            # Difficulty assessment (matching BaseTeacher.generate() behaviour)
            if self.config.include_difficulty:
                try:
                    sample.difficulty = self.assess_difficulty(sample)
                except Exception as e:
                    logger.warning(f"Difficulty assessment failed: {e}")
                    if sample.difficulty is None:
                        sample.difficulty = self._heuristic_difficulty_assessment(sample)
            
            return GenerationResult(
                success=True,
                sample=sample,
                raw_response=raw_response,
                attempts=retry_ctx.attempts,
                generation_time_seconds=generation_time,
            )
        else:
            self._total_failed += 1
            return GenerationResult(
                success=False,
                error=str(retry_ctx.last_exception),
                attempts=retry_ctx.attempts,
                generation_time_seconds=generation_time,
            )
    
    def warm_up(self, num_tokens: int = 10) -> float:
        """
        Warm up the model with a simple generation
        
        Useful for getting consistent timing measurements.
        
        Args:
            num_tokens: Number of tokens to generate
            
        Returns:
            Time taken in seconds
        """
        start = time.time()
        
        try:
            self.client.generate(
                model=self.model_name,
                prompt="Hello, how are you?",
                max_tokens=num_tokens,
                temperature=0.1,
            )
        except Exception as e:
            logger.warning(f"Warm-up failed: {e}")
        
        elapsed = time.time() - start
        logger.info(f"Model warm-up completed in {elapsed:.2f}s")
        return elapsed
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        try:
            response = self.client._client.post(
                f"{self.base_url}/api/show",
                json={"name": self.model_name},
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {}
    
    def close(self):
        """Close the Ollama client"""
        self.client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# =============================================================================
# Factory Function
# =============================================================================

def create_ollama_teacher(
    model_name: str = "llama3.1:8b",
    base_url: str = "http://localhost:11434",
    retriever: Optional[BaseRetriever] = None,
    temperature: float = 0.7,
    use_rag: bool = True,
    use_json_mode: bool = True,
    **kwargs,
) -> OllamaTeacher:
    """
    Factory function to create Ollama teacher
    
    Args:
        model_name: Ollama model name
        base_url: Ollama server URL
        retriever: RAG retriever
        temperature: Generation temperature
        use_rag: Enable RAG
        use_json_mode: Use JSON mode
        **kwargs: Additional GenerationConfig options
        
    Returns:
        Configured OllamaTeacher
    """
    config = GenerationConfig(
        temperature=temperature,
        use_rag=use_rag,
        **kwargs,
    )
    
    return OllamaTeacher(
        model_name=model_name,
        base_url=base_url,
        retriever=retriever,
        config=config,
        use_json_mode=use_json_mode,
    )


# =============================================================================
# Recommended Models
# =============================================================================

RECOMMENDED_MODELS = {
    "general": [
        "llama3.1:8b",        # Good balance of quality and speed
        "llama3.1:70b",       # Best quality (requires 48GB+ VRAM)
        "mistral:7b",         # Fast, good quality
        "mixtral:8x7b",       # MoE, good for complex tasks
    ],
    "medical": [
        "medllama2:7b",       # Medical fine-tuned
        "meditron:7b",        # Medical domain
    ],
    "small": [
        "phi3:mini",          # 3.8B, very fast
        "gemma:2b",           # Small but capable
    ],
    "coding": [
        "codellama:13b",      # Good for structured output
    ],
}


def get_recommended_model(
    use_case: str = "general",
    max_vram_gb: int = 16,
) -> str:
    """
    Get recommended model based on use case and hardware
    
    Args:
        use_case: "general", "medical", "small", or "coding"
        max_vram_gb: Maximum available VRAM
        
    Returns:
        Recommended model name
    """
    models = RECOMMENDED_MODELS.get(use_case, RECOMMENDED_MODELS["general"])
    
    # Filter by VRAM requirement (rough estimates)
    vram_requirements = {
        "llama3.1:8b": 8,
        "llama3.1:70b": 48,
        "mistral:7b": 6,
        "mixtral:8x7b": 32,
        "phi3:mini": 4,
        "medllama2:7b": 6,
    }
    
    for model in models:
        required = vram_requirements.get(model, 8)
        if required <= max_vram_gb:
            return model
    
    # Default to smallest
    return "phi3:mini"


if __name__ == "__main__":
    print("Ollama Teacher Module")
    print("=" * 60)
    
    # Check Ollama availability
    client = OllamaClient()
    
    if client.is_available():
        print("✓ Ollama server is running")
        print(f"  Available models: {client.list_models()}")
    else:
        print("✗ Ollama server not available")
        print("  Start with: ollama serve")
    
    print()
    print("Recommended models by use case:")
    for use_case, models in RECOMMENDED_MODELS.items():
        print(f"  {use_case}: {models}")
    
    client.close()