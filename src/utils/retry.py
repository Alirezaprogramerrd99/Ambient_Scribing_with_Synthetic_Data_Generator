"""
Retry Logic with Exponential Backoff

Provides robust retry mechanisms for LLM API calls and other fallible operations.
Essential for production-quality synthetic data generation.

Author: Alireza Rashidi
MSc Project: Trustworthy SLMs for Ambient Clinical Scribing
"""

import asyncio
import logging
import random
import time
from functools import wraps
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from tenacity import (
    retry,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
    wait_random_exponential,
    retry_if_exception_type,
    retry_if_result,
    before_sleep_log,
    after_log,
    RetryError,
)

logger = logging.getLogger(__name__)

# Type variable for generic return types
T = TypeVar("T")


# =============================================================================
# Exception Classes
# =============================================================================

class RetryableError(Exception):
    """Base class for errors that should trigger a retry"""
    pass


class RateLimitError(RetryableError):
    """Rate limit exceeded - should retry with backoff"""
    pass


class ModelOverloadedError(RetryableError):
    """Model is overloaded - should retry with backoff"""
    pass


class ConnectionError(RetryableError):
    """Connection failed - should retry"""
    pass


class InvalidResponseError(Exception):
    """Response was invalid and cannot be retried"""
    pass


class MaxRetriesExceededError(Exception):
    """Maximum retry attempts exceeded"""
    
    def __init__(self, message: str, last_exception: Optional[Exception] = None):
        super().__init__(message)
        self.last_exception = last_exception


# =============================================================================
# Retry Configurations
# =============================================================================

# Default retryable exceptions for LLM calls
DEFAULT_RETRYABLE_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    RetryableError,
    RateLimitError,
    ModelOverloadedError,
    ConnectionError,
    TimeoutError,
    # Add common HTTP/network errors
)


def is_retryable_error(exception: Exception) -> bool:
    """
    Check if an exception should trigger a retry
    
    Args:
        exception: The exception to check
        
    Returns:
        True if the error is retryable
    """
    # Check for known retryable types
    if isinstance(exception, DEFAULT_RETRYABLE_EXCEPTIONS):
        return True
    
    # Check for common error patterns in exception messages
    error_message = str(exception).lower()
    retryable_patterns = [
        "rate limit",
        "rate_limit",
        "too many requests",
        "429",
        "503",
        "502",
        "504",
        "overloaded",
        "timeout",
        "timed out",
        "connection",
        "temporarily unavailable",
        "service unavailable",
    ]
    
    return any(pattern in error_message for pattern in retryable_patterns)


def is_none_or_empty(result: Any) -> bool:
    """Check if result is None or empty (for retry_if_result)"""
    if result is None:
        return True
    if isinstance(result, (str, list, dict)) and len(result) == 0:
        return True
    return False


# =============================================================================
# Retry Decorators
# =============================================================================

def retry_with_exponential_backoff(
    max_attempts: int = 3,
    min_wait_seconds: float = 1.0,
    max_wait_seconds: float = 60.0,
    exponential_base: float = 2.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = DEFAULT_RETRYABLE_EXCEPTIONS,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
):
    """
    Decorator for retrying functions with exponential backoff
    
    Args:
        max_attempts: Maximum number of retry attempts
        min_wait_seconds: Minimum wait time between retries
        max_wait_seconds: Maximum wait time between retries
        exponential_base: Base for exponential backoff calculation
        retryable_exceptions: Tuple of exception types to retry on
        on_retry: Optional callback function called on each retry
        
    Returns:
        Decorated function with retry logic
        
    Example:
        @retry_with_exponential_backoff(max_attempts=3)
        def call_llm(prompt: str) -> str:
            return llm.generate(prompt)
    """
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception: Optional[Exception] = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                    
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts:
                        logger.error(
                            f"Max retries ({max_attempts}) exceeded for {func.__name__}: {e}"
                        )
                        raise MaxRetriesExceededError(
                            f"Failed after {max_attempts} attempts: {e}",
                            last_exception=e
                        )
                    
                    # Calculate wait time with jitter
                    wait_time = min(
                        max_wait_seconds,
                        min_wait_seconds * (exponential_base ** (attempt - 1))
                    )
                    # Add jitter (±25%)
                    wait_time = wait_time * (0.75 + random.random() * 0.5)
                    
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}. "
                        f"Retrying in {wait_time:.2f}s..."
                    )
                    
                    # Call on_retry callback if provided
                    if on_retry:
                        on_retry(e, attempt)
                    
                    time.sleep(wait_time)
                    
                except Exception as e:
                    # Non-retryable exception
                    logger.error(f"Non-retryable error in {func.__name__}: {e}")
                    raise
            
            # Should not reach here, but just in case
            raise MaxRetriesExceededError(
                f"Failed after {max_attempts} attempts",
                last_exception=last_exception
            )
        
        return wrapper
    return decorator


def async_retry_with_exponential_backoff(
    max_attempts: int = 3,
    min_wait_seconds: float = 1.0,
    max_wait_seconds: float = 60.0,
    exponential_base: float = 2.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = DEFAULT_RETRYABLE_EXCEPTIONS,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
):
    """
    Async version of retry decorator with exponential backoff
    
    Same parameters as retry_with_exponential_backoff but for async functions.
    """
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception: Optional[Exception] = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                    
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts:
                        logger.error(
                            f"Max retries ({max_attempts}) exceeded for {func.__name__}: {e}"
                        )
                        raise MaxRetriesExceededError(
                            f"Failed after {max_attempts} attempts: {e}",
                            last_exception=e
                        )
                    
                    # Calculate wait time with jitter
                    wait_time = min(
                        max_wait_seconds,
                        min_wait_seconds * (exponential_base ** (attempt - 1))
                    )
                    wait_time = wait_time * (0.75 + random.random() * 0.5)
                    
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}. "
                        f"Retrying in {wait_time:.2f}s..."
                    )
                    
                    if on_retry:
                        on_retry(e, attempt)
                    
                    await asyncio.sleep(wait_time)
                    
                except Exception as e:
                    logger.error(f"Non-retryable error in {func.__name__}: {e}")
                    raise
            
            raise MaxRetriesExceededError(
                f"Failed after {max_attempts} attempts",
                last_exception=last_exception
            )
        
        return wrapper
    return decorator


# =============================================================================
# Tenacity-based Retry Decorators (More Powerful)
# =============================================================================

def create_llm_retry_decorator(
    max_attempts: int = 3,
    max_delay_seconds: float = 60.0,
    log_level: int = logging.WARNING,
):
    """
    Create a tenacity retry decorator optimized for LLM API calls
    
    Features:
    - Random exponential backoff to avoid thundering herd
    - Logging before each retry
    - Handles common LLM API errors
    
    Args:
        max_attempts: Maximum number of attempts
        max_delay_seconds: Maximum delay between retries
        log_level: Logging level for retry messages
        
    Returns:
        Configured tenacity retry decorator
    """
    
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_random_exponential(multiplier=1, min=1, max=max_delay_seconds),
        retry=retry_if_exception_type(DEFAULT_RETRYABLE_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, log_level),
        after=after_log(logger, log_level),
        reraise=True,
    )


# Pre-configured decorators for common use cases
llm_retry = create_llm_retry_decorator(max_attempts=3, max_delay_seconds=60.0)
llm_retry_aggressive = create_llm_retry_decorator(max_attempts=5, max_delay_seconds=120.0)
llm_retry_light = create_llm_retry_decorator(max_attempts=2, max_delay_seconds=30.0)


# =============================================================================
# Retry Context Manager
# =============================================================================

class RetryContext:
    """
    Context manager for retry logic with detailed tracking
    
    Provides more control and visibility into retry behavior.
    
    Example:
        with RetryContext(max_attempts=3) as ctx:
            while ctx.should_retry():
                try:
                    result = risky_operation()
                    ctx.success(result)
                except Exception as e:
                    ctx.failed(e)
        
        print(f"Attempts: {ctx.attempts}, Success: {ctx.succeeded}")
    """
    
    def __init__(
        self,
        max_attempts: int = 3,
        min_wait_seconds: float = 1.0,
        max_wait_seconds: float = 60.0,
        exponential_base: float = 2.0,
    ):
        self.max_attempts = max_attempts
        self.min_wait_seconds = min_wait_seconds
        self.max_wait_seconds = max_wait_seconds
        self.exponential_base = exponential_base
        
        self.attempts = 0
        self.succeeded = False
        self.result: Any = None
        self.exceptions: List[Exception] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        return False  # Don't suppress exceptions
    
    def should_retry(self) -> bool:
        """Check if another retry attempt should be made"""
        return not self.succeeded and self.attempts < self.max_attempts
    
    def failed(self, exception: Exception):
        """Record a failed attempt"""
        self.attempts += 1
        self.exceptions.append(exception)
        
        if self.should_retry():
            # Calculate and apply backoff
            wait_time = min(
                self.max_wait_seconds,
                self.min_wait_seconds * (self.exponential_base ** (self.attempts - 1))
            )
            wait_time = wait_time * (0.75 + random.random() * 0.5)
            
            logger.warning(
                f"Attempt {self.attempts}/{self.max_attempts} failed: {exception}. "
                f"Retrying in {wait_time:.2f}s..."
            )
            time.sleep(wait_time)
    
    def success(self, result: Any):
        """Record a successful attempt"""
        self.attempts += 1
        self.succeeded = True
        self.result = result
    
    @property
    def elapsed_time(self) -> float:
        """Get total elapsed time"""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time
    
    @property
    def last_exception(self) -> Optional[Exception]:
        """Get the last exception that occurred"""
        return self.exceptions[-1] if self.exceptions else None


# =============================================================================
# Utility Functions
# =============================================================================

def retry_on_json_error(
    func: Callable[..., str],
    max_attempts: int = 3,
    *args,
    **kwargs
) -> dict:
    """
    Retry a function that should return valid JSON
    
    Useful for LLM calls that sometimes return malformed JSON.
    
    Args:
        func: Function that returns a JSON string
        max_attempts: Maximum retry attempts
        *args, **kwargs: Arguments to pass to func
        
    Returns:
        Parsed JSON as dictionary
        
    Raises:
        MaxRetriesExceededError: If all attempts fail
    """
    import json
    
    last_error: Optional[Exception] = None
    
    for attempt in range(1, max_attempts + 1):
        try:
            result = func(*args, **kwargs)
            
            # Try to parse as JSON
            if isinstance(result, str):
                # Clean up common issues
                result = result.strip()
                if result.startswith("```json"):
                    result = result[7:]
                if result.startswith("```"):
                    result = result[3:]
                if result.endswith("```"):
                    result = result[:-3]
                result = result.strip()
                
                return json.loads(result)
            elif isinstance(result, dict):
                return result
            else:
                raise ValueError(f"Expected string or dict, got {type(result)}")
                
        except json.JSONDecodeError as e:
            last_error = e
            logger.warning(
                f"JSON parse error on attempt {attempt}/{max_attempts}: {e}"
            )
            if attempt < max_attempts:
                time.sleep(1.0 * attempt)  # Linear backoff for JSON errors
                
        except Exception as e:
            last_error = e
            logger.warning(
                f"Error on attempt {attempt}/{max_attempts}: {e}"
            )
            if attempt < max_attempts:
                time.sleep(2.0 ** (attempt - 1))  # Exponential backoff
    
    raise MaxRetriesExceededError(
        f"Failed to get valid JSON after {max_attempts} attempts",
        last_exception=last_error
    )


def with_timeout(
    func: Callable[..., T],
    timeout_seconds: float,
    *args,
    **kwargs
) -> T:
    """
    Execute a function with a timeout
    
    Args:
        func: Function to execute
        timeout_seconds: Maximum execution time
        *args, **kwargs: Arguments to pass to func
        
    Returns:
        Function result
        
    Raises:
        TimeoutError: If execution exceeds timeout
    """
    import concurrent.futures
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(
                f"Function {func.__name__} timed out after {timeout_seconds}s"
            )


if __name__ == "__main__":
    # Test retry logic
    print("Testing Retry Logic")
    print("=" * 60)
    
    # Test 1: Successful after retries
    attempt_count = 0
    
    @retry_with_exponential_backoff(max_attempts=3, min_wait_seconds=0.1)
    def flaky_function():
        global attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise RateLimitError("Rate limited!")
        return "Success!"
    
    result = flaky_function()
    print(f"✓ Flaky function succeeded after {attempt_count} attempts: {result}")
    
    # Test 2: RetryContext
    print("\nTesting RetryContext...")
    
    ctx_attempts = 0
    with RetryContext(max_attempts=3, min_wait_seconds=0.1) as ctx:
        while ctx.should_retry():
            try:
                ctx_attempts += 1
                if ctx_attempts < 2:
                    raise ConnectionError("Connection failed")
                ctx.success("Connected!")
            except Exception as e:
                ctx.failed(e)
    
    print(f"✓ RetryContext: attempts={ctx.attempts}, succeeded={ctx.succeeded}")
    print(f"  Result: {ctx.result}")
    print(f"  Elapsed: {ctx.elapsed_time:.2f}s")
    
    print("\n✓ All retry tests passed!")