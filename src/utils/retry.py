import time
import random
import functools
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def retry_with_backoff(max_retries=5, initial_delay=1.0, backoff_factor=2.0, jitter=True, retry_exceptions=(Exception,)):
    """
    Decorator for retrying a function with exponential backoff and jitter.
    Useful for external API calls (OpenRouter, Voyage AI).
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retry_exceptions as e:
                    if attempt == max_retries:
                        logger.error(f"[Retry] Function '{func.__name__}' failed after {max_retries} attempts. Error: {e}")
                        raise e
                    
                    sleep_time = delay
                    if jitter:
                        sleep_time += random.uniform(0, delay * 0.5)
                    
                    logger.warning(f"[Retry] Attempt {attempt}/{max_retries} for '{func.__name__}' failed ({e}). Retrying in {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
                    delay *= backoff_factor
        return wrapper
    return decorator
