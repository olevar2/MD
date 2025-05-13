"""
Profiling module.

This module provides functionality for...
"""

import time
import logging
import functools
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@with_exception_handling
def log_and_time(func):
    """Decorator that logs function execution time and entry/exit."""

    @functools.wraps(func)
    @with_exception_handling
    def wrapper(*args, **kwargs):
    """
    Wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

        func_name = func.__name__
        logger.info(f'Entering {func_name}...')
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            duration = end_time - start_time
            logger.info(
                f'Exiting {func_name}. Duration: {duration:.4f} seconds.')
            return result
        except Exception as e:
            logger.exception(f'Exception in {func_name}: {e}')
            raise
    return wrapper
