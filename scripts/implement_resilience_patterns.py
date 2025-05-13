#!/usr/bin/env python3
"""
Script to implement uniform resilience patterns across services.
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set

def find_python_files(root_dir: str) -> List[str]:
    """Find all Python files in the given directory."""
    python_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip excluded directories
        dirnames[:] = [d for d in dirnames if not d.startswith('.') and d != '__pycache__']
        
        for filename in filenames:
            if filename.endswith('.py'):
                python_files.append(os.path.join(dirpath, filename))
    
    return python_files

def find_service_clients(file_path: str) -> List[Dict]:
    """Find service client classes in a Python file."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Find service client classes
    service_clients = []
    
    # Pattern to match class definitions
    class_pattern = r'class\s+(\w+)(?:\(([^)]+)\))?:'
    
    for match in re.finditer(class_pattern, content, re.DOTALL):
        class_name = match.group(1)
        parent_class = match.group(2) if match.group(2) else ''
        
        # Check if it's a service client
        if ('Client' in class_name or 'ServiceClient' in class_name) and 'Test' not in class_name:
            # Check if it already uses resilience patterns
            if 'with_circuit_breaker' in content or 'with_retry' in content or 'with_timeout' in content:
                continue
            
            service_clients.append({
                'file': file_path,
                'class_name': class_name,
                'parent_class': parent_class,
                'start': match.start(),
                'end': match.end()
            })
    
    return service_clients

def create_resilience_template() -> str:
    """Create a template for resilience patterns."""
    template = """#!/usr/bin/env python3
\"\"\"
Resilience patterns for service clients.
\"\"\"

import asyncio
import functools
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, cast

from common_lib.errors import ServiceError, TimeoutError, DependencyError

# Type variable for generic function
F = TypeVar('F', bound=Callable[..., Any])

logger = logging.getLogger(__name__)

def with_circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: int = 30,
    fallback_function: Optional[Callable] = None
) -> Callable[[F], F]:
    """
    With circuit breaker.
    
    Args:
        failure_threshold: Description of failure_threshold
        recovery_timeout: Description of recovery_timeout
        fallback_function: Description of fallback_function
    
    Returns:
        Callable[[F], F]: Description of return value
    
    """

    \"\"\"
    Circuit breaker decorator to prevent repeated calls to failing services.
    
    Args:
        failure_threshold: Number of failures before opening the circuit
        recovery_timeout: Seconds to wait before trying to close the circuit
        fallback_function: Function to call when the circuit is open
        
    Returns:
        Decorated function
    \"\"\"
    def decorator(func: F) -> F:
    """
    Decorator.
    
    Args:
        func: Description of func
    
    Returns:
        F: Description of return value
    
    """

        # State for the circuit breaker
        state = {
            'failures': 0,
            'is_open': False,
            'last_failure_time': 0
        }
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
    """
    Wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    Returns:
        Any: Description of return value
    
    """

            # Check if the circuit is open
            if state['is_open']:
                # Check if recovery timeout has elapsed
                if time.time() - state['last_failure_time'] > recovery_timeout:
                    logger.info(f"Circuit breaker: Attempting to close circuit for {func.__name__}")
                    state['is_open'] = False
                    state['failures'] = 0
                else:
                    logger.warning(f"Circuit breaker: Circuit open for {func.__name__}")
                    if fallback_function:
                        return fallback_function(*args, **kwargs)
                    raise DependencyError(f"Circuit breaker open for {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                # Reset failures on success
                state['failures'] = 0
                return result
            except Exception as e:
                # Increment failures
                state['failures'] += 1
                state['last_failure_time'] = time.time()
                
                # Open the circuit if threshold is reached
                if state['failures'] >= failure_threshold:
                    logger.error(f"Circuit breaker: Opening circuit for {func.__name__} after {state['failures']} failures")
                    state['is_open'] = True
                
                # Re-raise the exception
                raise
        
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
    """
    Async wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    Returns:
        Any: Description of return value
    
    """

            # Check if the circuit is open
            if state['is_open']:
                # Check if recovery timeout has elapsed
                if time.time() - state['last_failure_time'] > recovery_timeout:
                    logger.info(f"Circuit breaker: Attempting to close circuit for {func.__name__}")
                    state['is_open'] = False
                    state['failures'] = 0
                else:
                    logger.warning(f"Circuit breaker: Circuit open for {func.__name__}")
                    if fallback_function:
                        if asyncio.iscoroutinefunction(fallback_function):
                            return await fallback_function(*args, **kwargs)
                        return fallback_function(*args, **kwargs)
                    raise DependencyError(f"Circuit breaker open for {func.__name__}")
            
            try:
                result = await func(*args, **kwargs)
                # Reset failures on success
                state['failures'] = 0
                return result
            except Exception as e:
                # Increment failures
                state['failures'] += 1
                state['last_failure_time'] = time.time()
                
                # Open the circuit if threshold is reached
                if state['failures'] >= failure_threshold:
                    logger.error(f"Circuit breaker: Opening circuit for {func.__name__} after {state['failures']} failures")
                    state['is_open'] = True
                
                # Re-raise the exception
                raise
        
        return cast(F, async_wrapper if asyncio.iscoroutinefunction(func) else wrapper)
    
    return decorator

def with_retry(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception
) -> Callable[[F], F]:
    """
    With retry.
    
    Args:
        max_retries: Description of max_retries
        retry_delay: Description of retry_delay
        backoff_factor: Description of backoff_factor
        exceptions: Description of exceptions
        Tuple[Type[Exception]: Description of Tuple[Type[Exception]
        ...]]: Description of ...]]
    
    Returns:
        Callable[[F], F]: Description of return value
    
    """

    \"\"\"
    Retry decorator to retry failed operations.
    
    Args:
        max_retries: Maximum number of retries
        retry_delay: Initial delay between retries in seconds
        backoff_factor: Factor to increase delay between retries
        exceptions: Exception types to retry on
        
    Returns:
        Decorated function
    \"\"\"
    def decorator(func: F) -> F:
    """
    Decorator.
    
    Args:
        func: Description of func
    
    Returns:
        F: Description of return value
    
    """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
    """
    Wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    Returns:
        Any: Description of return value
    
    """

            last_exception = None
            delay = retry_delay
            
            for retry in range(max_retries + 1):
                try:
                    if retry > 0:
                        logger.info(f"Retry {retry}/{max_retries} for {func.__name__} after {delay:.2f}s delay")
                        time.sleep(delay)
                        delay *= backoff_factor
                    
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    logger.warning(f"Retry {retry + 1}/{max_retries} failed for {func.__name__}: {str(e)}")
                    if retry == max_retries:
                        break
            
            if last_exception:
                raise ServiceError(f"Failed after {max_retries} retries: {str(last_exception)}") from last_exception
            
            # This should never happen
            raise ServiceError(f"Failed after {max_retries} retries")
        
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
    """
    Async wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    Returns:
        Any: Description of return value
    
    """

            last_exception = None
            delay = retry_delay
            
            for retry in range(max_retries + 1):
                try:
                    if retry > 0:
                        logger.info(f"Retry {retry}/{max_retries} for {func.__name__} after {delay:.2f}s delay")
                        await asyncio.sleep(delay)
                        delay *= backoff_factor
                    
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    logger.warning(f"Retry {retry + 1}/{max_retries} failed for {func.__name__}: {str(e)}")
                    if retry == max_retries:
                        break
            
            if last_exception:
                raise ServiceError(f"Failed after {max_retries} retries: {str(last_exception)}") from last_exception
            
            # This should never happen
            raise ServiceError(f"Failed after {max_retries} retries")
        
        return cast(F, async_wrapper if asyncio.iscoroutinefunction(func) else wrapper)
    
    return decorator

def with_timeout(seconds: float = 30.0) -> Callable[[F], F]:
    """
    With timeout.
    
    Args:
        seconds: Description of seconds
    
    Returns:
        Callable[[F], F]: Description of return value
    
    """

    \"\"\"
    Timeout decorator to prevent operations from hanging.
    
    Args:
        seconds: Timeout in seconds
        
    Returns:
        Decorated function
    \"\"\"
    def decorator(func: F) -> F:
    """
    Decorator.
    
    Args:
        func: Description of func
    
    Returns:
        F: Description of return value
    
    """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
    """
    Wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    Returns:
        Any: Description of return value
    
    """

            # For synchronous functions, we can't easily implement timeout
            # without using threads, which can be problematic
            # This is a placeholder for a more robust implementation
            return func(*args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
    """
    Async wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    Returns:
        Any: Description of return value
    
    """

            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                raise TimeoutError(f"Operation {func.__name__} timed out after {seconds} seconds")
        
        return cast(F, async_wrapper if asyncio.iscoroutinefunction(func) else wrapper)
    
    return decorator

def with_bulkhead(max_concurrent: int = 10, max_queue: int = 5) -> Callable[[F], F]:
    """
    With bulkhead.
    
    Args:
        max_concurrent: Description of max_concurrent
        max_queue: Description of max_queue
    
    Returns:
        Callable[[F], F]: Description of return value
    
    """

    \"\"\"
    Bulkhead decorator to limit concurrent operations.
    
    Args:
        max_concurrent: Maximum number of concurrent operations
        max_queue: Maximum number of operations to queue
        
    Returns:
        Decorated function
    \"\"\"
    def decorator(func: F) -> F:
    """
    Decorator.
    
    Args:
        func: Description of func
    
    Returns:
        F: Description of return value
    
    """

        # Semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(max_concurrent)
        # Queue to track pending operations
        queue = []
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
    """
    Wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    Returns:
        Any: Description of return value
    
    """

            # For synchronous functions, we can't easily implement bulkhead
            # without using threads, which can be problematic
            # This is a placeholder for a more robust implementation
            return func(*args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
    """
    Async wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    Returns:
        Any: Description of return value
    
    """

            # Check if queue is full
            if len(queue) >= max_queue:
                raise ServiceError(f"Bulkhead queue full for {func.__name__}")
            
            # Add to queue
            queue_item = object()
            queue.append(queue_item)
            
            try:
                # Acquire semaphore
                async with semaphore:
                    # Remove from queue
                    queue.remove(queue_item)
                    
                    # Execute function
                    return await func(*args, **kwargs)
            except Exception as e:
                # Remove from queue on exception
                if queue_item in queue:
                    queue.remove(queue_item)
                raise
        
        return cast(F, async_wrapper if asyncio.iscoroutinefunction(func) else wrapper)
    
    return decorator

def with_resilience(
    circuit_breaker: bool = True,
    retry: bool = True,
    timeout: bool = True,
    bulkhead: bool = False,
    circuit_breaker_kwargs: Optional[Dict[str, Any]] = None,
    retry_kwargs: Optional[Dict[str, Any]] = None,
    timeout_kwargs: Optional[Dict[str, Any]] = None,
    bulkhead_kwargs: Optional[Dict[str, Any]] = None
) -> Callable[[F], F]:
    """
    With resilience.
    
    Args:
        circuit_breaker: Description of circuit_breaker
        retry: Description of retry
        timeout: Description of timeout
        bulkhead: Description of bulkhead
        circuit_breaker_kwargs: Description of circuit_breaker_kwargs
        Any]]: Description of Any]]
        retry_kwargs: Description of retry_kwargs
        Any]]: Description of Any]]
        timeout_kwargs: Description of timeout_kwargs
        Any]]: Description of Any]]
        bulkhead_kwargs: Description of bulkhead_kwargs
        Any]]: Description of Any]]
    
    Returns:
        Callable[[F], F]: Description of return value
    
    """

    \"\"\"
    Combined resilience decorator to apply multiple resilience patterns.
    
    Args:
        circuit_breaker: Whether to apply circuit breaker
        retry: Whether to apply retry
        timeout: Whether to apply timeout
        bulkhead: Whether to apply bulkhead
        circuit_breaker_kwargs: Arguments for circuit breaker
        retry_kwargs: Arguments for retry
        timeout_kwargs: Arguments for timeout
        bulkhead_kwargs: Arguments for bulkhead
        
    Returns:
        Decorated function
    \"\"\"
    def decorator(func: F) -> F:
    """
    Decorator.
    
    Args:
        func: Description of func
    
    Returns:
        F: Description of return value
    
    """

        result = func
        
        # Apply decorators in reverse order
        if bulkhead:
            result = with_bulkhead(**(bulkhead_kwargs or {}))(result)
        
        if timeout:
            result = with_timeout(**(timeout_kwargs or {}))(result)
        
        if retry:
            result = with_retry(**(retry_kwargs or {}))(result)
        
        if circuit_breaker:
            result = with_circuit_breaker(**(circuit_breaker_kwargs or {}))(result)
        
        return cast(F, result)
    
    return decorator
"""
    
    return template

def create_resilience_files(root_dir: str) -> None:
    """Create resilience files for services."""
    # Create resilience directory in common-lib
    resilience_dir = os.path.join(root_dir, 'common-lib', 'common_lib', 'resilience')
    os.makedirs(resilience_dir, exist_ok=True)
    
    # Create decorators.py
    decorators_file = os.path.join(resilience_dir, 'decorators.py')
    if not os.path.exists(decorators_file):
        with open(decorators_file, 'w') as f:
            f.write(create_resilience_template())
        print(f"Created resilience decorators file: {decorators_file}")
    
    # Create __init__.py
    init_file = os.path.join(resilience_dir, '__init__.py')
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write('"""Resilience patterns for service clients."""\n\nfrom .decorators import with_circuit_breaker, with_retry, with_timeout, with_bulkhead, with_resilience\n\n__all__ = ["with_circuit_breaker", "with_retry", "with_timeout", "with_bulkhead", "with_resilience"]\n')
        print(f"Created resilience __init__.py file: {init_file}")

def main():
    """Main function to implement uniform resilience patterns."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Find Python files
    python_files = find_python_files(root_dir)
    
    # Find service clients
    service_clients = []
    for file_path in python_files:
        clients = find_service_clients(file_path)
        service_clients.extend(clients)
    
    # Write service clients to file
    with open(os.path.join(root_dir, 'service_clients.json'), 'w') as f:
        json.dump(service_clients, f, indent=2)
    
    print(f"Found {len(service_clients)} service clients.")
    print(f"Results written to {os.path.join(root_dir, 'service_clients.json')}")
    
    # Create resilience files
    create_resilience_files(root_dir)
    
    print("Resilience patterns implementation completed.")

if __name__ == '__main__':
    main()
