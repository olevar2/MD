"""
Dependency Injection Decorators

This module provides decorators for dependency injection in the Forex Trading Platform.
"""

import asyncio
import functools
import inspect
from typing import Any, Callable, Dict, TypeVar, cast, get_type_hints

from .container import ServiceContainer

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


def inject(container: ServiceContainer) -> Callable[[F], F]:
    """
    Decorator to inject dependencies into a function.
    
    Args:
        container: The service container to use
        
    Returns:
        Decorated function with dependencies injected
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get the function signature
            signature = inspect.signature(func)
            type_hints = get_type_hints(func)
            
            # Create a dictionary of parameters to pass to the function
            params = kwargs.copy()
            
            # Add positional arguments
            positional_params = list(signature.parameters.keys())
            for i, arg in enumerate(args):
                if i < len(positional_params):
                    params[positional_params[i]] = arg
            
            # Inject dependencies for parameters that aren't provided
            for param_name, param in signature.parameters.items():
                if param_name not in params and param.default is inspect.Parameter.empty:
                    # Get the parameter type
                    param_type = type_hints.get(param_name, Any)
                    
                    # Try to resolve the parameter
                    try:
                        params[param_name] = container.get_service(param_type)
                    except KeyError:
                        # Let it fail when the function is called
                        pass
            
            return func(**params)
        
        return cast(F, wrapper)
    
    return decorator


def async_inject(container: ServiceContainer) -> Callable[[F], F]:
    """
    Decorator to inject dependencies into an async function.
    
    Args:
        container: The service container to use
        
    Returns:
        Decorated async function with dependencies injected
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get the function signature
            signature = inspect.signature(func)
            type_hints = get_type_hints(func)
            
            # Create a dictionary of parameters to pass to the function
            params = kwargs.copy()
            
            # Add positional arguments
            positional_params = list(signature.parameters.keys())
            for i, arg in enumerate(args):
                if i < len(positional_params):
                    params[positional_params[i]] = arg
            
            # Inject dependencies for parameters that aren't provided
            for param_name, param in signature.parameters.items():
                if param_name not in params and param.default is inspect.Parameter.empty:
                    # Get the parameter type
                    param_type = type_hints.get(param_name, Any)
                    
                    # Try to resolve the parameter
                    try:
                        params[param_name] = await container.get_service_async(param_type)
                    except KeyError:
                        # Let it fail when the function is called
                        pass
            
            return await func(**params)
        
        return cast(F, wrapper)
    
    return decorator
