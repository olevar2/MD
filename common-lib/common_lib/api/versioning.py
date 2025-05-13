"""
API Versioning Module

This module provides utilities for API versioning in the Forex Trading Platform.
"""

import re
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request, status


class APIVersion(str, Enum):
    """API versions supported by the platform."""
    V1 = "v1"
    V2 = "v2"
    V3 = "v3"
    LATEST = "latest"


def get_api_version(
    accept_version: Optional[str] = Header(None, alias="Accept-Version"),
    request: Request = None
) -> APIVersion:
    """
    Determine the API version from the Accept-Version header or URL path.
    
    Args:
        accept_version: The Accept-Version header value
        request: The FastAPI request object
        
    Returns:
        The API version to use
        
    Raises:
        HTTPException: If the requested version is not supported
    """
    # Default to latest version
    version = APIVersion.LATEST
    
    # Check Accept-Version header
    if accept_version:
        try:
            # Handle 'v1', 'v2', etc.
            if accept_version.lower() in [v.value for v in APIVersion]:
                version = APIVersion(accept_version.lower())
            # Handle '1', '2', etc.
            elif accept_version in [v.value[1:] for v in APIVersion if v.value.startswith('v')]:
                version = APIVersion(f"v{accept_version}")
            else:
                raise ValueError(f"Unsupported API version: {accept_version}")
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_406_NOT_ACCEPTABLE,
                detail=f"API version '{accept_version}' is not supported"
            )
    
    # Check URL path if no header or request is provided
    elif request:
        path = request.url.path
        match = re.search(r"/api/(v[0-9]+)/", path)
        if match:
            version_str = match.group(1)
            try:
                version = APIVersion(version_str)
            except ValueError:
                # If version in URL is not supported, continue with default
                pass
    
    # Map 'latest' to the actual latest version
    if version == APIVersion.LATEST:
        # Find the highest version
        versions = [v for v in APIVersion if v != APIVersion.LATEST]
        versions.sort(key=lambda v: int(v.value[1:]))
        version = versions[-1] if versions else APIVersion.V1
    
    return version


def version_route(
    app: FastAPI,
    versions: List[APIVersion],
    prefix: str = "/api",
    tags: Optional[List[str]] = None,
    dependencies: Optional[List[Depends]] = None,
    responses: Optional[Dict[Union[int, str], Dict[str, Any]]] = None
) -> Callable[[Callable], Callable]:
    """
    Decorator to create versioned API routes.
    
    Args:
        app: The FastAPI application
        versions: List of API versions this route supports
        prefix: Prefix for the API routes
        tags: OpenAPI tags for the route
        dependencies: Route dependencies
        responses: Response descriptions
        
    Returns:
        Decorator function for the route handler
    """
    def decorator(func: Callable) -> Callable:
    """
    Decorator.
    
    Args:
        func: Description of func
    
    Returns:
        Callable: Description of return value
    
    """

        # Create a router for each version
        for version in versions:
            version_prefix = f"{prefix}/{version}"
            router = APIRouter(
                prefix=version_prefix,
                tags=tags,
                dependencies=dependencies,
                responses=responses
            )
            
            # Add the route to the router
            router.add_api_route(
                path=func.__name__.replace("_", "-"),
                endpoint=func,
                methods=["GET"],
                response_model=func.__annotations__.get("return")
            )
            
            # Include the router in the app
            app.include_router(router)
        
        return func
    
    return decorator
