"""
Cookie Manager

This module provides utilities for secure cookie management.
"""

import logging
from typing import Dict, Any, Optional, Union

from starlette.responses import Response


class CookieManager:
    """
    Utility class for secure cookie management.
    
    This class provides methods for setting and deleting cookies with secure defaults.
    """
    
    def __init__(self, secure: bool = True, http_only: bool = True, same_site: str = "lax"):
        """
        Initialize the cookie manager.
        
        Args:
            secure: Whether to set the Secure flag on cookies
            http_only: Whether to set the HttpOnly flag on cookies
            same_site: SameSite policy for cookies (lax, strict, none)
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.secure = secure
        self.http_only = http_only
        self.same_site = same_site
    
    def set_cookie(
        self,
        response: Response,
        key: str,
        value: str,
        max_age: Optional[int] = None,
        expires: Optional[int] = None,
        path: str = "/",
        domain: Optional[str] = None,
        secure: Optional[bool] = None,
        http_only: Optional[bool] = None,
        same_site: Optional[str] = None
    ) -> Response:
        """
        Set a cookie with secure defaults.
        
        Args:
            response: Response to set cookie on
            key: Cookie name
            value: Cookie value
            max_age: Cookie max age in seconds
            expires: Cookie expiration time in seconds
            path: Cookie path
            domain: Cookie domain
            secure: Whether to set the Secure flag (overrides default)
            http_only: Whether to set the HttpOnly flag (overrides default)
            same_site: SameSite policy (overrides default)
            
        Returns:
            Response with cookie set
        """
        # Use instance defaults if not specified
        secure = secure if secure is not None else self.secure
        http_only = http_only if http_only is not None else self.http_only
        same_site = same_site if same_site is not None else self.same_site
        
        # Set cookie
        response.set_cookie(
            key=key,
            value=value,
            max_age=max_age,
            expires=expires,
            path=path,
            domain=domain,
            secure=secure,
            httponly=http_only,
            samesite=same_site
        )
        
        return response
    
    def delete_cookie(
        self,
        response: Response,
        key: str,
        path: str = "/",
        domain: Optional[str] = None
    ) -> Response:
        """
        Delete a cookie.
        
        Args:
            response: Response to delete cookie from
            key: Cookie name
            path: Cookie path
            domain: Cookie domain
            
        Returns:
            Response with cookie deleted
        """
        # Delete cookie
        response.delete_cookie(
            key=key,
            path=path,
            domain=domain
        )
        
        return response