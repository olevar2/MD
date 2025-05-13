"""
Security Package for Forex Trading Platform

This package provides comprehensive security functionality for the forex trading platform,
including authentication, authorization, security monitoring, and security middleware.
"""

from common_lib.security.cookie_manager import CookieManager
from common_lib.security.file_upload import SecureFileUploadHandler

# Import from monitoring module
from common_lib.security.monitoring import (
    SecurityMonitor,
    EnhancedSecurityEvent,
    SecurityEventCategory,
    SecurityEventSeverity,
    SecurityThreshold
)

# Import from middleware module
from common_lib.security.middleware import (
    SecurityLoggingMiddleware,
    RateLimitingMiddleware,
    SecurityHeadersMiddleware
)

# Import from MFA module
from common_lib.security.mfa import (
    MFAService,
    TOTPConfig,
    BackupCodes,
    MFAMethod,
    MFAStatus,
    MFAChallenge
)

# Import from RBAC module
from common_lib.security.rbac import (
    RBACService,
    Permission,
    Role,
    UserRoles,
    require_permission
)

# Re-export from common_lib.security
from common_lib.security import (
    validate_api_key,
    create_jwt_token,
    validate_token,
    get_security_headers,
    api_key_auth,
    jwt_auth,
    create_security_event,
    log_security_event,
    SecurityEvent
)

__all__ = [
    # Existing utilities
    "CookieManager",
    "SecureFileUploadHandler",

    # Security monitoring
    'SecurityMonitor',
    'EnhancedSecurityEvent',
    'SecurityEventCategory',
    'SecurityEventSeverity',
    'SecurityThreshold',

    # Security middleware
    'SecurityLoggingMiddleware',
    'RateLimitingMiddleware',
    'SecurityHeadersMiddleware',

    # Authentication and authorization
    'validate_api_key',
    'create_jwt_token',
    'validate_token',
    'get_security_headers',
    'api_key_auth',
    'jwt_auth',

    # Security events
    'create_security_event',
    'log_security_event',
    'SecurityEvent',

    # Multi-Factor Authentication
    'MFAService',
    'TOTPConfig',
    'BackupCodes',
    'MFAMethod',
    'MFAStatus',
    'MFAChallenge',

    # Role-Based Access Control
    'RBACService',
    'Permission',
    'Role',
    'UserRoles',
    'require_permission'
]