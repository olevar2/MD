"""
Service Template Package

This package provides a template for creating new services.
"""

from .config import (
    ServiceConfig,
    config_manager,
    get_service_config,
    get_database_config,
    get_logging_config,
    get_service_client_config,
    is_development,
    is_testing,
    is_production
)

from .logging_setup import setup_logging
from .service_clients import ServiceClients, service_clients
from .database import Database, database
from .error_handling import (
    handle_error,
    handle_exception,
    handle_async_exception,
    get_status_code
)

__all__ = [
    # Configuration
    'ServiceConfig',
    'config_manager',
    'get_service_config',
    'get_database_config',
    'get_logging_config',
    'get_service_client_config',
    'is_development',
    'is_testing',
    'is_production',
    
    # Logging
    'setup_logging',
    
    # Service Clients
    'ServiceClients',
    'service_clients',
    
    # Database
    'Database',
    'database',
    
    # Error Handling
    'handle_error',
    'handle_exception',
    'handle_async_exception',
    'get_status_code'
]
