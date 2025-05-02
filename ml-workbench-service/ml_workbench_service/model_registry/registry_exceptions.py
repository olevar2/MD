"""
Model Registry Exceptions

This module defines custom exceptions for the Model Registry.
"""

class ModelRegistryException(Exception):
    """Base exception for all Model Registry related errors."""
    pass

class ModelNotFoundException(ModelRegistryException):
    """Exception raised when a model is not found in the registry."""
    pass

class ModelVersionNotFoundException(ModelRegistryException):
    """Exception raised when a specific model version is not found."""
    pass

class InvalidModelException(ModelRegistryException):
    """Exception raised when an invalid model is provided for registration."""
    pass
