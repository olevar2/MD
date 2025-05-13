"""
Custom exceptions for the Model Registry Service.
"""

class ModelRegistryError(Exception):
    """Base class for all model registry exceptions"""
    pass

class ModelNotFoundError(ModelRegistryError):
    """Raised when a model is not found"""
    pass

class ModelVersionNotFoundError(ModelRegistryError):
    """Raised when a model version is not found"""
    pass

class InvalidModelError(ModelRegistryError):
    """Raised when model data is invalid"""
    pass

class ArtifactNotFoundError(ModelRegistryError):
    """Raised when a model artifact is not found"""
    pass

class DuplicateModelError(ModelRegistryError):
    """Raised when attempting to create a model with a name that already exists"""
    pass

class InvalidStageTransitionError(ModelRegistryError):
    """Raised when attempting an invalid stage transition"""
    pass

class StorageError(ModelRegistryError):
    """Raised when there's an error with the storage backend"""
    pass
