"""
FastAPI dependencies for the Model Registry Service.
"""
import os
from functools import lru_cache
from services.service import ModelRegistryService
from core.filesystem_repository import FilesystemModelRegistry

@lru_cache()
def get_model_registry_service() -> ModelRegistryService:
    """Get or create a ModelRegistryService instance"""
    # Get storage path from environment variable or use default
    storage_path = os.getenv("MODEL_REGISTRY_STORAGE_PATH", "./model_registry_storage")
    
    # Create repository
    repository = FilesystemModelRegistry(storage_path)
    
    # Create service
    return ModelRegistryService(repository)
