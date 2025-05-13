"""
Interface definition for model-registry-service service.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any


class ModelRegistryServiceInterface(ABC):
    """
    Interface for model-registry-service service.
    """

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the service.

        Returns:
            Service status information
        """
        pass

    @abstractmethod
    def get_model(model_id: str) -> Dict[str, Any]:
        """
        Get a model from the registry.

        Args:
            model_id: Model identifier
        Returns:
            Model information
        """
        pass

    @abstractmethod
    def list_models(tags: Optional[List[str]] = None, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        """
        List available models.

        Args:
            tags: Filter by tags
        Args:
            limit: Maximum number of results
        Args:
            offset: Result offset
        Returns:
            Dictionary with models and pagination information
        """
        pass

