"""
Dependency Injection Container for the ML Integration Service.

This module provides a dependency injection container for the ML Integration Service,
which manages the creation and lifecycle of service dependencies.
"""

from typing import Dict, Any, Optional, Type, Callable
import logging
from functools import lru_cache

from ml_integration_service.repositories.model_repository import ModelRepository
from ml_integration_service.services.feature_service import FeatureService
from ml_integration_service.validation.data_validator import DataValidator
from ml_integration_service.services.reconciliation_service import ReconciliationService
from ml_integration_service.config.enhanced_settings import enhanced_settings

logger = logging.getLogger(__name__)


class DIContainer:
    """
    Dependency Injection Container for the ML Integration Service.
    
    This class manages the creation and lifecycle of service dependencies.
    """
    
    def __init__(self):
        """Initialize the container."""
        self._instances: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable[[], Any]] = {}
        
        # Register factories
        self._register_factories()
    
    def _register_factories(self):
        """Register factories for creating dependencies."""
        # Repositories
        self._factories[ModelRepository] = self._create_model_repository
        
        # Services
        self._factories[FeatureService] = self._create_feature_service
        self._factories[ReconciliationService] = self._create_reconciliation_service
        
        # Validators
        self._factories[DataValidator] = self._create_data_validator
    
    def _create_model_repository(self) -> ModelRepository:
        """Create a model repository instance."""
        logger.debug("Creating ModelRepository instance")
        return ModelRepository()
    
    def _create_feature_service(self) -> FeatureService:
        """Create a feature service instance."""
        logger.debug("Creating FeatureService instance")
        return FeatureService()
    
    def _create_data_validator(self) -> DataValidator:
        """Create a data validator instance."""
        logger.debug("Creating DataValidator instance")
        return DataValidator()
    
    def _create_reconciliation_service(self) -> ReconciliationService:
        """Create a reconciliation service instance."""
        logger.debug("Creating ReconciliationService instance")
        model_repository = self.get(ModelRepository)
        feature_service = self.get(FeatureService)
        data_validator = self.get(DataValidator)
        
        return ReconciliationService(
            model_repository=model_repository,
            feature_service=feature_service,
            data_validator=data_validator
        )
    
    def get(self, dependency_type: Type) -> Any:
        """
        Get an instance of the specified dependency type.
        
        Args:
            dependency_type: Type of dependency to get
            
        Returns:
            Instance of the specified dependency type
        """
        # Check if instance already exists
        if dependency_type in self._instances:
            return self._instances[dependency_type]
        
        # Check if factory exists
        if dependency_type in self._factories:
            # Create instance
            instance = self._factories[dependency_type]()
            
            # Cache instance
            self._instances[dependency_type] = instance
            
            return instance
        
        # No factory found
        raise ValueError(f"No factory registered for dependency type: {dependency_type.__name__}")
    
    def clear(self):
        """Clear all cached instances."""
        self._instances.clear()


# Create a singleton instance of the container
@lru_cache(maxsize=1)
def get_container() -> DIContainer:
    """
    Get the singleton instance of the dependency injection container.
    
    Returns:
        Singleton instance of the dependency injection container
    """
    return DIContainer()


# Convenience functions for getting dependencies
def get_model_repository() -> ModelRepository:
    """
    Get a model repository instance.
    
    Returns:
        Model repository instance
    """
    return get_container().get(ModelRepository)


def get_feature_service() -> FeatureService:
    """
    Get a feature service instance.
    
    Returns:
        Feature service instance
    """
    return get_container().get(FeatureService)


def get_data_validator() -> DataValidator:
    """
    Get a data validator instance.
    
    Returns:
        Data validator instance
    """
    return get_container().get(DataValidator)


def get_reconciliation_service() -> ReconciliationService:
    """
    Get a reconciliation service instance.
    
    Returns:
        Reconciliation service instance
    """
    return get_container().get(ReconciliationService)
