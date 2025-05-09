"""
Core repository interface for the Model Registry Service.
"""
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from model_registry.domain.model import ModelMetadata, ModelVersion, ModelStage, ModelType

class ModelRegistryRepository(ABC):
    """Abstract interface for model registry storage operations"""
    
    @abstractmethod
    async def save_model_metadata(self, metadata: ModelMetadata) -> None:
        """Save model metadata"""
        pass
        
    @abstractmethod
    async def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID"""
        pass
        
    @abstractmethod
    async def list_models(self, 
                         name_filter: Optional[str] = None,
                         model_type: Optional[ModelType] = None,
                         tags: Optional[Dict[str, str]] = None,
                         business_domain: Optional[str] = None) -> List[ModelMetadata]:
        """List models matching criteria"""
        pass

    @abstractmethod
    async def save_model_version(self, version: ModelVersion) -> None:
        """Save model version"""
        pass

    @abstractmethod
    async def get_model_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get model version by ID"""
        pass

    @abstractmethod
    async def list_model_versions(self,
                                model_id: str,
                                stage: Optional[ModelStage] = None) -> List[ModelVersion]:
        """List versions for a model"""
        pass

    @abstractmethod
    async def save_model_artifact(self,
                                version_id: str,
                                artifact_data: bytes,
                                artifact_name: str) -> str:
        """Save a model artifact and return its URI"""
        pass

    @abstractmethod
    async def get_model_artifact(self, version_id: str, artifact_name: str) -> Optional[bytes]:
        """Get a model artifact by version ID and name"""
        pass

    @abstractmethod
    async def delete_model(self, model_id: str) -> bool:
        """Delete a model and all its versions"""
        pass

    @abstractmethod
    async def delete_model_version(self, version_id: str) -> bool:
        """Delete a specific model version"""
        pass

    @abstractmethod
    async def save_ab_test(self, test: "ABTest") -> None:
        """Save A/B test metadata"""
        pass

    @abstractmethod
    async def get_ab_test(self, test_id: str) -> Optional["ABTest"]:
        """Get A/B test by ID"""
        pass

    @abstractmethod
    async def list_ab_tests(self,
                           model_id: Optional[str] = None,
                           status: Optional[str] = None) -> List["ABTest"]:
        """List A/B tests matching criteria"""
        pass
