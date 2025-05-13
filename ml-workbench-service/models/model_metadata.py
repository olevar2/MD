"""
Model Metadata

This module defines the ModelMetadata class for tracking model metadata.
"""

from typing import Dict, List, Any, Optional
from models.model_version import ModelVersion

class ModelMetadata:
    """
    Class representing metadata for a model in the registry.
    """
    
    def __init__(
        self,
        name: str,
        model_type: str,
        description: str,
        tags: Dict[str, str],
        creation_time: str,
        latest_version: int,
        versions: List[ModelVersion]
    ):
        """
        Initialize model metadata.
        
        Args:
            name: Name of the model
            model_type: Type of model (e.g., "classification", "regression", "forecasting")
            description: Model description
            tags: Tags for categorization and filtering
            creation_time: ISO-formatted timestamp of initial creation
            latest_version: Latest version number
            versions: List of model versions
        """
        self.name = name
        self.model_type = model_type
        self.description = description
        self.tags = tags or {}
        self.creation_time = creation_time
        self.latest_version = latest_version
        self.versions = versions or []
    
    def __repr__(self) -> str:
        """String representation of the model metadata."""
        return f"ModelMetadata(name='{self.name}', type='{self.model_type}', versions={len(self.versions)})"
        
    def get_version(self, version: int) -> Optional[ModelVersion]:
        """
        Get a specific version of the model.
        
        Args:
            version: Version number to retrieve
            
        Returns:
            ModelVersion if found, None otherwise
        """
        for v in self.versions:
            if v.version == version:
                return v
        return None
