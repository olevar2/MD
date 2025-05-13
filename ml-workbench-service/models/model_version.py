"""
Model Version

This module defines the ModelVersion class for tracking individual model versions.
"""

from typing import Dict, List, Any, Optional
from models.model_stage import ModelStage
from models.model_metrics import ModelMetrics

class ModelVersion:
    """
    Class representing a specific version of a model in the registry.
    """
    
    def __init__(
        self,
        version: int,
        version_id: str,
        creation_time: str,
        description: str,
        metrics: ModelMetrics,
        parameters: Dict[str, Any],
        framework: str,
        stage: ModelStage = ModelStage.DEVELOPMENT,
        feature_names: List[str] = None,
        target_names: List[str] = None,
        artifact_paths: Dict[str, str] = None
    ):
        """
        Initialize a model version.
        
        Args:
            version: Version number (incremental integer)
            version_id: Unique identifier (UUID) for this version
            creation_time: ISO-formatted timestamp of creation
            description: Description of this specific version
            metrics: Performance metrics
            parameters: Model parameters/hyperparameters
            framework: ML framework used (e.g., "sklearn", "tensorflow", "pytorch")
            stage: Current lifecycle stage
            feature_names: Names of features used by the model
            target_names: Names of target variables predicted by the model
            artifact_paths: Paths to model artifacts
        """
        self.version = version
        self.version_id = version_id
        self.creation_time = creation_time
        self.description = description
        self.metrics = metrics
        self.parameters = parameters or {}
        self.framework = framework
        self.stage = stage
        self.feature_names = feature_names or []
        self.target_names = target_names or []
        self.artifact_paths = artifact_paths or {}
    
    def __repr__(self) -> str:
        """String representation of the model version."""
        return f"ModelVersion(version={self.version}, stage={self.stage.value}, framework={self.framework})"
