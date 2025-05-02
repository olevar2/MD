"""
Model Registry Core

This module defines the core components of the ML Model Registry, including the
ModelVersion, ModelMetadata, and ModelRegistry classes that enable versioned tracking
of machine learning models.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import uuid
import os
import json
import shutil
from enum import Enum
import logging
from pathlib import Path

from pydantic import BaseModel, Field, validator

from ml_workbench_service.models.base import BaseMLModel
from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)

# Enums for model registry

class ModelStatus(str, Enum):
    """Enum for model status in the registry"""
    DRAFT = "draft"
    STAGING = "staging"
    PRODUCTION = "production" 
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"
    FAILED = "failed"


class ModelFramework(str, Enum):
    """Supported ML frameworks"""
    SKLEARN = "sklearn"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    CUSTOM = "custom"


class ModelType(str, Enum):
    """Types of models supported in the registry"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TIME_SERIES = "time_series" 
    FORECASTING = "forecasting"
    ANOMALY_DETECTION = "anomaly_detection"
    CLUSTERING = "clustering"
    CUSTOM = "custom"


# Pydantic models for the registry

class ModelMetrics(BaseModel):
    """Metrics tracked for model versions"""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    mse: Optional[float] = None
    mape: Optional[float] = None
    r2: Optional[float] = None
    log_loss: Optional[float] = None
    custom_metrics: Dict[str, float] = Field(default_factory=dict)
    
    def get_primary_metric(self, metric_name: str) -> Optional[float]:
        """Get value of a specific metric by name"""
        if hasattr(self, metric_name) and getattr(self, metric_name) is not None:
            return getattr(self, metric_name)
        return self.custom_metrics.get(metric_name)
    
    @property
    def all_metrics(self) -> Dict[str, float]:
        """Get all metrics as a dictionary"""
        metrics = {}
        # Add standard metrics if they exist
        for field in self.__fields__:
            if field != 'custom_metrics':
                value = getattr(self, field)
                if value is not None:
                    metrics[field] = value
        
        # Add custom metrics
        metrics.update(self.custom_metrics)
        return metrics


class HyperParameters(BaseModel):
    """Model hyperparameters"""
    values: Dict[str, Any] = Field(default_factory=dict)
    tuning_method: Optional[str] = None
    search_space: Optional[Dict[str, Any]] = None
    optimization_metric: Optional[str] = None
    
    @property
    def as_dict(self) -> Dict[str, Any]:
        """Return hyperparameters as a flat dictionary"""
        return self.values


class ModelVersion(BaseModel):
    """Data model for a specific version of a model"""
    version_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str
    version_number: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str
    description: Optional[str] = None
    status: ModelStatus = ModelStatus.DRAFT
    framework: ModelFramework
    framework_version: str
    metrics: ModelMetrics = Field(default_factory=ModelMetrics)
    hyperparameters: HyperParameters = Field(default_factory=HyperParameters)
    training_dataset_id: Optional[str] = None
    validation_dataset_id: Optional[str] = None
    test_dataset_id: Optional[str] = None
    file_path: Optional[str] = None
    feature_columns: List[str] = Field(default_factory=list)
    target_column: Optional[str] = None
    preprocessing_steps: List[Dict[str, Any]] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None
    experiment_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model version to dictionary"""
        return {
            "version_id": self.version_id,
            "model_id": self.model_id,
            "version_number": self.version_number,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "description": self.description,
            "status": self.status.value,
            "framework": self.framework.value,
            "framework_version": self.framework_version,
            "metrics": self.metrics.all_metrics,
            "hyperparameters": self.hyperparameters.as_dict,
            "training_dataset_id": self.training_dataset_id,
            "validation_dataset_id": self.validation_dataset_id,
            "test_dataset_id": self.test_dataset_id,
            "file_path": self.file_path,
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "preprocessing_steps": self.preprocessing_steps,
            "tags": self.tags,
            "notes": self.notes,
            "experiment_id": self.experiment_id,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """Create model version from dictionary"""
        # Handle dates
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        
        if "updated_at" in data and isinstance(data["updated_at"], str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        
        # Handle enums
        if "status" in data and not isinstance(data["status"], ModelStatus):
            data["status"] = ModelStatus(data["status"])
        
        if "framework" in data and not isinstance(data["framework"], ModelFramework):
            data["framework"] = ModelFramework(data["framework"])
        
        # Handle nested objects
        if "metrics" in data and isinstance(data["metrics"], dict):
            custom_metrics = {k: v for k, v in data["metrics"].items() if k not in ModelMetrics.__fields__}
            standard_metrics = {k: v for k, v in data["metrics"].items() if k in ModelMetrics.__fields__}
            
            metrics_data = standard_metrics.copy()
            if custom_metrics:
                metrics_data["custom_metrics"] = custom_metrics
                
            data["metrics"] = ModelMetrics(**metrics_data)
            
        if "hyperparameters" in data and isinstance(data["hyperparameters"], dict):
            if not isinstance(data["hyperparameters"], HyperParameters):
                if "values" not in data["hyperparameters"]:
                    # If it's just a flat dict of hyperparameters
                    data["hyperparameters"] = HyperParameters(values=data["hyperparameters"])
                else:
                    data["hyperparameters"] = HyperParameters(**data["hyperparameters"])
        
        return cls(**data)


class ModelMetadata(BaseModel):
    """Metadata for a model in the registry"""
    model_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str
    model_type: ModelType
    tags: List[str] = Field(default_factory=list)
    latest_version_id: Optional[str] = None
    latest_version_number: int = 0
    production_version_id: Optional[str] = None
    staging_version_id: Optional[str] = None
    business_domain: Optional[str] = None
    purpose: Optional[str] = None
    training_frequency: Optional[str] = None
    monitoring_config: Dict[str, Any] = Field(default_factory=dict)
    versioning_strategy: Optional[str] = "semantic"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model metadata to dictionary"""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "model_type": self.model_type.value,
            "tags": self.tags,
            "latest_version_id": self.latest_version_id,
            "latest_version_number": self.latest_version_number,
            "production_version_id": self.production_version_id,
            "staging_version_id": self.staging_version_id,
            "business_domain": self.business_domain,
            "purpose": self.purpose,
            "training_frequency": self.training_frequency,
            "monitoring_config": self.monitoring_config,
            "versioning_strategy": self.versioning_strategy,
            "metadata": self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create model metadata from dictionary"""
        # Handle dates
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        
        if "updated_at" in data and isinstance(data["updated_at"], str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        
        # Handle enums
        if "model_type" in data and not isinstance(data["model_type"], ModelType):
            data["model_type"] = ModelType(data["model_type"])
        
        return cls(**data)


class ModelRegistry:
    """
    Core registry for ML models that handles versioning, metadata tracking,
    and model lifecycle management.
    
    Key capabilities:
    - Register and version ML models
    - Track model metadata, metrics, and hyperparameters
    - Manage model lifecycle (staging, production, deprecated)
    - Store and retrieve model artifacts
    - Compare model versions
    - Support A/B testing configurations
    """
    
    def __init__(self, storage_path: str = None):
        """
        Initialize the ModelRegistry.
        
        Args:
            storage_path: Path to store model files and metadata. If None, defaults
                        to a models directory in the current working directory.
        """
        self.storage_path = storage_path or os.path.join(os.getcwd(), "models")
        self.metadata_dir = os.path.join(self.storage_path, "metadata")
        self.version_dir = os.path.join(self.storage_path, "versions")
        self.artifacts_dir = os.path.join(self.storage_path, "artifacts")
        
        # Ensure directories exist
        for directory in [self.metadata_dir, self.version_dir, self.artifacts_dir]:
            os.makedirs(directory, exist_ok=True)
            
        logger.info(f"Initialized ModelRegistry with storage path: {self.storage_path}")
        
    def register_model(
        self, 
        name: str, 
        model_type: Union[ModelType, str], 
        description: Optional[str] = None,
        created_by: str = "system",
        tags: List[str] = None,
        business_domain: Optional[str] = None,
        purpose: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ) -> ModelMetadata:
        """
        Register a new model in the registry.
        
        Args:
            name: Name of the model
            model_type: Type of the model (classification, regression, etc.)
            description: Optional description
            created_by: User or system that created the model
            tags: Optional list of tags for the model
            business_domain: Business domain the model belongs to
            purpose: Purpose of the model
            metadata: Additional metadata for the model
            
        Returns:
            ModelMetadata: The created model metadata
        """
        # Convert string model_type to enum if needed
        if isinstance(model_type, str):
            model_type = ModelType(model_type)
            
        # Create model metadata
        model_metadata = ModelMetadata(
            name=name,
            description=description,
            created_by=created_by,
            model_type=model_type,
            tags=tags or [],
            business_domain=business_domain,
            purpose=purpose,
            metadata=metadata or {}
        )
        
        # Save metadata to disk
        self._save_model_metadata(model_metadata)
        
        logger.info(f"Registered new model: {name} (ID: {model_metadata.model_id})")
        
        return model_metadata
    
    def _save_model_metadata(self, metadata: ModelMetadata) -> None:
        """Save model metadata to disk"""
        metadata_path = os.path.join(self.metadata_dir, f"{metadata.model_id}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2)
    
    def _save_model_version(self, version: ModelVersion) -> None:
        """Save model version to disk"""
        version_path = os.path.join(self.version_dir, f"{version.version_id}.json")
        with open(version_path, 'w') as f:
            json.dump(version.to_dict(), f, indent=2)
    
    def create_model_version(
        self,
        model_id: str,
        framework: Union[ModelFramework, str],
        framework_version: str,
        created_by: str,
        model_file_path: Optional[str] = None,
        description: Optional[str] = None,
        hyperparameters: Optional[Union[Dict[str, Any], HyperParameters]] = None,
        metrics: Optional[Union[Dict[str, float], ModelMetrics]] = None,
        feature_columns: List[str] = None,
        target_column: Optional[str] = None,
        preprocessing_steps: List[Dict[str, Any]] = None,
        training_dataset_id: Optional[str] = None,
        validation_dataset_id: Optional[str] = None,
        test_dataset_id: Optional[str] = None,
        tags: List[str] = None,
        experiment_id: Optional[str] = None,
        notes: Optional[str] = None,
        metadata: Dict[str, Any] = None,
        status: ModelStatus = ModelStatus.DRAFT
    ) -> ModelVersion:
        """
        Create a new version of a model.
        
        Args:
            model_id: ID of the model
            framework: ML framework used (sklearn, pytorch, etc.)
            framework_version: Version of the framework
            created_by: User or system that created the version
            model_file_path: Path to the model file to store
            description: Optional description
            hyperparameters: Model hyperparameters
            metrics: Model performance metrics
            feature_columns: List of feature column names
            target_column: Target column name
            preprocessing_steps: List of preprocessing steps used
            training_dataset_id: ID of training dataset
            validation_dataset_id: ID of validation dataset
            test_dataset_id: ID of test dataset
            tags: Optional list of tags for the version
            experiment_id: Optional ID of associated experiment
            notes: Optional notes about the version
            metadata: Additional metadata
            status: Initial status of the version
            
        Returns:
            ModelVersion: The created model version
        """
        # Check if model exists
        model_metadata = self.get_model_metadata(model_id)
        if not model_metadata:
            raise ValueError(f"Model with ID {model_id} does not exist")
        
        # Convert string framework to enum if needed
        if isinstance(framework, str):
            framework = ModelFramework(framework)
            
        # Convert hyperparameters to HyperParameters if needed
        if hyperparameters is not None and not isinstance(hyperparameters, HyperParameters):
            hyperparameters = HyperParameters(values=hyperparameters)
        else:
            hyperparameters = hyperparameters or HyperParameters()
            
        # Convert metrics to ModelMetrics if needed
        if metrics is not None and not isinstance(metrics, ModelMetrics):
            # Split into standard and custom metrics
            standard_metrics = {k: v for k, v in metrics.items() if k in ModelMetrics.__fields__}
            custom_metrics = {k: v for k, v in metrics.items() if k not in ModelMetrics.__fields__}
            
            metrics_data = standard_metrics.copy()
            if custom_metrics:
                metrics_data["custom_metrics"] = custom_metrics
                
            metrics = ModelMetrics(**metrics_data)
        else:
            metrics = metrics or ModelMetrics()
        
        # Generate next version number
        version_number = model_metadata.latest_version_number + 1
        
        # Create file path for model artifact if provided
        artifact_path = None
        if model_file_path:
            # Create directory for this model's artifacts if it doesn't exist
            model_artifacts_dir = os.path.join(self.artifacts_dir, model_id)
            os.makedirs(model_artifacts_dir, exist_ok=True)
            
            # Define target path for the artifact
            file_ext = os.path.splitext(model_file_path)[1]
            artifact_path = os.path.join(
                model_artifacts_dir, 
                f"version_{version_number}{file_ext}"
            )
            
            # Copy the model file
            try:
                shutil.copy2(model_file_path, artifact_path)
                logger.info(f"Copied model artifact from {model_file_path} to {artifact_path}")
            except Exception as e:
                logger.error(f"Failed to copy model artifact: {str(e)}")
                artifact_path = None
        
        # Create model version
        version = ModelVersion(
            model_id=model_id,
            version_number=version_number,
            created_by=created_by,
            description=description,
            status=status,
            framework=framework,
            framework_version=framework_version,
            hyperparameters=hyperparameters,
            metrics=metrics,
            feature_columns=feature_columns or [],
            target_column=target_column,
            preprocessing_steps=preprocessing_steps or [],
            training_dataset_id=training_dataset_id,
            validation_dataset_id=validation_dataset_id,
            test_dataset_id=test_dataset_id,
            file_path=artifact_path,
            tags=tags or [],
            experiment_id=experiment_id,
            notes=notes,
            metadata=metadata or {}
        )
        
        # Save version to disk
        self._save_model_version(version)
        
        # Update model metadata
        model_metadata.latest_version_id = version.version_id
        model_metadata.latest_version_number = version_number
        model_metadata.updated_at = datetime.utcnow()
        
        # Save updated metadata
        self._save_model_metadata(model_metadata)
        
        logger.info(f"Created version {version_number} for model {model_metadata.name} (ID: {model_id})")
        
        return version
    
    def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """
        Get metadata for a model by ID.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Optional[ModelMetadata]: Model metadata or None if not found
        """
        metadata_path = os.path.join(self.metadata_dir, f"{model_id}.json")
        
        if not os.path.exists(metadata_path):
            return None
            
        try:
            with open(metadata_path, 'r') as f:
                data = json.load(f)
                return ModelMetadata.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading model metadata for {model_id}: {str(e)}")
            return None
    
    def get_model_version(self, version_id: str) -> Optional[ModelVersion]:
        """
        Get a model version by ID.
        
        Args:
            version_id: ID of the version
            
        Returns:
            Optional[ModelVersion]: Model version or None if not found
        """
        version_path = os.path.join(self.version_dir, f"{version_id}.json")
        
        if not os.path.exists(version_path):
            return None
            
        try:
            with open(version_path, 'r') as f:
                data = json.load(f)
                return ModelVersion.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading model version {version_id}: {str(e)}")
            return None
    
    def list_models(
        self, 
        name_filter: Optional[str] = None, 
        model_type: Optional[Union[ModelType, str]] = None,
        tags: Optional[List[str]] = None,
        business_domain: Optional[str] = None
    ) -> List[ModelMetadata]:
        """
        List models in the registry with optional filtering.
        
        Args:
            name_filter: Optional filter for model name (case-insensitive contains)
            model_type: Optional filter by model type
            tags: Optional filter by tags (models must have all listed tags)
            business_domain: Optional filter by business domain
            
        Returns:
            List[ModelMetadata]: List of model metadata matching filters
        """
        models = []
        
        # Convert string model_type to enum if needed
        if isinstance(model_type, str):
            model_type = ModelType(model_type)
        
        # Iterate through metadata files
        for filename in os.listdir(self.metadata_dir):
            if not filename.endswith('.json'):
                continue
                
            try:
                with open(os.path.join(self.metadata_dir, filename), 'r') as f:
                    data = json.load(f)
                    model = ModelMetadata.from_dict(data)
                    
                    # Apply filters
                    if name_filter and name_filter.lower() not in model.name.lower():
                        continue
                        
                    if model_type and model.model_type != model_type:
                        continue
                        
                    if tags and not all(tag in model.tags for tag in tags):
                        continue
                        
                    if business_domain and model.business_domain != business_domain:
                        continue
                    
                    models.append(model)
            except Exception as e:
                logger.error(f"Error loading model from {filename}: {str(e)}")
        
        return models
    
    def list_versions(
        self, 
        model_id: str,
        status_filter: Optional[Union[ModelStatus, str]] = None,
        min_version: Optional[int] = None,
        max_version: Optional[int] = None
    ) -> List[ModelVersion]:
        """
        List versions for a specific model with optional filtering.
        
        Args:
            model_id: ID of the model
            status_filter: Optional filter by status
            min_version: Optional minimum version number
            max_version: Optional maximum version number
            
        Returns:
            List[ModelVersion]: List of model versions matching filters
        """
        versions = []
        
        # Convert string status to enum if needed
        if isinstance(status_filter, str):
            status_filter = ModelStatus(status_filter)
        
        # Iterate through version files
        for filename in os.listdir(self.version_dir):
            if not filename.endswith('.json'):
                continue
                
            try:
                with open(os.path.join(self.version_dir, filename), 'r') as f:
                    data = json.load(f)
                    
                    # Skip if not matching model_id
                    if data.get('model_id') != model_id:
                        continue
                    
                    version = ModelVersion.from_dict(data)
                    
                    # Apply filters
                    if status_filter and version.status != status_filter:
                        continue
                        
                    if min_version is not None and version.version_number < min_version:
                        continue
                        
                    if max_version is not None and version.version_number > max_version:
                        continue
                    
                    versions.append(version)
            except Exception as e:
                logger.error(f"Error loading version from {filename}: {str(e)}")
        
        # Sort by version number
        versions.sort(key=lambda v: v.version_number)
        return versions
    
    def set_version_status(
        self, 
        version_id: str, 
        status: Union[ModelStatus, str]
    ) -> Optional[ModelVersion]:
        """
        Update the status of a model version.
        
        Args:
            version_id: ID of the version
            status: New status
            
        Returns:
            Optional[ModelVersion]: Updated model version or None if not found
        """
        # Convert string status to enum if needed
        if isinstance(status, str):
            status = ModelStatus(status)
        
        # Get the version
        version = self.get_model_version(version_id)
        if not version:
            logger.error(f"Version {version_id} not found")
            return None
        
        # Get model metadata
        model_metadata = self.get_model_metadata(version.model_id)
        if not model_metadata:
            logger.error(f"Model {version.model_id} not found")
            return None
        
        # Update status
        version.status = status
        version.updated_at = datetime.utcnow()
        
        # Handle special statuses
        if status == ModelStatus.PRODUCTION:
            # Update production version in metadata
            model_metadata.production_version_id = version_id
            logger.info(f"Set version {version.version_number} as production for model {model_metadata.name}")
        elif status == ModelStatus.STAGING:
            # Update staging version in metadata
            model_metadata.staging_version_id = version_id
            logger.info(f"Set version {version.version_number} as staging for model {model_metadata.name}")
        
        # Save changes
        self._save_model_version(version)
        self._save_model_metadata(model_metadata)
        
        return version
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model and all its versions.
        
        Args:
            model_id: ID of the model
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Delete metadata file
            metadata_path = os.path.join(self.metadata_dir, f"{model_id}.json")
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            
            # Delete all versions for this model
            versions = self.list_versions(model_id)
            for version in versions:
                version_path = os.path.join(self.version_dir, f"{version.version_id}.json")
                if os.path.exists(version_path):
                    os.remove(version_path)
            
            # Delete artifacts directory
            artifacts_dir = os.path.join(self.artifacts_dir, model_id)
            if os.path.exists(artifacts_dir):
                shutil.rmtree(artifacts_dir)
                
            logger.info(f"Deleted model {model_id} with {len(versions)} versions")
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {str(e)}")
            return False
    
    def delete_version(self, version_id: str) -> bool:
        """
        Delete a specific model version.
        
        Args:
            version_id: ID of the version
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get version details
            version = self.get_model_version(version_id)
            if not version:
                logger.warning(f"Version {version_id} not found")
                return False
            
            # Get model metadata
            model_id = version.model_id
            model_metadata = self.get_model_metadata(model_id)
            
            # Remove file
            version_path = os.path.join(self.version_dir, f"{version_id}.json")
            if os.path.exists(version_path):
                os.remove(version_path)
            
            # Delete artifact file if it exists
            if version.file_path and os.path.exists(version.file_path):
                os.remove(version.file_path)
            
            # Update model metadata if needed
            if model_metadata:
                updated = False
                
                if model_metadata.latest_version_id == version_id:
                    # Find the new latest version
                    versions = self.list_versions(model_id)
                    if versions:
                        latest = max(versions, key=lambda v: v.version_number)
                        model_metadata.latest_version_id = latest.version_id
                        model_metadata.latest_version_number = latest.version_number
                    else:
                        model_metadata.latest_version_id = None
                        model_metadata.latest_version_number = 0
                    updated = True
                
                if model_metadata.production_version_id == version_id:
                    model_metadata.production_version_id = None
                    updated = True
                
                if model_metadata.staging_version_id == version_id:
                    model_metadata.staging_version_id = None
                    updated = True
                
                if updated:
                    model_metadata.updated_at = datetime.utcnow()
                    self._save_model_metadata(model_metadata)
            
            logger.info(f"Deleted version {version.version_number} of model {model_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete version {version_id}: {str(e)}")
            return False
    
    def compare_versions(
        self, 
        version_id_1: str, 
        version_id_2: str
    ) -> Dict[str, Any]:
        """
        Compare two model versions.
        
        Args:
            version_id_1: ID of the first version
            version_id_2: ID of the second version
            
        Returns:
            Dict[str, Any]: Comparison results
        """
        # Get versions
        version1 = self.get_model_version(version_id_1)
        version2 = self.get_model_version(version_id_2)
        
        if not version1 or not version2:
            raise ValueError("One or both versions not found")
        
        # Basic validation - should be versions of the same model
        if version1.model_id != version2.model_id:
            logger.warning(f"Comparing versions from different models: {version1.model_id} vs {version2.model_id}")
        
        # Compare metrics
        metrics_diff = {}
        all_metrics = set(version1.metrics.all_metrics.keys()) | set(version2.metrics.all_metrics.keys())
        
        for metric in all_metrics:
            v1_value = version1.metrics.get_primary_metric(metric)
            v2_value = version2.metrics.get_primary_metric(metric)
            
            if v1_value is not None and v2_value is not None:
                abs_diff = v2_value - v1_value
                rel_diff = abs_diff / v1_value if v1_value != 0 else float('inf')
                
                metrics_diff[metric] = {
                    "version1": v1_value,
                    "version2": v2_value,
                    "absolute_diff": abs_diff,
                    "relative_diff": rel_diff
                }
        
        # Compare hyperparameters
        hyperparams_diff = {}
        all_params = set(version1.hyperparameters.as_dict.keys()) | set(version2.hyperparameters.as_dict.keys())
        
        for param in all_params:
            v1_value = version1.hyperparameters.values.get(param)
            v2_value = version2.hyperparameters.values.get(param)
            
            if v1_value != v2_value:
                hyperparams_diff[param] = {
                    "version1": v1_value,
                    "version2": v2_value
                }
        
        # Compare other attributes
        attributes_diff = {}
        
        # Compare feature columns
        if set(version1.feature_columns) != set(version2.feature_columns):
            attributes_diff["feature_columns"] = {
                "version1": version1.feature_columns,
                "version2": version2.feature_columns,
                "added": list(set(version2.feature_columns) - set(version1.feature_columns)),
                "removed": list(set(version1.feature_columns) - set(version2.feature_columns))
            }
        
        # Compare preprocessing steps
        if version1.preprocessing_steps != version2.preprocessing_steps:
            attributes_diff["preprocessing_steps"] = {
                "version1": version1.preprocessing_steps,
                "version2": version2.preprocessing_steps
            }
        
        return {
            "version1": {
                "version_id": version1.version_id,
                "version_number": version1.version_number,
                "created_at": version1.created_at.isoformat(),
                "status": version1.status.value
            },
            "version2": {
                "version_id": version2.version_id,
                "version_number": version2.version_number,
                "created_at": version2.created_at.isoformat(),
                "status": version2.status.value
            },
            "metrics_comparison": metrics_diff,
            "hyperparameters_comparison": hyperparams_diff,
            "attributes_comparison": attributes_diff,
            "same_framework": version1.framework == version2.framework,
            "same_training_dataset": version1.training_dataset_id == version2.training_dataset_id
        }
    
    def get_best_version(
        self, 
        model_id: str,
        metric: str,
        higher_is_better: bool = True,
        status_filter: Optional[Union[ModelStatus, str]] = None,
        min_version: Optional[int] = None
    ) -> Optional[ModelVersion]:
        """
        Get the best model version based on a specific metric.
        
        Args:
            model_id: ID of the model
            metric: Metric to use for comparison
            higher_is_better: Whether higher metric values are better
            status_filter: Optional filter by status
            min_version: Optional minimum version number
            
        Returns:
            Optional[ModelVersion]: Best model version or None if no versions found
        """
        versions = self.list_versions(
            model_id=model_id,
            status_filter=status_filter,
            min_version=min_version
        )
        
        if not versions:
            return None
        
        # Find best version
        best_version = None
        best_value = None
        
        for version in versions:
            value = version.metrics.get_primary_metric(metric)
            
            if value is None:
                continue
                
            if best_value is None or (higher_is_better and value > best_value) or (not higher_is_better and value < best_value):
                best_value = value
                best_version = version
        
        return best_version
    
    def load_model_artifact(self, version_id: str) -> Optional[str]:
        """
        Get the file path to a model artifact.
        
        Args:
            version_id: ID of the version
            
        Returns:
            Optional[str]: Path to the model artifact or None if not found
        """
        version = self.get_model_version(version_id)
        
        if not version or not version.file_path:
            return None
            
        if not os.path.exists(version.file_path):
            logger.error(f"Model artifact file not found: {version.file_path}")
            return None
            
        return version.file_path
    
    def update_version_metrics(
        self,
        version_id: str,
        metrics: Dict[str, float]
    ) -> Optional[ModelVersion]:
        """
        Update metrics for a model version.
        
        Args:
            version_id: ID of the version
            metrics: New metrics to add/update
            
        Returns:
            Optional[ModelVersion]: Updated model version or None if not found
        """
        # Get the version
        version = self.get_model_version(version_id)
        if not version:
            logger.error(f"Version {version_id} not found")
            return None
        
        # Update metrics
        current_metrics = version.metrics.all_metrics
        
        # Split into standard and custom metrics
        standard_metrics = {k: v for k, v in metrics.items() if k in ModelMetrics.__fields__ and k != 'custom_metrics'}
        custom_metrics = {k: v for k, v in metrics.items() if k not in ModelMetrics.__fields__}
        
        # Update standard metrics
        for k, v in standard_metrics.items():
            setattr(version.metrics, k, v)
            
        # Update custom metrics
        for k, v in custom_metrics.items():
            version.metrics.custom_metrics[k] = v
        
        # Update timestamp
        version.updated_at = datetime.utcnow()
        
        # Save changes
        self._save_model_version(version)
        
        logger.info(f"Updated metrics for version {version.version_number} of model {version.model_id}")
        
        return version
    
    def add_version_tags(
        self,
        version_id: str,
        tags: List[str]
    ) -> Optional[ModelVersion]:
        """
        Add tags to a model version.
        
        Args:
            version_id: ID of the version
            tags: Tags to add
            
        Returns:
            Optional[ModelVersion]: Updated model version or None if not found
        """
        # Get the version
        version = self.get_model_version(version_id)
        if not version:
            logger.error(f"Version {version_id} not found")
            return None
        
        # Add tags (ensure uniqueness)
        current_tags = set(version.tags)
        for tag in tags:
            current_tags.add(tag)
        
        version.tags = list(current_tags)
        version.updated_at = datetime.utcnow()
        
        # Save changes
        self._save_model_version(version)
        
        logger.info(f"Added tags to version {version.version_number} of model {version.model_id}: {tags}")
        
        return version
    
    def add_model_tags(
        self,
        model_id: str,
        tags: List[str]
    ) -> Optional[ModelMetadata]:
        """
        Add tags to a model.
        
        Args:
            model_id: ID of the model
            tags: Tags to add
            
        Returns:
            Optional[ModelMetadata]: Updated model metadata or None if not found
        """
        # Get the model
        model = self.get_model_metadata(model_id)
        if not model:
            logger.error(f"Model {model_id} not found")
            return None
        
        # Add tags (ensure uniqueness)
        current_tags = set(model.tags)
        for tag in tags:
            current_tags.add(tag)
        
        model.tags = list(current_tags)
        model.updated_at = datetime.utcnow()
        
        # Save changes
        self._save_model_metadata(model)
        
        logger.info(f"Added tags to model {model.name}: {tags}")
        
        return model
    
    def configure_ab_testing(
        self,
        model_id: str,
        version_ids: List[str],
        traffic_split: List[float]
    ) -> bool:
        """
        Configure A/B testing for a model.
        
        Args:
            model_id: ID of the model
            version_ids: IDs of versions to use in A/B testing
            traffic_split: Percentage of traffic for each version (must sum to 1.0)
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Basic validation
        if len(version_ids) != len(traffic_split):
            logger.error("Number of versions and traffic splits must match")
            return False
            
        if sum(traffic_split) != 1.0:
            logger.error("Traffic split must sum to 1.0")
            return False
        
        # Get the model
        model = self.get_model_metadata(model_id)
        if not model:
            logger.error(f"Model {model_id} not found")
            return False
        
        # Verify versions exist and belong to this model
        for version_id in version_ids:
            version = self.get_model_version(version_id)
            if not version:
                logger.error(f"Version {version_id} not found")
                return False
                
            if version.model_id != model_id:
                logger.error(f"Version {version_id} does not belong to model {model_id}")
                return False
        
        # Configure A/B testing
        ab_testing_config = {
            "enabled": True,
            "configured_at": datetime.utcnow().isoformat(),
            "versions": [
                {"version_id": vid, "traffic": split}
                for vid, split in zip(version_ids, traffic_split)
            ]
        }
        
        # Add to metadata
        if "ab_testing" not in model.metadata:
            model.metadata["ab_testing"] = {}
            
        model.metadata["ab_testing"]["config"] = ab_testing_config
        model.updated_at = datetime.utcnow()
        
        # Save changes
        self._save_model_metadata(model)
        
        logger.info(f"Configured A/B testing for model {model.name} with {len(version_ids)} versions")
        
        return True
