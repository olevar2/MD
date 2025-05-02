"""
Model Registry Service

This module provides the service layer for interacting with the model registry.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import os
import logging
import json

from ml_workbench_service.model_registry.registry import (
    ModelRegistry, ModelVersion, ModelMetadata, 
    ModelStatus, ModelType, ModelFramework
)
from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)

class ModelRegistryService:
    """
    Service for interacting with the model registry. This service provides
    higher-level functionality around the core ModelRegistry capabilities.
    """
    
    def __init__(self, registry: Optional[ModelRegistry] = None, storage_path: str = None):
        """
        Initialize the ModelRegistryService.
        
        Args:
            registry: Optional existing registry instance to use
            storage_path: Path for model storage if creating a new registry
        """
        self.registry = registry or ModelRegistry(storage_path=storage_path)
        logger.info("ModelRegistryService initialized")
        
    async def register_model(
        self,
        name: str, 
        model_type: Union[ModelType, str],
        description: Optional[str] = None,
        created_by: str = "system",
        tags: List[str] = None,
        business_domain: Optional[str] = None,
        purpose: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Register a new model in the registry.
        
        Args:
            name: Name of the model
            model_type: Type of the model
            description: Optional description
            created_by: User or system that created the model
            tags: Optional list of tags for the model
            business_domain: Business domain the model belongs to
            purpose: Purpose of the model
            metadata: Additional metadata for the model
            
        Returns:
            Dict[str, Any]: Registration result
        """
        try:
            # Check if model with this name already exists
            existing_models = self.registry.list_models(name_filter=name)
            exact_matches = [m for m in existing_models if m.name == name]
            
            if exact_matches:
                return {
                    "success": False,
                    "error": f"Model with name '{name}' already exists",
                    "existing_model_id": exact_matches[0].model_id
                }
            
            # Register the model
            model_metadata = self.registry.register_model(
                name=name,
                model_type=model_type,
                description=description,
                created_by=created_by,
                tags=tags,
                business_domain=business_domain,
                purpose=purpose,
                metadata=metadata
            )
            
            return {
                "success": True,
                "model_id": model_metadata.model_id,
                "name": model_metadata.name,
                "created_at": model_metadata.created_at.isoformat()
            }
            
        except Exception as e:
            error_message = f"Failed to register model: {str(e)}"
            logger.error(error_message)
            return {
                "success": False,
                "error": error_message
            }
    
    async def create_model_version(
        self,
        model_id: str,
        framework: Union[ModelFramework, str],
        framework_version: str,
        created_by: str,
        model_file_path: Optional[str] = None,
        description: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
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
    ) -> Dict[str, Any]:
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
            Dict[str, Any]: Version creation result
        """
        try:
            # Check if model exists
            model_metadata = self.registry.get_model_metadata(model_id)
            if not model_metadata:
                return {
                    "success": False,
                    "error": f"Model with ID {model_id} not found"
                }
                
            # Validate model file path if provided
            if model_file_path and not os.path.exists(model_file_path):
                return {
                    "success": False,
                    "error": f"Model file not found at {model_file_path}"
                }
            
            # Create the version
            version = self.registry.create_model_version(
                model_id=model_id,
                framework=framework,
                framework_version=framework_version,
                created_by=created_by,
                model_file_path=model_file_path,
                description=description,
                hyperparameters=hyperparameters,
                metrics=metrics,
                feature_columns=feature_columns,
                target_column=target_column,
                preprocessing_steps=preprocessing_steps,
                training_dataset_id=training_dataset_id,
                validation_dataset_id=validation_dataset_id,
                test_dataset_id=test_dataset_id,
                tags=tags,
                experiment_id=experiment_id,
                notes=notes,
                metadata=metadata,
                status=status
            )
            
            return {
                "success": True,
                "model_id": version.model_id,
                "version_id": version.version_id,
                "version_number": version.version_number,
                "status": version.status.value,
                "created_at": version.created_at.isoformat()
            }
            
        except Exception as e:
            error_message = f"Failed to create model version: {str(e)}"
            logger.error(error_message)
            return {
                "success": False,
                "error": error_message
            }
    
    async def get_model_details(self, model_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Dict[str, Any]: Model details
        """
        try:
            # Get model metadata
            model_metadata = self.registry.get_model_metadata(model_id)
            if not model_metadata:
                return {
                    "success": False,
                    "error": f"Model with ID {model_id} not found"
                }
                
            # Get versions
            versions = self.registry.list_versions(model_id)
            
            # Get production and staging versions
            production_version = None
            staging_version = None
            
            if model_metadata.production_version_id:
                production_version = self.registry.get_model_version(model_metadata.production_version_id)
                
            if model_metadata.staging_version_id:
                staging_version = self.registry.get_model_version(model_metadata.staging_version_id)
            
            # Format response
            result = {
                "success": True,
                "model_id": model_metadata.model_id,
                "name": model_metadata.name,
                "description": model_metadata.description,
                "model_type": model_metadata.model_type.value,
                "created_at": model_metadata.created_at.isoformat(),
                "updated_at": model_metadata.updated_at.isoformat(),
                "created_by": model_metadata.created_by,
                "tags": model_metadata.tags,
                "business_domain": model_metadata.business_domain,
                "purpose": model_metadata.purpose,
                "version_count": len(versions),
                "latest_version": model_metadata.latest_version_number,
                "has_production_version": model_metadata.production_version_id is not None,
                "has_staging_version": model_metadata.staging_version_id is not None,
                "metadata": model_metadata.metadata
            }
            
            if production_version:
                result["production_version"] = {
                    "version_id": production_version.version_id,
                    "version_number": production_version.version_number,
                    "created_at": production_version.created_at.isoformat()
                }
                
            if staging_version:
                result["staging_version"] = {
                    "version_id": staging_version.version_id,
                    "version_number": staging_version.version_number,
                    "created_at": staging_version.created_at.isoformat()
                }
                
            return result
            
        except Exception as e:
            error_message = f"Failed to get model details: {str(e)}"
            logger.error(error_message)
            return {
                "success": False,
                "error": error_message
            }
    
    async def get_version_details(self, version_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a model version.
        
        Args:
            version_id: ID of the version
            
        Returns:
            Dict[str, Any]: Version details
        """
        try:
            # Get version
            version = self.registry.get_model_version(version_id)
            if not version:
                return {
                    "success": False,
                    "error": f"Version with ID {version_id} not found"
                }
                
            # Get model metadata for context
            model_metadata = self.registry.get_model_metadata(version.model_id)
            
            # Format response
            result = {
                "success": True,
                "version_id": version.version_id,
                "model_id": version.model_id,
                "model_name": model_metadata.name if model_metadata else None,
                "version_number": version.version_number,
                "status": version.status.value,
                "created_at": version.created_at.isoformat(),
                "updated_at": version.updated_at.isoformat(),
                "created_by": version.created_by,
                "description": version.description,
                "framework": version.framework.value,
                "framework_version": version.framework_version,
                "metrics": version.metrics.all_metrics,
                "hyperparameters": version.hyperparameters.as_dict,
                "feature_columns": version.feature_columns,
                "target_column": version.target_column,
                "has_artifact": version.file_path is not None,
                "tags": version.tags,
                "experiment_id": version.experiment_id,
                "notes": version.notes,
                "metadata": version.metadata
            }
            
            if model_metadata:
                result["is_production"] = model_metadata.production_version_id == version_id
                result["is_staging"] = model_metadata.staging_version_id == version_id
                result["is_latest"] = model_metadata.latest_version_id == version_id
                
            return result
            
        except Exception as e:
            error_message = f"Failed to get version details: {str(e)}"
            logger.error(error_message)
            return {
                "success": False,
                "error": error_message
            }
    
    async def list_models(
        self,
        name_filter: Optional[str] = None,
        model_type: Optional[Union[ModelType, str]] = None,
        tags: Optional[List[str]] = None,
        business_domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List models in the registry with optional filtering.
        
        Args:
            name_filter: Optional filter for model name
            model_type: Optional filter by model type
            tags: Optional filter by tags
            business_domain: Optional filter by business domain
            
        Returns:
            Dict[str, Any]: List of models
        """
        try:
            models = self.registry.list_models(
                name_filter=name_filter,
                model_type=model_type,
                tags=tags,
                business_domain=business_domain
            )
            
            result = {
                "success": True,
                "count": len(models),
                "models": []
            }
            
            for model in models:
                model_summary = {
                    "model_id": model.model_id,
                    "name": model.name,
                    "model_type": model.model_type.value,
                    "description": model.description,
                    "created_at": model.created_at.isoformat(),
                    "updated_at": model.updated_at.isoformat(),
                    "created_by": model.created_by,
                    "tags": model.tags,
                    "business_domain": model.business_domain,
                    "latest_version_number": model.latest_version_number,
                    "has_production_version": model.production_version_id is not None,
                    "has_staging_version": model.staging_version_id is not None
                }
                
                result["models"].append(model_summary)
                
            return result
            
        except Exception as e:
            error_message = f"Failed to list models: {str(e)}"
            logger.error(error_message)
            return {
                "success": False,
                "error": error_message
            }
    
    async def list_versions(
        self,
        model_id: str,
        status_filter: Optional[Union[ModelStatus, str]] = None,
        min_version: Optional[int] = None,
        max_version: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        List versions for a specific model with optional filtering.
        
        Args:
            model_id: ID of the model
            status_filter: Optional filter by status
            min_version: Optional minimum version number
            max_version: Optional maximum version number
            
        Returns:
            Dict[str, Any]: List of versions
        """
        try:
            # Check if model exists
            model_metadata = self.registry.get_model_metadata(model_id)
            if not model_metadata:
                return {
                    "success": False,
                    "error": f"Model with ID {model_id} not found"
                }
                
            # Get versions
            versions = self.registry.list_versions(
                model_id=model_id,
                status_filter=status_filter,
                min_version=min_version,
                max_version=max_version
            )
            
            result = {
                "success": True,
                "model_id": model_id,
                "model_name": model_metadata.name,
                "count": len(versions),
                "versions": []
            }
            
            for version in versions:
                version_summary = {
                    "version_id": version.version_id,
                    "version_number": version.version_number,
                    "status": version.status.value,
                    "created_at": version.created_at.isoformat(),
                    "updated_at": version.updated_at.isoformat(),
                    "created_by": version.created_by,
                    "framework": version.framework.value,
                    "framework_version": version.framework_version,
                    "has_artifact": version.file_path is not None,
                    "tags": version.tags,
                    "is_production": model_metadata.production_version_id == version.version_id,
                    "is_staging": model_metadata.staging_version_id == version.version_id,
                    "is_latest": model_metadata.latest_version_id == version.version_id
                }
                
                # Add key metrics if available
                metrics = version.metrics.all_metrics
                if metrics:
                    important_metrics = ['accuracy', 'f1_score', 'auc', 'rmse', 'mae']
                    version_summary['metrics'] = {
                        metric: metrics.get(metric)
                        for metric in important_metrics
                        if metrics.get(metric) is not None
                    }
                
                result["versions"].append(version_summary)
                
            return result
            
        except Exception as e:
            error_message = f"Failed to list versions: {str(e)}"
            logger.error(error_message)
            return {
                "success": False,
                "error": error_message
            }
    
    async def update_version_status(
        self,
        version_id: str,
        status: Union[ModelStatus, str]
    ) -> Dict[str, Any]:
        """
        Update the status of a model version.
        
        Args:
            version_id: ID of the version
            status: New status
            
        Returns:
            Dict[str, Any]: Update result
        """
        try:
            # Update status
            version = self.registry.set_version_status(version_id, status)
            
            if not version:
                return {
                    "success": False,
                    "error": f"Version with ID {version_id} not found"
                }
                
            # Get model metadata for context
            model_metadata = self.registry.get_model_metadata(version.model_id)
            
            return {
                "success": True,
                "version_id": version.version_id,
                "model_id": version.model_id,
                "model_name": model_metadata.name if model_metadata else None,
                "version_number": version.version_number,
                "status": version.status.value,
                "updated_at": version.updated_at.isoformat(),
                "is_production": status == ModelStatus.PRODUCTION,
                "is_staging": status == ModelStatus.STAGING
            }
            
        except Exception as e:
            error_message = f"Failed to update version status: {str(e)}"
            logger.error(error_message)
            return {
                "success": False,
                "error": error_message
            }
    
    async def compare_versions(
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
        try:
            # Get comparison
            comparison = self.registry.compare_versions(version_id_1, version_id_2)
            
            return {
                "success": True,
                "comparison": comparison
            }
            
        except Exception as e:
            error_message = f"Failed to compare versions: {str(e)}"
            logger.error(error_message)
            return {
                "success": False,
                "error": error_message
            }
    
    async def get_best_version(
        self,
        model_id: str,
        metric: str,
        higher_is_better: bool = True,
        status_filter: Optional[Union[ModelStatus, str]] = None
    ) -> Dict[str, Any]:
        """
        Get the best model version based on a specific metric.
        
        Args:
            model_id: ID of the model
            metric: Metric to use for comparison
            higher_is_better: Whether higher metric values are better
            status_filter: Optional filter by status
            
        Returns:
            Dict[str, Any]: Best version details
        """
        try:
            # Check if model exists
            model_metadata = self.registry.get_model_metadata(model_id)
            if not model_metadata:
                return {
                    "success": False,
                    "error": f"Model with ID {model_id} not found"
                }
                
            # Get best version
            version = self.registry.get_best_version(
                model_id=model_id,
                metric=metric,
                higher_is_better=higher_is_better,
                status_filter=status_filter
            )
            
            if not version:
                return {
                    "success": True,
                    "found": False,
                    "message": f"No versions found with metric '{metric}'"
                }
                
            metric_value = version.metrics.get_primary_metric(metric)
            
            return {
                "success": True,
                "found": True,
                "model_id": model_id,
                "model_name": model_metadata.name,
                "version_id": version.version_id,
                "version_number": version.version_number,
                "status": version.status.value,
                "metric_name": metric,
                "metric_value": metric_value,
                "created_at": version.created_at.isoformat()
            }
            
        except Exception as e:
            error_message = f"Failed to get best version: {str(e)}"
            logger.error(error_message)
            return {
                "success": False,
                "error": error_message
            }
    
    async def configure_ab_testing(
        self,
        model_id: str,
        version_ids: List[str],
        traffic_split: List[float]
    ) -> Dict[str, Any]:
        """
        Configure A/B testing for a model.
        
        Args:
            model_id: ID of the model
            version_ids: IDs of versions to use in A/B testing
            traffic_split: Percentage of traffic for each version
            
        Returns:
            Dict[str, Any]: Configuration result
        """
        try:
            # Configure A/B testing
            success = self.registry.configure_ab_testing(
                model_id=model_id,
                version_ids=version_ids,
                traffic_split=traffic_split
            )
            
            if not success:
                return {
                    "success": False,
                    "error": "Failed to configure A/B testing"
                }
            
            # Get model metadata for response context
            model_metadata = self.registry.get_model_metadata(model_id)
            
            # Get versions for response context
            versions = []
            for version_id in version_ids:
                version = self.registry.get_model_version(version_id)
                if version:
                    versions.append({
                        "version_id": version.version_id,
                        "version_number": version.version_number,
                        "status": version.status.value
                    })
            
            return {
                "success": True,
                "model_id": model_id,
                "model_name": model_metadata.name if model_metadata else None,
                "versions": versions,
                "traffic_split": traffic_split,
                "configured_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            error_message = f"Failed to configure A/B testing: {str(e)}"
            logger.error(error_message)
            return {
                "success": False,
                "error": error_message
            }
    
    async def delete_model(self, model_id: str) -> Dict[str, Any]:
        """
        Delete a model and all its versions.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Dict[str, Any]: Deletion result
        """
        try:
            # Get model metadata for response context
            model_metadata = self.registry.get_model_metadata(model_id)
            if not model_metadata:
                return {
                    "success": False,
                    "error": f"Model with ID {model_id} not found"
                }
            
            model_name = model_metadata.name
            
            # Delete the model
            success = self.registry.delete_model(model_id)
            
            if not success:
                return {
                    "success": False,
                    "error": f"Failed to delete model {model_id}"
                }
                
            return {
                "success": True,
                "model_id": model_id,
                "model_name": model_name,
                "deleted_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            error_message = f"Failed to delete model: {str(e)}"
            logger.error(error_message)
            return {
                "success": False,
                "error": error_message
            }
    
    async def delete_version(self, version_id: str) -> Dict[str, Any]:
        """
        Delete a specific model version.
        
        Args:
            version_id: ID of the version
            
        Returns:
            Dict[str, Any]: Deletion result
        """
        try:
            # Get version for response context
            version = self.registry.get_model_version(version_id)
            if not version:
                return {
                    "success": False,
                    "error": f"Version with ID {version_id} not found"
                }
                
            model_id = version.model_id
            version_number = version.version_number
            
            # Get model name for context
            model_metadata = self.registry.get_model_metadata(model_id)
            model_name = model_metadata.name if model_metadata else None
            
            # Delete the version
            success = self.registry.delete_version(version_id)
            
            if not success:
                return {
                    "success": False,
                    "error": f"Failed to delete version {version_id}"
                }
                
            return {
                "success": True,
                "version_id": version_id,
                "version_number": version_number,
                "model_id": model_id,
                "model_name": model_name,
                "deleted_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            error_message = f"Failed to delete version: {str(e)}"
            logger.error(error_message)
            return {
                "success": False,
                "error": error_message
            }
