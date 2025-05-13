"""
ML Model Registry Service

This module implements the Model Registry Service that manages ML model versioning,
tracking, and deployment for the ML Workbench Service.
"""
from typing import Dict, List, Any, Optional, Union
import os
import json
import uuid
import shutil
import datetime
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import joblib
from ml_workbench_service.model_registry.model_metadata import ModelMetadata
from ml_workbench_service.model_registry.model_version import ModelVersion
from ml_workbench_service.model_registry.model_stage import ModelStage
from ml_workbench_service.model_registry.model_metrics import ModelMetrics
from ml_workbench_service.model_registry.registry_exceptions import ModelRegistryException, ModelNotFoundException, ModelVersionNotFoundException, InvalidModelException
from core_foundations.utils.logger import get_logger
logger = get_logger(__name__)


from ml_workbench_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class ModelRegistryService:
    """
    Service for managing ML model versioning, tracking, and deployment.
    
    Key capabilities:
    - Register models with versioning
    - Track model metadata and metrics
    - Manage model lifecycle stages (dev, staging, production, archived)
    - Facilitate A/B testing of models
    - Support model retrieval for inference
    """

    def __init__(self, registry_root_path: str):
        """
        Initialize the Model Registry Service.
        
        Args:
            registry_root_path: Directory path where model registry data will be stored
        """
        self.registry_root_path = Path(registry_root_path)
        self.models_path = self.registry_root_path / 'models'
        self.metadata_path = self.registry_root_path / 'metadata'
        self.models_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path.mkdir(parents=True, exist_ok=True)
        self._metadata_cache = {}
        logger.info(f'ModelRegistryService initialized at {registry_root_path}'
            )

    @with_exception_handling
    def register_model(self, model: Any, model_name: str, model_type: str,
        description: str, version_desc: Optional[str]=None, tags: Optional[
        Dict[str, str]]=None, metrics: Optional[Dict[str, float]]=None,
        parameters: Optional[Dict[str, Any]]=None, framework: str='sklearn',
        feature_names: Optional[List[str]]=None, target_names: Optional[
        List[str]]=None, artifacts: Optional[Dict[str, Any]]=None
        ) ->ModelMetadata:
        """
        Register a new model or a new version of an existing model.
        
        Args:
            model: The model object to register
            model_name: Name of the model
            model_type: Type of model (e.g., "classification", "regression", "forecasting")
            description: Model description
            version_desc: Optional description for this specific version
            tags: Optional tags for the model
            metrics: Optional performance metrics
            parameters: Optional model parameters/hyperparameters
            framework: ML framework used (e.g., "sklearn", "tensorflow", "pytorch")
            feature_names: Optional list of feature names
            target_names: Optional list of target variable names
            artifacts: Optional additional artifacts to save with the model
        
        Returns:
            ModelMetadata: Metadata for the registered model
        """
        if not model_name:
            raise ValueError('Model name is required')
        if not model_type:
            raise ValueError('Model type is required')
        safe_model_name = self._sanitize_name(model_name)
        model_dir = self.models_path / safe_model_name
        model_dir.mkdir(exist_ok=True)
        try:
            metadata = self.get_model_metadata(model_name)
            if description and description != metadata.description:
                metadata.description = description
            if tags:
                metadata.tags.update(tags)
        except ModelNotFoundException:
            metadata = ModelMetadata(name=model_name, model_type=model_type,
                description=description, tags=tags or {}, creation_time=
                datetime.datetime.utcnow().isoformat(), latest_version=0,
                versions=[])
        version = metadata.latest_version + 1
        version_id = str(uuid.uuid4())
        version_dir = model_dir / f'v{version}'
        version_dir.mkdir(exist_ok=True)
        version_info = ModelVersion(version=version, version_id=version_id,
            creation_time=datetime.datetime.utcnow().isoformat(),
            description=version_desc or f'Version {version}', metrics=
            ModelMetrics(**metrics or {}), parameters=parameters or {},
            framework=framework, stage=ModelStage.DEVELOPMENT,
            feature_names=feature_names or [], target_names=target_names or
            [], artifact_paths={})
        model_path = version_dir / 'model.joblib'
        try:
            joblib.dump(model, model_path)
            version_info.artifact_paths['model'] = str(model_path)
        except Exception as e:
            logger.error(f'Failed to save model: {e}')
            shutil.rmtree(version_dir)
            raise InvalidModelException(f'Failed to save model: {e}')
        if artifacts:
            artifacts_dir = version_dir / 'artifacts'
            artifacts_dir.mkdir(exist_ok=True)
            for name, artifact in artifacts.items():
                artifact_path = artifacts_dir / name
                try:
                    if isinstance(artifact, pd.DataFrame):
                        artifact.to_csv(artifact_path.with_suffix('.csv'),
                            index=False)
                        version_info.artifact_paths[name] = str(artifact_path
                            .with_suffix('.csv'))
                    elif isinstance(artifact, dict) or isinstance(artifact,
                        list):
                        with open(artifact_path.with_suffix('.json'), 'w'
                            ) as f:
                            json.dump(artifact, f, indent=2)
                        version_info.artifact_paths[name] = str(artifact_path
                            .with_suffix('.json'))
                    else:
                        joblib.dump(artifact, artifact_path.with_suffix(
                            '.joblib'))
                        version_info.artifact_paths[name] = str(artifact_path
                            .with_suffix('.joblib'))
                except Exception as e:
                    logger.warning(f"Failed to save artifact '{name}': {e}")
        metadata.versions.append(version_info)
        metadata.latest_version = version
        self._save_metadata(metadata)
        logger.info(
            f"Registered model '{model_name}' version {version} with ID {version_id}"
            )
        return metadata

    @with_exception_handling
    def get_model_metadata(self, model_name: str) ->ModelMetadata:
        """
        Get metadata for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelMetadata: Model metadata
            
        Raises:
            ModelNotFoundException: If model is not found
        """
        if model_name in self._metadata_cache:
            return self._metadata_cache[model_name]
        safe_model_name = self._sanitize_name(model_name)
        metadata_path = self.metadata_path / f'{safe_model_name}.json'
        if not metadata_path.exists():
            raise ModelNotFoundException(
                f"Model '{model_name}' not found in registry")
        try:
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            metadata = ModelMetadata(name=metadata_dict['name'], model_type
                =metadata_dict['model_type'], description=metadata_dict[
                'description'], tags=metadata_dict.get('tags', {}),
                creation_time=metadata_dict['creation_time'],
                latest_version=metadata_dict['latest_version'], versions=[])
            for version_dict in metadata_dict.get('versions', []):
                version_info = ModelVersion(version=version_dict['version'],
                    version_id=version_dict['version_id'], creation_time=
                    version_dict['creation_time'], description=version_dict
                    .get('description', ''), metrics=ModelMetrics(**
                    version_dict.get('metrics', {})), parameters=
                    version_dict.get('parameters', {}), framework=
                    version_dict.get('framework', 'unknown'), stage=
                    ModelStage(version_dict.get('stage', 'development')),
                    feature_names=version_dict.get('feature_names', []),
                    target_names=version_dict.get('target_names', []),
                    artifact_paths=version_dict.get('artifact_paths', {}))
                metadata.versions.append(version_info)
            self._metadata_cache[model_name] = metadata
            return metadata
        except Exception as e:
            logger.error(
                f"Failed to load metadata for model '{model_name}': {e}")
            raise ModelRegistryException(
                f"Failed to load metadata for model '{model_name}': {e}")

    @with_exception_handling
    def _save_metadata(self, metadata: ModelMetadata) ->None:
        """Save model metadata to disk."""
        safe_model_name = self._sanitize_name(metadata.name)
        metadata_path = self.metadata_path / f'{safe_model_name}.json'
        metadata_dict = {'name': metadata.name, 'model_type': metadata.
            model_type, 'description': metadata.description, 'tags':
            metadata.tags, 'creation_time': metadata.creation_time,
            'latest_version': metadata.latest_version, 'versions': []}
        for version in metadata.versions:
            version_dict = {'version': version.version, 'version_id':
                version.version_id, 'creation_time': version.creation_time,
                'description': version.description, 'metrics': version.
                metrics.as_dict(), 'parameters': version.parameters,
                'framework': version.framework, 'stage': version.stage.
                value, 'feature_names': version.feature_names,
                'target_names': version.target_names, 'artifact_paths':
                version.artifact_paths}
            metadata_dict['versions'].append(version_dict)
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
            self._metadata_cache[metadata.name] = metadata
        except Exception as e:
            logger.error(
                f"Failed to save metadata for model '{metadata.name}': {e}")
            raise ModelRegistryException(f'Failed to save metadata: {e}')

    @with_exception_handling
    def load_model(self, model_name: str, version: Optional[int]=None,
        stage: Optional[ModelStage]=None) ->Any:
        """
        Load a model from the registry.
        
        Args:
            model_name: Name of the model
            version: Specific version to load (if None, latest version for the given stage will be loaded)
            stage: Model stage to load (defaults to production if not specified and no version is given)
            
        Returns:
            The loaded model object
            
        Raises:
            ModelNotFoundException: If model is not found
            ModelVersionNotFoundException: If version is not found
        """
        metadata = self.get_model_metadata(model_name)
        version_info = None
        if version is not None:
            for v in metadata.versions:
                if v.version == version:
                    version_info = v
                    break
            if version_info is None:
                raise ModelVersionNotFoundException(
                    f"Version {version} not found for model '{model_name}'")
        elif stage is not None:
            candidates = [v for v in metadata.versions if v.stage == stage]
            if not candidates:
                raise ModelVersionNotFoundException(
                    f"No version found for model '{model_name}' in stage {stage.value}"
                    )
            version_info = max(candidates, key=lambda v: v.version)
        else:
            production_versions = [v for v in metadata.versions if v.stage ==
                ModelStage.PRODUCTION]
            if production_versions:
                version_info = max(production_versions, key=lambda v: v.version
                    )
            else:
                version_info = max(metadata.versions, key=lambda v: v.version)
        if 'model' not in version_info.artifact_paths:
            raise ModelRegistryException(
                f"Model artifact not found for '{model_name}' version {version_info.version}"
                )
        model_path = version_info.artifact_paths['model']
        try:
            model = joblib.load(model_path)
            logger.info(
                f"Loaded model '{model_name}' version {version_info.version} (stage: {version_info.stage.value})"
                )
            return model
        except Exception as e:
            logger.error(f'Failed to load model: {e}')
            raise ModelRegistryException(f'Failed to load model: {e}')

    @with_exception_handling
    def load_artifact(self, model_name: str, artifact_name: str, version:
        Optional[int]=None, stage: Optional[ModelStage]=None) ->Any:
        """
        Load a model artifact from the registry.
        
        Args:
            model_name: Name of the model
            artifact_name: Name of the artifact to load
            version: Specific version to load from (if None, latest version for the given stage will be used)
            stage: Model stage to load from (defaults to production if not specified and no version is given)
            
        Returns:
            The loaded artifact
            
        Raises:
            ModelNotFoundException: If model is not found
            ModelVersionNotFoundException: If version is not found
            ModelRegistryException: If artifact is not found
        """
        metadata = self.get_model_metadata(model_name)
        version_info = None
        if version is not None:
            for v in metadata.versions:
                if v.version == version:
                    version_info = v
                    break
            if version_info is None:
                raise ModelVersionNotFoundException(
                    f"Version {version} not found for model '{model_name}'")
        elif stage is not None:
            candidates = [v for v in metadata.versions if v.stage == stage]
            if not candidates:
                raise ModelVersionNotFoundException(
                    f"No version found for model '{model_name}' in stage {stage.value}"
                    )
            version_info = max(candidates, key=lambda v: v.version)
        else:
            production_versions = [v for v in metadata.versions if v.stage ==
                ModelStage.PRODUCTION]
            if production_versions:
                version_info = max(production_versions, key=lambda v: v.version
                    )
            else:
                version_info = max(metadata.versions, key=lambda v: v.version)
        if artifact_name not in version_info.artifact_paths:
            raise ModelRegistryException(
                f"Artifact '{artifact_name}' not found for '{model_name}' version {version_info.version}"
                )
        artifact_path = version_info.artifact_paths[artifact_name]
        path = Path(artifact_path)
        try:
            if path.suffix == '.csv':
                return pd.read_csv(path)
            elif path.suffix == '.json':
                with open(path, 'r') as f:
                    return json.load(f)
            else:
                return joblib.load(path)
        except Exception as e:
            logger.error(f"Failed to load artifact '{artifact_name}': {e}")
            raise ModelRegistryException(
                f"Failed to load artifact '{artifact_name}': {e}")

    def update_model_version_stage(self, model_name: str, version: int,
        stage: ModelStage) ->ModelMetadata:
        """
        Update the stage of a model version.
        
        Args:
            model_name: Name of the model
            version: Version to update
            stage: New stage for the version
            
        Returns:
            ModelMetadata: Updated model metadata
            
        Raises:
            ModelNotFoundException: If model is not found
            ModelVersionNotFoundException: If version is not found
        """
        metadata = self.get_model_metadata(model_name)
        version_found = False
        for v in metadata.versions:
            if v.version == version:
                v.stage = stage
                version_found = True
                break
        if not version_found:
            raise ModelVersionNotFoundException(
                f"Version {version} not found for model '{model_name}'")
        if stage == ModelStage.PRODUCTION:
            for v in metadata.versions:
                if v.version != version and v.stage == ModelStage.PRODUCTION:
                    v.stage = ModelStage.STAGING
                    logger.info(
                        f"Moved '{model_name}' version {v.version} from production to staging as version {version} is now in production"
                        )
        self._save_metadata(metadata)
        logger.info(
            f"Updated '{model_name}' version {version} stage to {stage.value}")
        return metadata

    def update_model_version_metrics(self, model_name: str, version: int,
        metrics: Dict[str, float]) ->ModelMetadata:
        """
        Update metrics for a model version.
        
        Args:
            model_name: Name of the model
            version: Version to update
            metrics: Metrics to update or add
            
        Returns:
            ModelMetadata: Updated model metadata
            
        Raises:
            ModelNotFoundException: If model is not found
            ModelVersionNotFoundException: If version is not found
        """
        metadata = self.get_model_metadata(model_name)
        version_found = False
        for v in metadata.versions:
            if v.version == version:
                for metric_name, value in metrics.items():
                    setattr(v.metrics, metric_name, value)
                version_found = True
                break
        if not version_found:
            raise ModelVersionNotFoundException(
                f"Version {version} not found for model '{model_name}'")
        self._save_metadata(metadata)
        logger.info(f"Updated metrics for '{model_name}' version {version}")
        return metadata

    @with_exception_handling
    def delete_model_version(self, model_name: str, version: int
        ) ->ModelMetadata:
        """
        Delete a model version.
        
        Args:
            model_name: Name of the model
            version: Version to delete
            
        Returns:
            ModelMetadata: Updated model metadata
            
        Raises:
            ModelNotFoundException: If model is not found
            ModelVersionNotFoundException: If version is not found
        """
        metadata = self.get_model_metadata(model_name)
        version_idx = None
        version_info = None
        for i, v in enumerate(metadata.versions):
            if v.version == version:
                version_idx = i
                version_info = v
                break
        if version_idx is None or version_info is None:
            raise ModelVersionNotFoundException(
                f"Version {version} not found for model '{model_name}'")
        if version_info.stage == ModelStage.PRODUCTION:
            raise ModelRegistryException(
                f"Cannot delete version {version} of model '{model_name}' because it is in production"
                )
        safe_model_name = self._sanitize_name(model_name)
        version_dir = self.models_path / safe_model_name / f'v{version}'
        if version_dir.exists():
            try:
                shutil.rmtree(version_dir)
            except Exception as e:
                logger.error(f'Failed to delete version directory: {e}')
        metadata.versions.pop(version_idx)
        if metadata.versions:
            metadata.latest_version = max(v.version for v in metadata.versions)
        else:
            metadata.latest_version = 0
        self._save_metadata(metadata)
        logger.info(f"Deleted version {version} of model '{model_name}'")
        return metadata

    @with_exception_handling
    def delete_model(self, model_name: str) ->bool:
        """
        Delete a model and all its versions.
        
        Args:
            model_name: Name of the model
            
        Returns:
            bool: True if successful
            
        Raises:
            ModelNotFoundException: If model is not found
        """
        try:
            metadata = self.get_model_metadata(model_name)
        except ModelNotFoundException:
            logger.warning(f"Model '{model_name}' not found, nothing to delete"
                )
            return False
        for version in metadata.versions:
            if version.stage == ModelStage.PRODUCTION:
                raise ModelRegistryException(
                    f"Cannot delete model '{model_name}' because version {version.version} is in production"
                    )
        safe_model_name = self._sanitize_name(model_name)
        model_dir = self.models_path / safe_model_name
        if model_dir.exists():
            try:
                shutil.rmtree(model_dir)
            except Exception as e:
                logger.error(f'Failed to delete model directory: {e}')
                return False
        metadata_path = self.metadata_path / f'{safe_model_name}.json'
        if metadata_path.exists():
            try:
                metadata_path.unlink()
            except Exception as e:
                logger.error(f'Failed to delete metadata file: {e}')
                return False
        if model_name in self._metadata_cache:
            del self._metadata_cache[model_name]
        logger.info(f"Deleted model '{model_name}' and all its versions")
        return True

    @with_exception_handling
    def list_models(self, model_type: Optional[str]=None, tag_filter:
        Optional[Dict[str, str]]=None) ->List[Dict[str, Any]]:
        """
        List all models in the registry with optional filtering.
        
        Args:
            model_type: Filter by model type
            tag_filter: Filter by tags (all specified tags must match)
            
        Returns:
            List[Dict[str, Any]]: List of model information dictionaries
        """
        results = []
        metadata_files = self.metadata_path.glob('*.json')
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                if model_type and metadata.get('model_type') != model_type:
                    continue
                if tag_filter:
                    tags = metadata.get('tags', {})
                    if not all(tags.get(k) == v for k, v in tag_filter.items()
                        ):
                        continue
                production_version = None
                for v in metadata.get('versions', []):
                    if v.get('stage') == 'production':
                        production_version = v.get('version')
                        break
                model_info = {'name': metadata.get('name'), 'model_type':
                    metadata.get('model_type'), 'description': metadata.get
                    ('description'), 'creation_time': metadata.get(
                    'creation_time'), 'tags': metadata.get('tags', {}),
                    'latest_version': metadata.get('latest_version', 0),
                    'production_version': production_version,
                    'version_count': len(metadata.get('versions', []))}
                results.append(model_info)
            except Exception as e:
                logger.error(
                    f'Failed to parse metadata file {metadata_file}: {e}')
        return results

    @with_exception_handling
    def search_models(self, name_contains: Optional[str]=None, model_type:
        Optional[str]=None, tag_filter: Optional[Dict[str, str]]=None,
        min_metric_filter: Optional[Dict[str, float]]=None, production_only:
        bool=False) ->List[Dict[str, Any]]:
        """
        Search for models in the registry with advanced filtering.
        
        Args:
            name_contains: Filter models whose name contains this string
            model_type: Filter by model type
            tag_filter: Filter by tags (all specified tags must match)
            min_metric_filter: Filter by minimum metric values (for production versions)
            production_only: Only include models with production versions
            
        Returns:
            List[Dict[str, Any]]: List of model information dictionaries
        """
        results = []
        metadata_files = self.metadata_path.glob('*.json')
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                if name_contains and name_contains.lower() not in metadata.get(
                    'name', '').lower():
                    continue
                if model_type and metadata.get('model_type') != model_type:
                    continue
                if tag_filter:
                    tags = metadata.get('tags', {})
                    if not all(tags.get(k) == v for k, v in tag_filter.items()
                        ):
                        continue
                production_version = None
                production_metrics = None
                for v in metadata.get('versions', []):
                    if v.get('stage') == 'production':
                        production_version = v.get('version')
                        production_metrics = v.get('metrics', {})
                        break
                if production_only and production_version is None:
                    continue
                if min_metric_filter and production_metrics:
                    if not all(production_metrics.get(k, 0) >= v for k, v in
                        min_metric_filter.items()):
                        continue
                model_info = {'name': metadata.get('name'), 'model_type':
                    metadata.get('model_type'), 'description': metadata.get
                    ('description'), 'creation_time': metadata.get(
                    'creation_time'), 'tags': metadata.get('tags', {}),
                    'latest_version': metadata.get('latest_version', 0),
                    'production_version': production_version,
                    'production_metrics': production_metrics,
                    'version_count': len(metadata.get('versions', []))}
                results.append(model_info)
            except Exception as e:
                logger.error(
                    f'Failed to parse metadata file {metadata_file}: {e}')
        return results

    def compare_versions(self, model_name: str, version1: int, version2: int
        ) ->Dict[str, Any]:
        """
        Compare two versions of a model.
        
        Args:
            model_name: Name of the model
            version1: First version to compare
            version2: Second version to compare
            
        Returns:
            Dict[str, Any]: Comparison results
            
        Raises:
            ModelNotFoundException: If model is not found
            ModelVersionNotFoundException: If either version is not found
        """
        metadata = self.get_model_metadata(model_name)
        v1_info = None
        v2_info = None
        for v in metadata.versions:
            if v.version == version1:
                v1_info = v
            if v.version == version2:
                v2_info = v
            if v1_info and v2_info:
                break
        if v1_info is None:
            raise ModelVersionNotFoundException(
                f"Version {version1} not found for model '{model_name}'")
        if v2_info is None:
            raise ModelVersionNotFoundException(
                f"Version {version2} not found for model '{model_name}'")
        metrics_comparison = {}
        all_metrics = set(vars(v1_info.metrics).keys()) | set(vars(v2_info.
            metrics).keys())
        for metric in all_metrics:
            v1_value = getattr(v1_info.metrics, metric, None)
            v2_value = getattr(v2_info.metrics, metric, None)
            if v1_value is not None and v2_value is not None:
                diff = v2_value - v1_value
                pct_change = diff / v1_value * 100 if v1_value != 0 else float(
                    'inf')
                metrics_comparison[metric] = {'version1': v1_value,
                    'version2': v2_value, 'difference': diff,
                    'percent_change': pct_change}
            else:
                metrics_comparison[metric] = {'version1': v1_value,
                    'version2': v2_value, 'difference': 'N/A',
                    'percent_change': 'N/A'}
        param_comparison = {}
        all_params = set(v1_info.parameters.keys()) | set(v2_info.
            parameters.keys())
        for param in all_params:
            v1_value = v1_info.parameters.get(param)
            v2_value = v2_info.parameters.get(param)
            param_comparison[param] = {'version1': v1_value, 'version2':
                v2_value, 'changed': v1_value != v2_value}
        feature_comparison = {'version1_only': [f for f in v1_info.
            feature_names if f not in v2_info.feature_names],
            'version2_only': [f for f in v2_info.feature_names if f not in
            v1_info.feature_names], 'common': [f for f in v1_info.
            feature_names if f in v2_info.feature_names]}
        artifact_comparison = {'version1_only': [a for a in v1_info.
            artifact_paths.keys() if a not in v2_info.artifact_paths],
            'version2_only': [a for a in v2_info.artifact_paths.keys() if a
             not in v1_info.artifact_paths], 'common': [a for a in v1_info.
            artifact_paths.keys() if a in v2_info.artifact_paths]}
        comparison = {'model_name': model_name, 'version1': {'version':
            version1, 'creation_time': v1_info.creation_time, 'stage':
            v1_info.stage.value, 'framework': v1_info.framework},
            'version2': {'version': version2, 'creation_time': v2_info.
            creation_time, 'stage': v2_info.stage.value, 'framework':
            v2_info.framework}, 'metrics_comparison': metrics_comparison,
            'parameter_comparison': param_comparison, 'feature_comparison':
            feature_comparison, 'artifact_comparison': artifact_comparison}
        return comparison

    @with_exception_handling
    def setup_ab_test(self, model_name: str, version_a: int, version_b: int,
        test_name: str, traffic_split: float=0.5, description: str='') ->Dict[
        str, Any]:
        """
        Set up an A/B test between two model versions.
        
        Args:
            model_name: Name of the model
            version_a: First version for testing (A)
            version_b: Second version for testing (B)
            test_name: Name for the A/B test
            traffic_split: Portion of traffic to send to version B (0.0-1.0)
            description: Description of the A/B test
            
        Returns:
            Dict[str, Any]: A/B test configuration
            
        Raises:
            ModelNotFoundException: If model is not found
            ModelVersionNotFoundException: If either version is not found
        """
        metadata = self.get_model_metadata(model_name)
        v_a = None
        v_b = None
        for v in metadata.versions:
            if v.version == version_a:
                v_a = v
            if v.version == version_b:
                v_b = v
        if v_a is None:
            raise ModelVersionNotFoundException(
                f"Version {version_a} not found for model '{model_name}'")
        if v_b is None:
            raise ModelVersionNotFoundException(
                f"Version {version_b} not found for model '{model_name}'")
        test_id = str(uuid.uuid4())
        ab_test = {'test_id': test_id, 'test_name': test_name,
            'description': description, 'model_name': model_name,
            'version_a': version_a, 'version_b': version_b, 'traffic_split':
            traffic_split, 'creation_time': datetime.datetime.utcnow().
            isoformat(), 'status': 'active'}
        tests_dir = self.registry_root_path / 'ab_tests'
        tests_dir.mkdir(exist_ok=True)
        test_path = tests_dir / f'{test_id}.json'
        try:
            with open(test_path, 'w') as f:
                json.dump(ab_test, f, indent=2)
            logger.info(
                f"Created A/B test '{test_name}' between '{model_name}' versions {version_a} and {version_b} with {traffic_split:.0%} traffic to B"
                )
            return ab_test
        except Exception as e:
            logger.error(f'Failed to setup A/B test: {e}')
            raise ModelRegistryException(f'Failed to setup A/B test: {e}')

    @with_exception_handling
    def get_ab_test(self, test_id: str) ->Dict[str, Any]:
        """
        Get A/B test configuration.
        
        Args:
            test_id: ID of the A/B test
            
        Returns:
            Dict[str, Any]: A/B test configuration
            
        Raises:
            ModelRegistryException: If A/B test is not found
        """
        test_path = self.registry_root_path / 'ab_tests' / f'{test_id}.json'
        if not test_path.exists():
            raise ModelRegistryException(
                f"A/B test with ID '{test_id}' not found")
        try:
            with open(test_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f'Failed to load A/B test: {e}')
            raise ModelRegistryException(f'Failed to load A/B test: {e}')

    @with_exception_handling
    def list_ab_tests(self, model_name: Optional[str]=None, status:
        Optional[str]=None) ->List[Dict[str, Any]]:
        """
        List A/B tests with optional filtering.
        
        Args:
            model_name: Filter tests for a specific model
            status: Filter by test status ("active", "completed", "cancelled")
            
        Returns:
            List[Dict[str, Any]]: List of A/B test configurations
        """
        tests_dir = self.registry_root_path / 'ab_tests'
        if not tests_dir.exists():
            return []
        tests = []
        for test_file in tests_dir.glob('*.json'):
            try:
                with open(test_file, 'r') as f:
                    test = json.load(f)
                if model_name and test.get('model_name') != model_name:
                    continue
                if status and test.get('status') != status:
                    continue
                tests.append(test)
            except Exception as e:
                logger.error(f'Failed to parse A/B test file {test_file}: {e}')
        return tests

    @with_exception_handling
    def update_ab_test(self, test_id: str, status: Optional[str]=None,
        traffic_split: Optional[float]=None) ->Dict[str, Any]:
        """
        Update an A/B test.
        
        Args:
            test_id: ID of the A/B test
            status: New status for the test
            traffic_split: New traffic split
            
        Returns:
            Dict[str, Any]: Updated A/B test configuration
            
        Raises:
            ModelRegistryException: If A/B test is not found
        """
        test = self.get_ab_test(test_id)
        if status:
            test['status'] = status
        if traffic_split is not None:
            test['traffic_split'] = traffic_split
        test['last_modified'] = datetime.datetime.utcnow().isoformat()
        test_path = self.registry_root_path / 'ab_tests' / f'{test_id}.json'
        try:
            with open(test_path, 'w') as f:
                json.dump(test, f, indent=2)
            logger.info(f"Updated A/B test '{test_id}'")
            return test
        except Exception as e:
            logger.error(f'Failed to update A/B test: {e}')
            raise ModelRegistryException(f'Failed to update A/B test: {e}')

    def _sanitize_name(self, name: str) ->str:
        """Convert model name to a safe filename."""
        return name.replace(' ', '_').replace('/', '_').replace('\\', '_')
