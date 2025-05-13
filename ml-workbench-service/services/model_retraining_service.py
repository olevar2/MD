"""
Model Drift Detection and Retraining Integration Module

This module provides integration between the Model Drift Detection system and 
automated retraining workflows, enabling continuous model improvement.
"""
import os
import json
import logging
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from enum import Enum
import uuid
import asyncio
import traceback
import requests
from pydantic import BaseModel, Field
from core_foundations.utils.logger import get_logger
from core_foundations.config.settings import get_settings
from services.model_registry_service import ModelRegistryService
from models.registry import ModelStatus, ModelVersion, ModelMetrics
from models.model_drift_detector import ModelDriftDetector, DriftMetrics, ModelPerformanceDrift
from core.auto_optimization_framework import AutoOptimizationFramework
logger = get_logger(__name__)
settings = get_settings()


from ml_workbench_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class RetrainingStatus(str, Enum):
    """Status of a retraining workflow."""
    PENDING = 'pending'
    IN_PROGRESS = 'in_progress'
    COMPLETED = 'completed'
    FAILED = 'failed'
    EVALUATED = 'evaluated'
    DEPLOYED = 'deployed'
    ABORTED = 'aborted'


class RetrainingTriggerType(str, Enum):
    """Types of triggers for model retraining."""
    FEATURE_DRIFT = 'feature_drift'
    PERFORMANCE_DEGRADATION = 'performance_degradation'
    SCHEDULED = 'scheduled'
    MANUAL = 'manual'


class RetrainingJob(BaseModel):
    """Model for a retraining job."""
    job_id: str = Field(default_factory=lambda :
        f"retrain_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )
    model_id: str
    model_version_id: str
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: RetrainingStatus = RetrainingStatus.PENDING
    trigger_type: RetrainingTriggerType
    trigger_details: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metrics_before: Dict[str, float] = Field(default_factory=dict)
    metrics_after: Dict[str, float] = Field(default_factory=dict)
    improvement: Dict[str, float] = Field(default_factory=dict)
    evaluation_result: Optional[str] = None
    new_model_version_id: Optional[str] = None


class ModelRetrainingService:
    """
    Service for managing model retraining based on drift detection
    
    This service integrates the Model Drift Detection system with automated
    retraining workflows, including optimization and deployment processes.
    """

    def __init__(self, registry_service: Optional[ModelRegistryService]=
        None, drift_detector: Optional[ModelDriftDetector]=None, optimizer:
        Optional[AutoOptimizationFramework]=None, retraining_config:
        Optional[Dict[str, Any]]=None):
        """
        Initialize the retraining service
        
        Args:
            registry_service: Model registry service instance or None to create new
            drift_detector: Model drift detector instance or None to create new
            optimizer: Auto-optimization framework instance or None to create new
            retraining_config: Configuration for retraining
        """
        self.registry_service = registry_service or ModelRegistryService()
        self.drift_detector = drift_detector or ModelDriftDetector()
        self.optimizer = optimizer or AutoOptimizationFramework()
        self.config = retraining_config or {}
        self.default_config = {'feature_drift_threshold': 0.05,
            'performance_drift_threshold': 0.1, 'retraining_cooldown_hours':
            24.0, 'auto_deploy_threshold': 0.05,
            'max_optimization_evaluations': 30, 'enable_notifications': 
            True, 'notify_urls': []}
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
        self.active_jobs = {}
        self.job_history = []
        self.jobs_dir = os.path.join('output', 'retraining_jobs')
        os.makedirs(self.jobs_dir, exist_ok=True)

    async def monitor_model_drift(self, model_id: str, version_id: str,
        feature_data: pd.DataFrame, target_data: Optional[pd.DataFrame]=
        None, reference_window: str='1w', current_window: str='1d',
        feature_importance_threshold: float=0.05) ->Dict[str, Any]:
        """
        Monitor a model for drift and trigger retraining if needed
        
        Args:
            model_id: ID of the model to monitor
            version_id: ID of the model version to monitor
            feature_data: DataFrame containing feature data
            target_data: Optional DataFrame containing target data
            reference_window: Time window for reference data (e.g., "1w" for 1 week)
            current_window: Time window for current data (e.g., "1d" for 1 day)
            feature_importance_threshold: Threshold for considering a feature important
            
        Returns:
            Dict with monitoring results and actions taken
        """
        model_metadata = await self.registry_service.get_model_by_id(model_id)
        if not model_metadata:
            error_msg = f'Model with ID {model_id} not found'
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
        version_metadata = await self.registry_service.get_model_version(
            model_id, version_id)
        if not version_metadata:
            error_msg = (
                f'Model version {version_id} not found for model {model_id}')
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
        logger.info(
            f'Monitoring drift for model {model_id}, version {version_id}')
        feature_drift_results = await self.drift_detector.detect_feature_drift(
            model_id=model_id, version_id=version_id, feature_data=
            feature_data, reference_window=reference_window, current_window
            =current_window, feature_importance_threshold=
            feature_importance_threshold)
        performance_drift_results = None
        if target_data is not None:
            performance_drift_results = (await self.drift_detector.
                detect_performance_drift(model_id=model_id, version_id=
                version_id, feature_data=feature_data, target_data=
                target_data, reference_window=reference_window,
                current_window=current_window))
        retraining_needed = False
        trigger_type = None
        trigger_details = {}
        if feature_drift_results.get('drift_detected', False):
            if feature_drift_results.get('drift_severity', 0) > self.config[
                'feature_drift_threshold']:
                retraining_needed = True
                trigger_type = RetrainingTriggerType.FEATURE_DRIFT
                trigger_details = {'feature_drift': feature_drift_results,
                    'drift_severity': feature_drift_results.get(
                    'drift_severity', 0)}
                logger.info(
                    f"Feature drift detected with severity {feature_drift_results.get('drift_severity', 0)}, exceeding threshold {self.config['feature_drift_threshold']}"
                    )
        if not retraining_needed and performance_drift_results:
            if performance_drift_results.get('drift_detected', False):
                if performance_drift_results.get('drift_severity', 0
                    ) > self.config['performance_drift_threshold']:
                    retraining_needed = True
                    trigger_type = (RetrainingTriggerType.
                        PERFORMANCE_DEGRADATION)
                    trigger_details = {'performance_drift':
                        performance_drift_results, 'drift_severity':
                        performance_drift_results.get('drift_severity', 0)}
                    logger.info(
                        f"Performance drift detected with severity {performance_drift_results.get('drift_severity', 0)}, exceeding threshold {self.config['performance_drift_threshold']}"
                        )
        results = {'success': True, 'model_id': model_id, 'version_id':
            version_id, 'feature_drift': feature_drift_results,
            'performance_drift': performance_drift_results,
            'retraining_needed': retraining_needed, 'trigger_type': 
            trigger_type.value if trigger_type else None, 'timestamp':
            datetime.now().isoformat()}
        last_retraining = await self._get_last_retraining_job(model_id)
        cooldown_hours = self.config['retraining_cooldown_hours']
        if last_retraining:
            last_time = last_retraining.start_time
            hours_since_last = (datetime.now() - last_time).total_seconds(
                ) / 3600
            if hours_since_last < cooldown_hours:
                logger.info(
                    f'Retraining needed, but in cooldown period. Hours since last retraining: {hours_since_last}, cooldown: {cooldown_hours}'
                    )
                results['cooldown_active'] = True
                retraining_needed = False
        if retraining_needed:
            job = RetrainingJob(model_id=model_id, model_version_id=
                version_id, trigger_type=trigger_type, trigger_details=
                trigger_details, parameters={'feature_data': feature_data.
                shape, 'target_data': target_data.shape if target_data is not
                None else None})
            if version_metadata.metrics:
                job.metrics_before = version_metadata.metrics
            await self._queue_retraining_job(job, feature_data, target_data)
            results['retraining_job_id'] = job.job_id
            logger.info(
                f'Retraining job {job.job_id} queued for model {model_id}, version {version_id}'
                )
            if self.config['enable_notifications']:
                await self._send_notification(
                    'Model drift detected and retraining triggered', {
                    'model_id': model_id, 'version_id': version_id,
                    'trigger_type': trigger_type.value, 'job_id': job.
                    job_id, 'drift_details': trigger_details})
        return results

    async def trigger_retraining(self, model_id: str, version_id: str,
        feature_data: pd.DataFrame, target_data: pd.DataFrame, trigger_type:
        RetrainingTriggerType=RetrainingTriggerType.MANUAL, parameters:
        Optional[Dict[str, Any]]=None) ->Dict[str, Any]:
        """
        Manually trigger model retraining
        
        Args:
            model_id: ID of the model to retrain
            version_id: ID of the model version to retrain
            feature_data: DataFrame containing feature data for training
            target_data: DataFrame containing target data for training
            trigger_type: Type of retraining trigger
            parameters: Additional parameters for retraining
            
        Returns:
            Dict with retraining job information
        """
        model_metadata = await self.registry_service.get_model_by_id(model_id)
        if not model_metadata:
            error_msg = f'Model with ID {model_id} not found'
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
        version_metadata = await self.registry_service.get_model_version(
            model_id, version_id)
        if not version_metadata:
            error_msg = (
                f'Model version {version_id} not found for model {model_id}')
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
        logger.info(
            f'Manually triggering retraining for model {model_id}, version {version_id}'
            )
        job = RetrainingJob(model_id=model_id, model_version_id=version_id,
            trigger_type=trigger_type, parameters=parameters or {})
        if version_metadata.metrics:
            job.metrics_before = version_metadata.metrics
        await self._queue_retraining_job(job, feature_data, target_data)
        return {'success': True, 'model_id': model_id, 'version_id':
            version_id, 'job_id': job.job_id, 'status': job.status.value}

    @async_with_exception_handling
    async def get_retraining_job(self, job_id: str) ->Optional[Dict[str, Any]]:
        """
        Get information about a specific retraining job
        
        Args:
            job_id: ID of the retraining job
            
        Returns:
            Dict with job information or None if not found
        """
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            return job.dict()
        for job in self.job_history:
            if job.job_id == job_id:
                return job.dict()
        job_path = os.path.join(self.jobs_dir, f'{job_id}.json')
        if os.path.exists(job_path):
            try:
                with open(job_path, 'r') as f:
                    job_data = json.load(f)
                job = RetrainingJob(**job_data)
                return job.dict()
            except Exception as e:
                logger.error(f'Error loading job file {job_path}: {str(e)}')
        return None

    async def list_retraining_jobs(self, model_id: Optional[str]=None,
        status: Optional[RetrainingStatus]=None, limit: int=100) ->List[Dict
        [str, Any]]:
        """
        List retraining jobs with optional filtering
        
        Args:
            model_id: Optional model ID to filter by
            status: Optional status to filter by
            limit: Maximum number of jobs to return
            
        Returns:
            List of job information dictionaries
        """
        all_jobs = list(self.active_jobs.values()) + self.job_history
        filtered_jobs = all_jobs
        if model_id is not None:
            filtered_jobs = [j for j in filtered_jobs if j.model_id == model_id
                ]
        if status is not None:
            filtered_jobs = [j for j in filtered_jobs if j.status == status]
        sorted_jobs = sorted(filtered_jobs, key=lambda j: j.start_time,
            reverse=True)
        limited_jobs = sorted_jobs[:limit]
        return [job.dict() for job in limited_jobs]

    async def _queue_retraining_job(self, job: RetrainingJob, feature_data:
        pd.DataFrame, target_data: pd.DataFrame) ->None:
        """
        Queue a retraining job for execution
        
        Args:
            job: RetrainingJob object
            feature_data: Feature data for training
            target_data: Target data for training
        """
        self.active_jobs[job.job_id] = job
        job_path = os.path.join(self.jobs_dir, f'{job.job_id}.json')
        with open(job_path, 'w') as f:
            json.dump(job.dict(), f, default=str, indent=2)
        asyncio.create_task(self._execute_retraining(job, feature_data,
            target_data))

    @async_with_exception_handling
    async def _execute_retraining(self, job: RetrainingJob, feature_data:
        pd.DataFrame, target_data: pd.DataFrame) ->None:
        """
        Execute a retraining job
        
        Args:
            job: RetrainingJob object
            feature_data: Feature data for training
            target_data: Target data for training
        """
        try:
            job.status = RetrainingStatus.IN_PROGRESS
            await self._update_job(job)
            model_id = job.model_id
            version_id = job.model_version_id
            model_metadata = await self.registry_service.get_model_by_id(
                model_id)
            version_metadata = await self.registry_service.get_model_version(
                model_id, version_id)
            if not version_metadata.training_func:
                raise ValueError(
                    f'No training function available for model {model_id}, version {version_id}'
                    )
            training_func = version_metadata.training_func
            param_space = version_metadata.hyperparameters or {}
            base_params = version_metadata.parameters or {}
            max_evals = self.config_manager.get('max_optimization_evaluations', 30)
            logger.info(
                f'Starting hyperparameter optimization for job {job.job_id} with {max_evals} evaluations'
                )

            def evaluate_params(params):
                """Evaluate model performance with given parameters"""
                all_params = {**base_params, **params}
                model = training_func(feature_data, target_data, all_params)
                if hasattr(model, 'evaluate'):
                    metrics = model.evaluate(feature_data, target_data)
                else:
                    from sklearn.metrics import mean_squared_error
                    predictions = model.predict(feature_data)
                    metrics = {'mse': mean_squared_error(target_data,
                        predictions)}
                return metrics
            optimization_results = await self.optimizer.optimize_async(
                evaluate_func=evaluate_params, parameter_space=param_space,
                algorithm='bayesian', objective='minimize' if
                model_metadata.minimize_metric else 'maximize',
                target_metric=model_metadata.primary_metric,
                max_evaluations=max_evals)
            best_params = optimization_results['best_parameters']
            best_score = optimization_results['best_score']
            final_params = {**base_params, **best_params}
            final_model = training_func(feature_data, target_data, final_params
                )
            if hasattr(final_model, 'evaluate'):
                final_metrics = final_model.evaluate(feature_data, target_data)
            else:
                from sklearn.metrics import mean_squared_error, r2_score
                predictions = final_model.predict(feature_data)
                final_metrics = {'mse': mean_squared_error(target_data,
                    predictions), 'r2': r2_score(target_data, predictions)}
            new_version = await self.registry_service.create_model_version(
                model_id=model_id, model=final_model, version_name=
                f"retrained_v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                description=
                f'Automatically retrained from version {version_id} due to {job.trigger_type.value}'
                , metrics=final_metrics, parameters=final_params,
                hyperparameters=param_space, training_func=training_func,
                parent_version_id=version_id)
            job.end_time = datetime.now()
            job.status = RetrainingStatus.COMPLETED
            job.new_model_version_id = new_version.version_id
            job.metrics_after = final_metrics
            improvement = {}
            primary_metric = model_metadata.primary_metric
            for metric, value in final_metrics.items():
                if metric in job.metrics_before:
                    if model_metadata.minimize_metric:
                        imp = (job.metrics_before[metric] - value
                            ) / job.metrics_before[metric]
                    else:
                        imp = (value - job.metrics_before[metric]
                            ) / job.metrics_before[metric]
                    improvement[metric] = imp * 100
            job.improvement = improvement
            auto_deploy = False
            if primary_metric in improvement:
                if improvement[primary_metric] > self.config[
                    'auto_deploy_threshold'] * 100:
                    auto_deploy = True
            if auto_deploy:
                await self.registry_service.transition_model_version(model_id
                    =model_id, version_id=new_version.version_id, stage=
                    ModelStatus.PRODUCTION)
                await self.registry_service.transition_model_version(model_id
                    =model_id, version_id=version_id, stage=ModelStatus.
                    ARCHIVED)
                job.status = RetrainingStatus.DEPLOYED
                logger.info(
                    f'Auto-deployed new model version {new_version.version_id} with {primary_metric} improvement of {improvement.get(primary_metric, 0):.2f}%'
                    )
            await self._update_job(job)
            if self.config['enable_notifications']:
                notification_data = {'job_id': job.job_id, 'model_id':
                    model_id, 'old_version_id': version_id,
                    'new_version_id': new_version.version_id, 'improvement':
                    improvement, 'auto_deployed': auto_deploy}
                await self._send_notification('Model retraining completed',
                    notification_data)
        except Exception as e:
            logger.error(
                f'Error executing retraining job {job.job_id}: {str(e)}')
            logger.error(traceback.format_exc())
            job.status = RetrainingStatus.FAILED
            job.end_time = datetime.now()
            job.parameters['error'] = str(e)
            await self._update_job(job)
            if self.config['enable_notifications']:
                await self._send_notification('Model retraining failed', {
                    'job_id': job.job_id, 'model_id': job.model_id,
                    'version_id': job.model_version_id, 'error': str(e)})
        if job.job_id in self.active_jobs:
            del self.active_jobs[job.job_id]
            self.job_history.append(job)

    async def _update_job(self, job: RetrainingJob) ->None:
        """
        Update and persist job information
        
        Args:
            job: RetrainingJob object to update
        """
        self.active_jobs[job.job_id] = job
        job_path = os.path.join(self.jobs_dir, f'{job.job_id}.json')
        with open(job_path, 'w') as f:
            json.dump(job.dict(), f, default=str, indent=2)

    async def _get_last_retraining_job(self, model_id: str) ->Optional[
        RetrainingJob]:
        """
        Get the most recent retraining job for a model
        
        Args:
            model_id: Model ID to check
            
        Returns:
            Most recent RetrainingJob or None if none found
        """
        model_jobs = []
        for job in self.active_jobs.values():
            if job.model_id == model_id:
                model_jobs.append(job)
        for job in self.job_history:
            if job.model_id == model_id:
                model_jobs.append(job)
        if not model_jobs:
            return None
        sorted_jobs = sorted(model_jobs, key=lambda j: j.start_time,
            reverse=True)
        return sorted_jobs[0] if sorted_jobs else None

    @async_with_exception_handling
    async def _send_notification(self, title: str, data: Dict[str, Any]
        ) ->None:
        """
        Send a notification about retraining events
        
        Args:
            title: Notification title
            data: Notification data
        """
        if not self.config['enable_notifications']:
            return
        payload = {'title': title, 'timestamp': datetime.now().isoformat(),
            'data': data}
        for url in self.config['notify_urls']:
            try:
                requests.post(url, json=payload, timeout=5)
            except Exception as e:
                logger.error(f'Error sending notification to {url}: {str(e)}')
