"""
Parallel Inference Module for ML Integration Service.

This module provides specialized parallel processing capabilities for
ML model inference, optimizing prediction across multiple models,
instruments, and timeframes.

Features:
- Parallel inference for multiple models
- Efficient batch processing of prediction requests
- Optimized memory usage for large feature sets
- Model-specific parallelization strategies
- Comprehensive error handling and reporting
"""
import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union
import numpy as np
import pandas as pd
from common_lib.exceptions import ModelPredictionError
from common_lib.parallel import ParallelProcessor, ParallelizationMethod, ResourceManager, TaskDefinition, TaskPriority, TaskResult, get_parallel_processor
logger = logging.getLogger(__name__)
T = TypeVar('T')


from ml_integration_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class ModelInferenceSpec:
    """
    Specification for a model inference request.
    
    This class provides a standardized way to specify model inference requests,
    including model ID, parameters, and inference method.
    """

    def __init__(self, model_id: str, version: Optional[str]=None, params:
        Dict[str, Any]=None, priority: TaskPriority=TaskPriority.MEDIUM,
        parallelization_method: ParallelizationMethod=ParallelizationMethod
        .AUTO):
        """
        Initialize a model inference specification.
        
        Args:
            model_id: Model identifier
            version: Optional model version
            params: Inference parameters
            priority: Priority for inference
            parallelization_method: Method for parallelization
        """
        self.model_id = model_id
        self.version = version
        self.params = params or {}
        self.priority = priority
        self.parallelization_method = parallelization_method

    def __str__(self) ->str:
        """String representation of the model inference spec."""
        version_str = f':{self.version}' if self.version else ''
        param_str = ', '.join([f'{k}={v}' for k, v in self.params.items()])
        return f'{self.model_id}{version_str}({param_str})'

    def to_dict(self) ->Dict[str, Any]:
        """Convert to dictionary representation."""
        return {'model_id': self.model_id, 'version': self.version,
            'params': self.params, 'priority': self.priority.value,
            'parallelization_method': self.parallelization_method.value}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) ->'ModelInferenceSpec':
        """Create from dictionary representation."""
        return cls(model_id=data['model_id'], version=data.get('version'),
            params=data.get('params', {}), priority=TaskPriority(data.get(
            'priority', TaskPriority.MEDIUM.value)), parallelization_method
            =ParallelizationMethod(data.get('parallelization_method',
            ParallelizationMethod.AUTO.value)))


class ParallelInferenceProcessor:
    """
    Processor for parallel ML model inference.
    
    This class provides optimized parallel processing for ML model inference
    operations that need to be performed with multiple models, instruments,
    or timeframes.
    """

    def __init__(self, resource_manager: Optional[ResourceManager]=None):
        """
        Initialize the parallel inference processor.
        
        Args:
            resource_manager: Optional resource manager for worker allocation
        """
        self.resource_manager = resource_manager or ResourceManager()
        self.executor = ParallelExecutor(resource_manager=self.resource_manager
            )

    async def run_inference(self, features: pd.DataFrame, models: List[
        ModelInferenceSpec], inference_func: Callable[[pd.DataFrame,
        ModelInferenceSpec], Any], timeout: Optional[float]=None) ->Dict[
        str, Any]:
        """
        Run inference for multiple models on a single feature set in parallel.
        
        Args:
            features: Input features
            models: List of model specifications
            inference_func: Function to run inference for each model
            timeout: Optional timeout in seconds
            
        Returns:
            Dictionary mapping model IDs to inference results
            
        Raises:
            ModelPredictionError: If inference fails
        """
        if features.empty or not models:
            return {}
        tasks = []
        for model in models:
            task_id = f'model_{model.model_id}_{uuid.uuid4().hex[:8]}'

            def create_wrapper(f, m):
    """
    Create wrapper.
    
    Args:
        f: Description of f
        m: Description of m
    
    """

                return lambda _: inference_func(f, m)
            wrapper_func = create_wrapper(features, model)
            tasks.append(TaskDefinition(id=task_id, func=wrapper_func,
                input_data=None, priority=model.priority,
                parallelization_method=model.parallelization_method,
                timeout=timeout))
        results = await self.executor.execute_tasks(tasks)
        processed_results = {}
        errors = []
        for task_id, task_result in results.items():
            model_id = task_id.split('_')[1]
            if task_result.success:
                processed_results[model_id] = task_result.result
            else:
                error_msg = str(task_result.error
                    ) if task_result.error else 'Unknown error'
                logger.error(
                    f'Error running inference for model {model_id}: {error_msg}'
                    )
                errors.append((model_id, error_msg))
        if errors and len(errors) == len(models):
            error_details = '\n'.join([f'{m}: {e}' for m, e in errors])
            raise ModelPredictionError(message=
                f'Inference failed for all models:\n{error_details}')
        return processed_results

    @async_with_exception_handling
    async def run_inference_for_instruments(self, instrument_features: Dict
        [str, pd.DataFrame], models: List[ModelInferenceSpec],
        inference_func: Callable[[pd.DataFrame, ModelInferenceSpec], Any],
        timeout: Optional[float]=None, batch_size: Optional[int]=None) ->Dict[
        str, Dict[str, Any]]:
        """
        Run inference for multiple instruments in parallel.
        
        Args:
            instrument_features: Dictionary mapping instrument symbols to feature DataFrames
            models: List of model specifications
            inference_func: Function to run inference for each model
            timeout: Optional timeout in seconds
            batch_size: Optional batch size for processing instruments
            
        Returns:
            Dictionary mapping instruments to dictionaries mapping model IDs to results
            
        Raises:
            ModelPredictionError: If inference fails for all instruments
        """
        if not instrument_features or not models:
            return {}
        if batch_size and len(instrument_features) > batch_size:
            all_results = {}
            instruments = list(instrument_features.keys())
            for i in range(0, len(instruments), batch_size):
                batch_instruments = instruments[i:i + batch_size]
                batch_features = {k: instrument_features[k] for k in
                    batch_instruments}
                logger.debug(
                    f'Processing instrument batch {i // batch_size + 1}/{(len(instruments) + batch_size - 1) // batch_size} with {len(batch_features)} instruments'
                    )
                batch_results = await self.run_inference_for_instruments(
                    instrument_features=batch_features, models=models,
                    inference_func=inference_func, timeout=timeout)
                all_results.update(batch_results)
            return all_results
        results = {}
        errors = []
        for instrument, features in instrument_features.items():
            try:
                instrument_results = await self.run_inference(features=
                    features, models=models, inference_func=inference_func,
                    timeout=timeout)
                results[instrument] = instrument_results
            except Exception as e:
                logger.error(
                    f'Error running inference for instrument {instrument}: {str(e)}'
                    )
                errors.append((instrument, str(e)))
        if errors and len(errors) == len(instrument_features):
            error_details = '\n'.join([f'{i}: {e}' for i, e in errors])
            raise ModelPredictionError(message=
                f"""Inference failed for all instruments:
{error_details}""")
        return results

    @async_with_exception_handling
    async def run_batch_inference(self, features_batch: List[pd.DataFrame],
        model: ModelInferenceSpec, inference_func: Callable[[List[pd.
        DataFrame], ModelInferenceSpec], List[Any]], timeout: Optional[
        float]=None) ->List[Any]:
        """
        Run batch inference for a single model on multiple feature sets.
        
        Args:
            features_batch: List of feature DataFrames
            model: Model specification
            inference_func: Function to run batch inference
            timeout: Optional timeout in seconds
            
        Returns:
            List of inference results
            
        Raises:
            ModelPredictionError: If batch inference fails
        """
        if not features_batch:
            return []
        try:
            start_time = datetime.now()
            results = inference_func(features_batch, model)
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.debug(
                f'Batch inference for model {model.model_id} completed in {execution_time:.2f}s'
                )
            return results
        except Exception as e:
            error_msg = str(e)
            logger.error(
                f'Error running batch inference for model {model.model_id}: {error_msg}'
                )
            raise ModelPredictionError(message=
                f'Batch inference failed for model {model.model_id}: {error_msg}'
                , model_id=model.model_id)

    async def run_multi_model_batch_inference(self, features_batch: List[pd
        .DataFrame], models: List[ModelInferenceSpec], inference_func:
        Callable[[List[pd.DataFrame], ModelInferenceSpec], List[Any]],
        timeout: Optional[float]=None) ->Dict[str, List[Any]]:
        """
        Run batch inference for multiple models on multiple feature sets.
        
        Args:
            features_batch: List of feature DataFrames
            models: List of model specifications
            inference_func: Function to run batch inference
            timeout: Optional timeout in seconds
            
        Returns:
            Dictionary mapping model IDs to lists of inference results
            
        Raises:
            ModelPredictionError: If batch inference fails for all models
        """
        if not features_batch or not models:
            return {}
        tasks = []
        for model in models:
            task_id = f'batch_model_{model.model_id}_{uuid.uuid4().hex[:8]}'

            def create_wrapper(fb, m):
    """
    Create wrapper.
    
    Args:
        fb: Description of fb
        m: Description of m
    
    """

                return lambda _: inference_func(fb, m)
            wrapper_func = create_wrapper(features_batch, model)
            tasks.append(TaskDefinition(id=task_id, func=wrapper_func,
                input_data=None, priority=model.priority,
                parallelization_method=model.parallelization_method,
                timeout=timeout))
        results = await self.executor.execute_tasks(tasks)
        processed_results = {}
        errors = []
        for task_id, task_result in results.items():
            model_id = task_id.split('_')[2]
            if task_result.success:
                processed_results[model_id] = task_result.result
            else:
                error_msg = str(task_result.error
                    ) if task_result.error else 'Unknown error'
                logger.error(
                    f'Error running batch inference for model {model_id}: {error_msg}'
                    )
                errors.append((model_id, error_msg))
        if errors and len(errors) == len(models):
            error_details = '\n'.join([f'{m}: {e}' for m, e in errors])
            raise ModelPredictionError(message=
                f"""Batch inference failed for all models:
{error_details}""")
        return processed_results
