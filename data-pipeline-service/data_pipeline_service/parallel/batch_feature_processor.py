"""
Batch Feature Processor for Data Pipeline Service.

This module provides specialized parallel processing capabilities for
batch feature engineering, optimizing computation of features across
multiple instruments, timeframes, and feature types.

Features:
- Parallel feature computation for multiple instruments and timeframes
- Efficient batch processing of feature data
- Optimized memory usage for large feature sets
- Dependency-aware feature calculation
- Comprehensive error handling and reporting
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union

import pandas as pd
from common_lib.exceptions import DataProcessingError

from data_pipeline_service.parallel.parallel_processing_framework import (
    ParallelExecutor,
    ParallelizationMethod,
    ResourceManager,
    TaskDefinition,
    TaskPriority,
    TaskResult,
)

logger = logging.getLogger(__name__)

# Type variables for generic typing
T = TypeVar('T')  # Result type for feature processing


class FeatureSpec:
    """
    Specification for a feature to be calculated.
    
    This class provides a standardized way to specify features for calculation,
    including parameters, dependencies, and calculation methods.
    """
    
    def __init__(self,
                 name: str,
                 params: Dict[str, Any] = None,
                 dependencies: Optional[List[str]] = None,
                 priority: TaskPriority = TaskPriority.MEDIUM,
                 parallelization_method: ParallelizationMethod = ParallelizationMethod.AUTO):
        """
        Initialize a feature specification.
        
        Args:
            name: Feature name
            params: Feature parameters
            dependencies: Names of features this feature depends on
            priority: Priority for feature calculation
            parallelization_method: Method for parallelization
        """
        self.name = name
        self.params = params or {}
        self.dependencies = dependencies or []
        self.priority = priority
        self.parallelization_method = parallelization_method
    
    def __str__(self) -> str:
        """String representation of the feature spec."""
        param_str = ", ".join([f"{k}={v}" for k, v in self.params.items()])
        return f"{self.name}({param_str})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "params": self.params,
            "dependencies": self.dependencies,
            "priority": self.priority.value,
            "parallelization_method": self.parallelization_method.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureSpec':
        """Create from dictionary representation."""
        return cls(
            name=data["name"],
            params=data.get("params", {}),
            dependencies=data.get("dependencies", []),
            priority=TaskPriority(data.get("priority", TaskPriority.MEDIUM.value)),
            parallelization_method=ParallelizationMethod(
                data.get("parallelization_method", ParallelizationMethod.AUTO.value)
            )
        )


class BatchFeatureProcessor:
    """
    Processor for parallel batch feature engineering.
    
    This class provides optimized parallel processing for feature engineering
    operations that need to be performed on multiple instruments, timeframes,
    and feature types.
    """
    
    def __init__(self, resource_manager: Optional[ResourceManager] = None):
        """
        Initialize the batch feature processor.
        
        Args:
            resource_manager: Optional resource manager for worker allocation
        """
        self.resource_manager = resource_manager or ResourceManager()
        self.executor = ParallelExecutor(resource_manager=self.resource_manager)
    
    async def calculate_features(self,
                          data: pd.DataFrame,
                          features: List[FeatureSpec],
                          calculate_func: Callable[[pd.DataFrame, FeatureSpec], pd.Series],
                          timeout: Optional[float] = None) -> Dict[str, pd.Series]:
        """
        Calculate multiple features for a DataFrame in parallel.
        
        Args:
            data: Input DataFrame
            features: List of feature specifications
            calculate_func: Function to calculate each feature
            timeout: Optional timeout in seconds
            
        Returns:
            Dictionary mapping feature names to calculated Series
            
        Raises:
            DataProcessingError: If feature calculation fails
        """
        if data.empty or not features:
            return {}
        
        # Create dependency map
        dependency_map = {feature.name: set(feature.dependencies) for feature in features}
        
        # Create task definitions
        tasks = []
        for feature in features:
            task_id = f"feature_{feature.name}_{uuid.uuid4().hex[:8]}"
            
            # Create a wrapper function to pass both data and feature spec
            def create_wrapper(d, f):
                return lambda _: calculate_func(d, f)
            
            wrapper_func = create_wrapper(data, feature)
            
            tasks.append(TaskDefinition(
                id=task_id,
                func=wrapper_func,
                input_data=None,  # Not used by the wrapper
                priority=feature.priority,
                dependencies=set(feature.dependencies),
                parallelization_method=feature.parallelization_method,
                timeout=timeout
            ))
        
        # Execute tasks
        results = await self.executor.execute_tasks(tasks)
        
        # Process results
        processed_results = {}
        errors = []
        
        for task_id, task_result in results.items():
            # Extract feature name from task ID
            feature_name = task_id.split('_')[1]
            
            if task_result.success:
                processed_results[feature_name] = task_result.result
            else:
                error_msg = str(task_result.error) if task_result.error else "Unknown error"
                logger.error(f"Error calculating feature {feature_name}: {error_msg}")
                errors.append((feature_name, error_msg))
        
        # If all features failed, raise an error
        if errors and len(errors) == len(features):
            error_details = "\n".join([f"{f}: {e}" for f, e in errors])
            raise DataProcessingError(
                message=f"Feature calculation failed for all features:\n{error_details}"
            )
        
        return processed_results
    
    async def calculate_features_for_instruments(self,
                                         instrument_data: Dict[str, pd.DataFrame],
                                         features: List[FeatureSpec],
                                         calculate_func: Callable[[pd.DataFrame, FeatureSpec], pd.Series],
                                         timeout: Optional[float] = None,
                                         batch_size: Optional[int] = None) -> Dict[str, Dict[str, pd.Series]]:
        """
        Calculate features for multiple instruments in parallel.
        
        Args:
            instrument_data: Dictionary mapping instrument symbols to DataFrames
            features: List of feature specifications
            calculate_func: Function to calculate each feature
            timeout: Optional timeout in seconds
            batch_size: Optional batch size for processing instruments
            
        Returns:
            Dictionary mapping instruments to dictionaries mapping feature names to Series
            
        Raises:
            DataProcessingError: If feature calculation fails for all instruments
        """
        if not instrument_data or not features:
            return {}
        
        # Process instruments in batches if specified
        if batch_size and len(instrument_data) > batch_size:
            all_results = {}
            instruments = list(instrument_data.keys())
            
            for i in range(0, len(instruments), batch_size):
                batch_instruments = instruments[i:i+batch_size]
                batch_data = {k: instrument_data[k] for k in batch_instruments}
                
                logger.debug(f"Processing instrument batch {i//batch_size + 1}/{(len(instruments) + batch_size - 1)//batch_size} "
                            f"with {len(batch_data)} instruments")
                
                batch_results = await self.calculate_features_for_instruments(
                    instrument_data=batch_data,
                    features=features,
                    calculate_func=calculate_func,
                    timeout=timeout
                )
                
                all_results.update(batch_results)
                
            return all_results
        
        # Process each instrument's features
        results = {}
        errors = []
        
        for instrument, data in instrument_data.items():
            try:
                instrument_results = await self.calculate_features(
                    data=data,
                    features=features,
                    calculate_func=calculate_func,
                    timeout=timeout
                )
                
                results[instrument] = instrument_results
            except Exception as e:
                logger.error(f"Error calculating features for instrument {instrument}: {str(e)}")
                errors.append((instrument, str(e)))
        
        # If all instruments failed, raise an error
        if errors and len(errors) == len(instrument_data):
            error_details = "\n".join([f"{i}: {e}" for i, e in errors])
            raise DataProcessingError(
                message=f"Feature calculation failed for all instruments:\n{error_details}"
            )
        
        return results
    
    async def calculate_features_for_timeframes(self,
                                        instrument: str,
                                        timeframe_data: Dict[str, pd.DataFrame],
                                        features: List[FeatureSpec],
                                        calculate_func: Callable[[pd.DataFrame, FeatureSpec], pd.Series],
                                        timeout: Optional[float] = None) -> Dict[str, Dict[str, pd.Series]]:
        """
        Calculate features for multiple timeframes of an instrument in parallel.
        
        Args:
            instrument: Instrument symbol
            timeframe_data: Dictionary mapping timeframes to DataFrames
            features: List of feature specifications
            calculate_func: Function to calculate each feature
            timeout: Optional timeout in seconds
            
        Returns:
            Dictionary mapping timeframes to dictionaries mapping feature names to Series
            
        Raises:
            DataProcessingError: If feature calculation fails for all timeframes
        """
        if not timeframe_data or not features:
            return {}
        
        # Process each timeframe's features
        results = {}
        errors = []
        
        for timeframe, data in timeframe_data.items():
            try:
                timeframe_results = await self.calculate_features(
                    data=data,
                    features=features,
                    calculate_func=calculate_func,
                    timeout=timeout
                )
                
                results[timeframe] = timeframe_results
            except Exception as e:
                logger.error(f"Error calculating features for timeframe {timeframe}: {str(e)}")
                errors.append((timeframe, str(e)))
        
        # If all timeframes failed, raise an error
        if errors and len(errors) == len(timeframe_data):
            error_details = "\n".join([f"{tf}: {e}" for tf, e in errors])
            raise DataProcessingError(
                message=f"Feature calculation failed for all timeframes of {instrument}:\n{error_details}"
            )
        
        return results
