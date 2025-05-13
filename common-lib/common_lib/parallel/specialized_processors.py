"""
Specialized Parallel Processors Module

This module provides specialized parallel processors for different use cases in the forex trading platform.
It includes processors for multi-instrument processing, multi-timeframe processing, and batch feature processing.

Features:
- Multi-instrument processing
- Multi-timeframe processing
- Batch feature processing
- Optimized for specific use cases
- Comprehensive error handling and reporting
"""

import asyncio
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Tuple, TypeVar, Union, cast

import pandas as pd

from common_lib.errors.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    BaseError,
    ServiceError
)

from common_lib.monitoring.performance_monitoring import (
    track_operation
)

from common_lib.parallel.parallel_processor import (
    ParallelProcessor,
    ParallelizationMethod,
    TaskDefinition,
    TaskPriority,
    TaskResult
)

# Type variables
T = TypeVar('T')  # Input type
R = TypeVar('R')  # Result type
D = TypeVar('D')  # Data type

# Create logger
logger = logging.getLogger(__name__)


class TimeframeHierarchy:
    """
    Manages timeframe hierarchies for optimized multi-timeframe processing.
    
    This class provides utilities for working with timeframe hierarchies,
    including determining parent-child relationships and conversion factors.
    """
    
    # Standard timeframe hierarchy
    STANDARD_HIERARCHY = {
        "1m": {"parent": None, "factor": 1},
        "5m": {"parent": "1m", "factor": 5},
        "15m": {"parent": "5m", "factor": 3},
        "30m": {"parent": "15m", "factor": 2},
        "1h": {"parent": "30m", "factor": 2},
        "4h": {"parent": "1h", "factor": 4},
        "1d": {"parent": "4h", "factor": 6},
        "1w": {"parent": "1d", "factor": 7},
        "1M": {"parent": "1w", "factor": 4}
    }
    
    def __init__(self, hierarchy: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize the timeframe hierarchy.
        
        Args:
            hierarchy: Custom timeframe hierarchy
        """
        self.hierarchy = hierarchy or self.STANDARD_HIERARCHY
    
    def get_parent(self, timeframe: str) -> Optional[str]:
        """
        Get the parent timeframe.
        
        Args:
            timeframe: Timeframe
            
        Returns:
            Parent timeframe or None if no parent
        """
        if timeframe not in self.hierarchy:
            return None
        
        return self.hierarchy[timeframe]["parent"]
    
    def get_children(self, timeframe: str) -> List[str]:
        """
        Get the child timeframes.
        
        Args:
            timeframe: Timeframe
            
        Returns:
            List of child timeframes
        """
        return [tf for tf, info in self.hierarchy.items() if info["parent"] == timeframe]
    
    def get_conversion_factor(self, from_timeframe: str, to_timeframe: str) -> Optional[int]:
        """
        Get the conversion factor between timeframes.
        
        Args:
            from_timeframe: Source timeframe
            to_timeframe: Target timeframe
            
        Returns:
            Conversion factor or None if conversion is not possible
        """
        if from_timeframe == to_timeframe:
            return 1
        
        # Check if from_timeframe is an ancestor of to_timeframe
        current = to_timeframe
        factor = 1
        
        while current != from_timeframe:
            parent = self.get_parent(current)
            
            if parent is None:
                # Check if to_timeframe is an ancestor of from_timeframe
                current = from_timeframe
                reverse_factor = 1
                
                while current != to_timeframe:
                    parent = self.get_parent(current)
                    
                    if parent is None:
                        return None
                    
                    reverse_factor *= self.hierarchy[current]["factor"]
                    current = parent
                
                return 1 / reverse_factor
            
            factor *= self.hierarchy[current]["factor"]
            current = parent
        
        return factor
    
    def sort_timeframes(self, timeframes: List[str], descending: bool = True) -> List[str]:
        """
        Sort timeframes by size.
        
        Args:
            timeframes: List of timeframes
            descending: Whether to sort in descending order
            
        Returns:
            Sorted list of timeframes
        """
        # Filter valid timeframes
        valid_timeframes = [tf for tf in timeframes if tf in self.hierarchy]
        
        # Create a mapping of timeframes to their absolute size
        sizes = {}
        for tf in valid_timeframes:
            size = 1
            current = tf
            
            while self.get_parent(current) is not None:
                parent = self.get_parent(current)
                size *= self.hierarchy[current]["factor"]
                current = parent
            
            sizes[tf] = size
        
        # Sort timeframes by size
        return sorted(valid_timeframes, key=lambda tf: sizes[tf], reverse=descending)


class FeatureSpec:
    """
    Specification for a feature to be calculated.
    
    This class provides a standardized way to specify features for batch processing,
    including feature name, parameters, and dependencies.
    """
    
    def __init__(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
        priority: TaskPriority = TaskPriority.MEDIUM
    ):
        """
        Initialize the feature specification.
        
        Args:
            name: Feature name
            parameters: Feature parameters
            dependencies: Feature dependencies
            priority: Feature calculation priority
        """
        self.name = name
        self.parameters = parameters or {}
        self.dependencies = dependencies or []
        self.priority = priority
    
    def __str__(self) -> str:
        """String representation."""
        return f"FeatureSpec(name={self.name}, parameters={self.parameters})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"FeatureSpec(name={self.name}, parameters={self.parameters}, dependencies={self.dependencies}, priority={self.priority})"


class MultiInstrumentProcessor:
    """
    Processor for parallel operations across multiple instruments.
    
    This processor optimizes parallel processing for operations that need to be
    performed on multiple instruments, such as data retrieval, feature calculation,
    and signal generation.
    """
    
    def __init__(self, parallel_processor: Optional[ParallelProcessor] = None):
        """
        Initialize the multi-instrument processor.
        
        Args:
            parallel_processor: Parallel processor to use
        """
        self.parallel_processor = parallel_processor or ParallelProcessor()
    
    @track_operation("parallel", "process_instruments")
    async def process_instruments(
        self,
        instruments: List[str],
        process_func: Callable[[str], T],
        priority: TaskPriority = TaskPriority.MEDIUM,
        parallelization_method: ParallelizationMethod = ParallelizationMethod.AUTO,
        timeout: Optional[float] = None,
        batch_size: Optional[int] = None
    ) -> Dict[str, T]:
        """
        Process multiple instruments in parallel.
        
        Args:
            instruments: List of instrument symbols to process
            process_func: Function to process each instrument
            priority: Priority for the processing tasks
            parallelization_method: Method for parallelization
            timeout: Optional timeout in seconds
            batch_size: Optional batch size for processing
            
        Returns:
            Dictionary mapping instrument symbols to processing results
        """
        if not instruments:
            return {}
        
        # If batch size is specified, process in batches
        if batch_size and len(instruments) > batch_size:
            return await self._process_in_batches(
                instruments=instruments,
                process_func=process_func,
                priority=priority,
                parallelization_method=parallelization_method,
                timeout=timeout,
                batch_size=batch_size
            )
        
        # Create task definitions
        tasks = []
        
        for instrument in instruments:
            task_id = f"instrument_{instrument}_{uuid.uuid4().hex[:8]}"
            
            tasks.append(
                TaskDefinition(
                    id=task_id,
                    func=process_func,
                    input_data=instrument,
                    priority=priority,
                    parallelization_method=parallelization_method,
                    timeout=timeout
                )
            )
        
        # Execute tasks
        results = await self.parallel_processor.execute_tasks(tasks)
        
        # Process results
        processed_results = {}
        errors = []
        
        for task_id, task_result in results.items():
            instrument = task_id.split("_")[1]
            
            if task_result.success:
                processed_results[instrument] = task_result.result
            else:
                error_msg = str(task_result.error) if task_result.error else "Unknown error"
                logger.error(f"Error processing instrument {instrument}: {error_msg}")
                errors.append((instrument, error_msg))
        
        # Log errors
        if errors:
            logger.warning(f"Failed to process {len(errors)} instruments: {', '.join(i for i, _ in errors)}")
        
        return processed_results
    
    @track_operation("parallel", "process_in_batches")
    async def _process_in_batches(
        self,
        instruments: List[str],
        process_func: Callable[[str], T],
        priority: TaskPriority,
        parallelization_method: ParallelizationMethod,
        timeout: Optional[float],
        batch_size: int
    ) -> Dict[str, T]:
        """
        Process instruments in batches.
        
        Args:
            instruments: List of instrument symbols to process
            process_func: Function to process each instrument
            priority: Priority for the processing tasks
            parallelization_method: Method for parallelization
            timeout: Optional timeout in seconds
            batch_size: Batch size for processing
            
        Returns:
            Dictionary mapping instrument symbols to processing results
        """
        all_results = {}
        
        # Process in batches
        for i in range(0, len(instruments), batch_size):
            batch = instruments[i:i+batch_size]
            
            logger.debug(
                f"Processing batch {i//batch_size + 1}/{(len(instruments) + batch_size - 1)//batch_size} "
                f"with {len(batch)} instruments"
            )
            
            batch_results = await self.process_instruments(
                instruments=batch,
                process_func=process_func,
                priority=priority,
                parallelization_method=parallelization_method,
                timeout=timeout
            )
            
            all_results.update(batch_results)
        
        return all_results
    
    @track_operation("parallel", "process_instrument_data")
    async def process_instrument_data(
        self,
        instrument_data: Dict[str, D],
        process_func: Callable[[str, D], T],
        priority: TaskPriority = TaskPriority.MEDIUM,
        parallelization_method: ParallelizationMethod = ParallelizationMethod.AUTO,
        timeout: Optional[float] = None,
        batch_size: Optional[int] = None
    ) -> Dict[str, T]:
        """
        Process data for multiple instruments in parallel.
        
        Args:
            instrument_data: Dictionary mapping instrument symbols to data
            process_func: Function to process each instrument's data
            priority: Priority for the processing tasks
            parallelization_method: Method for parallelization
            timeout: Optional timeout in seconds
            batch_size: Optional batch size for processing
            
        Returns:
            Dictionary mapping instrument symbols to processing results
        """
        if not instrument_data:
            return {}
        
        # If batch size is specified, process in batches
        if batch_size and len(instrument_data) > batch_size:
            return await self._process_data_in_batches(
                instrument_data=instrument_data,
                process_func=process_func,
                priority=priority,
                parallelization_method=parallelization_method,
                timeout=timeout,
                batch_size=batch_size
            )
        
        # Create task definitions
        tasks = []
        
        for instrument, data in instrument_data.items():
            task_id = f"instrument_data_{instrument}_{uuid.uuid4().hex[:8]}"
            
            # Create wrapper function
            def create_wrapper(i, d):
    """
    Create wrapper.
    
    Args:
        i: Description of i
        d: Description of d
    
    """

                return lambda _: process_func(i, d)
            
            wrapper_func = create_wrapper(instrument, data)
            
            tasks.append(
                TaskDefinition(
                    id=task_id,
                    func=wrapper_func,
                    input_data=None,
                    priority=priority,
                    parallelization_method=parallelization_method,
                    timeout=timeout
                )
            )
        
        # Execute tasks
        results = await self.parallel_processor.execute_tasks(tasks)
        
        # Process results
        processed_results = {}
        errors = []
        
        for task_id, task_result in results.items():
            instrument = task_id.split("_")[2]
            
            if task_result.success:
                processed_results[instrument] = task_result.result
            else:
                error_msg = str(task_result.error) if task_result.error else "Unknown error"
                logger.error(f"Error processing data for instrument {instrument}: {error_msg}")
                errors.append((instrument, error_msg))
        
        # Log errors
        if errors:
            logger.warning(f"Failed to process data for {len(errors)} instruments: {', '.join(i for i, _ in errors)}")
        
        return processed_results
    
    @track_operation("parallel", "process_data_in_batches")
    async def _process_data_in_batches(
        self,
        instrument_data: Dict[str, D],
        process_func: Callable[[str, D], T],
        priority: TaskPriority,
        parallelization_method: ParallelizationMethod,
        timeout: Optional[float],
        batch_size: int
    ) -> Dict[str, T]:
        """
        Process instrument data in batches.
        
        Args:
            instrument_data: Dictionary mapping instrument symbols to data
            process_func: Function to process each instrument's data
            priority: Priority for the processing tasks
            parallelization_method: Method for parallelization
            timeout: Optional timeout in seconds
            batch_size: Batch size for processing
            
        Returns:
            Dictionary mapping instrument symbols to processing results
        """
        all_results = {}
        instruments = list(instrument_data.keys())
        
        # Process in batches
        for i in range(0, len(instruments), batch_size):
            batch_instruments = instruments[i:i+batch_size]
            batch_data = {k: instrument_data[k] for k in batch_instruments}
            
            logger.debug(
                f"Processing data batch {i//batch_size + 1}/{(len(instruments) + batch_size - 1)//batch_size} "
                f"with {len(batch_data)} instruments"
            )
            
            batch_results = await self.process_instrument_data(
                instrument_data=batch_data,
                process_func=process_func,
                priority=priority,
                parallelization_method=parallelization_method,
                timeout=timeout
            )
            
            all_results.update(batch_results)
        
        return all_results


# Create singleton instances
_default_multi_instrument_processor = MultiInstrumentProcessor()


def get_multi_instrument_processor() -> MultiInstrumentProcessor:
    """
    Get the default multi-instrument processor.
    
    Returns:
        Default multi-instrument processor
    """
    return _default_multi_instrument_processor
