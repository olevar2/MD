"""
Multi-Timeframe Processor for Data Pipeline Service.

This module provides specialized parallel processing capabilities for
handling multiple timeframes simultaneously, optimizing data retrieval,
processing, and analysis across different timeframes.

Features:
- Parallel data retrieval for multiple timeframes
- Efficient batch processing of timeframe data
- Optimized memory usage for large timeframe sets
- Hierarchical timeframe processing (higher timeframes first)
- Comprehensive error handling and reporting
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
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
T = TypeVar('T')  # Result type for timeframe processing


class TimeframeHierarchy:
    """
    Manages timeframe hierarchies and relationships.
    
    This class provides utilities for working with timeframe hierarchies,
    including determining parent-child relationships, aggregation, and
    dependency ordering.
    """
    
    # Standard timeframe hierarchy (from smallest to largest)
    STANDARD_TIMEFRAMES = [
        "1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"
    ]
    
    # Mapping of timeframe to minutes
    TIMEFRAME_MINUTES = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,
        "1d": 1440,
        "1w": 10080,
        "1M": 43200  # Approximation
    }
    
    @classmethod
    def get_parent_timeframes(cls, timeframe: str) -> List[str]:
        """
        Get all parent timeframes for a given timeframe.
        
        Args:
            timeframe: The timeframe to get parents for
            
        Returns:
            List of parent timeframes (larger timeframes)
        """
        if timeframe not in cls.STANDARD_TIMEFRAMES:
            logger.warning(f"Unknown timeframe: {timeframe}")
            return []
        
        idx = cls.STANDARD_TIMEFRAMES.index(timeframe)
        return cls.STANDARD_TIMEFRAMES[idx+1:]
    
    @classmethod
    def get_child_timeframes(cls, timeframe: str) -> List[str]:
        """
        Get all child timeframes for a given timeframe.
        
        Args:
            timeframe: The timeframe to get children for
            
        Returns:
            List of child timeframes (smaller timeframes)
        """
        if timeframe not in cls.STANDARD_TIMEFRAMES:
            logger.warning(f"Unknown timeframe: {timeframe}")
            return []
        
        idx = cls.STANDARD_TIMEFRAMES.index(timeframe)
        return cls.STANDARD_TIMEFRAMES[:idx]
    
    @classmethod
    def is_parent_of(cls, parent: str, child: str) -> bool:
        """
        Check if one timeframe is a parent of another.
        
        Args:
            parent: Potential parent timeframe
            child: Potential child timeframe
            
        Returns:
            True if parent is a parent of child
        """
        if parent not in cls.STANDARD_TIMEFRAMES or child not in cls.STANDARD_TIMEFRAMES:
            logger.warning(f"Unknown timeframe: {parent} or {child}")
            return False
        
        parent_idx = cls.STANDARD_TIMEFRAMES.index(parent)
        child_idx = cls.STANDARD_TIMEFRAMES.index(child)
        
        return parent_idx > child_idx
    
    @classmethod
    def get_minutes(cls, timeframe: str) -> int:
        """
        Get the number of minutes in a timeframe.
        
        Args:
            timeframe: The timeframe
            
        Returns:
            Number of minutes
        """
        return cls.TIMEFRAME_MINUTES.get(timeframe, 0)
    
    @classmethod
    def sort_timeframes(cls, timeframes: List[str], ascending: bool = True) -> List[str]:
        """
        Sort timeframes by size.
        
        Args:
            timeframes: List of timeframes to sort
            ascending: If True, sort from smallest to largest
            
        Returns:
            Sorted list of timeframes
        """
        valid_timeframes = [tf for tf in timeframes if tf in cls.STANDARD_TIMEFRAMES]
        
        # Sort by index in standard timeframes
        sorted_timeframes = sorted(
            valid_timeframes,
            key=lambda tf: cls.STANDARD_TIMEFRAMES.index(tf)
        )
        
        if not ascending:
            sorted_timeframes.reverse()
            
        return sorted_timeframes


class MultiTimeframeProcessor:
    """
    Processor for parallel operations across multiple timeframes.
    
    This class provides optimized parallel processing for operations that need
    to be performed on multiple timeframes, such as data retrieval, feature
    calculation, or analysis.
    """
    
    def __init__(self, resource_manager: Optional[ResourceManager] = None):
        """
        Initialize the multi-timeframe processor.
        
        Args:
            resource_manager: Optional resource manager for worker allocation
        """
        self.resource_manager = resource_manager or ResourceManager()
        self.executor = ParallelExecutor(resource_manager=self.resource_manager)
        self.timeframe_hierarchy = TimeframeHierarchy()
    
    async def process_timeframes(self,
                          timeframes: List[str],
                          process_func: Callable[[str], T],
                          priority: TaskPriority = TaskPriority.MEDIUM,
                          parallelization_method: ParallelizationMethod = ParallelizationMethod.AUTO,
                          timeout: Optional[float] = None,
                          respect_hierarchy: bool = True) -> Dict[str, T]:
        """
        Process multiple timeframes in parallel.
        
        Args:
            timeframes: List of timeframes to process
            process_func: Function to process each timeframe
            priority: Priority for the processing tasks
            parallelization_method: Method for parallelization
            timeout: Optional timeout in seconds
            respect_hierarchy: If True, process larger timeframes first
            
        Returns:
            Dictionary mapping timeframes to processing results
            
        Raises:
            DataProcessingError: If processing fails for all timeframes
        """
        if not timeframes:
            return {}
        
        # Sort timeframes if respecting hierarchy
        if respect_hierarchy:
            # Process larger timeframes first (descending order)
            sorted_timeframes = TimeframeHierarchy.sort_timeframes(timeframes, ascending=False)
        else:
            sorted_timeframes = timeframes
        
        # Create task definitions
        tasks = []
        for timeframe in sorted_timeframes:
            task_id = f"timeframe_{timeframe}_{uuid.uuid4().hex[:8]}"
            
            # Assign higher priority to larger timeframes
            if respect_hierarchy:
                # Adjust priority based on timeframe size
                tf_idx = TimeframeHierarchy.STANDARD_TIMEFRAMES.index(timeframe)
                tf_priority = max(0, priority.value - (tf_idx // 2))
                task_priority = TaskPriority(tf_priority)
            else:
                task_priority = priority
            
            tasks.append(TaskDefinition(
                id=task_id,
                func=process_func,
                input_data=timeframe,
                priority=task_priority,
                parallelization_method=parallelization_method,
                timeout=timeout
            ))
        
        # Execute tasks
        results = await self.executor.execute_tasks(tasks)
        
        # Process results
        processed_results = {}
        errors = []
        
        for task_id, task_result in results.items():
            # Extract timeframe from task ID
            timeframe = next((t.input_data for t in tasks if t.id == task_id), None)
            
            if task_result.success:
                processed_results[timeframe] = task_result.result
            else:
                error_msg = str(task_result.error) if task_result.error else "Unknown error"
                logger.error(f"Error processing timeframe {timeframe}: {error_msg}")
                errors.append((timeframe, error_msg))
        
        # If all timeframes failed, raise an error
        if errors and len(errors) == len(timeframes):
            error_details = "\n".join([f"{tf}: {e}" for tf, e in errors])
            raise DataProcessingError(
                message=f"Processing failed for all timeframes:\n{error_details}"
            )
        
        return processed_results
    
    async def process_instrument_timeframes(self,
                                     instrument: str,
                                     timeframes: List[str],
                                     process_func: Callable[[str, str], T],
                                     priority: TaskPriority = TaskPriority.MEDIUM,
                                     parallelization_method: ParallelizationMethod = ParallelizationMethod.AUTO,
                                     timeout: Optional[float] = None,
                                     respect_hierarchy: bool = True) -> Dict[str, T]:
        """
        Process multiple timeframes for a specific instrument in parallel.
        
        Args:
            instrument: The instrument symbol
            timeframes: List of timeframes to process
            process_func: Function to process each timeframe for the instrument
            priority: Priority for the processing tasks
            parallelization_method: Method for parallelization
            timeout: Optional timeout in seconds
            respect_hierarchy: If True, process larger timeframes first
            
        Returns:
            Dictionary mapping timeframes to processing results
            
        Raises:
            DataProcessingError: If processing fails for all timeframes
        """
        if not timeframes:
            return {}
        
        # Create a wrapper function to include the instrument
        def create_wrapper(inst):
            return lambda tf: process_func(inst, tf)
        
        wrapper_func = create_wrapper(instrument)
        
        # Process timeframes with the wrapper function
        return await self.process_timeframes(
            timeframes=timeframes,
            process_func=wrapper_func,
            priority=priority,
            parallelization_method=parallelization_method,
            timeout=timeout,
            respect_hierarchy=respect_hierarchy
        )
    
    async def process_multi_instrument_timeframes(self,
                                          instruments: List[str],
                                          timeframes: List[str],
                                          process_func: Callable[[str, str], T],
                                          priority: TaskPriority = TaskPriority.MEDIUM,
                                          parallelization_method: ParallelizationMethod = ParallelizationMethod.AUTO,
                                          timeout: Optional[float] = None,
                                          respect_hierarchy: bool = True,
                                          batch_size: Optional[int] = None) -> Dict[str, Dict[str, T]]:
        """
        Process multiple timeframes for multiple instruments in parallel.
        
        Args:
            instruments: List of instrument symbols
            timeframes: List of timeframes to process
            process_func: Function to process each timeframe for each instrument
            priority: Priority for the processing tasks
            parallelization_method: Method for parallelization
            timeout: Optional timeout in seconds
            respect_hierarchy: If True, process larger timeframes first
            batch_size: Optional batch size for processing instruments
            
        Returns:
            Dictionary mapping instruments to dictionaries mapping timeframes to results
            
        Raises:
            DataProcessingError: If processing fails for all instruments
        """
        if not instruments or not timeframes:
            return {}
        
        # Process instruments in batches if specified
        if batch_size and len(instruments) > batch_size:
            all_results = {}
            
            for i in range(0, len(instruments), batch_size):
                batch = instruments[i:i+batch_size]
                logger.debug(f"Processing instrument batch {i//batch_size + 1}/{(len(instruments) + batch_size - 1)//batch_size} "
                            f"with {len(batch)} instruments")
                
                batch_results = await self.process_multi_instrument_timeframes(
                    instruments=batch,
                    timeframes=timeframes,
                    process_func=process_func,
                    priority=priority,
                    parallelization_method=parallelization_method,
                    timeout=timeout,
                    respect_hierarchy=respect_hierarchy
                )
                
                all_results.update(batch_results)
                
            return all_results
        
        # Process each instrument's timeframes
        results = {}
        errors = []
        
        for instrument in instruments:
            try:
                instrument_results = await self.process_instrument_timeframes(
                    instrument=instrument,
                    timeframes=timeframes,
                    process_func=process_func,
                    priority=priority,
                    parallelization_method=parallelization_method,
                    timeout=timeout,
                    respect_hierarchy=respect_hierarchy
                )
                
                results[instrument] = instrument_results
            except Exception as e:
                logger.error(f"Error processing timeframes for instrument {instrument}: {str(e)}")
                errors.append((instrument, str(e)))
        
        # If all instruments failed, raise an error
        if errors and len(errors) == len(instruments):
            error_details = "\n".join([f"{i}: {e}" for i, e in errors])
            raise DataProcessingError(
                message=f"Processing failed for all instruments:\n{error_details}"
            )
        
        return results
