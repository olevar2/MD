"""
Multi-Instrument Processor for Data Pipeline Service.

This module provides specialized parallel processing capabilities for
handling multiple financial instruments simultaneously, optimizing
data retrieval, processing, and analysis across instruments.

Features:
- Parallel data retrieval for multiple instruments
- Efficient batch processing of instrument data
- Optimized memory usage for large instrument sets
- Correlation-aware processing for related instruments
- Comprehensive error handling and reporting
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union

import pandas as pd
from common_lib.exceptions import DataProcessingError

from common_lib.parallel import ParallelProcessor, ParallelizationMethod, ResourceManager, TaskDefinition, TaskPriority, TaskResult, get_parallel_processor
    ParallelExecutor,
    ParallelizationMethod,
    ResourceManager,
    TaskDefinition,
    TaskPriority,
    TaskResult,
)

logger = logging.getLogger(__name__)

# Type variables for generic typing
T = TypeVar('T')  # Result type for instrument processing


class MultiInstrumentProcessor:
    """
    Processor for parallel operations across multiple financial instruments.
    
    This class provides optimized parallel processing for operations that need
    to be performed on multiple instruments, such as data retrieval, feature
    calculation, or analysis.
    """
    
    def __init__(self, resource_manager: Optional[ResourceManager] = None):
        """
        Initialize the multi-instrument processor.
        
        Args:
            resource_manager: Optional resource manager for worker allocation
        """
        self.resource_manager = resource_manager or ResourceManager()
        self.executor = ParallelExecutor(resource_manager=self.resource_manager)
        
    async def process_instruments(self,
                           instruments: List[str],
                           process_func: Callable[[str], T],
                           priority: TaskPriority = TaskPriority.MEDIUM,
                           parallelization_method: ParallelizationMethod = ParallelizationMethod.AUTO,
                           timeout: Optional[float] = None,
                           batch_size: Optional[int] = None) -> Dict[str, T]:
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
            
        Raises:
            DataProcessingError: If processing fails for all instruments
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
            tasks.append(TaskDefinition(
                id=task_id,
                func=process_func,
                input_data=instrument,
                priority=priority,
                parallelization_method=parallelization_method,
                timeout=timeout
            ))
        
        # Execute tasks
        results = await self.executor.execute_tasks(tasks)
        
        # Process results
        processed_results = {}
        errors = []
        
        for task_id, task_result in results.items():
            # Extract instrument from task ID
            instrument = next((t.input_data for t in tasks if t.id == task_id), None)
            
            if task_result.success:
                processed_results[instrument] = task_result.result
            else:
                error_msg = str(task_result.error) if task_result.error else "Unknown error"
                logger.error(f"Error processing instrument {instrument}: {error_msg}")
                errors.append((instrument, error_msg))
        
        # If all instruments failed, raise an error
        if errors and len(errors) == len(instruments):
            error_details = "\n".join([f"{i}: {e}" for i, e in errors])
            raise DataProcessingError(
                message=f"Processing failed for all instruments:\n{error_details}"
            )
        
        return processed_results
    
    async def _process_in_batches(self,
                           instruments: List[str],
                           process_func: Callable[[str], T],
                           priority: TaskPriority,
                           parallelization_method: ParallelizationMethod,
                           timeout: Optional[float],
                           batch_size: int) -> Dict[str, T]:
        """
        Process instruments in batches to manage memory usage.
        
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
            logger.debug(f"Processing batch {i//batch_size + 1}/{(len(instruments) + batch_size - 1)//batch_size} "
                        f"with {len(batch)} instruments")
            
            batch_results = await self.process_instruments(
                instruments=batch,
                process_func=process_func,
                priority=priority,
                parallelization_method=parallelization_method,
                timeout=timeout
            )
            
            all_results.update(batch_results)
        
        return all_results
    
    async def process_instrument_data(self,
                               instrument_data: Dict[str, Any],
                               process_func: Callable[[str, Any], T],
                               priority: TaskPriority = TaskPriority.MEDIUM,
                               parallelization_method: ParallelizationMethod = ParallelizationMethod.AUTO,
                               timeout: Optional[float] = None,
                               batch_size: Optional[int] = None) -> Dict[str, T]:
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
            
        Raises:
            DataProcessingError: If processing fails for all instruments
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
            
            # Create a wrapper function to pass both instrument and data
            def create_wrapper(inst, d):
    """
    Create wrapper.
    
    Args:
        inst: Description of inst
        d: Description of d
    
    """

                return lambda _: process_func(inst, d)
            
            wrapper_func = create_wrapper(instrument, data)
            
            tasks.append(TaskDefinition(
                id=task_id,
                func=wrapper_func,
                input_data=None,  # Not used by the wrapper
                priority=priority,
                parallelization_method=parallelization_method,
                timeout=timeout
            ))
        
        # Execute tasks
        results = await self.executor.execute_tasks(tasks)
        
        # Process results
        processed_results = {}
        errors = []
        
        for task_id, task_result in results.items():
            # Extract instrument from task ID
            instrument = task_id.split('_')[2]  # Extract from task_id format
            
            if task_result.success:
                processed_results[instrument] = task_result.result
            else:
                error_msg = str(task_result.error) if task_result.error else "Unknown error"
                logger.error(f"Error processing data for instrument {instrument}: {error_msg}")
                errors.append((instrument, error_msg))
        
        # If all instruments failed, raise an error
        if errors and len(errors) == len(instrument_data):
            error_details = "\n".join([f"{i}: {e}" for i, e in errors])
            raise DataProcessingError(
                message=f"Processing failed for all instrument data:\n{error_details}"
            )
        
        return processed_results
    
    async def _process_data_in_batches(self,
                                instrument_data: Dict[str, Any],
                                process_func: Callable[[str, Any], T],
                                priority: TaskPriority,
                                parallelization_method: ParallelizationMethod,
                                timeout: Optional[float],
                                batch_size: int) -> Dict[str, T]:
        """
        Process instrument data in batches to manage memory usage.
        
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
            
            logger.debug(f"Processing data batch {i//batch_size + 1}/{(len(instruments) + batch_size - 1)//batch_size} "
                        f"with {len(batch_data)} instruments")
            
            batch_results = await self.process_instrument_data(
                instrument_data=batch_data,
                process_func=process_func,
                priority=priority,
                parallelization_method=parallelization_method,
                timeout=timeout
            )
            
            all_results.update(batch_results)
        
        return all_results
