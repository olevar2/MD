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
from common_lib.parallel import ParallelProcessor, ParallelizationMethod, ResourceManager, TaskDefinition, TaskPriority, TaskResult, get_parallel_processor
logger = logging.getLogger(__name__)
T = TypeVar('T')


from data_pipeline_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class TimeframeHierarchy:
    """
    Manages timeframe hierarchies and relationships.

    This class provides utilities for working with timeframe hierarchies,
    including determining parent-child relationships, aggregation, and
    dependency ordering.
    """
    STANDARD_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w',
        '1M']
    TIMEFRAME_MINUTES = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60,
        '4h': 240, '1d': 1440, '1w': 10080, '1M': 43200}

    @classmethod
    def get_parent_timeframes(cls, timeframe: str) ->List[str]:
        """
        Get all parent timeframes for a given timeframe.

        Args:
            timeframe: The timeframe to get parents for

        Returns:
            List of parent timeframes (larger timeframes)
        """
        if timeframe not in cls.STANDARD_TIMEFRAMES:
            logger.warning(f'Unknown timeframe: {timeframe}')
            return []
        idx = cls.STANDARD_TIMEFRAMES.index(timeframe)
        return cls.STANDARD_TIMEFRAMES[idx + 1:]

    @classmethod
    def get_child_timeframes(cls, timeframe: str) ->List[str]:
        """
        Get all child timeframes for a given timeframe.

        Args:
            timeframe: The timeframe to get children for

        Returns:
            List of child timeframes (smaller timeframes)
        """
        if timeframe not in cls.STANDARD_TIMEFRAMES:
            logger.warning(f'Unknown timeframe: {timeframe}')
            return []
        idx = cls.STANDARD_TIMEFRAMES.index(timeframe)
        return cls.STANDARD_TIMEFRAMES[:idx]

    @classmethod
    def is_parent_of(cls, parent: str, child: str) ->bool:
        """
        Check if one timeframe is a parent of another.

        Args:
            parent: Potential parent timeframe
            child: Potential child timeframe

        Returns:
            True if parent is a parent of child
        """
        if (parent not in cls.STANDARD_TIMEFRAMES or child not in cls.
            STANDARD_TIMEFRAMES):
            logger.warning(f'Unknown timeframe: {parent} or {child}')
            return False
        parent_idx = cls.STANDARD_TIMEFRAMES.index(parent)
        child_idx = cls.STANDARD_TIMEFRAMES.index(child)
        return parent_idx > child_idx

    @classmethod
    def get_minutes(cls, timeframe: str) ->int:
        """
        Get the number of minutes in a timeframe.

        Args:
            timeframe: The timeframe

        Returns:
            Number of minutes
        """
        return cls.TIMEFRAME_MINUTES.get(timeframe, 0)

    @classmethod
    def sort_timeframes(cls, timeframes: List[str], ascending: bool=True
        ) ->List[str]:
        """
        Sort timeframes by size.

        Args:
            timeframes: List of timeframes to sort
            ascending: If True, sort from smallest to largest

        Returns:
            Sorted list of timeframes
        """
        valid_timeframes = [tf for tf in timeframes if tf in cls.
            STANDARD_TIMEFRAMES]
        sorted_timeframes = sorted(valid_timeframes, key=lambda tf: cls.
            STANDARD_TIMEFRAMES.index(tf))
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

    def __init__(self, resource_manager: Optional[ResourceManager]=None):
        """
        Initialize the multi-timeframe processor.

        Args:
            resource_manager: Optional resource manager for worker allocation
        """
        self.resource_manager = resource_manager or ResourceManager()
        self.executor = ParallelExecutor(resource_manager=self.resource_manager
            )
        self.timeframe_hierarchy = TimeframeHierarchy()

    async def process_timeframes(self, timeframes: List[str], process_func:
        Callable[[str], T], priority: TaskPriority=TaskPriority.MEDIUM,
        parallelization_method: ParallelizationMethod=ParallelizationMethod
        .AUTO, timeout: Optional[float]=None, respect_hierarchy: bool=True,
        use_incremental_updates: bool=True, cache_results: bool=True,
        cache_ttl: int=300) ->Dict[str, T]:
        """
        Process multiple timeframes in parallel with optimized performance.

        Args:
            timeframes: List of timeframes to process
            process_func: Function to process each timeframe
            priority: Priority for the processing tasks
            parallelization_method: Method for parallelization
            timeout: Optional timeout in seconds
            respect_hierarchy: If True, process larger timeframes first
            use_incremental_updates: If True, use incremental updates for related timeframes
            cache_results: If True, cache results for future use
            cache_ttl: Time-to-live for cached results in seconds

        Returns:
            Dictionary mapping timeframes to processing results

        Raises:
            DataProcessingError: If processing fails for all timeframes
        """
        if not timeframes:
            return {}
        if cache_results and not hasattr(self, '_timeframe_cache'):
            self._timeframe_cache = {}
            self._timeframe_cache_timestamps = {}
        if cache_results:
            current_time = datetime.now()
            cached_results = {}
            remaining_timeframes = []
            for tf in timeframes:
                cache_key = self._get_cache_key(tf, process_func)
                if cache_key in self._timeframe_cache:
                    cache_time = self._timeframe_cache_timestamps.get(cache_key
                        )
                    if cache_time and (current_time - cache_time
                        ).total_seconds() < cache_ttl:
                        cached_results[tf] = self._timeframe_cache[cache_key]
                        continue
                remaining_timeframes.append(tf)
            if not remaining_timeframes:
                return cached_results
            timeframes = remaining_timeframes
        else:
            cached_results = {}
        if respect_hierarchy:
            sorted_timeframes = TimeframeHierarchy.sort_timeframes(timeframes,
                ascending=False)
        else:
            sorted_timeframes = timeframes
        if use_incremental_updates and respect_hierarchy:
            timeframe_groups = self._group_related_timeframes(sorted_timeframes
                )
            all_results = {}
            for group in timeframe_groups:
                group_results = await self._process_timeframe_group(group,
                    process_func, priority, parallelization_method, timeout)
                all_results.update(group_results)
            processed_results = all_results
        else:
            tasks = []
            for timeframe in sorted_timeframes:
                task_id = f'timeframe_{timeframe}_{uuid.uuid4().hex[:8]}'
                if respect_hierarchy:
                    tf_idx = TimeframeHierarchy.STANDARD_TIMEFRAMES.index(
                        timeframe)
                    tf_priority = max(0, priority.value - tf_idx // 2)
                    task_priority = TaskPriority(tf_priority)
                else:
                    task_priority = priority
                tasks.append(TaskDefinition(id=task_id, func=process_func,
                    input_data=timeframe, priority=task_priority,
                    parallelization_method=parallelization_method, timeout=
                    timeout))
            results = await self.executor.execute_tasks(tasks)
            processed_results = {}
            errors = []
            for task_id, task_result in results.items():
                timeframe = next((t.input_data for t in tasks if t.id ==
                    task_id), None)
                if task_result.success:
                    processed_results[timeframe] = task_result.result
                    if cache_results:
                        cache_key = self._get_cache_key(timeframe, process_func
                            )
                        self._timeframe_cache[cache_key] = task_result.result
                        self._timeframe_cache_timestamps[cache_key
                            ] = datetime.now()
                else:
                    error_msg = str(task_result.error
                        ) if task_result.error else 'Unknown error'
                    logger.error(
                        f'Error processing timeframe {timeframe}: {error_msg}')
                    errors.append((timeframe, error_msg))
            if errors and len(errors) == len(timeframes):
                error_details = '\n'.join([f'{tf}: {e}' for tf, e in errors])
                raise DataProcessingError(message=
                    f"""Processing failed for all timeframes:
{error_details}"""
                    )
        combined_results = {**cached_results, **processed_results}
        if cache_results and len(self._timeframe_cache) > 1000:
            self._cleanup_cache(max_size=500)
        return combined_results

    def _get_cache_key(self, timeframe: str, func: Callable) ->str:
        """Generate a cache key for a timeframe and function."""
        func_name = func.__name__
        func_module = func.__module__
        return f'{func_module}.{func_name}_{timeframe}'

    def _cleanup_cache(self, max_size: int=500) ->None:
        """Clean up old cache entries to prevent memory issues."""
        if len(self._timeframe_cache) <= max_size:
            return
        sorted_keys = sorted(self._timeframe_cache_timestamps.keys(), key=
            lambda k: self._timeframe_cache_timestamps[k])
        keys_to_remove = sorted_keys[:len(sorted_keys) - max_size]
        for key in keys_to_remove:
            if key in self._timeframe_cache:
                del self._timeframe_cache[key]
            if key in self._timeframe_cache_timestamps:
                del self._timeframe_cache_timestamps[key]

    def _group_related_timeframes(self, timeframes: List[str]) ->List[List[str]
        ]:
        """Group timeframes by hierarchical relationships for optimized processing."""
        sorted_tfs = TimeframeHierarchy.sort_timeframes(timeframes,
            ascending=False)
        groups = []
        remaining = set(sorted_tfs)
        while remaining:
            current = next(iter(remaining))
            remaining.remove(current)
            group = [current]
            for tf in list(remaining):
                if TimeframeHierarchy.is_parent_of(current, tf):
                    group.append(tf)
                    remaining.remove(tf)
            groups.append(sorted(group, key=lambda x: TimeframeHierarchy.
                STANDARD_TIMEFRAMES.index(x)))
        return groups

    @async_with_exception_handling
    async def _process_timeframe_group(self, timeframes: List[str],
        process_func: Callable[[str], T], priority: TaskPriority,
        parallelization_method: ParallelizationMethod, timeout: Optional[float]
        ) ->Dict[str, T]:
        """Process a group of related timeframes, optimizing for hierarchical relationships."""
        if not timeframes:
            return {}
        sorted_tfs = TimeframeHierarchy.sort_timeframes(timeframes,
            ascending=False)
        largest_tf = sorted_tfs[0]
        task_id = f'timeframe_{largest_tf}_{uuid.uuid4().hex[:8]}'
        task = TaskDefinition(id=task_id, func=process_func, input_data=
            largest_tf, priority=priority, parallelization_method=
            parallelization_method, timeout=timeout)
        results = await self.executor.execute_tasks([task])
        processed_results = {}
        for task_id, task_result in results.items():
            if task_result.success:
                processed_results[largest_tf] = task_result.result
            else:
                error_msg = str(task_result.error
                    ) if task_result.error else 'Unknown error'
                logger.error(
                    f'Error processing timeframe {largest_tf}: {error_msg}')
                return await self._process_timeframes_individually(timeframes,
                    process_func, priority, parallelization_method, timeout)
        if len(sorted_tfs) > 1:

            @async_with_exception_handling
            async def process_with_parent(tf):
    """
    Process with parent.
    
    Args:
        tf: Description of tf
    
    """

                try:
                    if hasattr(process_func, 'with_parent_result'):
                        return await process_func.with_parent_result(tf,
                            largest_tf, processed_results[largest_tf])
                    else:
                        return await process_func(tf)
                except Exception as e:
                    logger.error(
                        f'Error processing timeframe {tf} with parent data: {str(e)}'
                        )
                    return await process_func(tf)
            remaining_tfs = sorted_tfs[1:]
            tasks = []
            for tf in remaining_tfs:
                task_id = f'timeframe_{tf}_{uuid.uuid4().hex[:8]}'
                tf_idx = TimeframeHierarchy.STANDARD_TIMEFRAMES.index(tf)
                tf_priority = max(0, priority.value - tf_idx // 2)
                task_priority = TaskPriority(tf_priority)
                tasks.append(TaskDefinition(id=task_id, func=
                    process_with_parent, input_data=tf, priority=
                    task_priority, parallelization_method=
                    parallelization_method, timeout=timeout))
            remaining_results = await self.executor.execute_tasks(tasks)
            for task_id, task_result in remaining_results.items():
                tf = next((t.input_data for t in tasks if t.id == task_id),
                    None)
                if task_result.success:
                    processed_results[tf] = task_result.result
                else:
                    error_msg = str(task_result.error
                        ) if task_result.error else 'Unknown error'
                    logger.error(
                        f'Error processing timeframe {tf}: {error_msg}')
                    try:
                        individual_result = await process_func(tf)
                        processed_results[tf] = individual_result
                    except Exception as e:
                        logger.error(
                            f'Error processing timeframe {tf} individually: {str(e)}'
                            )
        return processed_results

    async def _process_timeframes_individually(self, timeframes: List[str],
        process_func: Callable[[str], T], priority: TaskPriority,
        parallelization_method: ParallelizationMethod, timeout: Optional[float]
        ) ->Dict[str, T]:
        """Process timeframes individually as a fallback."""
        tasks = []
        for tf in timeframes:
            task_id = f'timeframe_{tf}_{uuid.uuid4().hex[:8]}'
            tf_idx = TimeframeHierarchy.STANDARD_TIMEFRAMES.index(tf)
            tf_priority = max(0, priority.value - tf_idx // 2)
            task_priority = TaskPriority(tf_priority)
            tasks.append(TaskDefinition(id=task_id, func=process_func,
                input_data=tf, priority=task_priority,
                parallelization_method=parallelization_method, timeout=timeout)
                )
        results = await self.executor.execute_tasks(tasks)
        processed_results = {}
        for task_id, task_result in results.items():
            tf = next((t.input_data for t in tasks if t.id == task_id), None)
            if task_result.success:
                processed_results[tf] = task_result.result
        return processed_results

    async def process_instrument_timeframes(self, instrument: str,
        timeframes: List[str], process_func: Callable[[str, str], T],
        priority: TaskPriority=TaskPriority.MEDIUM, parallelization_method:
        ParallelizationMethod=ParallelizationMethod.AUTO, timeout: Optional
        [float]=None, respect_hierarchy: bool=True, use_incremental_updates:
        bool=True, cache_results: bool=True, cache_ttl: int=300) ->Dict[str, T
        ]:
        """
        Process multiple timeframes for a specific instrument in parallel with optimized performance.

        Args:
            instrument: The instrument symbol
            timeframes: List of timeframes to process
            process_func: Function to process each timeframe for the instrument
            priority: Priority for the processing tasks
            parallelization_method: Method for parallelization
            timeout: Optional timeout in seconds
            respect_hierarchy: If True, process larger timeframes first
            use_incremental_updates: If True, use incremental updates for related timeframes
            cache_results: If True, cache results for future use
            cache_ttl: Time-to-live for cached results in seconds

        Returns:
            Dictionary mapping timeframes to processing results

        Raises:
            DataProcessingError: If processing fails for all timeframes
        """
        if not timeframes:
            return {}

        def create_wrapper(inst):
    """
    Create wrapper.
    
    Args:
        inst: Description of inst
    
    """

            wrapper = lambda tf: process_func(inst, tf)
            if use_incremental_updates:

                async def with_parent_result(tf, parent_tf, parent_result):
    """
    With parent result.
    
    Args:
        tf: Description of tf
        parent_tf: Description of parent_tf
        parent_result: Description of parent_result
    
    """

                    if hasattr(process_func, 'with_parent_result'):
                        return await process_func.with_parent_result(inst,
                            tf, parent_tf, parent_result)
                    else:
                        return await process_func(inst, tf)
                wrapper.with_parent_result = with_parent_result
            return wrapper
        wrapper_func = create_wrapper(instrument)
        cache_key_prefix = (f'instrument_{instrument}_' if cache_results else
            None)
        return await self.process_timeframes(timeframes=timeframes,
            process_func=wrapper_func, priority=priority,
            parallelization_method=parallelization_method, timeout=timeout,
            respect_hierarchy=respect_hierarchy, use_incremental_updates=
            use_incremental_updates, cache_results=cache_results, cache_ttl
            =cache_ttl)

    @async_with_exception_handling
    async def process_multi_instrument_timeframes(self, instruments: List[
        str], timeframes: List[str], process_func: Callable[[str, str], T],
        priority: TaskPriority=TaskPriority.MEDIUM, parallelization_method:
        ParallelizationMethod=ParallelizationMethod.AUTO, timeout: Optional
        [float]=None, respect_hierarchy: bool=True, use_incremental_updates:
        bool=True, cache_results: bool=True, cache_ttl: int=300, batch_size:
        Optional[int]=None, parallel_instruments: bool=True) ->Dict[str,
        Dict[str, T]]:
        """
        Process multiple timeframes for multiple instruments in parallel with optimized performance.

        Args:
            instruments: List of instrument symbols
            timeframes: List of timeframes to process
            process_func: Function to process each timeframe for each instrument
            priority: Priority for the processing tasks
            parallelization_method: Method for parallelization
            timeout: Optional timeout in seconds
            respect_hierarchy: If True, process larger timeframes first
            use_incremental_updates: If True, use incremental updates for related timeframes
            cache_results: If True, cache results for future use
            cache_ttl: Time-to-live for cached results in seconds
            batch_size: Optional batch size for processing instruments
            parallel_instruments: If True, process instruments in parallel

        Returns:
            Dictionary mapping instruments to dictionaries mapping timeframes to results

        Raises:
            DataProcessingError: If processing fails for all instruments
        """
        if not instruments or not timeframes:
            return {}
        if batch_size and len(instruments) > batch_size:
            all_results = {}
            for i in range(0, len(instruments), batch_size):
                batch = instruments[i:i + batch_size]
                logger.debug(
                    f'Processing instrument batch {i // batch_size + 1}/{(len(instruments) + batch_size - 1) // batch_size} with {len(batch)} instruments'
                    )
                batch_results = await self.process_multi_instrument_timeframes(
                    instruments=batch, timeframes=timeframes, process_func=
                    process_func, priority=priority, parallelization_method
                    =parallelization_method, timeout=timeout,
                    respect_hierarchy=respect_hierarchy,
                    use_incremental_updates=use_incremental_updates,
                    cache_results=cache_results, cache_ttl=cache_ttl,
                    parallel_instruments=parallel_instruments)
                all_results.update(batch_results)
            return all_results
        if parallel_instruments and len(instruments) > 1:
            import asyncio

            @async_with_exception_handling
            async def process_single_instrument(instrument):
    """
    Process single instrument.
    
    Args:
        instrument: Description of instrument
    
    """

                try:
                    return (instrument, await self.
                        process_instrument_timeframes(instrument=instrument,
                        timeframes=timeframes, process_func=process_func,
                        priority=priority, parallelization_method=
                        parallelization_method, timeout=timeout,
                        respect_hierarchy=respect_hierarchy,
                        use_incremental_updates=use_incremental_updates,
                        cache_results=cache_results, cache_ttl=cache_ttl))
                except Exception as e:
                    logger.error(
                        f'Error processing timeframes for instrument {instrument}: {str(e)}'
                        )
                    return instrument, {}
            instrument_tasks = [process_single_instrument(instrument) for
                instrument in instruments]
            instrument_results = await asyncio.gather(*instrument_tasks)
            results = {instrument: result for instrument, result in
                instrument_results if result}
            if not results and instruments:
                raise DataProcessingError(message=
                    f'Processing failed for all {len(instruments)} instruments'
                    )
            return results
        else:
            results = {}
            errors = []
            for instrument in instruments:
                try:
                    instrument_results = (await self.
                        process_instrument_timeframes(instrument=instrument,
                        timeframes=timeframes, process_func=process_func,
                        priority=priority, parallelization_method=
                        parallelization_method, timeout=timeout,
                        respect_hierarchy=respect_hierarchy,
                        use_incremental_updates=use_incremental_updates,
                        cache_results=cache_results, cache_ttl=cache_ttl))
                    results[instrument] = instrument_results
                except Exception as e:
                    logger.error(
                        f'Error processing timeframes for instrument {instrument}: {str(e)}'
                        )
                    errors.append((instrument, str(e)))
            if errors and len(errors) == len(instruments):
                error_details = '\n'.join([f'{i}: {e}' for i, e in errors])
                raise DataProcessingError(message=
                    f"""Processing failed for all instruments:
{error_details}"""
                    )
            return results
