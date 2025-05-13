"""
Incremental Indicator Processor

This module provides functionality to process multiple incremental indicators efficiently,
with support for parallel computation to minimize latency when processing batches.
"""
from typing import Dict, List, Optional, Any, Set, Union, Tuple
import multiprocessing
import concurrent.futures
import logging
from datetime import datetime
import time
import pandas as pd
import numpy as np
from feature_store_service.indicators.incremental.base import IncrementalIndicator


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class IncrementalProcessor:
    """
    Processor for efficiently calculating multiple incremental indicators
    
    This class manages a collection of incremental indicators and can update
    them all efficiently with new data, with support for parallel processing.
    """

    def __init__(self, max_workers: Optional[int]=None):
        """
        Initialize the incremental processor
        
        Args:
            max_workers: Maximum number of worker threads/processes for parallel computation
                        If None, defaults to CPU count - 1 (or 1 if single core)
        """
        self.indicators: Dict[str, IncrementalIndicator] = {}
        self.initialized: Set[str] = set()
        self.historical_data: List[Dict[str, Any]] = []
        self.max_workers = max_workers
        if max_workers is None:
            self.max_workers = max(1, multiprocessing.cpu_count() - 1)
        self.logger = logging.getLogger(__name__)

    def register_indicator(self, indicator: IncrementalIndicator) ->None:
        """
        Register an indicator with the processor
        
        Args:
            indicator: Indicator instance to register
        """
        self.indicators[indicator.name] = indicator
        self.logger.debug(f'Registered indicator: {indicator.name}')

    def register_indicators(self, indicators: List[IncrementalIndicator]
        ) ->None:
        """
        Register multiple indicators with the processor
        
        Args:
            indicators: List of indicator instances to register
        """
        for indicator in indicators:
            self.register_indicator(indicator)

    def initialize_indicators(self, historical_data: List[Dict[str, Any]],
        indicator_names: Optional[List[str]]=None) ->None:
        """
        Initialize indicators with historical data
        
        Args:
            historical_data: List of historical data points
            indicator_names: Optional list of specific indicator names to initialize
                           If None, all registered indicators will be initialized
        """
        self.historical_data = historical_data
        indicators_to_init = {}
        if indicator_names is None:
            indicators_to_init = self.indicators
        else:
            indicators_to_init = {name: self.indicators[name] for name in
                indicator_names if name in self.indicators}
        if not indicators_to_init:
            self.logger.warning('No indicators to initialize')
            return
        if len(indicators_to_init) > 1 and self.max_workers > 1:
            self._parallel_initialize(indicators_to_init, historical_data)
        else:
            self._sequential_initialize(indicators_to_init, historical_data)

    def _sequential_initialize(self, indicators: Dict[str,
        IncrementalIndicator], data: List[Dict[str, Any]]) ->None:
        """
        Initialize indicators sequentially
        
        Args:
            indicators: Dictionary of indicators to initialize
            data: Historical data for initialization
        """
        for name, indicator in indicators.items():
            start_time = time.time()
            indicator.initialize(data)
            end_time = time.time()
            if indicator.is_initialized:
                self.initialized.add(name)
                self.logger.debug(
                    f'Initialized {name} in {end_time - start_time:.4f} seconds'
                    )
            else:
                self.logger.warning(f'Failed to initialize {name}')

    def _parallel_initialize(self, indicators: Dict[str,
        IncrementalIndicator], data: List[Dict[str, Any]]) ->None:
        """
        Initialize indicators in parallel using ThreadPoolExecutor
        
        Args:
            indicators: Dictionary of indicators to initialize
            data: Historical data for initialization
        """

        def _init_worker(name: str, indicator: IncrementalIndicator):
    """
     init worker.
    
    Args:
        name: Description of name
        indicator: Description of indicator
    
    """

            start_time = time.time()
            indicator.initialize(data)
            end_time = time.time()
            if indicator.is_initialized:
                init_time = end_time - start_time
                return name, True, init_time
            else:
                return name, False, 0.0
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers
            ) as executor:
            futures = {executor.submit(_init_worker, name, indicator): name for
                name, indicator in indicators.items()}
            for future in concurrent.futures.as_completed(futures):
                name, success, init_time = future.result()
                if success:
                    self.initialized.add(name)
                    self.logger.debug(
                        f'Initialized {name} in {init_time:.4f} seconds')
                else:
                    self.logger.warning(f'Failed to initialize {name}')

    def update(self, data_point: Dict[str, Any], indicator_names: Optional[
        List[str]]=None) ->Dict[str, Dict[str, Any]]:
        """
        Update indicators with a new data point
        
        Args:
            data_point: New data point
            indicator_names: Optional list of specific indicator names to update
                          If None, all initialized indicators will be updated
                          
        Returns:
            Dictionary mapping indicator names to their results
        """
        indicators_to_update = {}
        if indicator_names is None:
            indicators_to_update = {name: self.indicators[name] for name in
                self.initialized}
        else:
            indicators_to_update = {name: self.indicators[name] for name in
                indicator_names if name in self.initialized and name in
                self.indicators}
        if not indicators_to_update:
            self.logger.warning('No initialized indicators to update')
            return {}
        if len(indicators_to_update) > 1 and self.max_workers > 1:
            return self._parallel_update(indicators_to_update, data_point)
        else:
            return self._sequential_update(indicators_to_update, data_point)

    @with_exception_handling
    def _sequential_update(self, indicators: Dict[str, IncrementalIndicator
        ], data_point: Dict[str, Any]) ->Dict[str, Dict[str, Any]]:
        """
        Update indicators sequentially
        
        Args:
            indicators: Dictionary of indicators to update
            data_point: New data point
            
        Returns:
            Dictionary with update results
        """
        results = {}
        for name, indicator in indicators.items():
            try:
                result = indicator.update(data_point)
                results[name] = result
            except Exception as e:
                self.logger.error(f'Error updating {name}: {str(e)}')
                results[name] = {'error': str(e)}
        return results

    @with_exception_handling
    def _parallel_update(self, indicators: Dict[str, IncrementalIndicator],
        data_point: Dict[str, Any]) ->Dict[str, Dict[str, Any]]:
        """
        Update indicators in parallel using ThreadPoolExecutor
        
        Args:
            indicators: Dictionary of indicators to update
            data_point: New data point
            
        Returns:
            Dictionary with update results
        """

        @with_exception_handling
        def _update_worker(name: str, indicator: IncrementalIndicator):
    """
     update worker.
    
    Args:
        name: Description of name
        indicator: Description of indicator
    
    """

            try:
                result = indicator.update(data_point)
                return name, result, None
            except Exception as e:
                return name, {'error': str(e)}, str(e)
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers
            ) as executor:
            futures = {executor.submit(_update_worker, name, indicator):
                name for name, indicator in indicators.items()}
            for future in concurrent.futures.as_completed(futures):
                name, result, error = future.result()
                results[name] = result
                if error:
                    self.logger.error(f'Error updating {name}: {error}')
        return results

    def update_many(self, data_points: List[Dict[str, Any]],
        indicator_names: Optional[List[str]]=None) ->List[Dict[str, Dict[
        str, Any]]]:
        """
        Update indicators with multiple data points sequentially
        
        Args:
            data_points: List of new data points
            indicator_names: Optional list of specific indicator names to update
            
        Returns:
            List of dictionaries mapping indicator names to their results for each data point
        """
        results = []
        for data_point in data_points:
            result = self.update(data_point, indicator_names)
            results.append(result)
        return results

    def get_indicator(self, name: str) ->Optional[IncrementalIndicator]:
        """
        Get an indicator by name
        
        Args:
            name: Name of the indicator
            
        Returns:
            Indicator instance or None if not found
        """
        return self.indicators.get(name)

    def get_state(self) ->Dict[str, Any]:
        """
        Get the current state of all indicators for persistence
        
        Returns:
            Dictionary of indicator states
        """
        state = {}
        for name, indicator in self.indicators.items():
            if name in self.initialized:
                state[name] = indicator.get_state()
        return state

    def set_state(self, state: Dict[str, Any]) ->None:
        """
        Restore indicators from saved state
        
        Args:
            state: Dictionary of indicator states
        """
        for name, indicator_state in state.items():
            if name in self.indicators:
                self.indicators[name].set_state(indicator_state)
                if self.indicators[name].is_initialized:
                    self.initialized.add(name)

    def reset_all(self) ->None:
        """Reset all indicators to their initial state"""
        for indicator in self.indicators.values():
            indicator.reset()
        self.initialized.clear()

    def reset_indicator(self, name: str) ->bool:
        """
        Reset a specific indicator
        
        Args:
            name: Name of the indicator to reset
            
        Returns:
            True if successful, False if indicator not found
        """
        if name in self.indicators:
            self.indicators[name].reset()
            if name in self.initialized:
                self.initialized.remove(name)
            return True
        return False

    def calculate_all(self, data: Union[List[Dict[str, Any]], pd.DataFrame],
        indicator_names: Optional[List[str]]=None, include_incomplete: bool
        =False) ->pd.DataFrame:
        """
        Calculate all specified indicators over the entire dataset
        
        This is useful for initializing a system with historical data and getting
        full indicator results in a DataFrame format.
        
        Args:
            data: List of data points or DataFrame with OHLCV data
            indicator_names: Optional list of indicator names to calculate
            include_incomplete: Whether to include incomplete initial values
            
        Returns:
            DataFrame with all indicator values
        """
        if isinstance(data, pd.DataFrame):
            records = data.to_dict('records')
        else:
            records = data
        if not records:
            return pd.DataFrame()
        self.initialize_indicators(records, indicator_names)
        results = []
        timestamps = []
        for i, record in enumerate(records):
            timestamp = record.get('timestamp', i)
            timestamps.append(timestamp)
            indicator_results = self.update(record, indicator_names)
            flat_results = {'timestamp': timestamp}
            for ind_name, ind_result in indicator_results.items():
                if isinstance(ind_result, dict) and 'value' in ind_result:
                    if include_incomplete or ind_result.get('complete', False):
                        flat_results[ind_name] = ind_result['value']
                elif isinstance(ind_result, dict):
                    for key, value in ind_result.items():
                        if include_incomplete or ind_result.get('complete',
                            False):
                            flat_results[f'{ind_name}_{key}'] = value
            results.append(flat_results)
        return pd.DataFrame(results)

    def memory_usage_estimate(self) ->Dict[str, Any]:
        """
        Estimate memory usage of all indicators
        
        Returns:
            Dictionary with memory usage information
        """
        total_size = 0
        indicator_sizes = {}
        for name, indicator in self.indicators.items():
            state = indicator.get_state()
            size = 0
            for key, value in state.items():
                if isinstance(value, list):
                    if value and isinstance(value[0], float):
                        size += len(value) * 8 + 56
                    else:
                        size += len(value) * 4 + 56
                elif isinstance(value, dict):
                    size += 64 + len(value) * 32
                elif isinstance(value, str):
                    size += len(value) + 49
                else:
                    size += 16
            indicator_sizes[name] = size
            total_size += size
        return {'total_bytes': total_size, 'total_mb': total_size / (1024 *
            1024), 'indicator_bytes': indicator_sizes, 'indicator_count':
            len(self.indicators), 'initialized_count': len(self.initialized)}


class StreamingIndicatorProcessor:
    """
    Specialized processor for streaming data use cases
    
    This class is optimized for the case where indicators are updated
    frequently with new tick or bar data in a streaming context.
    """

    def __init__(self, max_workers: Optional[int]=None,
        state_persistence_path: Optional[str]=None):
        """
        Initialize the streaming processor
        
        Args:
            max_workers: Maximum number of worker threads/processes for parallel computation
            state_persistence_path: Optional path to store/load indicator states
        """
        self.processor = IncrementalProcessor(max_workers)
        self.state_persistence_path = state_persistence_path
        self.last_update_time = None
        self.update_count = 0
        self.logger = logging.getLogger(__name__)

    def register_indicator(self, indicator: IncrementalIndicator) ->None:
        """Register an indicator with the processor"""
        self.processor.register_indicator(indicator)

    def register_indicators(self, indicators: List[IncrementalIndicator]
        ) ->None:
        """Register multiple indicators with the processor"""
        self.processor.register_indicators(indicators)

    def initialize_with_history(self, historical_data: List[Dict[str, Any]],
        indicator_names: Optional[List[str]]=None) ->None:
        """Initialize indicators with historical data"""
        self.processor.initialize_indicators(historical_data, indicator_names)
        self.last_update_time = datetime.now()

    def process_tick(self, tick_data: Dict[str, Any], indicator_names:
        Optional[List[str]]=None) ->Dict[str, Dict[str, Any]]:
        """
        Process a new tick of data
        
        Args:
            tick_data: New tick data
            indicator_names: Optional list of indicators to update
            
        Returns:
            Dictionary of indicator results
        """
        self.update_count += 1
        self.last_update_time = datetime.now()
        return self.processor.update(tick_data, indicator_names)

    @with_exception_handling
    def save_state(self, path: Optional[str]=None) ->bool:
        """
        Save the current state of all indicators
        
        Args:
            path: Optional path to save state
            
        Returns:
            True if successful, False otherwise
        """
        import json
        save_path = path or self.state_persistence_path
        if not save_path:
            self.logger.warning('No state persistence path specified')
            return False
        try:
            state = self.processor.get_state()
            state_with_meta = {'metadata': {'timestamp': datetime.now().
                isoformat(), 'update_count': self.update_count},
                'indicators': {}}
            for ind_name, ind_state in state.items():
                serializable_state = {}
                for key, value in ind_state.items():
                    if isinstance(value, np.ndarray):
                        serializable_state[key] = value.tolist()
                    elif isinstance(value, (np.int_, np.float_, np.bool_)):
                        serializable_state[key] = value.item()
                    elif isinstance(value, list) and value and isinstance(value
                        [0], np.ndarray):
                        serializable_state[key] = [arr.tolist() for arr in
                            value]
                    else:
                        serializable_state[key] = value
                state_with_meta['indicators'][ind_name] = serializable_state
            with open(save_path, 'w') as f:
                json.dump(state_with_meta, f, indent=2)
            self.logger.info(f'Saved state to {save_path}')
            return True
        except Exception as e:
            self.logger.error(f'Failed to save state: {str(e)}')
            return False

    @with_exception_handling
    def load_state(self, path: Optional[str]=None) ->bool:
        """
        Load indicator states from file
        
        Args:
            path: Optional path to load state
            
        Returns:
            True if successful, False otherwise
        """
        import json
        import os
        load_path = path or self.state_persistence_path
        if not load_path:
            self.logger.warning('No state persistence path specified')
            return False
        if not os.path.exists(load_path):
            self.logger.warning(f'State file not found: {load_path}')
            return False
        try:
            with open(load_path, 'r') as f:
                state_with_meta = json.load(f)
            if 'indicators' in state_with_meta:
                indicator_state = state_with_meta['indicators']
                if 'metadata' in state_with_meta:
                    metadata = state_with_meta['metadata']
                    self.update_count = metadata.get('update_count', 0)
            else:
                indicator_state = state_with_meta
            self.processor.set_state(indicator_state)
            self.logger.info(f'Loaded state from {load_path}')
            return True
        except Exception as e:
            self.logger.error(f'Failed to load state: {str(e)}')
            return False

    def get_status(self) ->Dict[str, Any]:
        """Get current status information"""
        memory_usage = self.processor.memory_usage_estimate()
        return {'update_count': self.update_count, 'last_update_time': self
            .last_update_time, 'indicator_count': len(self.processor.
            indicators), 'initialized_count': len(self.processor.
            initialized), 'memory_usage_mb': memory_usage['total_mb']}

    def reset(self) ->None:
        """Reset all indicators"""
        self.processor.reset_all()
        self.update_count = 0
        self.last_update_time = None
