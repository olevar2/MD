"""
Parallel Indicator Processor

This module provides functionality to calculate multiple indicators in parallel
for optimal performance with large datasets or many indicators.
"""
import concurrent.futures
from typing import Dict, List, Optional, Any, Union, Callable, Tuple, Set
import logging
from time import time
import numpy as np
import pandas as pd
import multiprocessing
from enum import Enum
import os
import psutil
from dataclasses import dataclass
from feature_store_service.indicators.incremental_indicators import IndicatorState
from feature_store_service.indicators.extended_incremental_indicators import RSIState, BollingerBandsState, MACDState, ATRState, StochasticState


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class ParallelizationMode(Enum):
    """Enumeration of supported parallelization modes."""
    THREAD = 'thread'
    PROCESS = 'process'
    AUTO = 'auto'


class IndicatorPriority(Enum):
    """Enumeration of indicator priority levels."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class IndicatorDependency:
    """Represents a dependency relationship between indicators."""
    dependent_indicator: str
    required_indicators: List[str]


@dataclass
class EnhancedIndicatorSpec:
    """Enhanced indicator specification with priority and dependency information."""
    indicator: IndicatorState
    priority: IndicatorPriority = IndicatorPriority.MEDIUM
    dependencies: List[str] = None
    parallelization_mode: ParallelizationMode = ParallelizationMode.AUTO

    def __post_init__(self):
        """Initialize default values after initialization."""
        if self.dependencies is None:
            self.dependencies = []

    @property
    def name(self) ->str:
        """Get the indicator name."""
        return self.indicator.name


class ParallelIndicatorProcessor:
    """
    Process multiple indicators in parallel for improved performance.
    This is especially beneficial for batch processing of historical data
    or when calculating many indicators simultaneously.
    """

    def __init__(self, max_workers: int=None, cpu_threshold: float=0.8,
        memory_threshold: float=0.9):
        """
        Initialize the parallel processor
        
        Args:
            max_workers: Maximum number of worker threads/processes (None = auto-determine based on CPU cores)
            cpu_threshold: CPU usage threshold (0.0-1.0) above which to reduce parallelism
            memory_threshold: Memory usage threshold (0.0-1.0) above which to reduce parallelism
        """
        self.max_workers = max_workers
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.logger = logging.getLogger(__name__)
        self.dependency_graph = {}
        self.reverse_dependency_graph = {}
        self.execution_order = []
        self._last_resource_check = 0
        self._resource_check_interval = 5

    def process_dataframe(self, data: pd.DataFrame, indicators: List[Union[
        IndicatorState, EnhancedIndicatorSpec]], process_func: Callable[[
        IndicatorState, pd.DataFrame], pd.Series]=None, respect_priorities:
        bool=True, respect_dependencies: bool=True) ->Dict[str, pd.Series]:
        """
        Calculate multiple indicators in parallel using a DataFrame as input
        
        Args:
            data: DataFrame with price data (must include columns required by indicators)
            indicators: List of indicator state objects or enhanced specs to calculate
            process_func: Optional custom function to process each indicator
            respect_priorities: Whether to respect indicator priorities during scheduling
            respect_dependencies: Whether to respect indicator dependencies
            
        Returns:
            Dictionary mapping indicator names to calculated Series
        """
        if not indicators:
            return {}
        enhanced_indicators = self._ensure_enhanced_specs(indicators)
        if not process_func:
            process_func = self._default_process_dataframe
        if respect_dependencies:
            execution_groups = self._create_execution_groups(
                enhanced_indicators)
        else:
            execution_groups = [enhanced_indicators]
        start_time = time()
        results = {}
        for group_index, indicator_group in enumerate(execution_groups):
            self.logger.debug(
                f'Processing execution group {group_index + 1}/{len(execution_groups)} with {len(indicator_group)} indicators'
                )
            if respect_priorities:
                sorted_indicators = sorted(indicator_group, key=lambda x: x
                    .priority.value)
            else:
                sorted_indicators = indicator_group
            thread_indicators = []
            process_indicators = []
            for ind_spec in sorted_indicators:
                if self._should_use_process_based(ind_spec):
                    process_indicators.append(ind_spec)
                else:
                    thread_indicators.append(ind_spec)
            available_workers = self._get_available_workers()
            if thread_indicators:
                thread_results = self._process_with_thread_pool(data, [spec
                    .indicator for spec in thread_indicators], process_func,
                    max(1, available_workers // 2))
                results.update(thread_results)
            if process_indicators:
                process_results = self._process_with_process_pool(data, [
                    spec.indicator for spec in process_indicators], 
                    available_workers - min(len(thread_indicators), 
                    available_workers // 2))
                results.update(process_results)
        elapsed = time() - start_time
        self.logger.info(
            f'Calculated {len(results)} indicators in {elapsed:.4f} seconds')
        return results

    def _ensure_enhanced_specs(self, indicators: List[Union[IndicatorState,
        EnhancedIndicatorSpec]]) ->List[EnhancedIndicatorSpec]:
        """
        Ensure all indicators are wrapped in EnhancedIndicatorSpec
        
        Args:
            indicators: List of indicators or specs
            
        Returns:
            List with all indicators as EnhancedIndicatorSpec
        """
        result = []
        for ind in indicators:
            if isinstance(ind, EnhancedIndicatorSpec):
                result.append(ind)
            else:
                result.append(EnhancedIndicatorSpec(indicator=ind))
        return result

    def _create_execution_groups(self, indicators: List[EnhancedIndicatorSpec]
        ) ->List[List[EnhancedIndicatorSpec]]:
        """
        Create execution groups based on dependencies
        
        Each group contains indicators that can be processed in parallel.
        Groups must be processed in sequence.
        
        Args:
            indicators: List of enhanced indicator specs
            
        Returns:
            List of indicator groups
        """
        dependency_graph = {}
        for ind in indicators:
            dependency_graph[ind.name] = set(ind.dependencies)
        reverse_graph = {ind.name: set() for ind in indicators}
        for ind in indicators:
            for dep in ind.dependencies:
                if dep in reverse_graph:
                    reverse_graph[dep].add(ind.name)
        visited = set()
        temp_visited = set()
        order = []

        def visit(node):
    """
    Visit.
    
    Args:
        node: Description of node
    
    """

            if node in temp_visited:
                raise ValueError(f'Cyclic dependency detected involving {node}'
                    )
            if node in visited:
                return
            temp_visited.add(node)
            for dep in dependency_graph.get(node, set()):
                visit(dep)
            temp_visited.remove(node)
            visited.add(node)
            order.append(node)
        for ind in indicators:
            if ind.name not in visited:
                visit(ind.name)
        order.reverse()
        levels = {}
        for node in order:
            level = 0
            for dep in dependency_graph.get(node, set()):
                if dep in levels:
                    level = max(level, levels[dep] + 1)
            levels[node] = level
        max_level = max(levels.values()) if levels else 0
        execution_groups = [[] for _ in range(max_level + 1)]
        for ind in indicators:
            if ind.name in levels:
                execution_groups[levels[ind.name]].append(ind)
        return [group for group in execution_groups if group]

    @with_exception_handling
    def _process_with_thread_pool(self, data: pd.DataFrame, indicators:
        List[IndicatorState], process_func: Callable, max_workers: int) ->Dict[
        str, pd.Series]:
        """
        Process indicators using ThreadPoolExecutor
        
        Args:
            data: Input DataFrame
            indicators: List of indicators to process
            process_func: Function to process each indicator
            max_workers: Maximum number of workers
            
        Returns:
            Dictionary of results
        """
        results = {}
        actual_workers = max(1, min(max_workers, len(indicators)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers
            ) as executor:
            future_to_indicator = {executor.submit(process_func, indicator,
                data): indicator for indicator in indicators}
            for future in concurrent.futures.as_completed(future_to_indicator):
                indicator = future_to_indicator[future]
                try:
                    name, series = future.result()
                    results[name] = series
                except Exception as e:
                    self.logger.error(
                        f'Error calculating {indicator.name}: {str(e)}')
        return results

    @with_exception_handling
    def _process_with_process_pool(self, data: pd.DataFrame, indicators:
        List[IndicatorState], max_workers: int) ->Dict[str, pd.Series]:
        """
        Process indicators using ProcessPoolExecutor
        
        Args:
            data: Input DataFrame
            indicators: List of indicators to process
            max_workers: Maximum number of workers
            
        Returns:
            Dictionary of results
        """
        if not indicators:
            return {}
        if max_workers < 1:
            self.logger.warning(
                'Insufficient resources for process-based execution, falling back to thread-based'
                )
            return self._process_with_thread_pool(data, indicators, self.
                _default_process_dataframe, 1)
        actual_workers = max(1, min(max_workers, len(indicators), os.
            cpu_count() or 4))
        results = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=actual_workers
            ) as executor:
            data_dict = {'values': data.values.tolist(), 'index': data.
                index.tolist(), 'columns': data.columns.tolist()}
            indicator_params = []
            for ind in indicators:
                if hasattr(ind, 'to_dict'):
                    indicator_params.append((ind.name, ind.to_dict()))
                else:
                    indicator_params.append((ind.name, {'type': ind.
                        __class__.__name__}))
            futures = []
            for i, (name, params) in enumerate(indicator_params):
                futures.append(executor.submit(self._process_in_subprocess,
                    data_dict, name, params))
            for i, future in enumerate(concurrent.futures.as_completed(futures)
                ):
                try:
                    name, result_values = future.result()
                    results[name] = pd.Series(result_values, index=data.index)
                except Exception as e:
                    self.logger.error(f'Error in process pool: {str(e)}')
        return results

    @staticmethod
    def _process_in_subprocess(data_dict: Dict, indicator_name: str,
        indicator_params: Dict) ->Tuple[str, List[Any]]:
        """
        Static method to process indicator calculation in a subprocess
        
        Args:
            data_dict: Dictionary with serialized DataFrame
            indicator_name: Name of the indicator
            indicator_params: Parameters to reconstruct the indicator
            
        Returns:
            Tuple of (indicator name, calculated values)
        """
        df = pd.DataFrame(data_dict['values'], index=data_dict['index'],
            columns=data_dict['columns'])
        if 'window' in indicator_params:
            window = indicator_params.get('window', 14)
            result = df['close'].rolling(window=window).mean().tolist()
        elif indicator_name.lower().startswith('rsi'):
            period = indicator_params.get('period', 14)
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            result = (100 - 100 / (1 + rs)).tolist()
        else:
            result = df['close'].ewm(span=12).mean().tolist()
        return indicator_name, result

    def _should_use_process_based(self, indicator_spec: EnhancedIndicatorSpec
        ) ->bool:
        """
        Determine if an indicator should use process-based parallelism
        
        Args:
            indicator_spec: Enhanced indicator specification
            
        Returns:
            True if the indicator should use process-based parallelism
        """
        if indicator_spec.parallelization_mode == ParallelizationMode.PROCESS:
            return True
        if indicator_spec.parallelization_mode == ParallelizationMode.THREAD:
            return False
        if indicator_spec.parallelization_mode == ParallelizationMode.AUTO:
            if hasattr(indicator_spec.indicator, 'is_cpu_intensive'):
                return indicator_spec.indicator.is_cpu_intensive
            indicator_type = type(indicator_spec.indicator).__name__.lower()
            cpu_intensive_indicators = ['fractal', 'hurst', 'wav',
                'fourier', 'ml', 'regression', 'montecarlo', 'simulation',
                'kalman']
            return any(keyword in indicator_type for keyword in
                cpu_intensive_indicators)
        return False

    def _get_available_workers(self) ->int:
        """
        Determine the number of workers to use based on system resources
        
        Returns:
            Number of workers to use
        """
        current_time = time()
        if (current_time - self._last_resource_check < self.
            _resource_check_interval):
            if hasattr(self, '_last_worker_count'):
                return self._last_worker_count
        self._last_resource_check = current_time
        cpu_percent = psutil.cpu_percent(interval=0.1) / 100
        memory_percent = psutil.virtual_memory().percent / 100
        base_workers = self.max_workers or (os.cpu_count() or 4)
        if cpu_percent > self.cpu_threshold:
            cpu_factor = 1 - (cpu_percent - self.cpu_threshold) / (1 - self
                .cpu_threshold)
            base_workers = max(1, int(base_workers * cpu_factor))
            self.logger.debug(
                f'Reducing workers due to high CPU usage ({cpu_percent:.2f}): {base_workers}'
                )
        if memory_percent > self.memory_threshold:
            memory_factor = 1 - (memory_percent - self.memory_threshold) / (
                1 - self.memory_threshold)
            base_workers = max(1, int(base_workers * memory_factor))
            self.logger.debug(
                f'Reducing workers due to high memory usage ({memory_percent:.2f}): {base_workers}'
                )
        self._last_worker_count = base_workers
        return base_workers

    def _default_process_dataframe(self, indicator: IndicatorState, data:
        pd.DataFrame) ->Tuple[str, pd.Series]:
        """
        Default processor function that applies an indicator to a DataFrame
        
        Args:
            indicator: Indicator state object
            data: DataFrame with price data
            
        Returns:
            Tuple of (indicator name, calculated Series)
        """
        indicator.reset()
        result_values = []
        for idx, row in data.iterrows():
            row_dict = row.to_dict()
            result = indicator.update(row_dict)
            result_values.append(result)
        result_series = pd.Series(result_values, index=data.index)
        return indicator.name, result_series

    def process_batch(self, data_batch: List[Dict[str, float]], indicators:
        List[Union[IndicatorState, EnhancedIndicatorSpec]],
        respect_priorities: bool=True, respect_dependencies: bool=True) ->Dict[
        str, List[Any]]:
        """
        Process a batch of data points for multiple indicators in parallel
        
        Args:
            data_batch: List of dictionaries with price data
            indicators: List of indicator state objects or specs
            respect_priorities: Whether to respect indicator priorities
            respect_dependencies: Whether to respect indicator dependencies
            
        Returns:
            Dictionary mapping indicator names to lists of calculated values
        """
        if not indicators or not data_batch:
            return {}
        enhanced_indicators = self._ensure_enhanced_specs(indicators)
        if respect_dependencies:
            execution_groups = self._create_execution_groups(
                enhanced_indicators)
        else:
            execution_groups = [enhanced_indicators]
        start_time = time()
        results = {indicator.name: [] for indicator in enhanced_indicators}
        for ind_spec in enhanced_indicators:
            ind_spec.indicator.reset()
        for group_index, indicator_group in enumerate(execution_groups):
            self.logger.debug(
                f'Processing execution group {group_index + 1}/{len(execution_groups)} with {len(indicator_group)} indicators'
                )
            if respect_priorities:
                sorted_indicators = sorted(indicator_group, key=lambda x: x
                    .priority.value)
            else:
                sorted_indicators = indicator_group
            thread_indicators = []
            process_indicators = []
            for ind_spec in sorted_indicators:
                if self._should_use_process_based(ind_spec):
                    process_indicators.append(ind_spec)
                else:
                    thread_indicators.append(ind_spec)
            for data_index, data_point in enumerate(data_batch):
                if thread_indicators:
                    thread_results = self._process_point_with_threads(
                        data_point, [spec.indicator for spec in
                        thread_indicators])
                    for name, value in thread_results.items():
                        if name in results:
                            results[name].append(value)
                if process_indicators:
                    process_results = self._process_point_with_processes(
                        data_point, [spec.indicator for spec in
                        process_indicators])
                    for name, value in process_results.items():
                        if name in results:
                            results[name].append(value)
        elapsed = time() - start_time
        self.logger.info(
            f'Processed batch of {len(data_batch)} points for {len(indicators)} indicators in {elapsed:.4f} seconds'
            )
        return results

    @with_exception_handling
    def _process_point_with_threads(self, data_point: Dict[str, float],
        indicators: List[IndicatorState]) ->Dict[str, Any]:
        """
        Process a single data point for multiple indicators using threads
        
        Args:
            data_point: Dictionary with price data
            indicators: List of indicators to update
            
        Returns:
            Dictionary mapping indicator names to calculated values
        """
        results = {}
        available_workers = self._get_available_workers()
        with concurrent.futures.ThreadPoolExecutor(max_workers=
            available_workers) as executor:
            future_to_indicator = {executor.submit(self._safe_update,
                indicator, data_point): indicator for indicator in indicators}
            for future in concurrent.futures.as_completed(future_to_indicator):
                indicator = future_to_indicator[future]
                try:
                    value = future.result()
                    results[indicator.name] = value
                except Exception as e:
                    self.logger.error(
                        f'Error updating {indicator.name}: {str(e)}')
                    results[indicator.name] = None
        return results

    def _process_point_with_processes(self, data_point: Dict[str, float],
        indicators: List[IndicatorState]) ->Dict[str, Any]:
        """
        Process a single data point for multiple indicators using processes
        
        Args:
            data_point: Dictionary with price data
            indicators: List of indicators to update
            
        Returns:
            Dictionary mapping indicator names to calculated values
        """
        self.logger.debug(
            'Process-based execution not efficient for single point updates, falling back to threads'
            )
        return self._process_point_with_threads(data_point, indicators)

    @with_exception_handling
    def _safe_update(self, indicator: IndicatorState, data: Dict[str, float]
        ) ->Any:
        """
        Safely update an indicator with error handling
        
        Args:
            indicator: Indicator state object
            data: Dictionary with price data
            
        Returns:
            Updated indicator value or None on error
        """
        try:
            return indicator.update(data)
        except Exception as e:
            self.logger.error(f'Error updating {indicator.name}: {str(e)}')
            return None

    def register_dependency(self, dependent: str, prerequisite: str) ->None:
        """
        Register a dependency between indicators
        
        Args:
            dependent: Name of the dependent indicator
            prerequisite: Name of the indicator it depends on
        """
        if dependent not in self.dependency_graph:
            self.dependency_graph[dependent] = set()
        self.dependency_graph[dependent].add(prerequisite)
        if prerequisite not in self.reverse_dependency_graph:
            self.reverse_dependency_graph[prerequisite] = set()
        self.reverse_dependency_graph[prerequisite].add(dependent)
        self.execution_order = []

    def register_dependencies(self, dependencies: List[IndicatorDependency]
        ) ->None:
        """
        Register multiple dependencies at once
        
        Args:
            dependencies: List of dependency objects
        """
        for dep in dependencies:
            for prerequisite in dep.required_indicators:
                self.register_dependency(dep.dependent_indicator, prerequisite)

    def get_dependent_indicators(self, indicator_name: str) ->Set[str]:
        """
        Get all indicators that depend on the specified indicator
        
        Args:
            indicator_name: Name of the indicator
            
        Returns:
            Set of indicator names that depend on the specified indicator
        """
        return self.reverse_dependency_graph.get(indicator_name, set())

    def get_prerequisites(self, indicator_name: str) ->Set[str]:
        """
        Get all prerequisites for the specified indicator
        
        Args:
            indicator_name: Name of the indicator
            
        Returns:
            Set of indicator names that are prerequisites for the specified indicator
        """
        return self.dependency_graph.get(indicator_name, set())

    @with_exception_handling
    def visualize_dependencies(self, output_file: str=None) ->None:
        """
        Visualize indicator dependencies as a graph
        
        Args:
            output_file: Optional file path to save the visualization
        """
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            G = nx.DiGraph()
            for dependent, prerequisites in self.dependency_graph.items():
                for prerequisite in prerequisites:
                    G.add_edge(prerequisite, dependent)
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True, node_color='lightblue',
                node_size=1500, arrowsize=15, font_size=10)
            if output_file:
                plt.savefig(output_file)
            else:
                plt.show()
            plt.close()
        except ImportError:
            self.logger.error(
                'Visualization requires networkx and matplotlib packages')


def create_standard_indicators(window_sizes: Dict[str, List[int]]=None,
    price_types: Dict[str, str]=None) ->List[EnhancedIndicatorSpec]:
    """
    Create a standard set of indicators with appropriate priorities and dependencies
    
    Args:
        window_sizes: Dictionary mapping indicator types to list of window sizes
        price_types: Dictionary mapping indicator types to price column names
        
    Returns:
        List of EnhancedIndicatorSpec objects
    """
    if window_sizes is None:
        window_sizes = {'sma': [5, 10, 20, 50, 200], 'ema': [5, 13, 21, 55],
            'rsi': [14], 'bb': [20], 'macd': [(12, 26, 9)], 'atr': [14]}
    if price_types is None:
        price_types = {'sma': 'close', 'ema': 'close', 'rsi': 'close', 'bb':
            'close', 'macd': 'close', 'atr': None}
    specs = []
    for window in window_sizes.get('sma', []):
        indicator = IndicatorState(f'SMA_{window}', window=window,
            price_type=price_types['sma'])
        priority = (IndicatorPriority.HIGH if window <= 20 else
            IndicatorPriority.MEDIUM)
        specs.append(EnhancedIndicatorSpec(indicator=indicator, priority=
            priority))
    for window in window_sizes.get('ema', []):
        indicator = IndicatorState(f'EMA_{window}', window=window,
            price_type=price_types['ema'])
        priority = (IndicatorPriority.HIGH if window <= 13 else
            IndicatorPriority.MEDIUM)
        specs.append(EnhancedIndicatorSpec(indicator=indicator, priority=
            priority))
    for period in window_sizes.get('rsi', []):
        indicator = RSIState(f'RSI_{period}', period=period, price_type=
            price_types['rsi'])
        specs.append(EnhancedIndicatorSpec(indicator=indicator, priority=
            IndicatorPriority.MEDIUM))
    for period in window_sizes.get('bb', []):
        indicator = BollingerBandsState(f'BB_{period}', window=period,
            price_type=price_types['bb'])
        specs.append(EnhancedIndicatorSpec(indicator=indicator, priority=
            IndicatorPriority.MEDIUM, dependencies=[f'SMA_{period}']))
    for fast, slow, signal in window_sizes.get('macd', []):
        indicator = MACDState(f'MACD_{fast}_{slow}_{signal}', fast_period=
            fast, slow_period=slow, signal_period=signal, price_type=
            price_types['macd'])
        specs.append(EnhancedIndicatorSpec(indicator=indicator, priority=
            IndicatorPriority.MEDIUM, dependencies=[f'EMA_{fast}',
            f'EMA_{slow}']))
    for period in window_sizes.get('atr', []):
        indicator = ATRState(f'ATR_{period}', period=period)
        specs.append(EnhancedIndicatorSpec(indicator=indicator, priority=
            IndicatorPriority.LOW))
    return specs
