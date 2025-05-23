"""
Effectiveness-Based Parameter Optimizer

This module provides functionality to automatically optimize technical indicator parameters
based on effectiveness metrics. It implements adaptive parameter tuning to optimize
trading performance by adjusting technical indicators based on their historical effectiveness.
"""
import logging
import numpy as np
import json
import os
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
import threading
import requests


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class EffectivenessBasedOptimizer:
    """
    Automatically optimizes indicator parameters based on effectiveness metrics.
    
    This class consumes tool effectiveness metrics and adjusts indicator parameters
    to improve trading performance over time.
    """

    def __init__(self, metrics_api_url: str, optimization_interval_sec: int
        =3600, min_sample_size: int=50, learning_rate: float=0.05,
        storage_path: str=None):
        """
        Initialize the effectiveness-based optimizer
        
        Args:
            metrics_api_url: URL of the API that provides tool effectiveness metrics
            optimization_interval_sec: How often to run optimization (seconds)
            min_sample_size: Minimum number of signals required before optimizing
            learning_rate: How aggressively to adjust parameters (0.01-0.1 recommended)
            storage_path: Directory to store optimization state
        """
        self.metrics_api_url = metrics_api_url
        self.optimization_interval_sec = optimization_interval_sec
        self.min_sample_size = min_sample_size
        self.learning_rate = learning_rate
        self.storage_path = storage_path or os.path.join(os.getcwd(),
            'optimization_state')
        os.makedirs(self.storage_path, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.current_parameters = {}
        self.parameter_bounds = {}
        self.optimization_history = {}
        self.registered_indicators = {}
        self.indicator_effectiveness_weights = {}
        self._load_state()
        self.stop_event = threading.Event()
        self.optimization_thread = threading.Thread(target=self.
            _run_optimization_loop, daemon=True)
        self.optimization_thread.start()

    def register_indicator(self, indicator_name: str, current_params: Dict[
        str, Any], param_bounds: Dict[str, Tuple[float, float]],
        param_types: Dict[str, str], update_callback: Callable[[Dict[str,
        Any]], None]):
        """
        Register a technical indicator for optimization
        
        Args:
            indicator_name: Name of the indicator (e.g., 'SMA', 'RSI')
            current_params: Current parameter values (e.g., {'window': 14})
            param_bounds: Min/max bounds for each parameter (e.g., {'window': (5, 50)})
            param_types: Type of each parameter ('int', 'float', 'bool')
            update_callback: Function to call when parameters are updated
        """
        self.logger.info(
            f'Registering indicator {indicator_name} for optimization')
        self.registered_indicators[indicator_name] = {'name':
            indicator_name, 'params': current_params.copy(), 'bounds':
            param_bounds, 'types': param_types, 'callback': update_callback,
            'last_updated': datetime.now(), 'update_count': 0}
        if indicator_name not in self.current_parameters:
            self.current_parameters[indicator_name] = current_params.copy()
            self.parameter_bounds[indicator_name] = param_bounds
            if indicator_name not in self.optimization_history:
                self.optimization_history[indicator_name] = []
        update_callback(self.current_parameters[indicator_name])

    @with_exception_handling
    def _run_optimization_loop(self):
        """Run the optimization loop in the background thread"""
        while not self.stop_event.is_set():
            try:
                self._perform_optimization_cycle()
            except Exception as e:
                self.logger.error(f'Error in optimization cycle: {str(e)}',
                    exc_info=True)
            self.stop_event.wait(self.optimization_interval_sec)

    @with_exception_handling
    def _perform_optimization_cycle(self):
        """Perform a single optimization cycle"""
        self.logger.info('Starting effectiveness-based optimization cycle')
        effectiveness_metrics = self._fetch_effectiveness_metrics()
        if not effectiveness_metrics:
            self.logger.warning(
                'No effectiveness metrics available, skipping optimization')
            return
        optimized_count = 0
        for indicator_name, indicator_config in self.registered_indicators.items(
            ):
            try:
                indicator_metrics = effectiveness_metrics.get(indicator_name)
                if not indicator_metrics:
                    indicator_metrics = effectiveness_metrics.get(
                        f'rt_{indicator_name}')
                if not indicator_metrics:
                    self.logger.debug(
                        f'No metrics found for indicator {indicator_name}')
                    continue
                sample_size = indicator_metrics.get('sample_size', 0)
                if sample_size < self.min_sample_size:
                    self.logger.debug(
                        f'Insufficient samples for {indicator_name}: {sample_size} < {self.min_sample_size}'
                        )
                    continue
                win_rate = indicator_metrics.get('win_rate', 0.5)
                profit_factor = indicator_metrics.get('profit_factor', 1.0)
                expected_payoff = indicator_metrics.get('expected_payoff', 0.0)
                composite_score = (win_rate - 0.5) * 2.0 + (profit_factor - 1.0
                    ) * 0.5 + expected_payoff * 0.1
                regime_performance = indicator_metrics.get('by_regime', {})
                current_regime = self._detect_current_market_regime()
                if current_regime in regime_performance:
                    regime_win_rate = regime_performance[current_regime].get(
                        'win_rate', win_rate)
                    composite_score = composite_score * 0.7 + (regime_win_rate
                         - 0.5) * 2.0 * 0.3
                self._optimize_indicator_parameters(indicator_name,
                    indicator_config, composite_score)
                optimized_count += 1
            except Exception as e:
                self.logger.error(
                    f'Error optimizing {indicator_name}: {str(e)}',
                    exc_info=True)
        self._save_state()
        self.logger.info(
            f'Completed optimization cycle. Optimized {optimized_count} indicators'
            )

    @with_exception_handling
    def _fetch_effectiveness_metrics(self) ->Dict[str, Any]:
        """Fetch effectiveness metrics from the API"""
        try:
            response = requests.get(
                f'{self.metrics_api_url}/api/v1/tool-effectiveness/metrics',
                timeout=10)
            if response.status_code == 200:
                data = response.json()
                metrics = {}
                for tool_name, tool_data in data.get('tools', {}).items():
                    tool_metrics = {}
                    win_rate = tool_data.get('metrics', {}).get('win_rate', {}
                        ).get('value')
                    if win_rate is not None:
                        tool_metrics['win_rate'] = win_rate
                    profit_factor = tool_data.get('metrics', {}).get(
                        'profit_factor', {}).get('value')
                    if profit_factor is not None:
                        tool_metrics['profit_factor'] = profit_factor
                    expected_payoff = tool_data.get('metrics', {}).get(
                        'expected_payoff', {}).get('value')
                    if expected_payoff is not None:
                        tool_metrics['expected_payoff'] = expected_payoff
                    sample_size = tool_data.get('sample_size', 0)
                    tool_metrics['sample_size'] = sample_size
                    if 'by_regime' in tool_data:
                        tool_metrics['by_regime'] = {}
                        for regime, regime_data in tool_data['by_regime'
                            ].items():
                            regime_win_rate = regime_data.get('metrics', {}
                                ).get('win_rate', {}).get('value')
                            if regime_win_rate is not None:
                                if 'by_regime' not in tool_metrics:
                                    tool_metrics['by_regime'] = {}
                                tool_metrics['by_regime'][regime] = {'win_rate'
                                    : regime_win_rate}
                    metrics[tool_name] = tool_metrics
                return metrics
            self.logger.warning(
                f'Failed to fetch metrics: HTTP {response.status_code}')
            return {}
        except Exception as e:
            self.logger.error(f'Error fetching effectiveness metrics: {str(e)}'
                , exc_info=True)
            return {}

    def _optimize_indicator_parameters(self, indicator_name: str,
        indicator_config: Dict[str, Any], performance_score: float):
        """
        Optimize parameters for an indicator based on its performance
        
        Args:
            indicator_name: Name of the indicator
            indicator_config: Indicator configuration
            performance_score: Composite performance score (-1 to +1 scale)
        """
        current_params = self.current_parameters[indicator_name]
        param_bounds = self.parameter_bounds[indicator_name]
        param_types = indicator_config_manager.get('types', {})
        adjustment_factor = self.learning_rate * (1.0 - min(1.0, max(0.0, (
            performance_score + 1.0) / 2.0)))
        new_params = {}
        for param_name, current_value in current_params.items():
            if param_name not in param_bounds:
                new_params[param_name] = current_value
                continue
            min_value, max_value = param_bounds[param_name]
            param_range = max_value - min_value
            adjustment = np.random.normal(0, adjustment_factor) * param_range
            new_value = current_value + adjustment
            new_value = max(min_value, min(max_value, new_value))
            if param_types.get(param_name) == 'int':
                new_value = int(round(new_value))
            elif param_types.get(param_name) == 'bool':
                new_value = bool(new_value > 0.5)
            new_params[param_name] = new_value
        self.optimization_history[indicator_name].append({'timestamp':
            datetime.now().isoformat(), 'old_params': current_params.copy(),
            'new_params': new_params.copy(), 'performance_score':
            performance_score, 'adjustment_factor': adjustment_factor})
        if len(self.optimization_history[indicator_name]) > 100:
            self.optimization_history[indicator_name
                ] = self.optimization_history[indicator_name][-100:]
        self.current_parameters[indicator_name] = new_params
        indicator_config['callback'](new_params)
        indicator_config['last_updated'] = datetime.now()
        indicator_config['update_count'] += 1
        self.logger.info(
            f'Optimized {indicator_name} parameters: {new_params} (score: {performance_score:.4f})'
            )

    def _detect_current_market_regime(self) ->str:
        """
        Detect the current market regime
        
        Returns:
            String representing the current market regime
            (e.g., 'trending_up', 'trending_down', 'ranging', 'volatile')
        """
        return 'unknown'

    @with_exception_handling
    def _load_state(self):
        """Load optimizer state from disk"""
        state_file = os.path.join(self.storage_path,
            'effectiveness_optimizer_state.json')
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                self.current_parameters = state.get('current_parameters', {})
                self.parameter_bounds = state.get('parameter_bounds', {})
                self.optimization_history = state.get('optimization_history',
                    {})
                self.indicator_effectiveness_weights = state.get(
                    'indicator_effectiveness_weights', {})
                self.logger.info(
                    f'Loaded optimizer state with {len(self.current_parameters)} indicators'
                    )
            except Exception as e:
                self.logger.error(f'Error loading optimizer state: {str(e)}',
                    exc_info=True)

    @with_exception_handling
    def _save_state(self):
        """Save optimizer state to disk"""
        state_file = os.path.join(self.storage_path,
            'effectiveness_optimizer_state.json')
        try:
            state = {'current_parameters': self.current_parameters,
                'parameter_bounds': self.parameter_bounds,
                'optimization_history': self.optimization_history,
                'indicator_effectiveness_weights': self.
                indicator_effectiveness_weights, 'last_saved': datetime.now
                ().isoformat()}
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
            current_hour = datetime.now().strftime('%Y%m%d%H')
            backup_file = os.path.join(self.storage_path,
                f'effectiveness_optimizer_state_{current_hour}.json')
            if not os.path.exists(backup_file):
                with open(backup_file, 'w') as f:
                    json.dump(state, f, indent=2)
        except Exception as e:
            self.logger.error(f'Error saving optimizer state: {str(e)}',
                exc_info=True)

    def get_optimization_stats(self, indicator_name: str=None) ->Dict[str, Any
        ]:
        """
        Get optimization statistics
        
        Args:
            indicator_name: Optional name of indicator to get stats for
                           (if None, returns stats for all indicators)
        
        Returns:
            Dictionary of optimization statistics
        """
        stats = {'total_indicators': len(self.registered_indicators),
            'last_optimization': None, 'indicators': {}}
        if indicator_name:
            if indicator_name in self.registered_indicators:
                indicator_config = self.registered_indicators[indicator_name]
                stats['indicators'][indicator_name] = {'current_params':
                    self.current_parameters.get(indicator_name, {}),
                    'update_count': indicator_config_manager.get('update_count', 0),
                    'last_updated': indicator_config.get('last_updated', ''
                    ).isoformat() if indicator_config_manager.get('last_updated') else
                    None, 'history_entries': len(self.optimization_history.
                    get(indicator_name, []))}
                history = self.optimization_history.get(indicator_name, [])
                if history:
                    stats['indicators'][indicator_name]['last_optimization'
                        ] = history[-1]
            else:
                stats['error'] = f'Indicator {indicator_name} not found'
        else:
            for ind_name, ind_config in self.registered_indicators.items():
                stats['indicators'][ind_name] = {'current_params': self.
                    current_parameters.get(ind_name, {}), 'update_count':
                    ind_config_manager.get('update_count', 0), 'last_updated': 
                    ind_config_manager.get('last_updated', '').isoformat() if
                    ind_config_manager.get('last_updated') else None}
        return stats

    def stop(self):
        """Stop the optimizer background thread"""
        self.stop_event.set()
        self._save_state()
        self.optimization_thread.join(timeout=5)
        self.logger.info('Stopped effectiveness-based optimizer')


class AdaptiveIndicatorWrapper:
    """
    Wrapper for technical indicators that enables automatic parameter optimization
    based on effectiveness metrics.
    
    This class wraps an indicator calculator and integrates it with the
    effectiveness-based optimizer.
    """

    def __init__(self, optimizer: EffectivenessBasedOptimizer,
        indicator_name: str, initial_params: Dict[str, Any], param_bounds:
        Dict[str, Tuple[float, float]], param_types: Dict[str, str],
        calculator_factory: Callable[..., Any]):
        """
        Initialize the adaptive indicator wrapper
        
        Args:
            optimizer: The effectiveness-based optimizer instance
            indicator_name: Name of the indicator (e.g., 'SMA', 'RSI')
            initial_params: Initial parameter values
            param_bounds: Min/max bounds for each parameter
            param_types: Type of each parameter ('int', 'float', 'bool')
            calculator_factory: Function that creates the indicator calculator instance
        """
        self.optimizer = optimizer
        self.indicator_name = indicator_name
        self.initial_params = initial_params
        self.param_bounds = param_bounds
        self.param_types = param_types
        self.calculator_factory = calculator_factory
        self.current_params = initial_params.copy()
        self.calculator = self.calculator_factory(**self.current_params)
        self.optimizer.register_indicator(indicator_name=indicator_name,
            current_params=initial_params, param_bounds=param_bounds,
            param_types=param_types, update_callback=self.
            _parameter_update_callback)

    def _parameter_update_callback(self, new_params: Dict[str, Any]):
        """
        Callback for when the optimizer updates parameters
        
        Args:
            new_params: The new parameter values
        """
        self.current_params = new_params.copy()
        self.calculator = self.calculator_factory(**self.current_params)

    def update(self, *args, **kwargs):
        """
        Update the indicator with new data
        
        This method delegates to the calculator's update method
        """
        return self.calculator.update(*args, **kwargs)

    def get_value(self, *args, **kwargs):
        """
        Get the current indicator value
        
        This method delegates to the calculator's get_value method
        """
        return self.calculator.get_value(*args, **kwargs)

    def get_current_params(self) ->Dict[str, Any]:
        """Get the current indicator parameters"""
        return self.current_params.copy()
