"""
Performance profiling and optimization service for indicator calculations.

Provides performance analysis, profiling, and optimization capabilities
for indicator calculations, including bottleneck detection and automated tuning.
"""
from typing import Dict, Any, List, Optional, Tuple, Callable
import logging
import time
import cProfile
import pstats
from datetime import datetime, timedelta
import threading
from pathlib import Path
import json
import numpy as np
from concurrent.futures import Future
import pandas as pd
logger = logging.getLogger(__name__)


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class PerformanceProfile:
    """Represents a performance profile of an indicator calculation."""

    def __init__(self, indicator_name: str, execution_time: float, cpu_time:
        float, memory_used: float, data_points: int, parameters: Dict[str, Any]
        ):
    """
      init  .
    
    Args:
        indicator_name: Description of indicator_name
        execution_time: Description of execution_time
        cpu_time: Description of cpu_time
        memory_used: Description of memory_used
        data_points: Description of data_points
        parameters: Description of parameters
        Any]: Description of Any]
    
    """

        self.indicator_name = indicator_name
        self.execution_time = execution_time
        self.cpu_time = cpu_time
        self.memory_used = memory_used
        self.data_points = data_points
        self.parameters = parameters
        self.timestamp = datetime.utcnow()

    def to_dict(self) ->Dict[str, Any]:
        """Convert profile to dictionary format."""
        return {'indicator_name': self.indicator_name, 'execution_time':
            self.execution_time, 'cpu_time': self.cpu_time, 'memory_used':
            self.memory_used, 'data_points': self.data_points, 'parameters':
            self.parameters, 'points_per_second': self.data_points / self.
            execution_time if self.execution_time > 0 else 0, 'timestamp':
            self.timestamp.isoformat()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) ->'PerformanceProfile':
        """Create profile from dictionary format."""
        profile = cls(indicator_name=data['indicator_name'], execution_time
            =data['execution_time'], cpu_time=data['cpu_time'], memory_used
            =data['memory_used'], data_points=data['data_points'],
            parameters=data['parameters'])
        profile.timestamp = datetime.fromisoformat(data['timestamp'])
        return profile


class PerformanceOptimizer:
    """Analyzes and optimizes indicator calculation performance."""

    def __init__(self, profile_dir: str='profiles'):
    """
      init  .
    
    Args:
        profile_dir: Description of profile_dir
    
    """

        self.profile_dir = Path(profile_dir)
        self.profile_dir.mkdir(exist_ok=True)
        self.profiles: Dict[str, List[PerformanceProfile]] = {}
        self._lock = threading.Lock()

    def profile_calculation(self, indicator_name: str, calc_func: Callable,
        data: pd.DataFrame, parameters: Dict[str, Any]) ->Tuple[Any,
        PerformanceProfile]:
        """
        Profile an indicator calculation.
        
        Args:
            indicator_name: Name of the indicator
            calc_func: Calculation function to profile
            data: Input data for calculation
            parameters: Calculation parameters
            
        Returns:
            Tuple of (calculation result, performance profile)
        """
        profiler = cProfile.Profile()
        start_time = time.time()
        start_memory = self._get_memory_usage()
        result = profiler.runcall(calc_func, data, **parameters)
        end_time = time.time()
        end_memory = self._get_memory_usage()
        stats = pstats.Stats(profiler)
        profile = PerformanceProfile(indicator_name=indicator_name,
            execution_time=end_time - start_time, cpu_time=stats.total_tt,
            memory_used=end_memory - start_memory, data_points=len(data),
            parameters=parameters)
        self._store_profile(profile)
        return result, profile

    def analyze_performance(self, indicator_name: str, time_window:
        Optional[timedelta]=None) ->Dict[str, Any]:
        """
        Analyze performance trends for an indicator.
        
        Args:
            indicator_name: Name of the indicator to analyze
            time_window: Optional time window for analysis
            
        Returns:
            Dictionary containing performance analysis
        """
        with self._lock:
            if indicator_name not in self.profiles:
                return {'message': 'No profiles found for indicator'}
            profiles = self.profiles[indicator_name]
            if time_window:
                cutoff = datetime.utcnow() - time_window
                profiles = [p for p in profiles if p.timestamp >= cutoff]
            if not profiles:
                return {'message': 'No profiles found in time window'}
            execution_times = [p.execution_time for p in profiles]
            memory_usage = [p.memory_used for p in profiles]
            points_per_second = [(p.data_points / p.execution_time if p.
                execution_time > 0 else 0) for p in profiles]
            param_impact = self._analyze_parameter_impact(profiles)
            trend = self._calculate_trend(execution_times)
            return {'sample_size': len(profiles), 'execution_time': {'mean':
                np.mean(execution_times), 'median': np.median(
                execution_times), 'std': np.std(execution_times), 'min':
                min(execution_times), 'max': max(execution_times)},
                'memory_usage': {'mean': np.mean(memory_usage), 'median':
                np.median(memory_usage), 'std': np.std(memory_usage), 'min':
                min(memory_usage), 'max': max(memory_usage)}, 'throughput':
                {'mean': np.mean(points_per_second), 'median': np.median(
                points_per_second), 'std': np.std(points_per_second)},
                'parameter_impact': param_impact, 'trend': {'direction': 
                'improving' if trend < 0 else 'degrading', 'magnitude': abs
                (trend)}, 'bottlenecks': self._detect_bottlenecks(profiles)}

    def optimize_parameters(self, indicator_name: str, current_params: Dict
        [str, Any]) ->Dict[str, Any]:
        """
        Suggest optimized parameters based on performance history.
        
        Args:
            indicator_name: Name of the indicator
            current_params: Current parameter values
            
        Returns:
            Dictionary of suggested parameter values
        """
        analysis = self.analyze_performance(indicator_name)
        if 'message' in analysis:
            return current_params
        param_impact = analysis['parameter_impact']
        optimized = current_params.copy()
        for param, impact in param_impact.items():
            if param in optimized:
                if impact['correlation'] > 0.5:
                    optimized[param] = max(impact['optimal_range'][0], 
                        optimized[param] * 0.8)
                elif impact['correlation'] < -0.5:
                    optimized[param] = min(impact['optimal_range'][1], 
                        optimized[param] * 1.2)
        return optimized

    @with_exception_handling
    def _store_profile(self, profile: PerformanceProfile) ->None:
        """Store a performance profile."""
        with self._lock:
            if profile.indicator_name not in self.profiles:
                self.profiles[profile.indicator_name] = []
            self.profiles[profile.indicator_name].append(profile)
            profile_file = self.profile_dir / f'{profile.indicator_name}.json'
            try:
                if profile_file.exists():
                    with open(profile_file, 'r') as f:
                        profiles = json.load(f)
                else:
                    profiles = []
                profiles.append(profile.to_dict())
                with open(profile_file, 'w') as f:
                    json.dump(profiles, f, indent=2)
            except Exception as e:
                logger.error(f'Error saving profile: {str(e)}')

    def _analyze_parameter_impact(self, profiles: List[PerformanceProfile]
        ) ->Dict[str, Dict[str, Any]]:
        """Analyze the impact of parameters on performance."""
        if not profiles:
            return {}
        param_names = set()
        for profile in profiles:
            param_names.update(profile.parameters.keys())
        results = {}
        for param in param_names:
            values = []
            times = []
            for profile in profiles:
                if param in profile.parameters:
                    values.append(profile.parameters[param])
                    times.append(profile.execution_time)
            if not values:
                continue
            correlation = np.corrcoef(values, times)[0, 1]
            sorted_data = sorted(zip(values, times), key=lambda x: x[1])
            optimal_values = [x[0] for x in sorted_data[:max(3, len(
                sorted_data) // 3)]]
            results[param] = {'correlation': correlation, 'optimal_range':
                (min(optimal_values), max(optimal_values)), 'impact_score':
                abs(correlation)}
        return results

    def _calculate_trend(self, values: List[float]) ->float:
        """Calculate performance trend."""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        return np.polyfit(x, values, 1)[0]

    def _detect_bottlenecks(self, profiles: List[PerformanceProfile]) ->List[
        Dict[str, Any]]:
        """Detect performance bottlenecks."""
        if not profiles:
            return []
        bottlenecks = []
        times = [p.execution_time for p in profiles]
        mean_time = np.mean(times)
        std_time = np.std(times)
        if std_time > mean_time * 0.5:
            bottlenecks.append({'type': 'variability', 'metric':
                'execution_time', 'severity': 'high', 'description':
                'High variability in execution times'})
        memory_trend = self._calculate_trend([p.memory_used for p in profiles])
        if memory_trend > 0:
            bottlenecks.append({'type': 'growth', 'metric': 'memory',
                'severity': 'medium', 'description':
                'Increasing memory usage trend'})
        time_trend = self._calculate_trend(times)
        if time_trend > 0:
            bottlenecks.append({'type': 'degradation', 'metric':
                'execution_time', 'severity': 'high', 'description':
                'Performance degrading over time'})
        return bottlenecks

    def _get_memory_usage(self) ->float:
        """Get current memory usage in MB."""
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024


class PerformanceMonitor:
    """Monitors and tracks indicator calculation performance."""

    def __init__(self):
    """
      init  .
    
    """

        self.optimizer = PerformanceOptimizer()
        self.active_calculations: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def start_calculation(self, calc_id: str, indicator_name: str,
        parameters: Dict[str, Any]) ->None:
        """Record the start of a calculation."""
        with self._lock:
            self.active_calculations[calc_id] = {'indicator':
                indicator_name, 'parameters': parameters, 'start_time':
                time.time(), 'start_memory': self._get_memory_usage()}

    def end_calculation(self, calc_id: str, data_points: int) ->Optional[
        PerformanceProfile]:
        """
        Record the end of a calculation.
        
        Args:
            calc_id: Calculation identifier
            data_points: Number of data points processed
            
        Returns:
            Optional performance profile for the calculation
        """
        with self._lock:
            if calc_id not in self.active_calculations:
                return None
            info = self.active_calculations.pop(calc_id)
            end_time = time.time()
            end_memory = self._get_memory_usage()
            profile = PerformanceProfile(indicator_name=info['indicator'],
                execution_time=end_time - info['start_time'], cpu_time=
                end_time - info['start_time'], memory_used=end_memory -
                info['start_memory'], data_points=data_points, parameters=
                info['parameters'])
            self.optimizer.optimize_parameters(info['indicator'], info[
                'parameters'])
            return profile

    def _get_memory_usage(self) ->float:
        """Get current memory usage in MB."""
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
