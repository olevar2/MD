"""
Indicator Monitoring Module.

This module provides centralized monitoring and reporting for the Feature Store service,
tracking performance, errors, dependencies, and usage patterns.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import pandas as pd

logger = logging.getLogger(__name__)

class IndicatorMonitoring:
    """Centralized monitoring for indicator calculations and usage"""
    
    def __init__(self):
        """Initialize monitoring system"""
        self.performance_metrics = {}
        self.error_tracking = {}
        self.usage_stats = {}
        self.dependency_violations = []
        self._initialize_metrics()
        
    def _initialize_metrics(self) -> None:
        """Initialize metric containers"""
        self.performance_metrics = {
            'calculation_times': {},
            'cache_hits': {},
            'data_volume': {}
        }
        
        self.error_tracking = {
            'calculation_errors': {},
            'validation_errors': {},
            'dependency_errors': {}
        }
        
        self.usage_stats = {
            'indicator_usage': {},
            'parameter_distributions': {},
            'timeframe_distribution': {}
        }
        
    def record_calculation(self, indicator: str, duration: float, success: bool) -> None:
        """Record a calculation attempt"""
        if indicator not in self.performance_metrics['calculation_times']:
            self.performance_metrics['calculation_times'][indicator] = []
        
        self.performance_metrics['calculation_times'][indicator].append({
            'timestamp': datetime.now(),
            'duration': duration,
            'success': success
        })
        
        # Keep only last 1000 calculations
        if len(self.performance_metrics['calculation_times'][indicator]) > 1000:
            self.performance_metrics['calculation_times'][indicator] = \
                self.performance_metrics['calculation_times'][indicator][-1000:]
                
    def record_error(self, error_type: str, indicator: str, error: Exception) -> None:
        """Record an error occurrence"""
        if indicator not in self.error_tracking[error_type]:
            self.error_tracking[error_type][indicator] = []
            
        self.error_tracking[error_type][indicator].append({
            'timestamp': datetime.now(),
            'error_class': error.__class__.__name__,
            'message': str(error)
        })
        
    def record_usage(self, indicator: str, parameters: Dict[str, Any]) -> None:
        """Record indicator usage"""
        if indicator not in self.usage_stats['indicator_usage']:
            self.usage_stats['indicator_usage'][indicator] = 0
        self.usage_stats['indicator_usage'][indicator] += 1
        
        # Track parameter distributions
        if indicator not in self.usage_stats['parameter_distributions']:
            self.usage_stats['parameter_distributions'][indicator] = {}
            
        for param, value in parameters.items():
            if param not in self.usage_stats['parameter_distributions'][indicator]:
                self.usage_stats['parameter_distributions'][indicator][param] = {}
            
            str_value = str(value)
            if str_value not in self.usage_stats['parameter_distributions'][indicator][param]:
                self.usage_stats['parameter_distributions'][indicator][param][str_value] = 0
            self.usage_stats['parameter_distributions'][indicator][param][str_value] += 1
            
    def record_dependency_violation(self, dependent: str, missing_prerequisite: str) -> None:
        """Record a dependency violation"""
        self.dependency_violations.append({
            'timestamp': datetime.now(),
            'dependent': dependent,
            'missing_prerequisite': missing_prerequisite
        })
        
    def get_monitoring_report(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive monitoring report
        
        Args:
            time_window: Optional time window to limit report to
            
        Returns:
            Dict containing monitoring statistics
        """
        if time_window is None:
            time_window = timedelta(hours=24)
            
        cutoff = datetime.now() - time_window
        
        report = {
            'time_window': str(time_window),
            'generated_at': datetime.now().isoformat(),
            'performance': self._get_performance_stats(cutoff),
            'errors': self._get_error_stats(cutoff),
            'usage': self._get_usage_stats(),
            'dependencies': {
                'violations': [v for v in self.dependency_violations 
                             if v['timestamp'] > cutoff],
                'violation_count': len([v for v in self.dependency_violations 
                                     if v['timestamp'] > cutoff])
            }
        }
        
        return report
        
    def _get_performance_stats(self, cutoff: datetime) -> Dict[str, Any]:
        """Generate performance statistics"""
        stats = {}
        for indicator, calculations in self.performance_metrics['calculation_times'].items():
            recent_calcs = [c for c in calculations if c['timestamp'] > cutoff]
            if recent_calcs:
                durations = [c['duration'] for c in recent_calcs]
                success_rate = sum(1 for c in recent_calcs if c['success']) / len(recent_calcs)
                
                stats[indicator] = {
                    'count': len(recent_calcs),
                    'avg_duration': sum(durations) / len(durations),
                    'max_duration': max(durations),
                    'min_duration': min(durations),
                    'success_rate': success_rate
                }
        return stats
        
    def _get_error_stats(self, cutoff: datetime) -> Dict[str, Any]:
        """Generate error statistics"""
        stats = {}
        for error_type, indicators in self.error_tracking.items():
            stats[error_type] = {}
            for indicator, errors in indicators.items():
                recent_errors = [e for e in errors if e['timestamp'] > cutoff]
                if recent_errors:
                    stats[error_type][indicator] = {
                        'count': len(recent_errors),
                        'most_recent': recent_errors[-1],
                        'error_classes': self._count_error_classes(recent_errors)
                    }
        return stats
        
    def _get_usage_stats(self) -> Dict[str, Any]:
        """Generate usage statistics"""
        return {
            'total_usage': self.usage_stats['indicator_usage'],
            'parameter_distributions': self.usage_stats['parameter_distributions']
        }
        
    @staticmethod
    def _count_error_classes(errors: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count occurrences of each error class"""
        counts = {}
        for error in errors:
            error_class = error['error_class']
            if error_class not in counts:
                counts[error_class] = 0
            counts[error_class] += 1
        return counts
