"""
Performance monitoring integration for feature store service.
Tracks latency and performance of feature calculations and data access.
"""

from typing import Dict, Any, Optional
import logging
from datetime import datetime
from core_foundations.monitoring.critical_component_monitor import CriticalComponentMonitor

logger = logging.getLogger(__name__)

class FeatureStoreMonitoring:
    """
    Integrates performance monitoring for the feature store service.
    Tracks performance metrics for feature calculations and data access.
    """
    
    def __init__(self, base_dir: str = "monitoring/feature_store"):
    """
      init  .
    
    Args:
        base_dir: Description of base_dir
    
    """

        self.monitor = CriticalComponentMonitor(
            base_dir=base_dir,
            config={
                'response_time_threshold': 0.5,  # 500ms max latency for feature calculations
                'error_rate_threshold': 0.01     # 1% max error rate
            }
        )
        
        # Register critical operations
        self.track_feature_calculation = self.monitor.track_operation(
            'feature_store', 'calculate_features')
        self.track_data_retrieval = self.monitor.track_operation(
            'feature_store', 'retrieve_data')
        self.track_batch_processing = self.monitor.track_operation(
            'feature_store', 'process_batch')
        self.track_cache_operation = self.monitor.track_operation(
            'feature_store', 'cache_operation')
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.monitor.get_component_metrics('feature_store')
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get component health status."""
        metrics = self.get_metrics()
        operations = metrics.get('operations', {})
        
        status = {
            'healthy': True,
            'issues': []
        }
        
        # Check feature calculation performance
        for op_name, op_stats in operations.items():
            stats = op_stats.get('statistics', {})
            
            # Check latency thresholds
            p95_latency = stats.get('p95', 0)
            if op_name == 'calculate_features' and p95_latency > 0.5:  # 500ms threshold
                status['healthy'] = False
                status['issues'].append(
                    f"High feature calculation latency: {p95_latency*1000:.1f}ms"
                )
            elif op_name == 'retrieve_data' and p95_latency > 0.2:  # 200ms threshold
                status['healthy'] = False
                status['issues'].append(
                    f"High data retrieval latency: {p95_latency*1000:.1f}ms"
                )
            elif op_name == 'process_batch' and p95_latency > 2.0:  # 2s threshold
                status['healthy'] = False
                status['issues'].append(
                    f"High batch processing latency: {p95_latency*1000:.1f}ms"
                )
                
            # Check error rates
            error_rate = op_stats.get('error_count', 0) / (op_stats.get('total_requests', 1))
            if error_rate > 0.01:  # 1% threshold
                status['healthy'] = False
                status['issues'].append(
                    f"High error rate in {op_name}: {error_rate*100:.2f}%"
                )
                
            # Check cache effectiveness if it's a cache operation
            if op_name == 'cache_operation':
                hit_rate = op_stats.get('hit_rate', 0)
                if hit_rate < 0.80:  # 80% minimum cache hit rate
                    status['issues'].append(
                        f"Low cache hit rate: {hit_rate*100:.1f}%"
                    )
                
        return status
