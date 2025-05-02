"""
Performance monitoring integration for analysis engine service.
Tracks latency and performance of market analysis operations.
"""

from typing import Dict, Any, Optional
import logging
from datetime import datetime
from core_foundations.monitoring.critical_component_monitor import CriticalComponentMonitor

logger = logging.getLogger(__name__)

class AnalysisEngineMonitoring:
    """
    Integrates performance monitoring for the analysis engine service.
    Tracks performance metrics for market analysis operations.
    """
    
    def __init__(self, base_dir: str = "monitoring/analysis_engine"):
        self.monitor = CriticalComponentMonitor(
            base_dir=base_dir,
            config={
                'response_time_threshold': 1.0,  # 1s max latency for analysis
                'error_rate_threshold': 0.01     # 1% max error rate
            }
        )
        
        # Register critical operations
        self.track_market_analysis = self.monitor.track_operation(
            'analysis_engine', 'analyze_market')
        self.track_pattern_detection = self.monitor.track_operation(
            'analysis_engine', 'detect_patterns')
        self.track_regime_analysis = self.monitor.track_operation(
            'analysis_engine', 'analyze_regime')
        self.track_indicator_calculation = self.monitor.track_operation(
            'analysis_engine', 'calculate_indicators')
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.monitor.get_component_metrics('analysis_engine')
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get component health status."""
        metrics = self.get_metrics()
        operations = metrics.get('operations', {})
        
        status = {
            'healthy': True,
            'issues': []
        }
        
        # Check analysis operation performance
        for op_name, op_stats in operations.items():
            stats = op_stats.get('statistics', {})
            
            # Check latency thresholds
            p95_latency = stats.get('p95', 0)
            if op_name == 'analyze_market' and p95_latency > 1.0:  # 1s threshold
                status['healthy'] = False
                status['issues'].append(
                    f"High market analysis latency: {p95_latency*1000:.1f}ms"
                )
            elif op_name == 'detect_patterns' and p95_latency > 0.5:  # 500ms threshold
                status['healthy'] = False
                status['issues'].append(
                    f"High pattern detection latency: {p95_latency*1000:.1f}ms"
                )
            elif op_name == 'analyze_regime' and p95_latency > 2.0:  # 2s threshold
                status['healthy'] = False
                status['issues'].append(
                    f"High regime analysis latency: {p95_latency*1000:.1f}ms"
                )
                
            # Check error rates
            error_rate = op_stats.get('error_count', 0) / (op_stats.get('total_requests', 1))
            if error_rate > 0.01:  # 1% threshold
                status['healthy'] = False
                status['issues'].append(
                    f"High error rate in {op_name}: {error_rate*100:.2f}%"
                )
                
        return status
