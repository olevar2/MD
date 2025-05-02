"""
Performance monitoring integration for trading gateway service.
"""

from typing import Dict, Any, Optional
import logging
from datetime import datetime
from core_foundations.monitoring.critical_component_monitor import CriticalComponentMonitor

logger = logging.getLogger(__name__)

class TradingGatewayMonitoring:
    """
    Integrates performance monitoring for the trading gateway service.
    Tracks latency and performance metrics for critical trading operations.
    """
    
    def __init__(self, base_dir: str = "monitoring/trading_gateway"):
        self.monitor = CriticalComponentMonitor(
            base_dir=base_dir,
            config={
                'response_time_threshold': 0.1,  # 100ms max latency for trading operations
                'error_rate_threshold': 0.001    # 0.1% max error rate
            }
        )
        
        # Register critical operations
        self.track_order_submission = self.monitor.track_operation(
            'trading_gateway', 'submit_order')
        self.track_market_data = self.monitor.track_operation(
            'trading_gateway', 'process_market_data')
        self.track_risk_check = self.monitor.track_operation(
            'trading_gateway', 'risk_check')
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.monitor.get_component_metrics('trading_gateway')
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get component health status."""
        metrics = self.get_metrics()
        operations = metrics.get('operations', {})
        
        status = {
            'healthy': True,
            'issues': []
        }
        
        # Check order submission performance
        if 'submit_order' in operations:
            stats = operations['submit_order'].get('statistics', {})
            if stats.get('p95', 0) > 0.1:  # 100ms threshold
                status['healthy'] = False
                status['issues'].append(
                    f"High order submission latency: {stats['p95']*1000:.1f}ms"
                )
                
        # Check error rates
        for op_name, op_stats in operations.items():
            error_rate = op_stats.get('error_count', 0) / (op_stats.get('total_requests', 1))
            if error_rate > 0.001:  # 0.1% threshold
                status['healthy'] = False
                status['issues'].append(
                    f"High error rate in {op_name}: {error_rate*100:.2f}%"
                )
                
        return status
