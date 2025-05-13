"""
Performance monitoring integration for trading gateway service.
"""
from typing import Dict, Any, Optional
import logging
from datetime import datetime
from core_foundations.monitoring.critical_component_monitor import CriticalComponentMonitor
logger = logging.getLogger(__name__)


from trading_gateway_service.resilience.utils import (
    with_broker_api_resilience,
    with_market_data_resilience,
    with_order_execution_resilience,
    with_risk_management_resilience,
    with_database_resilience
)

class TradingGatewayMonitoring:
    """
    Integrates performance monitoring for the trading gateway service.
    Tracks latency and performance metrics for critical trading operations.
    """

    def __init__(self, base_dir: str='monitoring/trading_gateway'):
    """
      init  .
    
    Args:
        base_dir: Description of base_dir
    
    """

        self.monitor = CriticalComponentMonitor(base_dir=base_dir, config={
            'response_time_threshold': 0.1, 'error_rate_threshold': 0.001})
        self.track_order_submission = self.monitor.track_operation(
            'trading_gateway', 'submit_order')
        self.track_market_data = self.monitor.track_operation('trading_gateway'
            , 'process_market_data')
        self.track_risk_check = self.monitor.track_operation('trading_gateway',
            'risk_check')

    @with_broker_api_resilience('get_metrics')
    def get_metrics(self) ->Dict[str, Any]:
        """Get current performance metrics."""
        return self.monitor.get_component_metrics('trading_gateway')

    @with_broker_api_resilience('get_health_status')
    def get_health_status(self) ->Dict[str, Any]:
        """Get component health status."""
        metrics = self.get_metrics()
        operations = metrics.get('operations', {})
        status = {'healthy': True, 'issues': []}
        if 'submit_order' in operations:
            stats = operations['submit_order'].get('statistics', {})
            if stats.get('p95', 0) > 0.1:
                status['healthy'] = False
                status['issues'].append(
                    f"High order submission latency: {stats['p95'] * 1000:.1f}ms"
                    )
        for op_name, op_stats in operations.items():
            error_rate = op_stats.get('error_count', 0) / op_stats.get(
                'total_requests', 1)
            if error_rate > 0.001:
                status['healthy'] = False
                status['issues'].append(
                    f'High error rate in {op_name}: {error_rate * 100:.2f}%')
        return status
