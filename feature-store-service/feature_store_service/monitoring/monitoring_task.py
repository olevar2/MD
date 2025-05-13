"""
Monitoring Task Module.

This module provides scheduled monitoring tasks to collect and report system health metrics.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import json
import os
logger = logging.getLogger(__name__)


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class MonitoringTask:
    """Scheduled task for monitoring system health"""

    def __init__(self, indicator_service, monitoring_interval: int=300,
        report_dir: Optional[str]=None):
        """Initialize monitoring task"""
        self.indicator_service = indicator_service
        self.monitoring_interval = monitoring_interval
        self.report_dir = report_dir or os.path.join(os.getcwd(),
            'monitoring_reports')
        self._ensure_report_dir()

    def _ensure_report_dir(self) ->None:
        """Ensure the report directory exists"""
        if not os.path.exists(self.report_dir):
            os.makedirs(self.report_dir)

    @async_with_exception_handling
    async def start(self) ->None:
        """Start the monitoring task"""
        logger.info('Starting monitoring task')
        while True:
            try:
                await self._collect_and_report()
            except Exception as e:
                logger.error(f'Error in monitoring task: {str(e)}')
            await asyncio.sleep(self.monitoring_interval)

    async def _collect_and_report(self) ->None:
        """Collect and save monitoring report"""
        monitoring_report = (self.indicator_service.monitoring.
            get_monitoring_report(time_window=timedelta(minutes=self.
            monitoring_interval)))
        error_metrics = self.indicator_service.get_error_metrics()
        dependency_info = (self.indicator_service.dependency_tracker.
            export_dependencies())
        full_report = {'timestamp': datetime.now().isoformat(),
            'monitoring': monitoring_report, 'errors': error_metrics,
            'dependencies': dependency_info}
        filename = (
            f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
        filepath = os.path.join(self.report_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(full_report, f, indent=2)
        logger.info(f'Saved monitoring report: {filepath}')
        self._check_alerts(full_report)

    def _check_alerts(self, report: Dict[str, Any]) ->None:
        """Check monitoring data for alert conditions"""
        for indicator, stats in report['monitoring']['errors'].get(
            'calculation_errors', {}).items():
            if stats['count'] > 10:
                logger.warning(
                    f"High error rate for indicator {indicator}: {stats['count']} errors"
                    )
        for indicator, stats in report['monitoring']['performance'].items():
            if stats['success_rate'] < 0.95:
                logger.warning(
                    f"Low success rate for indicator {indicator}: {stats['success_rate']:.2%}"
                    )
            if stats['avg_duration'] > 1.0:
                logger.warning(
                    f"High calculation time for indicator {indicator}: {stats['avg_duration']:.2f}s"
                    )
