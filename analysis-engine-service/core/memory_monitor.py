"""
Memory Monitoring Module

This module provides memory monitoring functionality for the Analysis Engine Service.
"""
import os
import psutil
import asyncio
from typing import Dict, Optional
from prometheus_client import Gauge, Counter
from analysis_engine.core.logging import get_logger
from analysis_engine.core.config import get_settings
logger = get_logger(__name__)
MEMORY_USAGE = Gauge('memory_usage_bytes', 'Current memory usage in bytes',
    ['type'])
MEMORY_LIMIT = Gauge('memory_limit_bytes', 'Memory limit in bytes', ['type'])
MEMORY_WARNING = Counter('memory_warning_total',
    'Number of memory warnings', ['type'])
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class MemoryMonitor:
    """Monitors memory usage and provides warnings when thresholds are exceeded."""

    def __init__(self):
        """Initialize the memory monitor."""
        self._settings = get_settings()
        self._process = psutil.Process(os.getpid())
        self._warning_threshold = 0.8
        self._critical_threshold = 0.9
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None

    async def start_monitoring(self, interval: int=60) ->None:
        """
        Start memory monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self._monitoring:
            return
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitor_loop(interval))
        logger.info('Memory monitoring started')

    @async_with_exception_handling
    async def stop_monitoring(self) ->None:
        """Stop memory monitoring."""
        if not self._monitoring:
            return
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info('Memory monitoring stopped')

    @async_with_exception_handling
    async def _monitor_loop(self, interval: int) ->None:
        """
        Main monitoring loop.
        
        Args:
            interval: Monitoring interval in seconds
        """
        while self._monitoring:
            try:
                await self._check_memory_usage()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f'Error in memory monitoring loop: {e}')

    @async_with_exception_handling
    async def _check_memory_usage(self) ->None:
        """Check current memory usage and update metrics."""
        try:
            process_memory = self._process.memory_info()
            MEMORY_USAGE.labels(type='process').set(process_memory.rss)
            system_memory = psutil.virtual_memory()
            MEMORY_USAGE.labels(type='system').set(system_memory.used)
            MEMORY_LIMIT.labels(type='system').set(system_memory.total)
            memory_percent = system_memory.percent / 100
            if memory_percent >= self._critical_threshold:
                MEMORY_WARNING.labels(type='critical').inc()
                logger.critical(
                    f'Critical memory usage: {memory_percent:.1%} ({system_memory.used / 1024 / 1024:.0f}MB used)'
                    )
            elif memory_percent >= self._warning_threshold:
                MEMORY_WARNING.labels(type='warning').inc()
                logger.warning(
                    f'High memory usage: {memory_percent:.1%} ({system_memory.used / 1024 / 1024:.0f}MB used)'
                    )
        except Exception as e:
            logger.error(f'Error checking memory usage: {e}')

    @with_resilience('get_memory_stats')
    @with_exception_handling
    def get_memory_stats(self) ->Dict[str, float]:
        """
        Get current memory statistics.
        
        Returns:
            Dict[str, float]: Memory statistics
        """
        try:
            process_memory = self._process.memory_info()
            system_memory = psutil.virtual_memory()
            return {'process_rss': process_memory.rss / 1024 / 1024,
                'process_vms': process_memory.vms / 1024 / 1024,
                'system_used': system_memory.used / 1024 / 1024,
                'system_total': system_memory.total / 1024 / 1024,
                'system_percent': system_memory.percent}
        except Exception as e:
            logger.error(f'Error getting memory stats: {e}')
            return {}


_memory_monitor: Optional[MemoryMonitor] = None


def get_memory_monitor() ->MemoryMonitor:
    """
    Get the global memory monitor instance.
    
    Returns:
        MemoryMonitor: The global memory monitor instance
    """
    global _memory_monitor
    if _memory_monitor is None:
        _memory_monitor = MemoryMonitor()
    return _memory_monitor
