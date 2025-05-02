"""
Central performance monitoring configuration and initialization for critical components.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta
import threading
import time
from core_foundations.performance.multi_level_cache import MultiLevelCache
from core_foundations.performance.adaptive_resource_manager import AdaptiveResourceManager
from core_foundations.performance.performance_optimizer import (
    PerformanceOptimizer,
    PerformanceMetrics
)

logger = logging.getLogger(__name__)

class CriticalComponentMonitor:
    """
    Monitors performance of critical system components.
    Tracks:
    - Trading Gateway performance
    - Analysis Engine latency
    - Feature Store calculations
    - ML model inference times
    - Database query performance
    """

    def __init__(
        self,
        base_dir: str = "monitoring",
        config: Optional[Dict[str, Any]] = None
    ):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Initialize core monitoring components
        self.resource_manager = AdaptiveResourceManager(config)
        self.cache = MultiLevelCache(
            cache_dir=str(self.base_dir / "cache"),
            l1_max_size_mb=1024,  # 1GB L1 cache
            l2_max_size_gb=10,    # 10GB L2 cache
            l3_ttl_hours=48       # 2 days L3 TTL
        )
        self.performance_metrics = PerformanceMetrics()
        
        # Component-specific metrics
        self.components = {
            'trading_gateway': PerformanceOptimizer(
                resource_manager=self.resource_manager,
                cache_manager=self.cache,
                config={'response_time_threshold': 0.1}  # 100ms max latency for trading
            ),
            'analysis_engine': PerformanceOptimizer(
                resource_manager=self.resource_manager,
                cache_manager=self.cache,
                config={'response_time_threshold': 1.0}  # 1s threshold for analysis
            ),
            'feature_store': PerformanceOptimizer(
                resource_manager=self.resource_manager,
                cache_manager=self.cache,
                config={'response_time_threshold': 0.5}  # 500ms for feature calculation
            ),
            'ml_inference': PerformanceOptimizer(
                resource_manager=self.resource_manager,
                cache_manager=self.cache,
                config={'response_time_threshold': 0.2}  # 200ms for ML inference
            ),
            'database': PerformanceOptimizer(
                resource_manager=self.resource_manager,
                cache_manager=self.cache,
                config={'response_time_threshold': 0.3}  # 300ms for DB queries
            )
        }

        # Start monitoring thread
        self._start_monitoring()

    def _start_monitoring(self):
        """Start the monitoring thread for all components."""
        def monitor():
            while True:
                try:
                    self._collect_metrics()
                    self._analyze_performance()
                    time.sleep(10)  # Check every 10 seconds
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    time.sleep(30)  # Back off on error

        threading.Thread(target=monitor, daemon=True).start()

    def _collect_metrics(self):
        """Collect metrics from all monitored components."""
        system_status = self.resource_manager.get_system_status()
        cache_metrics = self.cache.get_metrics()

        # Log system-wide metrics
        logger.info(f"System Status - CPU: {system_status['resources']['cpu_avg']:.1f}%, "
                   f"Memory: {system_status['resources']['memory_avg']:.1f}%")
        
        # Log cache performance
        for level, stats in cache_metrics.items():
            logger.info(f"Cache {level} - Hit Rate: {stats['hit_rate']:.1f}%, "
                       f"Hits: {stats['hits']}, Misses: {stats['misses']}")

    def _analyze_performance(self):
        """Analyze performance metrics and trigger optimizations if needed."""
        for component_name, optimizer in self.components.items():
            report = optimizer.get_performance_report()
            
            # Check for performance issues
            for operation, stats in report.get('operations', {}).items():
                if stats.get('statistics', {}).get('p95', 0) > optimizer.response_time_threshold:
                    logger.warning(
                        f"{component_name} operation {operation} exceeded threshold: "
                        f"{stats['statistics']['p95']:.3f}s"
                    )

    def track_operation(self, component: str, operation: str):
        """
        Decorator to track operation performance.
        
        Usage:
            @monitor.track_operation('trading_gateway', 'submit_order')
            def submit_order():
                ...
        """
        if component not in self.components:
            raise ValueError(f"Unknown component: {component}")
            
        optimizer = self.components[component]
        return optimizer.wrap_operation(operation)

    def get_component_metrics(self, component: str) -> Dict[str, Any]:
        """Get detailed metrics for a specific component."""
        if component not in self.components:
            raise ValueError(f"Unknown component: {component}")
            
        optimizer = self.components[component]
        return optimizer.get_performance_report()

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system performance status."""
        return {
            'system': self.resource_manager.get_system_status(),
            'cache': self.cache.get_metrics(),
            'components': {
                name: optimizer.get_performance_report()
                for name, optimizer in self.components.items()
            }
        }
