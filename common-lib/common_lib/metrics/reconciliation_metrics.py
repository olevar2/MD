"""
Metrics collection for data reconciliation processes.

This module provides functions for collecting metrics about data reconciliation processes,
such as number of reconciliations, discrepancies, resolutions, and durations.
"""

import time
from typing import Dict, Any, Optional
from functools import wraps
import logging
from prometheus_client import Counter, Histogram, Gauge

# Initialize logger
logger = logging.getLogger(__name__)

# Define metrics
RECONCILIATION_COUNT = Counter(
    'reconciliation_total',
    'Total number of reconciliation processes',
    ['service', 'reconciliation_type', 'status']
)

DISCREPANCY_COUNT = Counter(
    'reconciliation_discrepancies_total',
    'Total number of discrepancies found during reconciliation',
    ['service', 'reconciliation_type', 'severity']
)

RESOLUTION_COUNT = Counter(
    'reconciliation_resolutions_total',
    'Total number of discrepancies resolved during reconciliation',
    ['service', 'reconciliation_type', 'strategy']
)

RECONCILIATION_DURATION = Histogram(
    'reconciliation_duration_seconds',
    'Duration of reconciliation processes in seconds',
    ['service', 'reconciliation_type'],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0)
)

ACTIVE_RECONCILIATIONS = Gauge(
    'reconciliation_active',
    'Number of currently active reconciliation processes',
    ['service', 'reconciliation_type']
)


def track_reconciliation(service: str, reconciliation_type: str):
    """
    Decorator for tracking reconciliation processes.
    
    Args:
        service: Name of the service performing the reconciliation
        reconciliation_type: Type of reconciliation being performed
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Increment active reconciliations
            ACTIVE_RECONCILIATIONS.labels(service=service, reconciliation_type=reconciliation_type).inc()
            
            # Record start time
            start_time = time.time()
            
            try:
                # Call the original function
                result = await func(*args, **kwargs)
                
                # Record metrics
                RECONCILIATION_COUNT.labels(
                    service=service,
                    reconciliation_type=reconciliation_type,
                    status=result.status.name
                ).inc()
                
                DISCREPANCY_COUNT.labels(
                    service=service,
                    reconciliation_type=reconciliation_type,
                    severity="ALL"
                ).inc(result.discrepancy_count)
                
                RESOLUTION_COUNT.labels(
                    service=service,
                    reconciliation_type=reconciliation_type,
                    strategy=result.strategy.name
                ).inc(result.resolution_count)
                
                # Record duration
                duration = time.time() - start_time
                RECONCILIATION_DURATION.labels(
                    service=service,
                    reconciliation_type=reconciliation_type
                ).observe(duration)
                
                return result
            except Exception as e:
                # Record failure
                RECONCILIATION_COUNT.labels(
                    service=service,
                    reconciliation_type=reconciliation_type,
                    status="FAILED"
                ).inc()
                
                # Re-raise the exception
                raise
            finally:
                # Decrement active reconciliations
                ACTIVE_RECONCILIATIONS.labels(service=service, reconciliation_type=reconciliation_type).dec()
        
        return wrapper
    
    return decorator


def record_discrepancy(
    service: str,
    reconciliation_type: str,
    severity: str,
    count: int = 1
):
    """
    Record discrepancies found during reconciliation.
    
    Args:
        service: Name of the service performing the reconciliation
        reconciliation_type: Type of reconciliation being performed
        severity: Severity of the discrepancies
        count: Number of discrepancies to record
    """
    DISCREPANCY_COUNT.labels(
        service=service,
        reconciliation_type=reconciliation_type,
        severity=severity
    ).inc(count)


def record_resolution(
    service: str,
    reconciliation_type: str,
    strategy: str,
    count: int = 1
):
    """
    Record resolutions applied during reconciliation.
    
    Args:
        service: Name of the service performing the reconciliation
        reconciliation_type: Type of reconciliation being performed
        strategy: Strategy used for resolution
        count: Number of resolutions to record
    """
    RESOLUTION_COUNT.labels(
        service=service,
        reconciliation_type=reconciliation_type,
        strategy=strategy
    ).inc(count)


def get_reconciliation_metrics(
    service: Optional[str] = None,
    reconciliation_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get metrics for reconciliation processes.
    
    Args:
        service: Optional filter by service
        reconciliation_type: Optional filter by reconciliation type
        
    Returns:
        Dictionary of metrics
    """
    # This is a simplified implementation that would need to be expanded
    # to actually query the Prometheus metrics
    return {
        "reconciliation_count": {
            "total": 0,
            "completed": 0,
            "failed": 0
        },
        "discrepancy_count": {
            "total": 0,
            "by_severity": {}
        },
        "resolution_count": {
            "total": 0,
            "by_strategy": {}
        },
        "duration": {
            "average": 0,
            "p50": 0,
            "p95": 0,
            "p99": 0
        },
        "active": 0
    }
