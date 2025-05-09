"""
Standardized Monitoring API for Analysis Engine Service.

This module provides standardized API endpoints for monitoring the service,
including async performance metrics and memory usage.

All endpoints follow the platform's standardized API design patterns.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, Query, HTTPException, status, Request
from pydantic import BaseModel, Field

from analysis_engine.core.monitoring.async_performance_monitor import get_async_monitor
from analysis_engine.core.monitoring.memory_monitor import get_memory_monitor
from analysis_engine.core.exceptions_bridge import (
    ForexTradingPlatformError,
    AnalysisError,
    MonitoringError,
    get_correlation_id_from_request
)
from analysis_engine.monitoring.structured_logging import get_structured_logger

# --- API Request/Response Models ---

class OperationMetric(BaseModel):
    """Model for an operation metric"""
    operation: str
    count: int
    avg_duration_ms: float
    min_duration_ms: float
    max_duration_ms: float
    p95_duration_ms: float
    p99_duration_ms: float
    error_count: int
    error_rate: float
    last_execution_time: Optional[datetime] = None

class AsyncPerformanceResponse(BaseModel):
    """Response model for async performance metrics"""
    metrics: Dict[str, OperationMetric]
    operation_count: int
    timestamp: datetime

class MemoryMetrics(BaseModel):
    """Model for memory metrics"""
    total_allocated_mb: float
    available_mb: float
    used_mb: float
    percent_used: float
    peak_used_mb: float
    gc_collections: Dict[str, int]

class MemoryResponse(BaseModel):
    """Response model for memory metrics"""
    metrics: MemoryMetrics
    timestamp: datetime

class ReportResponse(BaseModel):
    """Response model for triggering a report"""
    status: str
    message: str
    timestamp: datetime

# --- API Router Setup ---

router = APIRouter(
    prefix="/v1/analysis/monitoring",
    tags=["Monitoring"]
)

# Setup logging
logger = get_structured_logger(__name__)

# --- API Endpoints ---

@router.get(
    "/async-performance",
    response_model=AsyncPerformanceResponse,
    summary="Get async performance metrics",
    description="Get performance metrics for asynchronous operations."
)
async def get_async_performance_metrics(
    request_obj: Request,
    operation: Optional[str] = Query(None, description="Filter by operation name")
):
    """
    Get async performance metrics.

    This endpoint returns performance metrics for asynchronous operations,
    including average duration, error rates, and percentiles.
    """
    # Get correlation ID from request or generate a new one
    correlation_id = get_correlation_id_from_request(request_obj)

    try:
        monitor = get_async_monitor()
        metrics = monitor.get_metrics(operation)

        if operation and not metrics:
            # Log not found
            logger.warning(
                f"No metrics found for operation: {operation}",
                extra={
                    "correlation_id": correlation_id,
                    "operation": operation
                }
            )

            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No metrics found for operation: {operation}"
            )

        # Log successful retrieval
        logger.info(
            f"Retrieved async performance metrics for {1 if operation else len(metrics)} operations",
            extra={
                "correlation_id": correlation_id,
                "operation": operation,
                "operation_count": 1 if operation else len(metrics)
            }
        )

        return AsyncPerformanceResponse(
            metrics=metrics,
            operation_count=1 if operation else len(metrics),
            timestamp=datetime.utcnow()
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise

    except Exception as e:
        # Log error
        logger.error(
            f"Error retrieving async performance metrics: {str(e)}",
            extra={"correlation_id": correlation_id},
            exc_info=True
        )

        # Raise standardized error
        raise MonitoringError(
            message=f"Error retrieving async performance metrics: {str(e)}",
            correlation_id=correlation_id
        )

@router.get(
    "/memory",
    response_model=MemoryResponse,
    summary="Get memory metrics",
    description="Get memory usage metrics for the service."
)
async def get_memory_metrics(
    request_obj: Request
):
    """
    Get memory usage metrics.

    This endpoint returns memory usage metrics for the service,
    including total allocated memory, used memory, and garbage collection statistics.
    """
    # Get correlation ID from request or generate a new one
    correlation_id = get_correlation_id_from_request(request_obj)

    try:
        monitor = get_memory_monitor()
        metrics = await monitor.get_memory_usage()

        # Log successful retrieval
        logger.info(
            "Retrieved memory metrics",
            extra={
                "correlation_id": correlation_id,
                "total_allocated_mb": metrics.get("total_allocated_mb"),
                "used_mb": metrics.get("used_mb"),
                "percent_used": metrics.get("percent_used")
            }
        )

        return MemoryResponse(
            metrics=metrics,
            timestamp=datetime.utcnow()
        )

    except Exception as e:
        # Log error
        logger.error(
            f"Error retrieving memory metrics: {str(e)}",
            extra={"correlation_id": correlation_id},
            exc_info=True
        )

        # Raise standardized error
        raise MonitoringError(
            message=f"Error retrieving memory metrics: {str(e)}",
            correlation_id=correlation_id
        )

@router.post(
    "/async-performance/report",
    response_model=ReportResponse,
    summary="Trigger async performance report",
    description="Trigger an immediate async performance report."
)
async def trigger_async_performance_report(
    request_obj: Request
):
    """
    Trigger an immediate async performance report.

    This endpoint triggers the generation of an async performance report,
    which is logged to the configured logging system.
    """
    # Get correlation ID from request or generate a new one
    correlation_id = get_correlation_id_from_request(request_obj)

    try:
        monitor = get_async_monitor()
        monitor._log_metrics_report()

        # Log successful trigger
        logger.info(
            "Triggered async performance report",
            extra={
                "correlation_id": correlation_id
            }
        )

        return ReportResponse(
            status="success",
            message="Async performance report triggered",
            timestamp=datetime.utcnow()
        )

    except Exception as e:
        # Log error
        logger.error(
            f"Error triggering async performance report: {str(e)}",
            extra={"correlation_id": correlation_id},
            exc_info=True
        )

        # Raise standardized error
        raise MonitoringError(
            message=f"Error triggering async performance report: {str(e)}",
            correlation_id=correlation_id
        )

@router.get(
    "/health",
    summary="Get service health",
    description="Get detailed health status of the service and its dependencies."
)
async def get_service_health(
    request_obj: Request
):
    """
    Get detailed health status of the service and its dependencies.

    This endpoint returns the health status of the service and its dependencies,
    including database connections, external services, and resource usage.
    """
    # Get correlation ID from request or generate a new one
    correlation_id = get_correlation_id_from_request(request_obj)

    try:
        # Get memory metrics for health check
        monitor = get_memory_monitor()
        memory_metrics = await monitor.get_memory_usage()

        # Check if memory usage is critical
        memory_status = "healthy"
        if memory_metrics.get("percent_used", 0) > 90:
            memory_status = "critical"
        elif memory_metrics.get("percent_used", 0) > 75:
            memory_status = "warning"

        # Get async performance metrics for health check
        async_monitor = get_async_monitor()
        async_metrics = async_monitor.get_metrics()

        # Check if any operations have high error rates
        operations_with_high_error_rate = []
        for op_name, op_metrics in async_metrics.items():
            if op_metrics.get("error_rate", 0) > 0.05:  # 5% error rate threshold
                operations_with_high_error_rate.append(op_name)

        async_status = "healthy"
        if operations_with_high_error_rate:
            async_status = "warning"
            if any(async_metrics[op].get("error_rate", 0) > 0.2 for op in operations_with_high_error_rate):
                async_status = "critical"  # 20% error rate threshold for critical

        # Overall health status
        overall_status = "healthy"
        if memory_status == "critical" or async_status == "critical":
            overall_status = "critical"
        elif memory_status == "warning" or async_status == "warning":
            overall_status = "warning"

        # Log health check
        logger.info(
            f"Health check: {overall_status}",
            extra={
                "correlation_id": correlation_id,
                "memory_status": memory_status,
                "async_status": async_status,
                "memory_percent_used": memory_metrics.get("percent_used", 0),
                "operations_with_high_error_rate": operations_with_high_error_rate
            }
        )

        return {
            "status": overall_status,
            "timestamp": datetime.utcnow(),
            "components": {
                "memory": {
                    "status": memory_status,
                    "metrics": {
                        "percent_used": memory_metrics.get("percent_used", 0),
                        "used_mb": memory_metrics.get("used_mb", 0),
                        "total_allocated_mb": memory_metrics.get("total_allocated_mb", 0)
                    }
                },
                "async_operations": {
                    "status": async_status,
                    "metrics": {
                        "operation_count": len(async_metrics),
                        "operations_with_high_error_rate": operations_with_high_error_rate
                    }
                }
            }
        }

    except Exception as e:
        # Log error
        logger.error(
            f"Error checking service health: {str(e)}",
            extra={"correlation_id": correlation_id},
            exc_info=True
        )

        # Return degraded status instead of raising an exception
        # This ensures the health check endpoint always returns a response
        return {
            "status": "critical",
            "timestamp": datetime.utcnow(),
            "error": str(e),
            "components": {
                "health_check_system": {
                    "status": "critical",
                    "error": str(e)
                }
            }
        }

# --- Legacy Router for Backward Compatibility ---

legacy_router = APIRouter(
    prefix="/api/v1/monitoring",
    tags=["Monitoring (Legacy)"]
)

# Add legacy endpoints that redirect to the standardized endpoints
# This ensures backward compatibility while encouraging migration to the new endpoints

@legacy_router.get("/async-performance", response_model=Dict[str, Any])
async def legacy_get_async_performance_metrics(
    operation: Optional[str] = Query(None, description="Filter by operation name"),
    request_obj: Request = None
):
    """
    Legacy endpoint for getting async performance metrics.
    Consider migrating to /api/v1/analysis/monitoring/async-performance
    """
    logger.info("Legacy endpoint called - consider migrating to /api/v1/analysis/monitoring/async-performance")
    result = await get_async_performance_metrics(request_obj, operation)
    return result.dict()

@legacy_router.get("/memory", response_model=Dict[str, Any])
async def legacy_get_memory_metrics(
    request_obj: Request = None
):
    """
    Legacy endpoint for getting memory metrics.
    Consider migrating to /api/v1/analysis/monitoring/memory
    """
    logger.info("Legacy endpoint called - consider migrating to /api/v1/analysis/monitoring/memory")
    result = await get_memory_metrics(request_obj)
    return result.dict()

@legacy_router.post("/async-performance/report", response_model=Dict[str, Any])
async def legacy_trigger_async_performance_report(
    request_obj: Request = None
):
    """
    Legacy endpoint for triggering an async performance report.
    Consider migrating to /api/v1/analysis/monitoring/async-performance/report
    """
    logger.info("Legacy endpoint called - consider migrating to /api/v1/analysis/monitoring/async-performance/report")
    result = await trigger_async_performance_report(request_obj)
    return result.dict()

# --- Setup Function for Router Registration ---

def setup_monitoring_routes(app: FastAPI) -> None:
    """
    Set up monitoring routes.

    Args:
        app: FastAPI application
    """
    # Include standardized router
    app.include_router(router, prefix="/api")

    # Include legacy router for backward compatibility
    app.include_router(legacy_router)
