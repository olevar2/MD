"""
Standardized Monitoring Module for Data Pipeline Service.

This module provides standardized monitoring capabilities for the Data Pipeline Service,
including health checks, metrics collection, and distributed tracing.
"""
import time
import functools
import logging
from typing import Dict, Any, Optional, Callable, List, Union
from contextlib import contextmanager
from fastapi import FastAPI, Request, Response
from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry
import prometheus_client
logger = logging.getLogger(__name__)
metrics_registry = CollectorRegistry()
http_requests_total = Counter('data_pipeline_http_requests_total',
    'Total number of HTTP requests', ['method', 'endpoint', 'status'],
    registry=metrics_registry)
http_request_duration_seconds = Histogram(
    'data_pipeline_http_request_duration_seconds',
    'HTTP request duration in seconds', ['method', 'endpoint'], registry=
    metrics_registry)
database_query_duration_seconds = Histogram(
    'data_pipeline_database_query_duration_seconds',
    'Database query duration in seconds', ['operation', 'table'], registry=
    metrics_registry)
service_client_request_duration_seconds = Histogram(
    'data_pipeline_service_client_request_duration_seconds',
    'Service client request duration in seconds', ['service', 'method'],
    registry=metrics_registry)
pipeline_processing_duration_seconds = Histogram(
    'data_pipeline_processing_duration_seconds',
    'Pipeline processing duration in seconds', ['pipeline_name', 'stage'],
    registry=metrics_registry)
pipeline_records_processed = Counter('data_pipeline_records_processed_total',
    'Total number of records processed', ['pipeline_name', 'stage'],
    registry=metrics_registry)
pipeline_errors_total = Counter('data_pipeline_errors_total',
    'Total number of pipeline errors', ['pipeline_name', 'stage',
    'error_type'], registry=metrics_registry)
pipeline_backlog_size = Gauge('data_pipeline_backlog_size',
    'Current size of the pipeline backlog', ['pipeline_name'], registry=
    metrics_registry)
data_quality_score = Gauge('data_pipeline_data_quality_score',
    'Data quality score', ['pipeline_name', 'metric'], registry=
    metrics_registry)


from data_pipeline_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class HealthCheck:
    """Health check for the service."""

    def __init__(self):
        """Initialize the health check."""
        self.checks = {}

    def add_check(self, name: str, check_func: Callable[[], bool],
        description: str='') ->None:
        """
        Add a health check.

        Args:
            name: Name of the health check
            check_func: Function that returns True if the check passes, False otherwise
            description: Description of the health check
        """
        self.checks[name] = {'check_func': check_func, 'description':
            description}

    @with_exception_handling
    def check(self) ->Dict[str, Any]:
        """
        Run all health checks.

        Returns:
            Health check results
        """
        results = {'status': 'ok', 'timestamp': time.time(), 'checks': {}}
        for name, check_info in self.checks.items():
            try:
                check_passed = check_info['check_func']()
                results['checks'][name] = {'status': 'ok' if check_passed else
                    'error', 'description': check_info['description']}
                if not check_passed:
                    results['status'] = 'error'
            except Exception as e:
                logger.exception(f'Health check {name} failed: {str(e)}')
                results['checks'][name] = {'status': 'error', 'description':
                    check_info['description'], 'error': str(e)}
                results['status'] = 'error'
        return results


health_check = HealthCheck()


def register_health_check(name: str, check_func: Callable[[], bool],
    description: str='') ->None:
    """
    Register a health check.

    Args:
        name: Name of the health check
        check_func: Function that returns True if the check passes, False otherwise
        description: Description of the health check
    """
    health_check.add_check(name, check_func, description)


def start_metrics_collection() ->None:
    """Start metrics collection."""
    logger.info('Starting metrics collection')


def stop_metrics_collection() ->None:
    """Stop metrics collection."""
    logger.info('Stopping metrics collection')


@with_exception_handling
def track_database_query(operation: str, table: Optional[str]=None):
    """
    Decorator for tracking database query metrics.

    Args:
        operation: Database operation (e.g., select, insert, update, delete)
        table: Database table

    Returns:
        Decorator function
    """

    @with_exception_handling
    def decorator(func):
    """
    Decorator.
    
    Args:
        func: Description of func
    
    """


        @functools.wraps(func)
        @async_with_exception_handling
        async def wrapper(*args, **kwargs):
    """
    Wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

            table_name = table or kwargs.get('table', 'unknown')
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                database_query_duration_seconds.labels(operation=operation,
                    table=table_name).observe(duration)
        return wrapper
    return decorator


@with_exception_handling
def track_service_client_request(service: str, method: str):
    """
    Decorator for tracking service client request metrics.

    Args:
        service: Service name
        method: HTTP method

    Returns:
        Decorator function
    """

    @with_exception_handling
    def decorator(func):
    """
    Decorator.
    
    Args:
        func: Description of func
    
    """


        @functools.wraps(func)
        @async_with_exception_handling
        async def wrapper(*args, **kwargs):
    """
    Wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                service_client_request_duration_seconds.labels(service=
                    service, method=method).observe(duration)
        return wrapper
    return decorator


@with_exception_handling
def track_pipeline_processing(pipeline_name: str, stage: str):
    """
    Decorator for tracking pipeline processing metrics.

    Args:
        pipeline_name: Pipeline name
        stage: Pipeline stage

    Returns:
        Decorator function
    """

    @with_exception_handling
    def decorator(func):
    """
    Decorator.
    
    Args:
        func: Description of func
    
    """


        @functools.wraps(func)
        @async_with_exception_handling
        async def wrapper(*args, **kwargs):
    """
    Wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                if isinstance(result, list):
                    pipeline_records_processed.labels(pipeline_name=
                        pipeline_name, stage=stage).inc(len(result))
                return result
            except Exception as e:
                error_type = type(e).__name__
                pipeline_errors_total.labels(pipeline_name=pipeline_name,
                    stage=stage, error_type=error_type).inc()
                raise
            finally:
                duration = time.time() - start_time
                pipeline_processing_duration_seconds.labels(pipeline_name=
                    pipeline_name, stage=stage).observe(duration)
        return wrapper
    return decorator


@contextmanager
@with_exception_handling
def track_pipeline_stage(pipeline_name: str, stage: str):
    """
    Context manager for tracking pipeline stage metrics.

    Args:
        pipeline_name: Pipeline name
        stage: Pipeline stage

    Yields:
        None
    """
    start_time = time.time()
    try:
        yield
    except Exception as e:
        error_type = type(e).__name__
        pipeline_errors_total.labels(pipeline_name=pipeline_name, stage=
            stage, error_type=error_type).inc()
        raise
    finally:
        duration = time.time() - start_time
        pipeline_processing_duration_seconds.labels(pipeline_name=
            pipeline_name, stage=stage).observe(duration)


def update_backlog_size(pipeline_name: str, size: int) ->None:
    """
    Update the pipeline backlog size.

    Args:
        pipeline_name: Pipeline name
        size: Backlog size
    """
    pipeline_backlog_size.labels(pipeline_name=pipeline_name).set(size)


def update_data_quality_score(pipeline_name: str, metric: str, score: float
    ) ->None:
    """
    Update the data quality score.

    Args:
        pipeline_name: Pipeline name
        metric: Quality metric name
        score: Quality score
    """
    data_quality_score.labels(pipeline_name=pipeline_name, metric=metric).set(
        score)


async def metrics_middleware(request: Request, call_next):
    """
    Middleware for tracking HTTP request metrics.

    Args:
        request: FastAPI request
        call_next: Next middleware or route handler

    Returns:
        Response
    """
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    method = request.method
    endpoint = request.url.path
    status = response.status_code
    http_requests_total.labels(method=method, endpoint=endpoint, status=status
        ).inc()
    http_request_duration_seconds.labels(method=method, endpoint=endpoint
        ).observe(duration)
    return response


def setup_monitoring(app: FastAPI) ->None:
    """
    Set up monitoring for a FastAPI application.

    Args:
        app: FastAPI application
    """
    app.middleware('http')(metrics_middleware)

    @app.get('/health')
    async def health():
    """
    Health.
    
    """

        return health_check.check()

    @app.get('/ready')
    async def ready():
    """
    Ready.
    
    """

        return {'status': 'ok'}

    @app.get('/metrics')
    async def metrics():
    """
    Metrics.
    
    """

        return Response(content=prometheus_client.generate_latest(
            metrics_registry), media_type='text/plain')
    register_health_check(name='default', check_func=lambda : True,
        description='Default health check')
    start_metrics_collection()

    @app.on_event('shutdown')
    async def shutdown_event():
    """
    Shutdown event.
    
    """

        stop_metrics_collection()
