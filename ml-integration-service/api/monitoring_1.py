"""
Standardized Monitoring Module for ML Integration Service.

This module provides standardized monitoring capabilities for the ML Integration Service,
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
http_requests_total = Counter('ml_integration_http_requests_total',
    'Total number of HTTP requests', ['method', 'endpoint', 'status'],
    registry=metrics_registry)
http_request_duration_seconds = Histogram(
    'ml_integration_http_request_duration_seconds',
    'HTTP request duration in seconds', ['method', 'endpoint'], registry=
    metrics_registry)
database_query_duration_seconds = Histogram(
    'ml_integration_database_query_duration_seconds',
    'Database query duration in seconds', ['operation', 'table'], registry=
    metrics_registry)
service_client_request_duration_seconds = Histogram(
    'ml_integration_service_client_request_duration_seconds',
    'Service client request duration in seconds', ['service', 'method'],
    registry=metrics_registry)
model_prediction_duration_seconds = Histogram(
    'ml_integration_model_prediction_duration_seconds',
    'Model prediction duration in seconds', ['model_name', 'model_version'],
    registry=metrics_registry)
model_predictions_total = Counter('ml_integration_model_predictions_total',
    'Total number of model predictions', ['model_name', 'model_version'],
    registry=metrics_registry)
model_errors_total = Counter('ml_integration_model_errors_total',
    'Total number of model errors', ['model_name', 'model_version',
    'error_type'], registry=metrics_registry)
model_performance_score = Gauge('ml_integration_model_performance_score',
    'Model performance score', ['model_name', 'model_version', 'metric'],
    registry=metrics_registry)
feature_importance = Gauge('ml_integration_feature_importance',
    'Feature importance', ['model_name', 'model_version', 'feature'],
    registry=metrics_registry)


from ml_integration_service.error.exceptions_bridge import (
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
def track_model_prediction(model_name: str, model_version: str):
    """
    Decorator for tracking model prediction metrics.

    Args:
        model_name: Model name
        model_version: Model version

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
                model_predictions_total.labels(model_name=model_name,
                    model_version=model_version).inc()
                return result
            except Exception as e:
                error_type = type(e).__name__
                model_errors_total.labels(model_name=model_name,
                    model_version=model_version, error_type=error_type).inc()
                raise
            finally:
                duration = time.time() - start_time
                model_prediction_duration_seconds.labels(model_name=
                    model_name, model_version=model_version).observe(duration)
        return wrapper
    return decorator


@contextmanager
@with_exception_handling
def track_model_operation(model_name: str, model_version: str, operation: str):
    """
    Context manager for tracking model operation metrics.

    Args:
        model_name: Model name
        model_version: Model version
        operation: Operation name

    Yields:
        None
    """
    start_time = time.time()
    try:
        yield
    except Exception as e:
        error_type = type(e).__name__
        model_errors_total.labels(model_name=model_name, model_version=
            model_version, error_type=error_type).inc()
        raise
    finally:
        duration = time.time() - start_time
        model_prediction_duration_seconds.labels(model_name=model_name,
            model_version=model_version).observe(duration)


def update_model_performance(model_name: str, model_version: str, metric:
    str, value: float) ->None:
    """
    Update the model performance score.

    Args:
        model_name: Model name
        model_version: Model version
        metric: Performance metric name
        value: Performance value
    """
    model_performance_score.labels(model_name=model_name, model_version=
        model_version, metric=metric).set(value)


def update_feature_importance(model_name: str, model_version: str, feature:
    str, importance: float) ->None:
    """
    Update the feature importance.

    Args:
        model_name: Model name
        model_version: Model version
        feature: Feature name
        importance: Importance value
    """
    feature_importance.labels(model_name=model_name, model_version=
        model_version, feature=feature).set(importance)


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
