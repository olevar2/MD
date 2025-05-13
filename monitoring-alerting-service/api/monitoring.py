"""
Standardized Monitoring Module for Monitoring Alerting Service

This module provides standardized monitoring and observability that follows the
common-lib pattern for monitoring and metrics collection.
"""
import time
import logging
import functools
import socket
import os
import threading
from typing import Dict, Any, Optional, Callable, List, Union, Type, TypeVar, cast
from contextlib import contextmanager
import asyncio
from fastapi import FastAPI, Request, Response
from prometheus_client import Counter, Histogram, Gauge, Summary, Info, CollectorRegistry, push_to_gateway, start_http_server
from prometheus_client.exposition import basic_auth_handler
import psutil
from common_lib.monitoring import MetricsRegistry, HealthCheck, HealthStatus
from config.standardized_config_1 import get_monitoring_settings, settings
from core.logging_setup_1 import get_logger, get_correlation_id
F = TypeVar('F', bound=Callable[..., Any])
logger = get_logger(__name__)
monitoring_settings = get_monitoring_settings()
metrics_registry = MetricsRegistry(namespace='monitoring_alerting',
    subsystem='service', enable_metrics=monitoring_settings['enable_metrics'])
health_check = HealthCheck(service_name='monitoring-alerting-service',
    version=os.environ.get('SERVICE_VERSION', 'unknown'))
http_requests_total = metrics_registry.counter(name='http_requests_total',
    description='Total number of HTTP requests', labels=['method',
    'endpoint', 'status_code'])
http_request_duration_seconds = metrics_registry.histogram(name=
    'http_request_duration_seconds', description=
    'HTTP request duration in seconds', labels=['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0])
http_request_size_bytes = metrics_registry.histogram(name=
    'http_request_size_bytes', description='HTTP request size in bytes',
    labels=['method', 'endpoint'], buckets=[10, 100, 1000, 10000, 100000, 
    1000000])
http_response_size_bytes = metrics_registry.histogram(name=
    'http_response_size_bytes', description='HTTP response size in bytes',
    labels=['method', 'endpoint'], buckets=[10, 100, 1000, 10000, 100000, 
    1000000])
active_requests = metrics_registry.gauge(name='active_requests',
    description='Number of active HTTP requests', labels=['method', 'endpoint']
    )
system_memory_usage_bytes = metrics_registry.gauge(name=
    'system_memory_usage_bytes', description='System memory usage in bytes',
    labels=['type'])
system_cpu_usage_percent = metrics_registry.gauge(name=
    'system_cpu_usage_percent', description='System CPU usage in percent',
    labels=['cpu'])
process_memory_usage_bytes = metrics_registry.gauge(name=
    'process_memory_usage_bytes', description=
    'Process memory usage in bytes', labels=['type'])
process_cpu_usage_percent = metrics_registry.gauge(name=
    'process_cpu_usage_percent', description='Process CPU usage in percent')
database_connections = metrics_registry.gauge(name='database_connections',
    description='Number of database connections', labels=['state'])
database_query_duration_seconds = metrics_registry.histogram(name=
    'database_query_duration_seconds', description=
    'Database query duration in seconds', labels=['operation'], buckets=[
    0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0])
service_client_requests_total = metrics_registry.counter(name=
    'service_client_requests_total', description=
    'Total number of service client requests', labels=['service', 'method',
    'status_code'])
service_client_request_duration_seconds = metrics_registry.histogram(name=
    'service_client_request_duration_seconds', description=
    'Service client request duration in seconds', labels=['service',
    'method'], buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0])
alert_evaluations_total = metrics_registry.counter(name=
    'alert_evaluations_total', description=
    'Total number of alert evaluations', labels=['alert_name', 'status'])
alert_evaluation_duration_seconds = metrics_registry.histogram(name=
    'alert_evaluation_duration_seconds', description=
    'Alert evaluation duration in seconds', labels=['alert_name'], buckets=
    [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0])
alerts_triggered_total = metrics_registry.counter(name=
    'alerts_triggered_total', description=
    'Total number of alerts triggered', labels=['alert_name', 'severity'])
alerts_resolved_total = metrics_registry.counter(name=
    'alerts_resolved_total', description='Total number of alerts resolved',
    labels=['alert_name', 'severity'])
notifications_sent_total = metrics_registry.counter(name=
    'notifications_sent_total', description=
    'Total number of notifications sent', labels=['channel', 'status'])
notification_send_duration_seconds = metrics_registry.histogram(name=
    'notification_send_duration_seconds', description=
    'Notification send duration in seconds', labels=['channel'], buckets=[
    0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0])
prometheus_query_duration_seconds = metrics_registry.histogram(name=
    'prometheus_query_duration_seconds', description=
    'Prometheus query duration in seconds', labels=['query_type'], buckets=
    [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0])
prometheus_queries_total = metrics_registry.counter(name=
    'prometheus_queries_total', description=
    'Total number of Prometheus queries', labels=['query_type', 'status'])
alertmanager_operations_total = metrics_registry.counter(name=
    'alertmanager_operations_total', description=
    'Total number of Alertmanager operations', labels=['operation', 'status'])
alertmanager_operation_duration_seconds = metrics_registry.histogram(name=
    'alertmanager_operation_duration_seconds', description=
    'Alertmanager operation duration in seconds', labels=['operation'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0])
grafana_operations_total = metrics_registry.counter(name=
    'grafana_operations_total', description=
    'Total number of Grafana operations', labels=['operation', 'status'])
grafana_operation_duration_seconds = metrics_registry.histogram(name=
    'grafana_operation_duration_seconds', description=
    'Grafana operation duration in seconds', labels=['operation'], buckets=
    [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0])
system_metrics_thread = None
system_metrics_stop_event = threading.Event()


from core.exceptions_bridge_1 import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

@with_exception_handling
def collect_system_metrics() ->None:
    """Collect system metrics periodically."""
    while not system_metrics_stop_event.is_set():
        try:
            memory = psutil.virtual_memory()
            system_memory_usage_bytes.labels(type='total').set(memory.total)
            system_memory_usage_bytes.labels(type='available').set(memory.
                available)
            system_memory_usage_bytes.labels(type='used').set(memory.used)
            system_memory_usage_bytes.labels(type='free').set(memory.free)
            for i, percentage in enumerate(psutil.cpu_percent(percpu=True)):
                system_cpu_usage_percent.labels(cpu=f'cpu{i}').set(percentage)
            process = psutil.Process(os.getpid())
            process_memory_info = process.memory_info()
            process_memory_usage_bytes.labels(type='rss').set(
                process_memory_info.rss)
            process_memory_usage_bytes.labels(type='vms').set(
                process_memory_info.vms)
            process_cpu_usage_percent.set(process.cpu_percent(interval=1))
            system_metrics_stop_event.wait(15)
        except Exception as e:
            logger.error(f'Error collecting system metrics: {str(e)}',
                exc_info=True)
            system_metrics_stop_event.wait(60)


def start_metrics_collection() ->None:
    """Start collecting metrics."""
    global system_metrics_thread
    if monitoring_settings['enable_metrics']:
        start_http_server(monitoring_settings['metrics_port'])
        logger.info(
            f"Started metrics server on port {monitoring_settings['metrics_port']}"
            , extra={'metrics_port': monitoring_settings['metrics_port']})
        system_metrics_thread = threading.Thread(target=
            collect_system_metrics, daemon=True)
        system_metrics_thread.start()
        logger.info('Started system metrics collection thread')


def stop_metrics_collection() ->None:
    """Stop collecting metrics."""
    global system_metrics_thread
    if system_metrics_thread is not None:
        system_metrics_stop_event.set()
        system_metrics_thread.join(timeout=5)
        system_metrics_thread = None
        logger.info('Stopped system metrics collection thread')


@with_exception_handling
def setup_monitoring(app: FastAPI) ->None:
    """
    Set up monitoring for the FastAPI application.

    Args:
        app: FastAPI application
    """
    start_metrics_collection()

    @app.get('/health')
    async def health() ->Dict[str, Any]:
        """
        Health check endpoint.

        Returns:
            Health check result
        """
        return health_check.check()

    @app.get('/ready')
    async def ready() ->Dict[str, Any]:
        """
        Readiness check endpoint.

        Returns:
            Readiness check result
        """
        return health_check.check()

    @app.middleware('http')
    @async_with_exception_handling
    async def metrics_middleware(request: Request, call_next: Callable
        ) ->Response:
        """
        Middleware to collect metrics for HTTP requests.

        Args:
            request: HTTP request
            call_next: Next middleware or route handler

        Returns:
            HTTP response
        """
        path = request.url.path
        method = request.method
        if path in ['/health', '/ready', '/metrics']:
            return await call_next(request)
        active_requests.labels(method=method, endpoint=path).inc()
        content_length = request.headers.get('content-length')
        if content_length:
            http_request_size_bytes.labels(method=method, endpoint=path
                ).observe(int(content_length))
        start_time = time.time()
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            active_requests.labels(method=method, endpoint=path).dec()
            raise e
        active_requests.labels(method=method, endpoint=path).dec()
        duration = time.time() - start_time
        http_request_duration_seconds.labels(method=method, endpoint=path
            ).observe(duration)
        http_requests_total.labels(method=method, endpoint=path,
            status_code=status_code).inc()
        response_content_length = response.headers.get('content-length')
        if response_content_length:
            http_response_size_bytes.labels(method=method, endpoint=path
                ).observe(int(response_content_length))
        return response

    @app.on_event('shutdown')
    def shutdown_event() ->None:
        """Shutdown event handler."""
        stop_metrics_collection()


@with_exception_handling
def track_database_query(operation: str) ->Callable[[F], F]:
    """
    Decorator to track database query metrics.

    Args:
        operation: Database operation name

    Returns:
        Decorated function
    """

    @with_exception_handling
    def decorator(func: F) ->F:
    """
    Decorator.
    
    Args:
        func: Description of func
    
    Returns:
        F: Description of return value
    
    """


        @functools.wraps(func)
        @async_with_exception_handling
        async def async_wrapper(*args: Any, **kwargs: Any) ->Any:
    """
    Async wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    Returns:
        Any: Description of return value
    
    """

            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                database_query_duration_seconds.labels(operation=operation
                    ).observe(duration)

        @functools.wraps(func)
        @with_exception_handling
        def sync_wrapper(*args: Any, **kwargs: Any) ->Any:
    """
    Sync wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    Returns:
        Any: Description of return value
    
    """

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                database_query_duration_seconds.labels(operation=operation
                    ).observe(duration)
        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        else:
            return cast(F, sync_wrapper)
    return decorator


@with_exception_handling
def track_service_client_request(service: str, method: str) ->Callable[[F], F]:
    """
    Decorator to track service client request metrics.

    Args:
        service: Service name
        method: HTTP method

    Returns:
        Decorated function
    """

    @with_exception_handling
    def decorator(func: F) ->F:
    """
    Decorator.
    
    Args:
        func: Description of func
    
    Returns:
        F: Description of return value
    
    """


        @functools.wraps(func)
        @async_with_exception_handling
        async def async_wrapper(*args: Any, **kwargs: Any) ->Any:
    """
    Async wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    Returns:
        Any: Description of return value
    
    """

            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                status_code = result.status_code if hasattr(result,
                    'status_code') else 200
                service_client_requests_total.labels(service=service,
                    method=method, status_code=status_code).inc()
                return result
            except Exception as e:
                status_code = getattr(e, 'status_code', 500)
                service_client_requests_total.labels(service=service,
                    method=method, status_code=status_code).inc()
                raise
            finally:
                duration = time.time() - start_time
                service_client_request_duration_seconds.labels(service=
                    service, method=method).observe(duration)

        @functools.wraps(func)
        @with_exception_handling
        def sync_wrapper(*args: Any, **kwargs: Any) ->Any:
    """
    Sync wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    Returns:
        Any: Description of return value
    
    """

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                status_code = result.status_code if hasattr(result,
                    'status_code') else 200
                service_client_requests_total.labels(service=service,
                    method=method, status_code=status_code).inc()
                return result
            except Exception as e:
                status_code = getattr(e, 'status_code', 500)
                service_client_requests_total.labels(service=service,
                    method=method, status_code=status_code).inc()
                raise
            finally:
                duration = time.time() - start_time
                service_client_request_duration_seconds.labels(service=
                    service, method=method).observe(duration)
        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        else:
            return cast(F, sync_wrapper)
    return decorator


@with_exception_handling
def track_alert_evaluation(alert_name: str) ->Callable[[F], F]:
    """
    Decorator to track alert evaluation metrics.

    Args:
        alert_name: Alert name

    Returns:
        Decorated function
    """

    @with_exception_handling
    def decorator(func: F) ->F:
    """
    Decorator.
    
    Args:
        func: Description of func
    
    Returns:
        F: Description of return value
    
    """


        @functools.wraps(func)
        @async_with_exception_handling
        async def async_wrapper(*args: Any, **kwargs: Any) ->Any:
    """
    Async wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    Returns:
        Any: Description of return value
    
    """

            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                status = 'triggered' if result else 'normal'
                alert_evaluations_total.labels(alert_name=alert_name,
                    status=status).inc()
                return result
            except Exception as e:
                alert_evaluations_total.labels(alert_name=alert_name,
                    status='error').inc()
                raise
            finally:
                duration = time.time() - start_time
                alert_evaluation_duration_seconds.labels(alert_name=alert_name
                    ).observe(duration)

        @functools.wraps(func)
        @with_exception_handling
        def sync_wrapper(*args: Any, **kwargs: Any) ->Any:
    """
    Sync wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    Returns:
        Any: Description of return value
    
    """

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                status = 'triggered' if result else 'normal'
                alert_evaluations_total.labels(alert_name=alert_name,
                    status=status).inc()
                return result
            except Exception as e:
                alert_evaluations_total.labels(alert_name=alert_name,
                    status='error').inc()
                raise
            finally:
                duration = time.time() - start_time
                alert_evaluation_duration_seconds.labels(alert_name=alert_name
                    ).observe(duration)
        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        else:
            return cast(F, sync_wrapper)
    return decorator


@with_exception_handling
def track_notification_send(channel: str) ->Callable[[F], F]:
    """
    Decorator to track notification send metrics.

    Args:
        channel: Notification channel

    Returns:
        Decorated function
    """

    @with_exception_handling
    def decorator(func: F) ->F:
    """
    Decorator.
    
    Args:
        func: Description of func
    
    Returns:
        F: Description of return value
    
    """


        @functools.wraps(func)
        @async_with_exception_handling
        async def async_wrapper(*args: Any, **kwargs: Any) ->Any:
    """
    Async wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    Returns:
        Any: Description of return value
    
    """

            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                notifications_sent_total.labels(channel=channel, status=
                    'success').inc()
                return result
            except Exception as e:
                notifications_sent_total.labels(channel=channel, status='error'
                    ).inc()
                raise
            finally:
                duration = time.time() - start_time
                notification_send_duration_seconds.labels(channel=channel
                    ).observe(duration)

        @functools.wraps(func)
        @with_exception_handling
        def sync_wrapper(*args: Any, **kwargs: Any) ->Any:
    """
    Sync wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    Returns:
        Any: Description of return value
    
    """

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                notifications_sent_total.labels(channel=channel, status=
                    'success').inc()
                return result
            except Exception as e:
                notifications_sent_total.labels(channel=channel, status='error'
                    ).inc()
                raise
            finally:
                duration = time.time() - start_time
                notification_send_duration_seconds.labels(channel=channel
                    ).observe(duration)
        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        else:
            return cast(F, sync_wrapper)
    return decorator


@with_exception_handling
def track_prometheus_query(query_type: str) ->Callable[[F], F]:
    """
    Decorator to track Prometheus query metrics.

    Args:
        query_type: Query type

    Returns:
        Decorated function
    """

    @with_exception_handling
    def decorator(func: F) ->F:
    """
    Decorator.
    
    Args:
        func: Description of func
    
    Returns:
        F: Description of return value
    
    """


        @functools.wraps(func)
        @async_with_exception_handling
        async def async_wrapper(*args: Any, **kwargs: Any) ->Any:
    """
    Async wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    Returns:
        Any: Description of return value
    
    """

            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                prometheus_queries_total.labels(query_type=query_type,
                    status='success').inc()
                return result
            except Exception as e:
                prometheus_queries_total.labels(query_type=query_type,
                    status='error').inc()
                raise
            finally:
                duration = time.time() - start_time
                prometheus_query_duration_seconds.labels(query_type=query_type
                    ).observe(duration)

        @functools.wraps(func)
        @with_exception_handling
        def sync_wrapper(*args: Any, **kwargs: Any) ->Any:
    """
    Sync wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    Returns:
        Any: Description of return value
    
    """

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                prometheus_queries_total.labels(query_type=query_type,
                    status='success').inc()
                return result
            except Exception as e:
                prometheus_queries_total.labels(query_type=query_type,
                    status='error').inc()
                raise
            finally:
                duration = time.time() - start_time
                prometheus_query_duration_seconds.labels(query_type=query_type
                    ).observe(duration)
        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        else:
            return cast(F, sync_wrapper)
    return decorator


@with_exception_handling
def track_alertmanager_operation(operation: str) ->Callable[[F], F]:
    """
    Decorator to track Alertmanager operation metrics.

    Args:
        operation: Operation name

    Returns:
        Decorated function
    """

    @with_exception_handling
    def decorator(func: F) ->F:
    """
    Decorator.
    
    Args:
        func: Description of func
    
    Returns:
        F: Description of return value
    
    """


        @functools.wraps(func)
        @async_with_exception_handling
        async def async_wrapper(*args: Any, **kwargs: Any) ->Any:
    """
    Async wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    Returns:
        Any: Description of return value
    
    """

            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                alertmanager_operations_total.labels(operation=operation,
                    status='success').inc()
                return result
            except Exception as e:
                alertmanager_operations_total.labels(operation=operation,
                    status='error').inc()
                raise
            finally:
                duration = time.time() - start_time
                alertmanager_operation_duration_seconds.labels(operation=
                    operation).observe(duration)

        @functools.wraps(func)
        @with_exception_handling
        def sync_wrapper(*args: Any, **kwargs: Any) ->Any:
    """
    Sync wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    Returns:
        Any: Description of return value
    
    """

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                alertmanager_operations_total.labels(operation=operation,
                    status='success').inc()
                return result
            except Exception as e:
                alertmanager_operations_total.labels(operation=operation,
                    status='error').inc()
                raise
            finally:
                duration = time.time() - start_time
                alertmanager_operation_duration_seconds.labels(operation=
                    operation).observe(duration)
        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        else:
            return cast(F, sync_wrapper)
    return decorator


@with_exception_handling
def track_grafana_operation(operation: str) ->Callable[[F], F]:
    """
    Decorator to track Grafana operation metrics.

    Args:
        operation: Operation name

    Returns:
        Decorated function
    """

    @with_exception_handling
    def decorator(func: F) ->F:
    """
    Decorator.
    
    Args:
        func: Description of func
    
    Returns:
        F: Description of return value
    
    """


        @functools.wraps(func)
        @async_with_exception_handling
        async def async_wrapper(*args: Any, **kwargs: Any) ->Any:
    """
    Async wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    Returns:
        Any: Description of return value
    
    """

            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                grafana_operations_total.labels(operation=operation, status
                    ='success').inc()
                return result
            except Exception as e:
                grafana_operations_total.labels(operation=operation, status
                    ='error').inc()
                raise
            finally:
                duration = time.time() - start_time
                grafana_operation_duration_seconds.labels(operation=operation
                    ).observe(duration)

        @functools.wraps(func)
        @with_exception_handling
        def sync_wrapper(*args: Any, **kwargs: Any) ->Any:
    """
    Sync wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    Returns:
        Any: Description of return value
    
    """

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                grafana_operations_total.labels(operation=operation, status
                    ='success').inc()
                return result
            except Exception as e:
                grafana_operations_total.labels(operation=operation, status
                    ='error').inc()
                raise
            finally:
                duration = time.time() - start_time
                grafana_operation_duration_seconds.labels(operation=operation
                    ).observe(duration)
        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        else:
            return cast(F, sync_wrapper)
    return decorator


def register_health_check(name: str, check_func: Callable[[], Union[bool,
    HealthStatus]], description: Optional[str]=None) ->None:
    """
    Register a health check.

    Args:
        name: Health check name
        check_func: Health check function
        description: Health check description
    """
    health_check.register(name, check_func, description)
