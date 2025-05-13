"""
Error Handlers Module

This module provides FastAPI exception handlers for standardized error responses.
"""
from typing import Dict, Any, List, Union
import uuid
import time
from datetime import datetime
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from fastapi.responses import JSONResponse
import traceback
from common_lib.exceptions import ForexTradingPlatformError, DataError, ServiceError, ModelError, SecurityError, ResilienceError, TradingError
from monitoring_alerting_service.error.exceptions_bridge import MonitoringAlertingError, AlertNotFoundError, NotificationError, AlertStorageError, MetricsExporterError, DashboardError, AlertRuleError, ThresholdValidationError
from monitoring_alerting_service.metrics import record_error, record_http_error, initialize_error_metrics
import logging
logger = logging.getLogger(__name__)


from monitoring_alerting_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

def get_correlation_id(request: Request) ->str:
    """
    Get correlation ID from request or generate a new one.

    Args:
        request: The FastAPI request

    Returns:
        Correlation ID string
    """
    correlation_id = getattr(request.state, 'correlation_id', None)
    if not correlation_id:
        correlation_id = request.headers.get('X-Correlation-ID')
    if not correlation_id:
        correlation_id = str(uuid.uuid4())
    return correlation_id


def format_error_response(error_code: str, message: str, details: Dict[str,
    Any]=None, correlation_id: str=None, service: str=
    'monitoring-alerting-service') ->Dict[str, Any]:
    """
    Format a standardized error response.

    Args:
        error_code: Error code
        message: Error message
        details: Additional error details
        correlation_id: Request correlation ID
        service: Service name

    Returns:
        Formatted error response dictionary
    """
    return {'error': {'code': error_code, 'message': message, 'details': 
        details or {}, 'correlation_id': correlation_id or str(uuid.uuid4()
        ), 'timestamp': datetime.utcnow().isoformat(), 'service': service}}


def format_validation_errors(errors: List[Dict[str, Any]]) ->Dict[str, Any]:
    """
    Format validation errors into a structured format.

    Args:
        errors: List of validation errors

    Returns:
        Formatted validation errors
    """
    formatted_errors = {}
    for error in errors:
        loc = error.get('loc', [])
        field = '.'.join(str(item) for item in loc) if loc else 'request'
        msg = error.get('msg', 'Validation error')
        formatted_errors[field] = msg
    return formatted_errors


@with_exception_handling
def register_exception_handlers(app: FastAPI) ->None:
    """
    Register all exception handlers with the FastAPI application.

    Args:
        app: FastAPI application instance
    """
    error_metrics = initialize_error_metrics('monitoring-alerting-service')

    @app.exception_handler(ForexTradingPlatformError)
    @async_with_exception_handling
    async def forex_platform_exception_handler(request: Request, exc:
        ForexTradingPlatformError):
        """Handle custom ForexTradingPlatformError exceptions."""
        correlation_id = get_correlation_id(request)
        start_time = time.time()
        logger.error(f'ForexTradingPlatformError: {exc.message}', extra={
            'error_code': exc.error_code, 'details': exc.details, 'path':
            request.url.path, 'method': request.method, 'correlation_id':
            correlation_id})
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        if isinstance(exc, DataError):
            status_code = status.HTTP_400_BAD_REQUEST
        elif isinstance(exc, SecurityError):
            status_code = status.HTTP_401_UNAUTHORIZED
        elif isinstance(exc, ServiceError):
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        elif isinstance(exc, ModelError):
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        elif isinstance(exc, ResilienceError):
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        elif isinstance(exc, TradingError):
            status_code = status.HTTP_400_BAD_REQUEST
        if isinstance(exc, AlertNotFoundError):
            status_code = status.HTTP_404_NOT_FOUND
        elif isinstance(exc, NotificationError) or isinstance(exc,
            MetricsExporterError):
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        elif isinstance(exc, AlertStorageError):
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
        elif isinstance(exc, AlertRuleError) or isinstance(exc,
            ThresholdValidationError):
            status_code = status.HTTP_400_BAD_REQUEST
        elif isinstance(exc, DashboardError):
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        try:
            error_type = exc.__class__.__name__
            path_parts = request.url.path.strip('/').split('/')
            component = path_parts[0] if path_parts else 'api'
            record_error(error_code=exc.error_code, error_type=error_type,
                component=component, details={'path': request.url.path,
                'method': request.method, 'status_code': status_code,
                'correlation_id': correlation_id})
            record_http_error(endpoint=request.url.path, method=request.
                method, status_code=status_code, details={'error_code': exc
                .error_code, 'error_type': error_type, 'correlation_id':
                correlation_id})
        except Exception as metrics_exc:
            logger.error(f'Failed to record error metrics: {str(metrics_exc)}',
                extra={'correlation_id': correlation_id}, exc_info=True)
        return JSONResponse(status_code=status_code, content=
            format_error_response(exc.error_code, exc.message, exc.details,
            correlation_id, 'monitoring-alerting-service'))

    @app.exception_handler(RequestValidationError)
    @app.exception_handler(ValidationError)
    @async_with_exception_handling
    async def validation_exception_handler(request: Request, exc: Union[
        RequestValidationError, ValidationError]):
        """Handle validation errors from FastAPI and Pydantic."""
        correlation_id = get_correlation_id(request)
        errors = exc.errors() if hasattr(exc, 'errors') else [{'msg': str(exc)}
            ]
        logger.warning(
            f'Validation error for {request.method} {request.url.path}',
            extra={'errors': errors, 'correlation_id': correlation_id})
        try:
            path_parts = request.url.path.strip('/').split('/')
            component = path_parts[0] if path_parts else 'api'
            record_error(error_code='VALIDATION_ERROR', error_type=
                'ValidationError', component=component, details={'path':
                request.url.path, 'method': request.method, 'errors':
                errors, 'correlation_id': correlation_id})
            record_http_error(endpoint=request.url.path, method=request.
                method, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                details={'error_code': 'VALIDATION_ERROR', 'error_type':
                'ValidationError', 'correlation_id': correlation_id})
        except Exception as metrics_exc:
            logger.error(
                f'Failed to record validation error metrics: {str(metrics_exc)}'
                , extra={'correlation_id': correlation_id}, exc_info=True)
        return JSONResponse(status_code=status.
            HTTP_422_UNPROCESSABLE_ENTITY, content=format_error_response(
            'VALIDATION_ERROR', 'Request validation failed',
            format_validation_errors(errors), correlation_id,
            'monitoring-alerting-service'))

    @app.exception_handler(Exception)
    @async_with_exception_handling
    async def generic_exception_handler(request: Request, exc: Exception):
        """Handle all other unhandled exceptions."""
        correlation_id = get_correlation_id(request)
        logger.error(f'Unhandled exception: {str(exc)}', extra={'path':
            request.url.path, 'method': request.method, 'traceback':
            traceback.format_exc(), 'correlation_id': correlation_id})
        try:
            path_parts = request.url.path.strip('/').split('/')
            component = path_parts[0] if path_parts else 'api'
            record_error(error_code='INTERNAL_SERVER_ERROR', error_type=exc
                .__class__.__name__, component=component, details={'path':
                request.url.path, 'method': request.method, 'error': str(
                exc), 'correlation_id': correlation_id})
            record_http_error(endpoint=request.url.path, method=request.
                method, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                details={'error_code': 'INTERNAL_SERVER_ERROR',
                'error_type': exc.__class__.__name__, 'correlation_id':
                correlation_id})
        except Exception as metrics_exc:
            logger.error(
                f'Failed to record unhandled exception metrics: {str(metrics_exc)}'
                , extra={'correlation_id': correlation_id}, exc_info=True)
        return JSONResponse(status_code=status.
            HTTP_500_INTERNAL_SERVER_ERROR, content=format_error_response(
            'INTERNAL_SERVER_ERROR', 'An unexpected error occurred', {
            'error': str(exc)} if logger.level <= logging.DEBUG else None,
            correlation_id, 'monitoring-alerting-service'))
