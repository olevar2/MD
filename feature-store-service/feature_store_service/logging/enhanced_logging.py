"""
Enhanced logging configuration for the Feature Store Service.

This module provides a comprehensive logging setup with structured logs,
performance tracking, and integration with monitoring systems.
"""
import logging
import json
import time
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional, Callable, Union
from functools import wraps
import traceback
import uuid
DEFAULT_LOG_FORMAT = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
JSON_LOG_FORMAT = (
    '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s", "extra": %(extra)s}'
    )
DEFAULT_LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class StructuredLogAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds structured data to log records.
    """

    def process(self, msg, kwargs):
        """
        Process the logging message and keyword arguments.
        
        Args:
            msg: The log message
            kwargs: Additional logging parameters
            
        Returns:
            Tuple of (msg, kwargs) with extra data added
        """
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        if hasattr(self.extra, 'get') and self.extra.get('request_id'):
            kwargs['extra']['request_id'] = self.extra.get('request_id')
        if 'timestamp' not in kwargs['extra']:
            kwargs['extra']['timestamp'] = datetime.utcnow().isoformat()
        return msg, kwargs


class JsonFormatter(logging.Formatter):
    """
    Custom formatter that outputs logs as JSON objects.
    """

    def format(self, record):
        """
        Format the log record as a JSON string.
        
        Args:
            record: The log record to format
            
        Returns:
            Formatted JSON string
        """
        message = super().format(record)
        log_object = {'timestamp': datetime.utcnow().isoformat(), 'level':
            record.levelname, 'logger': record.name, 'message': record.
            getMessage(), 'module': record.module, 'function': record.
            funcName, 'line': record.lineno}
        if hasattr(record, 'extra'):
            log_object['extra'] = record.extra
        if record.exc_info:
            log_object['exception'] = {'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]), 'traceback': traceback.
                format_exception(*record.exc_info)}
        return json.dumps(log_object)


def configure_logging(service_name: str='feature-store-service', log_level:
    str=DEFAULT_LOG_LEVEL, use_json: bool=False, log_file: Optional[str]=None
    ) ->None:
    """
    Configure logging for the service.
    
    Args:
        service_name: Name of the service
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        use_json: Whether to use JSON formatting
        log_file: Optional file path for logging to file
    """
    numeric_level = getattr(logging, log_level, logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    if use_json:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    service_logger = logging.getLogger(service_name)
    service_logger.setLevel(numeric_level)
    service_logger.info(f'Logging configured for {service_name}', extra={
        'service': service_name, 'log_level': log_level, 'use_json':
        use_json, 'log_file': log_file})


def get_logger(name: str, **extra) ->logging.Logger:
    """
    Get a logger with the given name and extra context.
    
    Args:
        name: Logger name
        **extra: Additional context to include in logs
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    if extra:
        return StructuredLogAdapter(logger, extra)
    return logger


@with_exception_handling
def log_execution_time(logger: Optional[logging.Logger]=None):
    """
    Decorator to log function execution time.
    
    Args:
        logger: Logger to use, or None to use function's module logger
        
    Returns:
        Decorator function
    """

    @with_exception_handling
    def decorator(func: Callable) ->Callable:
    """
    Decorator.
    
    Args:
        func: Description of func
    
    Returns:
        Callable: Description of return value
    
    """


        @wraps(func)
        @with_exception_handling
        def wrapper(*args, **kwargs):
    """
    Wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

            log = logger or logging.getLogger(func.__module__)
            exec_id = str(uuid.uuid4())[:8]
            log.debug(f'Starting execution of {func.__name__} [{exec_id}]',
                extra={'execution_id': exec_id})
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                log.debug(
                    f'Completed {func.__name__} [{exec_id}] in {execution_time:.4f}s'
                    , extra={'execution_id': exec_id, 'execution_time':
                    execution_time})
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                log.error(
                    f'Exception in {func.__name__} [{exec_id}] after {execution_time:.4f}s: {str(e)}'
                    , extra={'execution_id': exec_id, 'execution_time':
                    execution_time, 'exception': str(e), 'exception_type':
                    e.__class__.__name__}, exc_info=True)
                raise
        return wrapper
    return decorator


@with_exception_handling
def log_async_execution_time(logger: Optional[logging.Logger]=None):
    """
    Decorator to log async function execution time.
    
    Args:
        logger: Logger to use, or None to use function's module logger
        
    Returns:
        Decorator function
    """

    @with_exception_handling
    def decorator(func: Callable) ->Callable:
    """
    Decorator.
    
    Args:
        func: Description of func
    
    Returns:
        Callable: Description of return value
    
    """


        @wraps(func)
        @async_with_exception_handling
        async def wrapper(*args, **kwargs):
    """
    Wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

            log = logger or logging.getLogger(func.__module__)
            exec_id = str(uuid.uuid4())[:8]
            log.debug(
                f'Starting async execution of {func.__name__} [{exec_id}]',
                extra={'execution_id': exec_id})
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                log.debug(
                    f'Completed async {func.__name__} [{exec_id}] in {execution_time:.4f}s'
                    , extra={'execution_id': exec_id, 'execution_time':
                    execution_time})
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                log.error(
                    f'Exception in async {func.__name__} [{exec_id}] after {execution_time:.4f}s: {str(e)}'
                    , extra={'execution_id': exec_id, 'execution_time':
                    execution_time, 'exception': str(e), 'exception_type':
                    e.__class__.__name__}, exc_info=True)
                raise
        return wrapper
    return decorator
