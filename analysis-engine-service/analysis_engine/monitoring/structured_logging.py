"""
Structured Logging Module for Analysis Engine Service.

This module provides structured logging capabilities with correlation IDs,
context data, and consistent formatting across all components.
"""
import logging
import json
import uuid
import time
import inspect
import threading
from typing import Dict, Any, Optional, Union
from datetime import datetime
from contextvars import ContextVar
from functools import wraps
request_id_var: ContextVar[str] = ContextVar('request_id', default='')
correlation_id_var: ContextVar[str] = ContextVar('correlation_id', default='')
session_id_var: ContextVar[str] = ContextVar('session_id', default='')


from analysis_engine.core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class StructuredLogRecord(logging.LogRecord):
    """Extended LogRecord that includes structured data."""

    @with_exception_handling
    def __init__(self, *args, **kwargs):
    """
      init  .
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

        super().__init__(*args, **kwargs)
        self.request_id = request_id_var.get()
        self.correlation_id = correlation_id_var.get()
        self.session_id = session_id_var.get()
        frame = inspect.currentframe()
        if frame:
            try:
                caller_frame = inspect.getouterframes(frame, 2)[2]
                self.function_name = caller_frame.function
                self.line_number = caller_frame.lineno
                self.file_path = caller_frame.filename
            finally:
                del frame
        self.timestamp = datetime.utcnow().isoformat() + 'Z'
        self.thread_id = threading.get_ident()
        self.thread_name = threading.current_thread().name


class JsonFormatter(logging.Formatter):
    """Formatter that outputs JSON strings."""

    def format(self, record):
        """Format the log record as JSON."""
        log_data = {'timestamp': getattr(record, 'timestamp', datetime.
            utcnow().isoformat() + 'Z'), 'level': record.levelname,
            'logger': record.name, 'message': record.getMessage(), 'module':
            record.module, 'function': getattr(record, 'function_name',
            record.funcName), 'line': getattr(record, 'line_number', record
            .lineno), 'file': getattr(record, 'file_path', record.pathname),
            'thread_id': getattr(record, 'thread_id', threading.get_ident()
            ), 'thread_name': getattr(record, 'thread_name', threading.
            current_thread().name)}
        if hasattr(record, 'request_id') and record.request_id:
            log_data['request_id'] = record.request_id
        if hasattr(record, 'correlation_id') and record.correlation_id:
            log_data['correlation_id'] = record.correlation_id
        if hasattr(record, 'session_id') and record.session_id:
            log_data['session_id'] = record.session_id
        if hasattr(record, 'data') and record.data:
            log_data['data'] = record.data
        if record.exc_info:
            log_data['exception'] = {'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]), 'traceback': self.
                formatException(record.exc_info)}
        return json.dumps(log_data)


class StructuredLogger:
    """Logger that provides structured logging capabilities."""

    def __init__(self, name: str):
        """
        Initialize the structured logger.
        
        Args:
            name: Logger name
        """
        self.logger = logging.getLogger(name)
        self.name = name

    def _log(self, level: int, msg: str, data: Optional[Dict[str, Any]]=
        None, *args, **kwargs):
        """
        Log a message with structured data.
        
        Args:
            level: Log level
            msg: Log message
            data: Additional structured data
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        extra = kwargs.get('extra', {})
        extra['data'] = data
        kwargs['extra'] = extra
        self.logger.log(level, msg, *args, **kwargs)

    def debug(self, msg: str, data: Optional[Dict[str, Any]]=None, *args,
        **kwargs):
        """Log a debug message."""
        self._log(logging.DEBUG, msg, data, *args, **kwargs)

    def info(self, msg: str, data: Optional[Dict[str, Any]]=None, *args, **
        kwargs):
        """Log an info message."""
        self._log(logging.INFO, msg, data, *args, **kwargs)

    def warning(self, msg: str, data: Optional[Dict[str, Any]]=None, *args,
        **kwargs):
        """Log a warning message."""
        self._log(logging.WARNING, msg, data, *args, **kwargs)

    def error(self, msg: str, data: Optional[Dict[str, Any]]=None, *args,
        **kwargs):
        """Log an error message."""
        self._log(logging.ERROR, msg, data, *args, **kwargs)

    def critical(self, msg: str, data: Optional[Dict[str, Any]]=None, *args,
        **kwargs):
        """Log a critical message."""
        self._log(logging.CRITICAL, msg, data, *args, **kwargs)

    def exception(self, msg: str, data: Optional[Dict[str, Any]]=None, *
        args, **kwargs):
        """Log an exception message."""
        kwargs['exc_info'] = kwargs.get('exc_info', True)
        self._log(logging.ERROR, msg, data, *args, **kwargs)


def configure_structured_logging():
    """Configure structured logging for the application."""
    logging.setLogRecordFactory(StructuredLogRecord)
    json_formatter = JsonFormatter()
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(json_formatter)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)


def get_structured_logger(name: str) ->StructuredLogger:
    """
    Get a structured logger.
    
    Args:
        name: Logger name
        
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name)


def set_correlation_id(correlation_id: Optional[str]=None) ->str:
    """
    Set the correlation ID for the current context.
    
    Args:
        correlation_id: Correlation ID to set, or None to generate a new one
        
    Returns:
        The correlation ID that was set
    """
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    correlation_id_var.set(correlation_id)
    return correlation_id


def get_correlation_id() ->str:
    """
    Get the correlation ID for the current context.
    
    Returns:
        The correlation ID, or an empty string if not set
    """
    return correlation_id_var.get()


def set_request_id(request_id: Optional[str]=None) ->str:
    """
    Set the request ID for the current context.
    
    Args:
        request_id: Request ID to set, or None to generate a new one
        
    Returns:
        The request ID that was set
    """
    if request_id is None:
        request_id = str(uuid.uuid4())
    request_id_var.set(request_id)
    return request_id


def get_request_id() ->str:
    """
    Get the request ID for the current context.
    
    Returns:
        The request ID, or an empty string if not set
    """
    return request_id_var.get()


def set_session_id(session_id: str) ->None:
    """
    Set the session ID for the current context.
    
    Args:
        session_id: Session ID to set
    """
    session_id_var.set(session_id)


def get_session_id() ->str:
    """
    Get the session ID for the current context.
    
    Returns:
        The session ID, or an empty string if not set
    """
    return session_id_var.get()


@with_exception_handling
def with_correlation_id(func):
    """
    Decorator that ensures a correlation ID is set for the function call.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
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

        correlation_id = kwargs.pop('correlation_id', None
            ) or get_correlation_id() or str(uuid.uuid4())
        token = correlation_id_var.set(correlation_id)
        try:
            return func(*args, correlation_id=correlation_id, **kwargs)
        finally:
            correlation_id_var.reset(token)
    return wrapper


@with_exception_handling
def log_execution_time(logger: Union[StructuredLogger, str]):
    """
    Decorator that logs the execution time of a function.
    
    Args:
        logger: StructuredLogger instance or logger name
        
    Returns:
        Decorator function
    """
    if isinstance(logger, str):
        logger = get_structured_logger(logger)

    @with_exception_handling
    def decorator(func):
    """
    Decorator.
    
    Args:
        func: Description of func
    
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

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.time()
                duration = end_time - start_time
                logger.info(
                    f'Function {func.__name__} executed in {duration:.6f} seconds'
                    , {'function': func.__name__, 'duration': duration,
                    'module': func.__module__})
        return wrapper
    return decorator
