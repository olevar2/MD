"""
Logging and reporting configuration for the indicator system.

Provides structured logging and reporting capabilities for tracking validation
and error events in the indicator calculation system.
"""
import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd


from feature_store_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class IndicatorLogger:
    """
    Manages logging configuration and formatting for the indicator system.
    
    This class provides structured logging with different log levels, log rotation,
    and formatting specific to indicator calculations and validations.
    """

    def __init__(self, log_dir: str='logs'):
    """
      init  .
    
    Args:
        log_dir: Description of log_dir
    
    """

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger('indicator_system')
        self.logger.setLevel(logging.INFO)
        self.console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(extra)s')
        self._setup_handlers()

    def _setup_handlers(self):
        """Set up console and file handlers for logging."""
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(self.console_formatter)
        self.logger.addHandler(console_handler)
        handlers = {'validation': (logging.INFO, 'validation.log'), 'error':
            (logging.ERROR, 'error.log'), 'performance': (logging.INFO,
            'performance.log')}
        for handler_name, (level, filename) in handlers.items():
            handler = logging.FileHandler(self.log_dir / filename, encoding
                ='utf-8')
            handler.setLevel(level)
            handler.setFormatter(self.file_formatter)
            self.logger.addHandler(handler)

    def log_validation(self, indicator_name: str, validation_type: str,
        is_valid: bool, details: Optional[Dict[str, Any]]=None) ->None:
        """
        Log a validation event.
        
        Args:
            indicator_name: Name of the indicator
            validation_type: Type of validation performed
            is_valid: Whether validation passed
            details: Optional validation details
        """
        details = details or {}
        self.logger.info(f'Validation: {indicator_name} - {validation_type}',
            extra={'indicator': indicator_name, 'validation_type':
            validation_type, 'is_valid': is_valid, 'details': json.dumps(
            details), 'timestamp': datetime.utcnow().isoformat()})

    def log_error(self, indicator_name: str, error_type: str, message: str,
        details: Optional[Dict[str, Any]]=None) ->None:
        """
        Log an error event.
        
        Args:
            indicator_name: Name of the indicator
            error_type: Type of error
            message: Error message
            details: Optional error details
        """
        details = details or {}
        self.logger.error(f'Error: {indicator_name} - {message}', extra={
            'indicator': indicator_name, 'error_type': error_type,
            'details': json.dumps(details), 'timestamp': datetime.utcnow().
            isoformat()})

    def log_performance(self, indicator_name: str, execution_time: float,
        data_points: int, details: Optional[Dict[str, Any]]=None) ->None:
        """
        Log performance metrics.
        
        Args:
            indicator_name: Name of the indicator
            execution_time: Time taken for calculation (seconds)
            data_points: Number of data points processed
            details: Optional performance details
        """
        details = details or {}
        self.logger.info(f'Performance: {indicator_name}', extra={
            'indicator': indicator_name, 'execution_time': execution_time,
            'data_points': data_points, 'points_per_second': data_points /
            execution_time if execution_time > 0 else 0, 'details': json.
            dumps(details), 'timestamp': datetime.utcnow().isoformat()})


class IndicatorReport:
    """
    Generates reports from logged indicator system events.
    
    This class provides functionality to analyze logs and generate 
    various reports about indicator performance, validation, and errors.
    """

    def __init__(self, log_dir: str='logs'):
        self.log_dir = Path(log_dir)

    @with_exception_handling
    def _read_log_file(self, filename: str) ->List[Dict[str, Any]]:
        """Read and parse a log file."""
        entries = []
        log_path = self.log_dir / filename
        if not log_path.exists():
            return entries
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    parts = line.strip().split(' - ')
                    timestamp = datetime.strptime(parts[0],
                        '%Y-%m-%d %H:%M:%S,%f')
                    extra = json.loads(parts[-1])
                    entry = {'timestamp': timestamp, **extra}
                    entries.append(entry)
                except Exception:
                    continue
        return entries

    def generate_validation_report(self, start_time: Optional[datetime]=
        None, end_time: Optional[datetime]=None) ->Dict[str, Any]:
        """
        Generate a validation statistics report.
        
        Args:
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            Dictionary containing validation statistics
        """
        entries = self._read_log_file('validation.log')
        if start_time:
            entries = [e for e in entries if e['timestamp'] >= start_time]
        if end_time:
            entries = [e for e in entries if e['timestamp'] <= end_time]
        total_validations = len(entries)
        if total_validations == 0:
            return {'message': 'No validation entries found'}
        validations_by_type = {}
        validations_by_indicator = {}
        for entry in entries:
            v_type = entry['validation_type']
            indicator = entry['indicator']
            is_valid = entry['is_valid']
            if v_type not in validations_by_type:
                validations_by_type[v_type] = {'total': 0, 'valid': 0}
            validations_by_type[v_type]['total'] += 1
            if is_valid:
                validations_by_type[v_type]['valid'] += 1
            if indicator not in validations_by_indicator:
                validations_by_indicator[indicator] = {'total': 0, 'valid': 0}
            validations_by_indicator[indicator]['total'] += 1
            if is_valid:
                validations_by_indicator[indicator]['valid'] += 1
        return {'total_validations': total_validations, 'by_type':
            validations_by_type, 'by_indicator': validations_by_indicator,
            'time_range': {'start': min(e['timestamp'] for e in entries),
            'end': max(e['timestamp'] for e in entries)}}

    def generate_error_report(self, start_time: Optional[datetime]=None,
        end_time: Optional[datetime]=None) ->Dict[str, Any]:
        """
        Generate an error statistics report.
        
        Args:
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            Dictionary containing error statistics
        """
        entries = self._read_log_file('error.log')
        if start_time:
            entries = [e for e in entries if e['timestamp'] >= start_time]
        if end_time:
            entries = [e for e in entries if e['timestamp'] <= end_time]
        total_errors = len(entries)
        if total_errors == 0:
            return {'message': 'No error entries found'}
        errors_by_type = {}
        errors_by_indicator = {}
        for entry in entries:
            error_type = entry['error_type']
            indicator = entry['indicator']
            errors_by_type[error_type] = errors_by_type.get(error_type, 0) + 1
            errors_by_indicator[indicator] = errors_by_indicator.get(indicator,
                0) + 1
        return {'total_errors': total_errors, 'by_type': errors_by_type,
            'by_indicator': errors_by_indicator, 'time_range': {'start':
            min(e['timestamp'] for e in entries), 'end': max(e['timestamp'] for
            e in entries)}}

    def generate_performance_report(self, start_time: Optional[datetime]=
        None, end_time: Optional[datetime]=None) ->Dict[str, Any]:
        """
        Generate a performance statistics report.
        
        Args:
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            Dictionary containing performance statistics
        """
        entries = self._read_log_file('performance.log')
        if start_time:
            entries = [e for e in entries if e['timestamp'] >= start_time]
        if end_time:
            entries = [e for e in entries if e['timestamp'] <= end_time]
        if not entries:
            return {'message': 'No performance entries found'}
        performance_by_indicator = {}
        for entry in entries:
            indicator = entry['indicator']
            exec_time = entry['execution_time']
            data_points = entry['data_points']
            points_per_sec = entry['points_per_second']
            if indicator not in performance_by_indicator:
                performance_by_indicator[indicator] = {'count': 0,
                    'total_time': 0.0, 'total_points': 0, 'min_time': float
                    ('inf'), 'max_time': float('-inf'),
                    'min_points_per_sec': float('inf'),
                    'max_points_per_sec': float('-inf')}
            stats = performance_by_indicator[indicator]
            stats['count'] += 1
            stats['total_time'] += exec_time
            stats['total_points'] += data_points
            stats['min_time'] = min(stats['min_time'], exec_time)
            stats['max_time'] = max(stats['max_time'], exec_time)
            stats['min_points_per_sec'] = min(stats['min_points_per_sec'],
                points_per_sec)
            stats['max_points_per_sec'] = max(stats['max_points_per_sec'],
                points_per_sec)
        for stats in performance_by_indicator.values():
            stats['avg_time'] = stats['total_time'] / stats['count']
            stats['avg_points_per_sec'] = stats['total_points'] / stats[
                'total_time']
        return {'total_executions': len(entries), 'by_indicator':
            performance_by_indicator, 'time_range': {'start': min(e[
            'timestamp'] for e in entries), 'end': max(e['timestamp'] for e in
            entries)}}

    def generate_summary_report(self, start_time: Optional[datetime]=None,
        end_time: Optional[datetime]=None) ->Dict[str, Any]:
        """
        Generate a comprehensive summary report.
        
        Args:
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            Dictionary containing summary statistics
        """
        validation_report = self.generate_validation_report(start_time,
            end_time)
        error_report = self.generate_error_report(start_time, end_time)
        performance_report = self.generate_performance_report(start_time,
            end_time)
        return {'validation_summary': validation_report, 'error_summary':
            error_report, 'performance_summary': performance_report,
            'generated_at': datetime.utcnow().isoformat()}
