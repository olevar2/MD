"""
Log Collector and Exporter for Loki Integration

This module collects logs from various services and exports them to Loki
while also generating Prometheus metrics about logging patterns.
"""
import logging
import time
from typing import Dict, List, Any
from prometheus_client import Counter, Gauge, Summary
import os
from datetime import datetime, timedelta
import requests
LOG_LINES_TOTAL = Counter('log_lines_total',
    'Total number of log lines processed', ['service', 'level'])
LOG_ERROR_TOTAL = Counter('log_error_total',
    'Total number of error log entries', ['service', 'error_type'])
LOG_PATTERN_MATCHES = Counter('log_critical_pattern_matched',
    'Number of critical pattern matches in logs', ['service', 'pattern'])
LOG_PROCESSING_TIME = Summary('log_processing_seconds',
    'Time spent processing log entries', ['service'])


from core.exceptions_bridge_1 import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class StructuredLogCollector:
    """
    StructuredLogCollector class.
    
    Attributes:
        Add attributes here
    """


    def __init__(self, loki_url: str):
        self.loki_url = loki_url
        self.retention_rules = {'error': '30d', 'warning': '14d', 'info': '7d'}

    def process_log_entry(self, log_entry: Dict[str, Any]) ->None:
        """Process a structured log entry and export to Loki."""
        service = log_entry.get('service', 'unknown')
        level = log_entry.get('level', 'unknown')
        with LOG_PROCESSING_TIME.labels(service=service).time():
            LOG_LINES_TOTAL.labels(service=service, level=level).inc()
            if level in ('ERROR', 'CRITICAL'):
                error_type = log_entry.get('error_type', 'unknown')
                LOG_ERROR_TOTAL.labels(service=service, error_type=error_type
                    ).inc()
            self._check_critical_patterns(log_entry)
            self._export_to_loki(log_entry)

    def _check_critical_patterns(self, log_entry: Dict[str, Any]) ->None:
        """Check log entry against defined critical patterns."""
        patterns = {'data_corruption': 'data.*corrupt', 'auth_failure':
            'authentication.*failed', 'trade_error':
            'trade.*failed|order.*rejected', 'connection_lost':
            'connection.*lost|connection.*timeout'}
        message = log_entry.get('message', '')
        service = log_entry.get('service', 'unknown')
        for pattern_name, pattern in patterns.items():
            if re.search(pattern, message, re.IGNORECASE):
                LOG_PATTERN_MATCHES.labels(service=service, pattern=
                    pattern_name).inc()

    @with_exception_handling
    def _export_to_loki(self, log_entry: Dict[str, Any]) ->None:
        """Export log entry to Loki with proper labels and retention."""
        try:
            labels = {'service': log_entry.get('service', 'unknown'),
                'level': log_entry.get('level', 'unknown'), 'env':
                log_entry.get('env', 'prod')}
            if 'trace_id' in log_entry:
                labels['trace_id'] = log_entry['trace_id']
            timestamp = int(time.time() * 1000000000.0)
            payload = {'streams': [{'stream': labels, 'values': [[str(
                timestamp), str(log_entry['message'])]]}]}
            response = requests.post(f'{self.loki_url}/loki/api/v1/push',
                json=payload)
            response.raise_for_status()
        except Exception as e:
            logging.error(f'Failed to export log to Loki: {str(e)}')

    @with_exception_handling
    def configure_retention(self) ->None:
        """Configure retention policies in Loki."""
        retention_config = {'config': {'rules': [{'selector':
            '{level="ERROR"}', 'retention': self.retention_rules['error']},
            {'selector': '{level="WARNING"}', 'retention': self.
            retention_rules['warning']}, {'selector': '{level="INFO"}',
            'retention': self.retention_rules['info']}]}}
        try:
            response = requests.post(f'{self.loki_url}/loki/api/v1/config',
                json=retention_config)
            response.raise_for_status()
        except Exception as e:
            logging.error(f'Failed to configure Loki retention: {str(e)}')
