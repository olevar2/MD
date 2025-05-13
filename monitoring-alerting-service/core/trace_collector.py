"""
Distributed Tracing Collector and Exporter

This module handles trace collection and exports trace data to Tempo/Jaeger
while generating Prometheus metrics about trace patterns and latencies.
"""
import logging
import time
from typing import Dict, List, Any, Optional
from prometheus_client import Counter, Gauge, Histogram, Summary
import os
from datetime import datetime, timedelta
import requests
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
TRACE_DURATION_SECONDS = Histogram('trace_duration_seconds',
    'Duration of traced operations', ['operation', 'service'], buckets=(
    0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 
    7.5, 10.0))
TRACE_ERROR_TOTAL = Counter('trace_error_total',
    'Total number of errors in traced operations', ['operation', 'service',
    'error_type'])
CRITICAL_PATH_DURATION = Gauge('critical_path_duration_seconds',
    'Duration of critical path operations', ['path_name'])
TRACE_DEPENDENCY_LATENCY = Histogram('trace_dependency_latency_seconds',
    'Latency of dependencies in traced operations', ['dependency',
    'operation'], buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 
    0.25, 0.5))


from core.exceptions_bridge_1 import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class TraceCollector:
    """
    TraceCollector class.
    
    Attributes:
        Add attributes here
    """


    def __init__(self, jaeger_host: str, jaeger_port: int):
    """
      init  .
    
    Args:
        jaeger_host: Description of jaeger_host
        jaeger_port: Description of jaeger_port
    
    """

        self.critical_paths = {'order_execution': ['validate_order',
            'check_risk_limits', 'execute_trade', 'update_position'],
            'signal_processing': ['collect_market_data', 'analyze_signals',
            'generate_order']}
        trace.set_tracer_provider(TracerProvider())
        jaeger_exporter = JaegerExporter(agent_host_name=jaeger_host,
            agent_port=jaeger_port)
        trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(
            jaeger_exporter))
        self.tracer = trace.get_tracer(__name__)

    def process_trace(self, trace_data: Dict[str, Any]) ->None:
        """Process a trace and update metrics."""
        operation = trace_data.get('operation', 'unknown')
        service = trace_data.get('service', 'unknown')
        duration = trace_data.get('duration_ms', 0) / 1000
        TRACE_DURATION_SECONDS.labels(operation=operation, service=service
            ).observe(duration)
        if trace_data.get('error'):
            error_type = trace_data.get('error_type', 'unknown')
            TRACE_ERROR_TOTAL.labels(operation=operation, service=service,
                error_type=error_type).inc()
        self._process_dependencies(trace_data)
        self._update_critical_path_metrics(trace_data)

    def _process_dependencies(self, trace_data: Dict[str, Any]) ->None:
        """Process and record dependency latencies from trace."""
        for dep in trace_data.get('dependencies', []):
            TRACE_DEPENDENCY_LATENCY.labels(dependency=dep['name'],
                operation=trace_data.get('operation', 'unknown')).observe(
                dep.get('duration_ms', 0) / 1000)

    def _update_critical_path_metrics(self, trace_data: Dict[str, Any]) ->None:
        """Update metrics for critical path operations."""
        operation = trace_data.get('operation')
        for path_name, operations in self.critical_paths.items():
            if operation in operations:
                duration = trace_data.get('duration_ms', 0) / 1000
                CRITICAL_PATH_DURATION.labels(path_name=path_name).set(duration
                    )

    def start_span(self, operation: str, context: Optional[Dict[str, Any]]=None
        ) ->Any:
        """Start a new trace span."""
        return self.tracer.start_span(operation, attributes=context or {})

    @with_exception_handling
    def configure_sampling(self, rules: Dict[str, float]) ->None:
        """Configure trace sampling rules."""
        try:
            sampling_config = {'service_strategies': [{'service':
                'forex-trading-platform', 'type': 'probabilistic', 'param':
                rule_value, 'operation': rule_name} for rule_name,
                rule_value in rules.items()]}
            response = requests.post(
                f'http://{self.jaeger_host}:{self.jaeger_port}/api/sampling',
                json=sampling_config)
            response.raise_for_status()
        except Exception as e:
            logging.error(f'Failed to configure trace sampling: {str(e)}')
