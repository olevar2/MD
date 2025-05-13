"""
Service-specific metrics for the Data Pipeline Service.

This module defines service-specific metrics for the Data Pipeline Service,
extending the standardized metrics defined in common_lib.monitoring.metrics_standards.
"""

import logging
from typing import Dict, List, Optional, Any, Callable

# Import standard metrics
from common_lib.monitoring.metrics import (
    get_counter, get_gauge, get_histogram, get_summary,
    track_execution_time, track_memory_usage
)
from common_lib.monitoring.metrics_standards import (
    StandardMetrics, LATENCY_BUCKETS_FAST, LATENCY_BUCKETS_MEDIUM, LATENCY_BUCKETS_SLOW,
    PREFIX_PIPELINE, PREFIX_SYSTEM, PREFIX_DATA,
    LABEL_SERVICE, LABEL_PIPELINE, LABEL_OPERATION, LABEL_STAGE
)

# Configure logging
logger = logging.getLogger(__name__)

# Service name
SERVICE_NAME = "data-pipeline-service"

class DataPipelineMetrics(StandardMetrics):
    """
    Service-specific metrics for the Data Pipeline Service.
    
    This class extends the StandardMetrics class to provide service-specific metrics
    for the Data Pipeline Service.
    """
    
    def __init__(self):
        """Initialize the Data Pipeline metrics."""
        super().__init__(SERVICE_NAME)
        
        # Initialize service-specific metrics
        self._init_pipeline_metrics()
        self._init_data_source_metrics()
        self._init_data_processing_metrics()
        self._init_data_quality_metrics()
        
        logger.info(f"Initialized Data Pipeline metrics")
    
    def _init_pipeline_metrics(self) -> None:
        """Initialize pipeline-related metrics."""
        # Pipeline operations counter
        self.pipeline_operations_total = get_counter(
            name=f"{PREFIX_PIPELINE}_operations_total",
            description="Total number of pipeline operations",
            labels=[LABEL_SERVICE, LABEL_PIPELINE, LABEL_OPERATION]
        )
        
        # Pipeline operation duration
        self.pipeline_operation_duration_seconds = get_histogram(
            name=f"{PREFIX_PIPELINE}_operation_duration_seconds",
            description="Pipeline operation duration in seconds",
            labels=[LABEL_SERVICE, LABEL_PIPELINE, LABEL_OPERATION],
            buckets=LATENCY_BUCKETS_MEDIUM
        )
        
        # Pipeline errors
        self.pipeline_errors_total = get_counter(
            name=f"{PREFIX_PIPELINE}_errors_total",
            description="Total number of pipeline errors",
            labels=[LABEL_SERVICE, LABEL_PIPELINE, LABEL_OPERATION, "error_type"]
        )
        
        # Pipeline count
        self.pipeline_count = get_gauge(
            name=f"{PREFIX_PIPELINE}_count",
            description="Number of pipelines",
            labels=[LABEL_SERVICE, "pipeline_type", "status"]
        )
        
        # Pipeline execution time
        self.pipeline_execution_time_seconds = get_histogram(
            name=f"{PREFIX_PIPELINE}_execution_time_seconds",
            description="Pipeline execution time in seconds",
            labels=[LABEL_SERVICE, LABEL_PIPELINE, "execution_type"],
            buckets=LATENCY_BUCKETS_SLOW
        )
        
        # Pipeline throughput
        self.pipeline_throughput_records_per_second = get_gauge(
            name=f"{PREFIX_PIPELINE}_throughput_records_per_second",
            description="Pipeline throughput in records per second",
            labels=[LABEL_SERVICE, LABEL_PIPELINE, LABEL_STAGE]
        )
    
    def _init_data_source_metrics(self) -> None:
        """Initialize data source-related metrics."""
        # Data source operations counter
        self.data_source_operations_total = get_counter(
            name="data_source_operations_total",
            description="Total number of data source operations",
            labels=[LABEL_SERVICE, "source_type", LABEL_OPERATION]
        )
        
        # Data source operation duration
        self.data_source_operation_duration_seconds = get_histogram(
            name="data_source_operation_duration_seconds",
            description="Data source operation duration in seconds",
            labels=[LABEL_SERVICE, "source_type", LABEL_OPERATION],
            buckets=LATENCY_BUCKETS_MEDIUM
        )
        
        # Data source errors
        self.data_source_errors_total = get_counter(
            name="data_source_errors_total",
            description="Total number of data source errors",
            labels=[LABEL_SERVICE, "source_type", LABEL_OPERATION, "error_type"]
        )
        
        # Data source latency
        self.data_source_latency_seconds = get_histogram(
            name="data_source_latency_seconds",
            description="Data source latency in seconds",
            labels=[LABEL_SERVICE, "source_type", "data_type"],
            buckets=LATENCY_BUCKETS_MEDIUM
        )
        
        # Data source availability
        self.data_source_availability = get_gauge(
            name="data_source_availability",
            description="Data source availability (0-1)",
            labels=[LABEL_SERVICE, "source_type", "data_type"]
        )
        
        # Data source throughput
        self.data_source_throughput_bytes_per_second = get_gauge(
            name="data_source_throughput_bytes_per_second",
            description="Data source throughput in bytes per second",
            labels=[LABEL_SERVICE, "source_type", "data_type"]
        )
    
    def _init_data_processing_metrics(self) -> None:
        """Initialize data processing-related metrics."""
        # Data processing operations counter
        self.data_processing_operations_total = get_counter(
            name=f"{PREFIX_DATA}_processing_operations_total",
            description="Total number of data processing operations",
            labels=[LABEL_SERVICE, "processor_type", LABEL_OPERATION]
        )
        
        # Data processing operation duration
        self.data_processing_operation_duration_seconds = get_histogram(
            name=f"{PREFIX_DATA}_processing_operation_duration_seconds",
            description="Data processing operation duration in seconds",
            labels=[LABEL_SERVICE, "processor_type", LABEL_OPERATION],
            buckets=LATENCY_BUCKETS_MEDIUM
        )
        
        # Data processing errors
        self.data_processing_errors_total = get_counter(
            name=f"{PREFIX_DATA}_processing_errors_total",
            description="Total number of data processing errors",
            labels=[LABEL_SERVICE, "processor_type", LABEL_OPERATION, "error_type"]
        )
        
        # Data processing throughput
        self.data_processing_throughput_records_per_second = get_gauge(
            name=f"{PREFIX_DATA}_processing_throughput_records_per_second",
            description="Data processing throughput in records per second",
            labels=[LABEL_SERVICE, "processor_type", "data_type"]
        )
        
        # Data processing batch size
        self.data_processing_batch_size = get_histogram(
            name=f"{PREFIX_DATA}_processing_batch_size",
            description="Data processing batch size",
            labels=[LABEL_SERVICE, "processor_type", "data_type"],
            buckets=[1, 10, 100, 1000, 10000, 100000, 1000000]
        )
        
        # Data processing memory usage
        self.data_processing_memory_usage_bytes = get_gauge(
            name=f"{PREFIX_DATA}_processing_memory_usage_bytes",
            description="Data processing memory usage in bytes",
            labels=[LABEL_SERVICE, "processor_type", "data_type"]
        )
    
    def _init_data_quality_metrics(self) -> None:
        """Initialize data quality-related metrics."""
        # Data quality operations counter
        self.data_quality_operations_total = get_counter(
            name="data_quality_operations_total",
            description="Total number of data quality operations",
            labels=[LABEL_SERVICE, "quality_check_type", LABEL_OPERATION]
        )
        
        # Data quality operation duration
        self.data_quality_operation_duration_seconds = get_histogram(
            name="data_quality_operation_duration_seconds",
            description="Data quality operation duration in seconds",
            labels=[LABEL_SERVICE, "quality_check_type", LABEL_OPERATION],
            buckets=LATENCY_BUCKETS_MEDIUM
        )
        
        # Data quality errors
        self.data_quality_errors_total = get_counter(
            name="data_quality_errors_total",
            description="Total number of data quality errors",
            labels=[LABEL_SERVICE, "quality_check_type", LABEL_OPERATION, "error_type"]
        )
        
        # Data quality score
        self.data_quality_score = get_gauge(
            name="data_quality_score",
            description="Data quality score (0-1)",
            labels=[LABEL_SERVICE, "data_type", "quality_dimension"]
        )
        
        # Data quality check failures
        self.data_quality_check_failures_total = get_counter(
            name="data_quality_check_failures_total",
            description="Total number of data quality check failures",
            labels=[LABEL_SERVICE, "data_type", "quality_check_type", "severity"]
        )
        
        # Data quality check duration
        self.data_quality_check_duration_seconds = get_histogram(
            name="data_quality_check_duration_seconds",
            description="Data quality check duration in seconds",
            labels=[LABEL_SERVICE, "data_type", "quality_check_type"],
            buckets=LATENCY_BUCKETS_MEDIUM
        )


# Singleton instance
data_pipeline_metrics = DataPipelineMetrics()
