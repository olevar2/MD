"""
Service-specific metrics for the Feature Store Service.

This module defines service-specific metrics for the Feature Store Service,
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
    PREFIX_FEATURE, PREFIX_SYSTEM, PREFIX_DB,
    LABEL_SERVICE, LABEL_FEATURE, LABEL_OPERATION, LABEL_DATASET
)

# Configure logging
logger = logging.getLogger(__name__)

# Service name
SERVICE_NAME = "feature-store-service"

class FeatureStoreMetrics(StandardMetrics):
    """
    Service-specific metrics for the Feature Store Service.
    
    This class extends the StandardMetrics class to provide service-specific metrics
    for the Feature Store Service.
    """
    
    def __init__(self):
        """Initialize the Feature Store metrics."""
        super().__init__(SERVICE_NAME)
        
        # Initialize service-specific metrics
        self._init_feature_metrics()
        self._init_dataset_metrics()
        self._init_computation_metrics()
        self._init_storage_metrics()
        
        logger.info(f"Initialized Feature Store metrics")
    
    def _init_feature_metrics(self) -> None:
        """Initialize feature-related metrics."""
        # Feature operations counter
        self.feature_operations_total = get_counter(
            name=f"{PREFIX_FEATURE}_operations_total",
            description="Total number of feature operations",
            labels=[LABEL_SERVICE, LABEL_FEATURE, LABEL_OPERATION]
        )
        
        # Feature operation duration
        self.feature_operation_duration_seconds = get_histogram(
            name=f"{PREFIX_FEATURE}_operation_duration_seconds",
            description="Feature operation duration in seconds",
            labels=[LABEL_SERVICE, LABEL_FEATURE, LABEL_OPERATION],
            buckets=LATENCY_BUCKETS_MEDIUM
        )
        
        # Feature errors
        self.feature_errors_total = get_counter(
            name=f"{PREFIX_FEATURE}_errors_total",
            description="Total number of feature errors",
            labels=[LABEL_SERVICE, LABEL_FEATURE, LABEL_OPERATION, "error_type"]
        )
        
        # Feature count
        self.feature_count = get_gauge(
            name=f"{PREFIX_FEATURE}_count",
            description="Number of features",
            labels=[LABEL_SERVICE, "feature_type", "feature_group"]
        )
        
        # Feature computation time
        self.feature_computation_time_seconds = get_histogram(
            name=f"{PREFIX_FEATURE}_computation_time_seconds",
            description="Feature computation time in seconds",
            labels=[LABEL_SERVICE, LABEL_FEATURE, "complexity"],
            buckets=LATENCY_BUCKETS_MEDIUM
        )
        
        # Feature cache hit ratio
        self.feature_cache_hit_ratio = get_gauge(
            name=f"{PREFIX_FEATURE}_cache_hit_ratio",
            description="Feature cache hit ratio (0-1)",
            labels=[LABEL_SERVICE, LABEL_FEATURE, "cache_level"]
        )
    
    def _init_dataset_metrics(self) -> None:
        """Initialize dataset-related metrics."""
        # Dataset operations counter
        self.dataset_operations_total = get_counter(
            name="dataset_operations_total",
            description="Total number of dataset operations",
            labels=[LABEL_SERVICE, LABEL_DATASET, LABEL_OPERATION]
        )
        
        # Dataset operation duration
        self.dataset_operation_duration_seconds = get_histogram(
            name="dataset_operation_duration_seconds",
            description="Dataset operation duration in seconds",
            labels=[LABEL_SERVICE, LABEL_DATASET, LABEL_OPERATION],
            buckets=LATENCY_BUCKETS_MEDIUM
        )
        
        # Dataset errors
        self.dataset_errors_total = get_counter(
            name="dataset_errors_total",
            description="Total number of dataset errors",
            labels=[LABEL_SERVICE, LABEL_DATASET, LABEL_OPERATION, "error_type"]
        )
        
        # Dataset size
        self.dataset_size_bytes = get_gauge(
            name="dataset_size_bytes",
            description="Dataset size in bytes",
            labels=[LABEL_SERVICE, LABEL_DATASET, "storage_type"]
        )
        
        # Dataset record count
        self.dataset_record_count = get_gauge(
            name="dataset_record_count",
            description="Number of records in dataset",
            labels=[LABEL_SERVICE, LABEL_DATASET]
        )
        
        # Dataset freshness
        self.dataset_freshness_seconds = get_gauge(
            name="dataset_freshness_seconds",
            description="Dataset freshness in seconds (time since last update)",
            labels=[LABEL_SERVICE, LABEL_DATASET]
        )
    
    def _init_computation_metrics(self) -> None:
        """Initialize computation-related metrics."""
        # Computation operations counter
        self.computation_operations_total = get_counter(
            name="computation_operations_total",
            description="Total number of computation operations",
            labels=[LABEL_SERVICE, "computation_type", LABEL_OPERATION]
        )
        
        # Computation operation duration
        self.computation_operation_duration_seconds = get_histogram(
            name="computation_operation_duration_seconds",
            description="Computation operation duration in seconds",
            labels=[LABEL_SERVICE, "computation_type", LABEL_OPERATION],
            buckets=LATENCY_BUCKETS_SLOW
        )
        
        # Computation errors
        self.computation_errors_total = get_counter(
            name="computation_errors_total",
            description="Total number of computation errors",
            labels=[LABEL_SERVICE, "computation_type", LABEL_OPERATION, "error_type"]
        )
        
        # Computation resource usage
        self.computation_cpu_usage_percent = get_gauge(
            name="computation_cpu_usage_percent",
            description="Computation CPU usage in percent",
            labels=[LABEL_SERVICE, "computation_type", "worker_id"]
        )
        
        self.computation_memory_usage_bytes = get_gauge(
            name="computation_memory_usage_bytes",
            description="Computation memory usage in bytes",
            labels=[LABEL_SERVICE, "computation_type", "worker_id"]
        )
        
        # Computation queue length
        self.computation_queue_length = get_gauge(
            name="computation_queue_length",
            description="Computation queue length",
            labels=[LABEL_SERVICE, "computation_type", "priority"]
        )
    
    def _init_storage_metrics(self) -> None:
        """Initialize storage-related metrics."""
        # Storage operations counter
        self.storage_operations_total = get_counter(
            name="storage_operations_total",
            description="Total number of storage operations",
            labels=[LABEL_SERVICE, "storage_type", LABEL_OPERATION]
        )
        
        # Storage operation duration
        self.storage_operation_duration_seconds = get_histogram(
            name="storage_operation_duration_seconds",
            description="Storage operation duration in seconds",
            labels=[LABEL_SERVICE, "storage_type", LABEL_OPERATION],
            buckets=LATENCY_BUCKETS_MEDIUM
        )
        
        # Storage errors
        self.storage_errors_total = get_counter(
            name="storage_errors_total",
            description="Total number of storage errors",
            labels=[LABEL_SERVICE, "storage_type", LABEL_OPERATION, "error_type"]
        )
        
        # Storage usage
        self.storage_usage_bytes = get_gauge(
            name="storage_usage_bytes",
            description="Storage usage in bytes",
            labels=[LABEL_SERVICE, "storage_type", "storage_tier"]
        )
        
        # Storage capacity
        self.storage_capacity_bytes = get_gauge(
            name="storage_capacity_bytes",
            description="Storage capacity in bytes",
            labels=[LABEL_SERVICE, "storage_type", "storage_tier"]
        )
        
        # Storage utilization
        self.storage_utilization_percent = get_gauge(
            name="storage_utilization_percent",
            description="Storage utilization in percent",
            labels=[LABEL_SERVICE, "storage_type", "storage_tier"]
        )


# Singleton instance
feature_store_metrics = FeatureStoreMetrics()
