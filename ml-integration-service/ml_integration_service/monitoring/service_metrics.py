"""
Service-specific metrics for the ML Integration Service.

This module defines service-specific metrics for the ML Integration Service,
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
    PREFIX_MODEL, PREFIX_SYSTEM, PREFIX_TRAINING,
    LABEL_SERVICE, LABEL_MODEL, LABEL_OPERATION, LABEL_VERSION
)

# Configure logging
logger = logging.getLogger(__name__)

# Service name
SERVICE_NAME = "ml-integration-service"

class MLIntegrationMetrics(StandardMetrics):
    """
    Service-specific metrics for the ML Integration Service.
    
    This class extends the StandardMetrics class to provide service-specific metrics
    for the ML Integration Service.
    """
    
    def __init__(self):
        """Initialize the ML Integration metrics."""
        super().__init__(SERVICE_NAME)
        
        # Initialize service-specific metrics
        self._init_model_metrics()
        self._init_training_metrics()
        self._init_inference_metrics()
        self._init_optimization_metrics()
        
        logger.info(f"Initialized ML Integration metrics")
    
    def _init_model_metrics(self) -> None:
        """Initialize model-related metrics."""
        # Model operations counter
        self.model_operations_total = get_counter(
            name=f"{PREFIX_MODEL}_operations_total",
            description="Total number of model operations",
            labels=[LABEL_SERVICE, LABEL_MODEL, LABEL_OPERATION]
        )
        
        # Model operation duration
        self.model_operation_duration_seconds = get_histogram(
            name=f"{PREFIX_MODEL}_operation_duration_seconds",
            description="Model operation duration in seconds",
            labels=[LABEL_SERVICE, LABEL_MODEL, LABEL_OPERATION],
            buckets=LATENCY_BUCKETS_MEDIUM
        )
        
        # Model errors
        self.model_errors_total = get_counter(
            name=f"{PREFIX_MODEL}_errors_total",
            description="Total number of model errors",
            labels=[LABEL_SERVICE, LABEL_MODEL, LABEL_OPERATION, "error_type"]
        )
        
        # Model count
        self.model_count = get_gauge(
            name=f"{PREFIX_MODEL}_count",
            description="Number of models",
            labels=[LABEL_SERVICE, "model_type", "status"]
        )
        
        # Model size
        self.model_size_bytes = get_gauge(
            name=f"{PREFIX_MODEL}_size_bytes",
            description="Model size in bytes",
            labels=[LABEL_SERVICE, LABEL_MODEL, LABEL_VERSION]
        )
        
        # Model performance metrics
        self.model_accuracy = get_gauge(
            name=f"{PREFIX_MODEL}_accuracy",
            description="Model accuracy (0-1)",
            labels=[LABEL_SERVICE, LABEL_MODEL, LABEL_VERSION, "dataset"]
        )
        
        self.model_f1_score = get_gauge(
            name=f"{PREFIX_MODEL}_f1_score",
            description="Model F1 score (0-1)",
            labels=[LABEL_SERVICE, LABEL_MODEL, LABEL_VERSION, "dataset"]
        )
        
        self.model_precision = get_gauge(
            name=f"{PREFIX_MODEL}_precision",
            description="Model precision (0-1)",
            labels=[LABEL_SERVICE, LABEL_MODEL, LABEL_VERSION, "dataset"]
        )
        
        self.model_recall = get_gauge(
            name=f"{PREFIX_MODEL}_recall",
            description="Model recall (0-1)",
            labels=[LABEL_SERVICE, LABEL_MODEL, LABEL_VERSION, "dataset"]
        )
    
    def _init_training_metrics(self) -> None:
        """Initialize training-related metrics."""
        # Training operations counter
        self.training_operations_total = get_counter(
            name=f"{PREFIX_TRAINING}_operations_total",
            description="Total number of training operations",
            labels=[LABEL_SERVICE, LABEL_MODEL, LABEL_OPERATION]
        )
        
        # Training operation duration
        self.training_operation_duration_seconds = get_histogram(
            name=f"{PREFIX_TRAINING}_operation_duration_seconds",
            description="Training operation duration in seconds",
            labels=[LABEL_SERVICE, LABEL_MODEL, LABEL_OPERATION],
            buckets=LATENCY_BUCKETS_SLOW
        )
        
        # Training errors
        self.training_errors_total = get_counter(
            name=f"{PREFIX_TRAINING}_errors_total",
            description="Total number of training errors",
            labels=[LABEL_SERVICE, LABEL_MODEL, LABEL_OPERATION, "error_type"]
        )
        
        # Training resource usage
        self.training_cpu_usage_percent = get_gauge(
            name=f"{PREFIX_TRAINING}_cpu_usage_percent",
            description="Training CPU usage in percent",
            labels=[LABEL_SERVICE, LABEL_MODEL, "worker_id"]
        )
        
        self.training_memory_usage_bytes = get_gauge(
            name=f"{PREFIX_TRAINING}_memory_usage_bytes",
            description="Training memory usage in bytes",
            labels=[LABEL_SERVICE, LABEL_MODEL, "worker_id"]
        )
        
        self.training_gpu_usage_percent = get_gauge(
            name=f"{PREFIX_TRAINING}_gpu_usage_percent",
            description="Training GPU usage in percent",
            labels=[LABEL_SERVICE, LABEL_MODEL, "worker_id", "gpu_id"]
        )
        
        # Training progress
        self.training_progress_percent = get_gauge(
            name=f"{PREFIX_TRAINING}_progress_percent",
            description="Training progress in percent",
            labels=[LABEL_SERVICE, LABEL_MODEL, "job_id"]
        )
        
        # Training metrics
        self.training_loss = get_gauge(
            name=f"{PREFIX_TRAINING}_loss",
            description="Training loss",
            labels=[LABEL_SERVICE, LABEL_MODEL, "job_id", "epoch"]
        )
        
        self.training_validation_loss = get_gauge(
            name=f"{PREFIX_TRAINING}_validation_loss",
            description="Training validation loss",
            labels=[LABEL_SERVICE, LABEL_MODEL, "job_id", "epoch"]
        )
    
    def _init_inference_metrics(self) -> None:
        """Initialize inference-related metrics."""
        # Inference operations counter
        self.inference_operations_total = get_counter(
            name="inference_operations_total",
            description="Total number of inference operations",
            labels=[LABEL_SERVICE, LABEL_MODEL, LABEL_VERSION]
        )
        
        # Inference operation duration
        self.inference_operation_duration_seconds = get_histogram(
            name="inference_operation_duration_seconds",
            description="Inference operation duration in seconds",
            labels=[LABEL_SERVICE, LABEL_MODEL, LABEL_VERSION],
            buckets=LATENCY_BUCKETS_FAST
        )
        
        # Inference errors
        self.inference_errors_total = get_counter(
            name="inference_errors_total",
            description="Total number of inference errors",
            labels=[LABEL_SERVICE, LABEL_MODEL, LABEL_VERSION, "error_type"]
        )
        
        # Inference batch size
        self.inference_batch_size = get_histogram(
            name="inference_batch_size",
            description="Inference batch size",
            labels=[LABEL_SERVICE, LABEL_MODEL, LABEL_VERSION],
            buckets=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        )
        
        # Inference resource usage
        self.inference_cpu_usage_percent = get_gauge(
            name="inference_cpu_usage_percent",
            description="Inference CPU usage in percent",
            labels=[LABEL_SERVICE, LABEL_MODEL, LABEL_VERSION]
        )
        
        self.inference_memory_usage_bytes = get_gauge(
            name="inference_memory_usage_bytes",
            description="Inference memory usage in bytes",
            labels=[LABEL_SERVICE, LABEL_MODEL, LABEL_VERSION]
        )
    
    def _init_optimization_metrics(self) -> None:
        """Initialize optimization-related metrics."""
        # Optimization operations counter
        self.optimization_operations_total = get_counter(
            name="optimization_operations_total",
            description="Total number of optimization operations",
            labels=[LABEL_SERVICE, "optimization_type", LABEL_OPERATION]
        )
        
        # Optimization operation duration
        self.optimization_operation_duration_seconds = get_histogram(
            name="optimization_operation_duration_seconds",
            description="Optimization operation duration in seconds",
            labels=[LABEL_SERVICE, "optimization_type", LABEL_OPERATION],
            buckets=LATENCY_BUCKETS_SLOW
        )
        
        # Optimization errors
        self.optimization_errors_total = get_counter(
            name="optimization_errors_total",
            description="Total number of optimization errors",
            labels=[LABEL_SERVICE, "optimization_type", LABEL_OPERATION, "error_type"]
        )
        
        # Optimization progress
        self.optimization_progress_percent = get_gauge(
            name="optimization_progress_percent",
            description="Optimization progress in percent",
            labels=[LABEL_SERVICE, "optimization_type", "job_id"]
        )
        
        # Optimization metrics
        self.optimization_objective_value = get_gauge(
            name="optimization_objective_value",
            description="Optimization objective value",
            labels=[LABEL_SERVICE, "optimization_type", "job_id", "iteration"]
        )
        
        # Hyperparameter optimization metrics
        self.hyperparameter_optimization_trials = get_gauge(
            name="hyperparameter_optimization_trials",
            description="Number of hyperparameter optimization trials",
            labels=[LABEL_SERVICE, LABEL_MODEL, "job_id"]
        )
        
        self.hyperparameter_optimization_best_score = get_gauge(
            name="hyperparameter_optimization_best_score",
            description="Best score from hyperparameter optimization",
            labels=[LABEL_SERVICE, LABEL_MODEL, "job_id", "metric"]
        )


# Singleton instance
ml_integration_metrics = MLIntegrationMetrics()
