"""
Model Retraining Service

This module implements the service responsible for integrating feedback
into the model retraining processes.
"""

from typing import Dict, List, Any, Optional, Union, Set, Tuple
import logging
from datetime import datetime, timedelta
import asyncio
import uuid
import json

from core_foundations.models.feedback import TradeFeedback, FeedbackSource, FeedbackCategory, FeedbackStatus
from core_foundations.utils.logger import get_logger
from core_foundations.events.event_publisher import EventPublisher
from core_foundations.events.event_schema import Event, EventType
from core_foundations.exceptions.feedback_exceptions import (
    ModelRetrainingError, FeedbackProcessingError, ModelPerformanceError
)
from core_foundations.resilience.circuit_breaker import CircuitBreaker, CircuitBreakerState

logger = get_logger(__name__)


class FeedbackClassifier:
    """
    Classifies feedback items to determine their relevance and priority
    for model retraining.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the feedback classifier.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Set default configuration
        self._set_default_config()
        
    def _set_default_config(self):
        """Set default configuration parameters."""
        defaults = {
            "high_priority_error_threshold": 0.1,  # 10% error considered high priority
            "low_priority_error_threshold": 0.05,  # 5% error considered low priority
            "min_samples_for_significance": 30,     # Minimum samples needed for statistical significance
            "recency_weight_factor": 0.8,          # Weight factor for recent samples (0-1)
            "recency_window_days": 7,              # How many days back to consider "recent"
            "error_type_weights": {
                "prediction_error": 1.0,
                "slippage_anomaly": 0.8,
                "execution_failure": 0.5,
                "market_regime_change": 1.2
            }
        }
        
        # Apply defaults for missing config values
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def classify(self, feedback: TradeFeedback) -> Dict[str, Any]:
        """
        Classify a feedback item.
        
        Args:
            feedback: The feedback to classify
            
        Returns:
            Dict[str, Any]: Classification results
        """
        # Default classification
        classification = {
            "priority": "low",
            "relevance_score": 0.5,
            "error_type": None,
            "retraining_recommended": False,
            "confidence": 0.5
        }
        
        # Extract error type if present
        error_type = None
        if hasattr(feedback, "metadata") and feedback.metadata:
            if isinstance(feedback.metadata, str):
                try:
                    metadata = json.loads(feedback.metadata)
                    error_type = metadata.get("error_type")
                except:
                    pass
            elif isinstance(feedback.metadata, dict):
                error_type = feedback.metadata.get("error_type")
        
        classification["error_type"] = error_type
        
        # Adjust relevance score based on error type
        if error_type and error_type in self.config["error_type_weights"]:
            classification["relevance_score"] = self.config["error_type_weights"][error_type]
        
        # Determine priority based on error magnitude if available
        error_magnitude = None
        if hasattr(feedback, "error_magnitude") and feedback.error_magnitude is not None:
            error_magnitude = feedback.error_magnitude
            
            if error_magnitude >= self.config["high_priority_error_threshold"]:
                classification["priority"] = "high"
                classification["relevance_score"] *= 2.0
            elif error_magnitude >= self.config["low_priority_error_threshold"]:
                classification["priority"] = "medium"
                classification["relevance_score"] *= 1.5
        
        # Adjust for recency
        if hasattr(feedback, "timestamp") and feedback.timestamp:
            try:
                fb_time = datetime.fromisoformat(feedback.timestamp)
                now = datetime.utcnow()
                days_old = (now - fb_time).days
                
                if days_old <= self.config["recency_window_days"]:
                    recency_factor = 1.0 - (days_old / self.config["recency_window_days"]) * (1.0 - self.config["recency_weight_factor"])
                    classification["relevance_score"] *= recency_factor
            except:
                pass
        
        # Determine if retraining is recommended
        if classification["priority"] == "high" or (classification["priority"] == "medium" and classification["relevance_score"] > 0.7):
            classification["retraining_recommended"] = True
        
        # Set confidence based on various factors
        if error_magnitude is not None and error_magnitude > 0:
            # Higher confidence when we have concrete error measurements
            classification["confidence"] = min(0.9, 0.5 + error_magnitude)
        
        return classification
    
    def classify_batch(self, feedback_items: List[TradeFeedback]) -> Dict[str, Any]:
        """
        Classify a batch of feedback items.
        
        Args:
            feedback_items: List of feedback items to classify
            
        Returns:
            Dict[str, Any]: Batch classification results
        """
        if not feedback_items:
            return {
                "priority": "low",
                "relevance_score": 0.0,
                "retraining_recommended": False,
                "confidence": 0.0,
                "sample_size": 0
            }
        
        # Classify each item
        classifications = [self.classify(fb) for fb in feedback_items]
        
        # Aggregate results
        result = {
            "sample_size": len(classifications),
            "high_priority_count": len([c for c in classifications if c["priority"] == "high"]),
            "medium_priority_count": len([c for c in classifications if c["priority"] == "medium"]),
            "low_priority_count": len([c for c in classifications if c["priority"] == "low"]),
            "avg_relevance_score": sum(c["relevance_score"] for c in classifications) / len(classifications),
            "error_types": {}
        }
        
        # Count error types
        for c in classifications:
            if c["error_type"]:
                error_type = c["error_type"]
                result["error_types"][error_type] = result["error_types"].get(error_type, 0) + 1
        
        # Determine overall priority
        if result["high_priority_count"] >= max(1, len(classifications) * 0.2):
            result["priority"] = "high"
        elif result["medium_priority_count"] >= max(1, len(classifications) * 0.3):
            result["priority"] = "medium"
        else:
            result["priority"] = "low"
        
        # Determine if retraining is recommended
        result["retraining_recommended"] = (
            result["priority"] == "high" or 
            (result["priority"] == "medium" and result["avg_relevance_score"] > 0.7)
        )
        
        # Confidence based on sample size
        if result["sample_size"] >= self.config["min_samples_for_significance"]:
            result["confidence"] = min(0.95, 0.5 + (result["sample_size"] / (2 * self.config["min_samples_for_significance"])))
        else:
            result["confidence"] = max(0.3, result["sample_size"] / self.config["min_samples_for_significance"])
        
        return result


class ModelPerformanceEvaluator:
    """
    Evaluates model performance based on feedback data.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the model performance evaluator.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Set default configuration
        self._set_default_config()
        
    def _set_default_config(self):
        """Set default configuration parameters."""
        defaults = {
            "critical_error_threshold": 0.15,    # 15% error is critical
            "warning_error_threshold": 0.08,     # 8% error is warning
            "min_samples_for_evaluation": 50,    # Minimum samples needed for reliable evaluation
            "confidence_interval": 0.95,         # 95% confidence interval for metrics
            "recency_weighted_eval": True,       # Whether to weight recent samples more heavily
            "recency_weight_factor": 0.7,        # Weight factor for recent samples (0-1)
            "recency_window_days": 14,           # How many days back to consider for recency weighting
        }
        
        # Apply defaults for missing config values
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def evaluate_model_performance(
        self,
        feedback_items: List[TradeFeedback],
        model_id: str,
        baseline_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model performance based on feedback.
        
        Args:
            feedback_items: Feedback items for the model
            model_id: ID of the model to evaluate
            baseline_metrics: Optional baseline metrics for comparison
            
        Returns:
            Dict[str, Any]: Performance evaluation results
        """
        if not feedback_items:
            return {
                "model_id": model_id,
                "status": "insufficient_data",
                "sample_size": 0,
                "confidence": 0.0,
                "metrics": {},
                "recommendation": "collect_more_data"
            }
        
        # Filter to relevant feedback for this model
        model_feedback = [
            fb for fb in feedback_items 
            if hasattr(fb, "model_id") and fb.model_id == model_id
        ]
        
        if not model_feedback:
            return {
                "model_id": model_id,
                "status": "no_relevant_data",
                "sample_size": 0,
                "confidence": 0.0,
                "metrics": {},
                "recommendation": "collect_relevant_data"
            }
            
        # Check if we have enough samples
        if len(model_feedback) < self.config["min_samples_for_evaluation"]:
            return {
                "model_id": model_id,
                "status": "insufficient_data",
                "sample_size": len(model_feedback),
                "confidence": min(0.5, len(model_feedback) / self.config["min_samples_for_evaluation"]),
                "metrics": self._calculate_basic_metrics(model_feedback),
                "recommendation": "collect_more_data"
            }
        
        # Calculate metrics
        metrics = self._calculate_metrics(model_feedback)
        
        # Compare with baseline if available
        if baseline_metrics:
            metrics["baseline_comparison"] = self._compare_with_baseline(metrics, baseline_metrics)
        
        # Determine overall status
        avg_error = metrics.get("avg_error", 0.0)
        if avg_error >= self.config["critical_error_threshold"]:
            status = "critical"
            recommendation = "retrain_immediately"
        elif avg_error >= self.config["warning_error_threshold"]:
            status = "warning"
            recommendation = "schedule_retraining"
        else:
            status = "good"
            recommendation = "monitor"
        
        # Confidence based on sample size and data consistency
        confidence = min(0.95, 0.5 + (len(model_feedback) / (2 * self.config["min_samples_for_evaluation"])))
        if "error_std_dev" in metrics and metrics["avg_error"] > 0:
            # Lower confidence if error variance is high relative to mean
            coefficient_of_variation = metrics["error_std_dev"] / metrics["avg_error"]
            if coefficient_of_variation > 1.0:
                confidence *= (1.0 / coefficient_of_variation)
        
        return {
            "model_id": model_id,
            "status": status,
            "sample_size": len(model_feedback),
            "confidence": confidence,
            "metrics": metrics,
            "recommendation": recommendation,
            "evaluation_timestamp": datetime.utcnow().isoformat()
        }
    
    def _calculate_basic_metrics(self, feedback_items: List[TradeFeedback]) -> Dict[str, float]:
        """Calculate basic metrics when we have limited data."""
        metrics = {}
        
        # Extract error magnitudes if available
        error_magnitudes = []
        for fb in feedback_items:
            if hasattr(fb, "error_magnitude") and fb.error_magnitude is not None:
                error_magnitudes.append(fb.error_magnitude)
        
        if error_magnitudes:
            metrics["avg_error"] = sum(error_magnitudes) / len(error_magnitudes)
            
            # Simple min/max
            metrics["min_error"] = min(error_magnitudes)
            metrics["max_error"] = max(error_magnitudes)
        
        return metrics
    
    def _calculate_metrics(self, feedback_items: List[TradeFeedback]) -> Dict[str, Any]:
        """Calculate comprehensive metrics from feedback data."""
        metrics = self._calculate_basic_metrics(feedback_items)
        
        # Extract error magnitudes if available
        error_magnitudes = []
        timestamps = []
        
        for fb in feedback_items:
            if hasattr(fb, "error_magnitude") and fb.error_magnitude is not None:
                error_magnitudes.append(fb.error_magnitude)
                
                if hasattr(fb, "timestamp") and fb.timestamp:
                    try:
                        timestamps.append(datetime.fromisoformat(fb.timestamp))
                    except:
                        timestamps.append(None)
                else:
                    timestamps.append(None)
        
        if not error_magnitudes:
            return metrics
        
        # Calculate standard deviation
        if len(error_magnitudes) > 1:
            mean = metrics["avg_error"]
            variance = sum((x - mean) ** 2 for x in error_magnitudes) / len(error_magnitudes)
            metrics["error_std_dev"] = variance ** 0.5
            
            # Calculate 95% confidence interval using simple approximation
            # For small samples, should use t-distribution, but this is a reasonable approximation
            std_error = metrics["error_std_dev"] / (len(error_magnitudes) ** 0.5)
            z_value = 1.96  # Approximately 95% confidence interval
            metrics["error_ci_lower"] = max(0, metrics["avg_error"] - z_value * std_error)
            metrics["error_ci_upper"] = metrics["avg_error"] + z_value * std_error
        
        # Calculate recency-weighted metrics if timestamps available
        if self.config["recency_weighted_eval"] and all(t is not None for t in timestamps):
            weighted_errors = []
            weights = []
            
            now = datetime.utcnow()
            max_age = timedelta(days=self.config["recency_window_days"])
            
            for error, timestamp in zip(error_magnitudes, timestamps):
                if timestamp:
                    age = now - timestamp
                    if age <= max_age:
                        # More weight for recent samples
                        weight = 1.0 - ((1.0 - self.config["recency_weight_factor"]) * (age / max_age))
                        weighted_errors.append(error * weight)
                        weights.append(weight)
            
            if weighted_errors and sum(weights) > 0:
                metrics["recency_weighted_avg_error"] = sum(weighted_errors) / sum(weights)
        
        # Try to calculate error trend
        if all(t is not None for t in timestamps) and len(timestamps) >= 3:
            # Group by day for trend analysis
            daily_errors = {}
            
            for error, timestamp in zip(error_magnitudes, timestamps):
                day_key = timestamp.date()
                if day_key not in daily_errors:
                    daily_errors[day_key] = []
                daily_errors[day_key].append(error)
            
            # Calculate daily averages
            daily_avgs = [(day, sum(errors)/len(errors)) for day, errors in daily_errors.items()]
            daily_avgs.sort()  # Sort by date
            
            if len(daily_avgs) >= 3:
                # Calculate simple linear regression slope for trend
                x = list(range(len(daily_avgs)))
                y = [avg for _, avg in daily_avgs]
                
                n = len(x)
                sum_x = sum(x)
                sum_y = sum(y)
                sum_xy = sum(x_i * y_i for x_i, y_i in zip(x, y))
                sum_x_squared = sum(x_i ** 2 for x_i in x)
                
                try:
                    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
                    metrics["error_trend_slope"] = slope
                    
                    # Interpret the trend
                    if slope > 0.01:  # Error increasing
                        metrics["error_trend"] = "increasing"
                    elif slope < -0.01:  # Error decreasing
                        metrics["error_trend"] = "decreasing"
                    else:  # Stable
                        metrics["error_trend"] = "stable"
                except:
                    # Division by zero or other numerical issues
                    metrics["error_trend"] = "unknown"
        
        return metrics
    
    def _compare_with_baseline(self, current_metrics: Dict[str, float], baseline_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Compare current metrics with baseline."""
        comparison = {}
        
        for key in ["avg_error", "error_std_dev", "recency_weighted_avg_error"]:
            if key in current_metrics and key in baseline_metrics and baseline_metrics[key] > 0:
                change_pct = (current_metrics[key] - baseline_metrics[key]) / baseline_metrics[key] * 100
                comparison[f"{key}_change_pct"] = change_pct
                
                # Interpret the change
                if key == "avg_error" or key == "recency_weighted_avg_error":
                    if change_pct <= -10:  # At least 10% improvement
                        comparison[f"{key}_change"] = "significant_improvement"
                    elif change_pct <= -5:  # 5-10% improvement
                        comparison[f"{key}_change"] = "improvement"
                    elif change_pct >= 10:  # At least 10% worse
                        comparison[f"{key}_change"] = "significant_degradation"
                    elif change_pct >= 5:  # 5-10% worse
                        comparison[f"{key}_change"] = "degradation"
                    else:  # Within ±5%
                        comparison[f"{key}_change"] = "stable"
        
        # Overall assessment
        if "avg_error_change_pct" in comparison:
            if comparison["avg_error_change_pct"] <= -10:
                comparison["overall_assessment"] = "significantly_better"
            elif comparison["avg_error_change_pct"] <= -5:
                comparison["overall_assessment"] = "better"
            elif comparison["avg_error_change_pct"] >= 10:
                comparison["overall_assessment"] = "significantly_worse"
            elif comparison["avg_error_change_pct"] >= 5:
                comparison["overall_assessment"] = "worse"
            else:
                comparison["overall_assessment"] = "comparable"
        
        return comparison


class TrainingPipelineIntegrator:
    """
    Integrates feedback into training pipelines for model retraining.
    """
    
    def __init__(
        self,
        event_publisher: Optional[EventPublisher] = None,
        adaptation_engine: Optional['AdaptationEngine'] = None,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the training pipeline integrator.
        
        Args:
            event_publisher: Event publisher for broadcasting events
            adaptation_engine: Adaptation engine for triggering model retraining
            config: Configuration parameters
        """
        self.event_publisher = event_publisher
        self.adaptation_engine = adaptation_engine
        self.config = config or {}
        
        # Track retraining jobs
        self.active_retraining_jobs = {}
        
        # Set default configuration
        self._set_default_config()
        
        # Circuit breakers for external services
        self.ml_workbench_circuit_breaker = CircuitBreaker(
            name="ml_workbench_service",
            failure_threshold=self.config.get("circuit_breaker_failure_threshold", 5),
            recovery_timeout=self.config.get("circuit_breaker_recovery_timeout", 30),
            half_open_success_threshold=self.config.get("circuit_breaker_success_threshold", 3)
        )
        
        logger.info("TrainingPipelineIntegrator initialized")
    
    def _set_default_config(self):
        """Set default configuration parameters."""
        defaults = {
            "retry_attempts": 3,               # Number of retry attempts for service calls
            "retry_delay": 1.0,                # Initial delay between retries (seconds)
            "feedback_weights": {              # Weights for different feedback types
                "prediction_error": 1.0,
                "slippage_anomaly": 0.8,
                "execution_failure": 0.5,
                "market_regime_change": 1.2
            },
            "ml_workbench_service_url": None,  # URL for ML Workbench service
            "workbench_service_timeout": 30,    # Timeout for ML Workbench service calls (seconds)
            "circuit_breaker_failure_threshold": 5,  # Failures before circuit breaker trips
            "circuit_breaker_recovery_timeout": 30,  # Seconds before attempting recovery
            "circuit_breaker_success_threshold": 3,  # Successes needed to close circuit breaker
        }
        
        # Apply defaults for missing config values
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    async def prepare_feedback_for_training(
        self,
        feedback_items: List[TradeFeedback],
        model_id: str
    ) -> Dict[str, Any]:
        """
        Prepare feedback data for model training.
        
        Args:
            feedback_items: Feedback items to prepare
            model_id: ID of the model to prepare for
            
        Returns:
            Dict[str, Any]: Prepared training data
        """
        # Filter to relevant feedback for this model
        model_feedback = [
            fb for fb in feedback_items 
            if hasattr(fb, "model_id") and fb.model_id == model_id
        ]
        
        if not model_feedback:
            logger.warning(f"No feedback items found for model {model_id}")
            return {
                "model_id": model_id,
                "status": "no_data",
                "prepared_data": None
            }
        
        try:
            # Group feedback by source
            grouped = {}
            for fb in model_feedback:
                source = fb.source.value
                if source not in grouped:
                    grouped[source] = []
                grouped[source].append(fb)
            
            # Transform feedback into training features and targets
            transformed_data = []
            
            for fb in model_feedback:
                # This is where we would extract features from the feedback
                # For now, we'll create a simple placeholder structure
                sample = {
                    "feedback_id": fb.id,
                    "timestamp": fb.timestamp,
                    "features": {},
                    "target": None
                }
                
                # Extract features from feedback metadata
                if hasattr(fb, "metadata") and fb.metadata:
                    metadata = {}
                    if isinstance(fb.metadata, str):
                        try:
                            metadata = json.loads(fb.metadata)
                        except:
                            pass
                    elif isinstance(fb.metadata, dict):
                        metadata = fb.metadata
                    
                    # Copy relevant metadata to features
                    for key, value in metadata.items():
                        if key not in ["id", "timestamp", "source", "category"]:
                            sample["features"][key] = value
                
                # Set target value based on feedback type
                if hasattr(fb, "error_magnitude") and fb.error_magnitude is not None:
                    sample["target"] = fb.error_magnitude
                
                transformed_data.append(sample)
            
            # Apply weights based on feedback type
            weighted_data = []
            for item in transformed_data:
                weight = 1.0
                if "error_type" in item["features"]:
                    error_type = item["features"]["error_type"]
                    if error_type in self.config["feedback_weights"]:
                        weight = self.config["feedback_weights"][error_type]
                
                weighted_data.append({
                    **item,
                    "weight": weight
                })
            
            return {
                "model_id": model_id,
                "status": "prepared",
                "sample_count": len(weighted_data),
                "prepared_data": weighted_data,
                "data_stats": {
                    "by_source": {source: len(items) for source, items in grouped.items()},
                    "total": len(model_feedback)
                }
            }
            
        except Exception as e:
            logger.error(f"Error preparing feedback for model {model_id}: {str(e)}", exc_info=True)
            raise FeedbackProcessingError(f"Failed to prepare feedback: {str(e)}")
    
    async def trigger_model_retraining(
        self,
        model_id: str,
        feedback_items: List[TradeFeedback],
        retraining_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Triggers model retraining based on feedback, waits for completion,
        and emits an event with the results.
        
        Args:
            model_id: ID of the model to retrain
            feedback_items: Feedback items that triggered the retraining
            retraining_params: Additional parameters for retraining
            
        Returns:
            Dict[str, Any]: Status of the retraining job
            
        Raises:
            ModelRetrainingError: If retraining fails
        """
        if not self.adaptation_engine:
            error_msg = f"Cannot trigger retraining for model {model_id}: Adaptation engine not available"
            logger.error(error_msg)
            raise ModelRetrainingError(error_msg)
            
        logger.info(f"Triggering retraining for model {model_id} based on {len(feedback_items)} feedback items")
        
        # Prepare parameters for retraining
        prepared_data = await self.prepare_feedback_for_training(feedback_items, model_id)
        
        # Combine prepared data with additional parameters
        params = retraining_params or {}
        params.update({
            "feedback_summary": prepared_data.get("feedback_summary", {}),
            "training_data_hints": prepared_data.get("training_data_hints", {}),
            "triggered_by": "feedback_loop",
            "feedback_count": len(feedback_items),
            "timestamp": datetime.utcnow().isoformat()
        })
        
        try:
            # Call the adaptation engine to trigger retraining
            job_id = await self.adaptation_engine.trigger_model_retraining(model_id, params)
            
            if not job_id:
                error_msg = f"Failed to get job ID when triggering retraining for model {model_id}"
                logger.error(error_msg)
                raise ModelRetrainingError(error_msg)
                
            logger.info(f"Successfully triggered retraining for model {model_id}. Job ID: {job_id}")
            
            # Store job details
            self.active_retraining_jobs[job_id] = {
                "model_id": model_id,
                "start_time": datetime.utcnow(),
                "status": "running",
                "feedback_ids": [fb.id for fb in feedback_items if hasattr(fb, 'id')]
            }
            
            # Start async task to monitor job status
            asyncio.create_task(self._monitor_retraining_job(job_id, model_id))
            
            return {
                "job_id": job_id,
                "model_id": model_id,
                "status": "triggered",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error triggering retraining for model {model_id}: {e}", exc_info=True)
            if isinstance(e, ModelRetrainingError):
                raise
            raise ModelRetrainingError(f"Failed to trigger retraining: {str(e)}")
    
    async def _monitor_retraining_job(self, job_id: str, model_id: str) -> None:
        """
        Monitors a retraining job until completion and publishes the result event.
        
        Args:
            job_id: ID of the retraining job
            model_id: ID of the model being retrained
        """
        if not self.adaptation_engine or not hasattr(self.adaptation_engine, 'ml_client'):
            logger.error(f"Cannot monitor job {job_id}: Adaptation engine or ML client not available")
            return
            
        ml_client = self.adaptation_engine.ml_client
        poll_interval = self.config.get("job_poll_interval_seconds", 30)
        max_polls = self.config.get("max_job_polls", 120)  # Default 1 hour timeout at 30s intervals
        
        polls = 0
        while polls < max_polls:
            try:
                # Wait for next polling interval
                await asyncio.sleep(poll_interval)
                polls += 1
                
                # Check job status
                job_status = await ml_client.get_job_status(job_id)
                status = job_status.get("status", "").lower()
                
                if status in ["completed", "succeeded", "finished"]:
                    logger.info(f"Retraining job {job_id} for model {model_id} completed successfully")
                    await self._handle_successful_retraining(job_id, model_id, job_status)
                    return
                    
                elif status in ["failed", "error"]:
                    error_msg = job_status.get("error", "Unknown error")
                    logger.error(f"Retraining job {job_id} for model {model_id} failed: {error_msg}")
                    await self._handle_failed_retraining(job_id, model_id, error_msg)
                    return
                    
                else:
                    # Still running, update progress if available
                    progress = job_status.get("progress")
                    if progress:
                        logger.debug(f"Retraining job {job_id} for model {model_id} in progress: {progress}%")
                    
                    # Update job tracking info
                    if job_id in self.active_retraining_jobs:
                        self.active_retraining_jobs[job_id]["last_poll"] = datetime.utcnow()
                        self.active_retraining_jobs[job_id]["status"] = status
            
            except Exception as e:
                logger.error(f"Error polling status for job {job_id}: {e}")
                
                # Continue polling despite errors
                if polls % 5 == 0:  # Log less frequently after errors to avoid log spam
                    logger.info(f"Still monitoring job {job_id} after polling errors")
        
        # If we get here, we've reached the maximum polls without completion
        logger.warning(f"Retraining job {job_id} for model {model_id} timed out after {polls} polls")
        await self._handle_timeout_retraining(job_id, model_id)
    
    async def _handle_successful_retraining(self, job_id: str, model_id: str, job_status: Dict[str, Any]) -> None:
        """
        Handles successful completion of a retraining job.
        
        Args:
            job_id: ID of the completed job
            model_id: ID of the model that was retrained
            job_status: Status information from the job
        """
        # Extract job tracking info
        job_info = self.active_retraining_jobs.get(job_id, {})
        feedback_ids = job_info.get("feedback_ids", [])
        
        # Extract model performance metrics if available
        model_metrics = job_status.get("metrics", {})
        if not model_metrics and hasattr(self.adaptation_engine, 'ml_client'):
            try:
                # Try to get detailed performance metrics
                performance_data = await self.adaptation_engine.ml_client.get_model_performance(model_id)
                model_metrics = performance_data.get("metrics", {})
            except Exception as e:
                logger.error(f"Failed to get performance metrics for model {model_id}: {e}")
        
        # Create event data
        event_data = {
            "event_type": "model_training_completed",
            "job_id": job_id,
            "model_id": model_id,
            "status": "success",
            "start_time": job_info.get("start_time", datetime.utcnow()).isoformat(),
            "completion_time": datetime.utcnow().isoformat(),
            "feedback_ids": feedback_ids,
            "model_metrics": model_metrics
        }
        
        # Emit event
        if self.event_publisher:
            try:
                await self.event_publisher.publish(
                    EventType.MODEL_TRAINING_COMPLETED,
                    event_data
                )
                logger.info(f"Published model_training_completed event for model {model_id}")
            except Exception as e:
                logger.error(f"Failed to publish model_training_completed event: {e}")
        
        # Clean up job tracking
        self.active_retraining_jobs.pop(job_id, None)
    
    async def _handle_failed_retraining(self, job_id: str, model_id: str, error_msg: str) -> None:
        """
        Handles failure of a retraining job.
        
        Args:
            job_id: ID of the failed job
            model_id: ID of the model that was being retrained
            error_msg: Error message from the job
        """
        # Extract job tracking info
        job_info = self.active_retraining_jobs.get(job_id, {})
        feedback_ids = job_info.get("feedback_ids", [])
        
        # Create event data
        event_data = {
            "event_type": "model_training_failed",
            "job_id": job_id,
            "model_id": model_id,
            "status": "failed",
            "start_time": job_info.get("start_time", datetime.utcnow()).isoformat(),
            "failure_time": datetime.utcnow().isoformat(),
            "feedback_ids": feedback_ids,
            "error": error_msg
        }
        
        # Emit event
        if self.event_publisher:
            try:
                await self.event_publisher.publish(
                    EventType.MODEL_TRAINING_FAILED,
                    event_data
                )
                logger.info(f"Published model_training_failed event for model {model_id}")
            except Exception as e:
                logger.error(f"Failed to publish model_training_failed event: {e}")
        
        # Clean up job tracking
        self.active_retraining_jobs.pop(job_id, None)
    
    async def _handle_timeout_retraining(self, job_id: str, model_id: str) -> None:
        """
        Handles timeout of a retraining job.
        
        Args:
            job_id: ID of the timed out job
            model_id: ID of the model that was being retrained
        """
        # Similar logic to failed retraining but with timeout status
        job_info = self.active_retraining_jobs.get(job_id, {})
        feedback_ids = job_info.get("feedback_ids", [])
        
        event_data = {
            "event_type": "model_training_timeout",
            "job_id": job_id,
            "model_id": model_id,
            "status": "timeout",
            "start_time": job_info.get("start_time", datetime.utcnow()).isoformat(),
            "timeout_time": datetime.utcnow().isoformat(),
            "feedback_ids": feedback_ids
        }
        
        if self.event_publisher:
            try:
                await self.event_publisher.publish(
                    EventType.MODEL_TRAINING_FAILED,  # Use same event type as failure
                    event_data
                )
                logger.info(f"Published model_training_timeout event for model {model_id}")
            except Exception as e:
                logger.error(f"Failed to publish model_training_timeout event: {e}")
        
        self.active_retraining_jobs.pop(job_id, None)


class ModelRetrainingService:
    """
    Service that integrates feedback into model retraining processes.
    """
    
    def __init__(
        self,
        event_publisher: Optional[EventPublisher] = None,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the model retraining service.
        
        Args:
            event_publisher: Event publisher for broadcasting events
            config: Configuration parameters
        """
        self.event_publisher = event_publisher
        self.config = config or {}
        
        # Initialize sub-components
        self.classifier = FeedbackClassifier(config.get("classifier_config"))
        self.evaluator = ModelPerformanceEvaluator(config.get("evaluator_config"))
        self.integrator = TrainingPipelineIntegrator(event_publisher, config.get("integrator_config"))
        
        # Set default configuration
        self._set_default_config()
        
        # Track retraining jobs
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        
        # Background task
        self._background_task = None
        self._is_running = False
        
        logger.info("ModelRetrainingService initialized")
    
    def _set_default_config(self):
        """Set default configuration parameters."""
        defaults = {
            "job_check_interval": 60,         # Check jobs every 60 seconds
            "max_concurrent_jobs": 3,         # Maximum concurrent retraining jobs
            "job_timeout": 3600,              # Job timeout in seconds (1 hour)
            "enable_background_task": True,   # Whether to enable background task
        }
        
        # Apply defaults for missing config values
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    async def start(self):
        """Start the model retraining service."""
        self._is_running = True
        
        # Start background task
        if self.config["enable_background_task"]:
            self._background_task = asyncio.create_task(self._background_loop())
            logger.info("Background task started")
            
        logger.info("ModelRetrainingService started")
    
    async def stop(self):
        """Stop the model retraining service."""
        self._is_running = False
        
        # Cancel background task
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
            
        logger.info("ModelRetrainingService stopped")
    
    async def _background_loop(self):
        """Background task for monitoring retraining jobs."""
        try:
            while self._is_running:
                # Sleep at the beginning to avoid immediate processing
                await asyncio.sleep(self.config["job_check_interval"])
                
                # Check active jobs
                await self._check_active_jobs()
                
        except asyncio.CancelledError:
            logger.info("Background task canceled")
        except Exception as e:
            logger.error(f"Error in background task: {str(e)}", exc_info=True)
    
    async def _check_active_jobs(self):
        """Check the status of active retraining jobs."""
        if not self.active_jobs:
            return
        
        # List of jobs to remove (completed or failed)
        jobs_to_remove = []
        
        for job_id, job_info in self.active_jobs.items():
            try:
                # Skip jobs that are already completed or failed
                if job_info["status"] in ["completed", "failed"]:
                    continue
                
                # Check job timeout
                start_time = datetime.fromisoformat(job_info["start_time"])
                if (datetime.utcnow() - start_time).total_seconds() > self.config["job_timeout"]:
                    logger.warning(f"Job {job_id} for model {job_info['model_id']} timed out")
                    job_info["status"] = "failed"
                    job_info["error"] = "Job timed out"
                    jobs_to_remove.append(job_id)
                    
                    # Publish event if possible
                    if self.event_publisher:
                        await self.event_publisher.publish(
                            "model.retraining.failed",
                            {
                                "job_id": job_id,
                                "model_id": job_info["model_id"],
                                "error": "Job timed out",
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        )
                    
                    continue
                
                # Check job status
                job_status = await self.integrator.check_retraining_job_status(job_id)
                
                # Update job info with latest status
                job_info.update(job_status)
                
                # If job completed or failed, mark for removal
                if job_status["status"] in ["completed", "failed"]:
                    jobs_to_remove.append(job_id)
                    
                    # Publish event if possible
                    if self.event_publisher:
                        event_type = f"model.retraining.{job_status['status']}"
                        await self.event_publisher.publish(
                            event_type,
                            {
                                "job_id": job_id,
                                "model_id": job_info["model_id"],
                                **{k: v for k, v in job_status.items() if k not in ["job_id", "status"]}
                            }
                        )
                
            except Exception as e:
                logger.error(f"Error checking job {job_id}: {str(e)}", exc_info=True)
        
        # Remove completed or failed jobs
        for job_id in jobs_to_remove:
            del self.active_jobs[job_id]
    
    async def process_feedback(
        self,
        model_id: str,
        feedback_items: List[TradeFeedback]
    ) -> Dict[str, Any]:
        """
        Process feedback for a model.
        
        Args:
            model_id: ID of the model
            feedback_items: Feedback items to process
            
        Returns:
            Dict[str, Any]: Processing result
        """
        try:
            # Step 1: Classify feedback
            classification = self.classifier.classify_batch(feedback_items)
            logger.info(f"Classified {len(feedback_items)} feedback items for model {model_id}: {classification['priority']} priority")
            
            # Step 2: Evaluate model performance
            evaluation = await self.evaluator.evaluate_model_performance(feedback_items, model_id)
            logger.info(f"Evaluated model {model_id} performance: {evaluation['status']}")
            
            # Step 3: Determine if retraining is needed
            retraining_needed = (
                classification["retraining_recommended"] or 
                evaluation["recommendation"] in ["retrain_immediately", "schedule_retraining"]
            )
            
            result = {
                "model_id": model_id,
                "feedback_count": len(feedback_items),
                "classification": classification,
                "evaluation": evaluation,
                "retraining_recommended": retraining_needed,
                "processed_at": datetime.utcnow().isoformat()
            }
            
            # Early return if retraining is not needed
            if not retraining_needed:
                logger.info(f"Retraining not recommended for model {model_id}")
                return result
            
            # Check if we are already retraining this model
            active_model_jobs = [
                job_info for job_info in self.active_jobs.values() 
                if job_info["model_id"] == model_id and job_info["status"] not in ["completed", "failed"]
            ]
            
            if active_model_jobs:
                logger.info(f"Model {model_id} already has active retraining job(s): {len(active_model_jobs)}")
                
                # Add references to active jobs
                result["active_jobs"] = [
                    {
                        "job_id": job_info["job_id"],
                        "status": job_info["status"],
                        "start_time": job_info["start_time"]
                    }
                    for job_info in active_model_jobs
                ]
                
                return result
            
            # Check if we have capacity for a new job
            if len(self.active_jobs) >= self.config["max_concurrent_jobs"]:
                logger.warning(f"Cannot schedule retraining for model {model_id}, maximum concurrent jobs reached")
                result["retraining_status"] = "deferred"
                result["deferred_reason"] = "max_concurrent_jobs_reached"
                return result
            
            # Step 4: Prepare feedback data for retraining
            prepared_data = await self.integrator.prepare_feedback_for_training(feedback_items, model_id)
            logger.info(f"Prepared feedback data for model {model_id}: {prepared_data['status']}")
            
            if prepared_data["status"] != "prepared" or not prepared_data["prepared_data"]:
                result["retraining_status"] = "failed"
                result["error"] = "Failed to prepare feedback data"
                return result
            
            # Step 5: Trigger retraining
            retraining_urgency = "normal"
            if evaluation["recommendation"] == "retrain_immediately":
                retraining_urgency = "high"
            
            retraining_params = {
                "urgency": retraining_urgency,
                "feedback_sample_count": prepared_data["sample_count"]
            }
            
            retraining_job = await self.integrator.trigger_model_retraining(
                model_id, prepared_data, retraining_params
            )
            
            logger.info(f"Triggered retraining for model {model_id}: {retraining_job['status']}")
            
            # Track the job
            if retraining_job["status"] == "submitted":
                self.active_jobs[retraining_job["job_id"]] = {
                    **retraining_job,
                    "model_id": model_id,
                    "start_time": datetime.utcnow().isoformat()
                }
                
                result["retraining_status"] = "submitted"
                result["retraining_job"] = retraining_job
            else:
                result["retraining_status"] = "failed"
                result["error"] = "Failed to submit retraining job"
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing feedback for model {model_id}: {str(e)}", exc_info=True)
            raise ModelRetrainingError(f"Failed to process feedback: {str(e)}")
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a retraining job.
        
        Args:
            job_id: ID of the retraining job
            
        Returns:
            Dict[str, Any]: Job status information
        """
        # Check if we're tracking this job
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]
        
        # Try to get status from integrator
        try:
            return await self.integrator.check_retraining_job_status(job_id)
        except Exception as e:
            logger.error(f"Error getting status for job {job_id}: {str(e)}", exc_info=True)
            raise ModelRetrainingError(f"Failed to get job status: {str(e)}")
    
    async def get_active_jobs(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about active retraining jobs.
        
        Args:
            model_id: Optional model ID to filter by
            
        Returns:
            Dict[str, Any]: Active jobs information
        """
        if model_id:
            filtered_jobs = {
                job_id: job_info for job_id, job_info in self.active_jobs.items()
                if job_info["model_id"] == model_id
            }
            
            return {
                "model_id": model_id,
                "active_jobs_count": len(filtered_jobs),
                "jobs": filtered_jobs
            }
        else:
            # Group by model ID
            by_model = {}
            for job_info in self.active_jobs.values():
                model_id = job_info["model_id"]
                if model_id not in by_model:
                    by_model[model_id] = []
                by_model[model_id].append(job_info)
            
            return {
                "active_jobs_count": len(self.active_jobs),
                "models_count": len(by_model),
                "by_model": {
                    model_id: len(jobs) for model_id, jobs in by_model.items()
                },
                "jobs": self.active_jobs
            }
