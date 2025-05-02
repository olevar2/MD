"""
Model Training Feedback Integration

This module implements the ModelTrainingFeedbackIntegrator, which connects trading 
outcomes with the model training process for continuous improvement.
"""

from typing import Dict, List, Any, Optional, Union
import asyncio
import uuid
from datetime import datetime, timedelta
import numpy as np
import logging

from core_foundations.utils.logger import get_logger
from analysis_engine.adaptive_layer.trading_feedback_collector import (
    TradeFeedback, FeedbackCategory, FeedbackSource
)

logger = get_logger(__name__)


class ModelFeedbackConfig:
    """Configuration for model feedback processing."""
    def __init__(
        self, 
        model_id: str,
        retraining_threshold: float = 0.1,
        min_feedback_samples: int = 100,
        validation_split: float = 0.2,
        max_feedback_age_days: int = 30,
        importance_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize model feedback configuration.
        
        Args:
            model_id: ID of the model
            retraining_threshold: Performance degradation threshold to trigger retraining
            min_feedback_samples: Minimum samples required before retraining
            validation_split: Portion of feedback to use for validation
            max_feedback_age_days: Maximum age of feedback to consider
            importance_weights: Weights for different feedback categories
        """
        self.model_id = model_id
        self.retraining_threshold = retraining_threshold
        self.min_feedback_samples = min_feedback_samples
        self.validation_split = validation_split
        self.max_feedback_age_days = max_feedback_age_days
        
        # Default weights if not provided
        self.importance_weights = importance_weights or {
            FeedbackCategory.SUCCESS.value: 1.0,
            FeedbackCategory.PARTIAL_SUCCESS.value: 1.5,
            FeedbackCategory.FAILURE.value: 2.0,
            FeedbackCategory.TECHNICAL_ERROR.value: 0.5,
            FeedbackCategory.MARKET_CONDITION.value: 1.0,
        }


class ModelPerformanceTracker:
    """Tracks model performance based on feedback."""
    def __init__(self, model_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize model performance tracker.
        
        Args:
            model_id: ID of the model
            config: Configuration parameters
        """
        self.model_id = model_id
        self.config = config or {}
        
        # Initialize performance metrics
        self.metrics = {
            "total_predictions": 0,
            "successful_predictions": 0,
            "failed_predictions": 0,
            "success_rate": 0.0,
            "mean_absolute_error": 0.0,
            "mean_squared_error": 0.0,
            "directional_accuracy": 0.0,
            "by_instrument": {},
            "by_timeframe": {},
            "by_market_regime": {},
            "recent_trend": [],  # Track recent performance trend
        }
        
        # Store recent errors for trend analysis
        self.recent_errors = []
        self.max_recent_samples = self.config.get("max_recent_samples", 500)
    
    def update_metrics(self, feedback: TradeFeedback):
        """
        Update performance metrics based on feedback.
        
        Args:
            feedback: Model prediction feedback
        """
        # Basic checks
        if not feedback.model_id or feedback.model_id != self.model_id:
            return
        
        if feedback.source != FeedbackSource.MODEL_PREDICTION:
            return
        
        # Extract prediction metrics
        metrics = feedback.outcome_metrics
        prediction_value = metrics.get("prediction_value")
        actual_value = metrics.get("actual_value")
        
        if prediction_value is None or actual_value is None:
            logger.warning(f"Missing prediction or actual values in feedback {feedback.feedback_id}")
            return
        
        # Calculate errors
        error = actual_value - prediction_value
        abs_error = abs(error)
        squared_error = error ** 2
        
        # Store for trend analysis
        self.recent_errors.append({
            "timestamp": feedback.timestamp,
            "error": error,
            "abs_error": abs_error,
            "squared_error": squared_error
        })
        
        # Limit size of recent errors list
        if len(self.recent_errors) > self.max_recent_samples:
            self.recent_errors = self.recent_errors[-self.max_recent_samples:]
        
        # Update overall metrics
        self.metrics["total_predictions"] += 1
        
        is_successful = metrics.get("is_successful", feedback.category == FeedbackCategory.SUCCESS)
        if is_successful:
            self.metrics["successful_predictions"] += 1
        else:
            self.metrics["failed_predictions"] += 1
        
        # Update success rate
        if self.metrics["total_predictions"] > 0:
            self.metrics["success_rate"] = (
                self.metrics["successful_predictions"] / self.metrics["total_predictions"]
            )
        
        # Update error metrics
        prev_mae = self.metrics["mean_absolute_error"]
        prev_mse = self.metrics["mean_squared_error"]
        count = self.metrics["total_predictions"]
        
        # Incremental average update
        self.metrics["mean_absolute_error"] = prev_mae + (abs_error - prev_mae) / count
        self.metrics["mean_squared_error"] = prev_mse + (squared_error - prev_mse) / count
        
        # Update directional accuracy
        correct_direction = (prediction_value > 0 and actual_value > 0) or (prediction_value < 0 and actual_value < 0)
        self.metrics["directional_accuracy"] = (
            (self.metrics["directional_accuracy"] * (count - 1) + int(correct_direction)) / count
        )
        
        # Update instrument-specific metrics
        instrument = feedback.instrument
        if instrument not in self.metrics["by_instrument"]:
            self.metrics["by_instrument"][instrument] = {
                "count": 0,
                "success_count": 0,
                "success_rate": 0.0,
                "mean_absolute_error": 0.0
            }
        
        instr_metrics = self.metrics["by_instrument"][instrument]
        instr_metrics["count"] += 1
        if is_successful:
            instr_metrics["success_count"] += 1
        instr_metrics["success_rate"] = instr_metrics["success_count"] / instr_metrics["count"]
        
        # Update MAE for instrument
        instr_metrics["mean_absolute_error"] = (
            instr_metrics["mean_absolute_error"] * (instr_metrics["count"] - 1) + abs_error
        ) / instr_metrics["count"]
        
        # Update timeframe-specific metrics
        timeframe = feedback.timeframe
        if timeframe not in self.metrics["by_timeframe"]:
            self.metrics["by_timeframe"][timeframe] = {
                "count": 0,
                "success_count": 0,
                "success_rate": 0.0,
                "mean_absolute_error": 0.0
            }
        
        tf_metrics = self.metrics["by_timeframe"][timeframe]
        tf_metrics["count"] += 1
        if is_successful:
            tf_metrics["success_count"] += 1
        tf_metrics["success_rate"] = tf_metrics["success_count"] / tf_metrics["count"]
        
        # Update MAE for timeframe
        tf_metrics["mean_absolute_error"] = (
            tf_metrics["mean_absolute_error"] * (tf_metrics["count"] - 1) + abs_error
        ) / tf_metrics["count"]
        
        # Update market regime metrics if available
        market_regime = feedback.market_regime
        if market_regime:
            if market_regime not in self.metrics["by_market_regime"]:
                self.metrics["by_market_regime"][market_regime] = {
                    "count": 0,
                    "success_count": 0,
                    "success_rate": 0.0,
                    "mean_absolute_error": 0.0
                }
            
            regime_metrics = self.metrics["by_market_regime"][market_regime]
            regime_metrics["count"] += 1
            if is_successful:
                regime_metrics["success_count"] += 1
            regime_metrics["success_rate"] = regime_metrics["success_count"] / regime_metrics["count"]
            
            # Update MAE for market regime
            regime_metrics["mean_absolute_error"] = (
                regime_metrics["mean_absolute_error"] * (regime_metrics["count"] - 1) + abs_error
            ) / regime_metrics["count"]
        
        # Update trend data (calculate rolling metrics)
        self._update_performance_trend()
    
    def _update_performance_trend(self):
        """Update the rolling performance trend metrics."""
        if len(self.recent_errors) < 20:  # Need minimum samples
            return
        
        # Get the last month's worth of errors
        now = datetime.utcnow()
        month_ago = now - timedelta(days=30)
        
        # Create weekly performance snapshots for trend analysis
        weekly_snapshots = []
        for week in range(4):  # 4 weeks
            week_start = month_ago + timedelta(days=week*7)
            week_end = week_start + timedelta(days=7)
            
            # Filter errors for this week
            week_errors = [
                e for e in self.recent_errors 
                if week_start <= e["timestamp"] < week_end
            ]
            
            if week_errors:
                avg_abs_error = np.mean([e["abs_error"] for e in week_errors])
                avg_squared_error = np.mean([e["squared_error"] for e in week_errors])
                
                weekly_snapshots.append({
                    "week": week + 1,
                    "sample_count": len(week_errors),
                    "mean_absolute_error": avg_abs_error,
                    "mean_squared_error": avg_squared_error,
                    "start_date": week_start.isoformat(),
                    "end_date": week_end.isoformat()
                })
        
        self.metrics["recent_trend"] = weekly_snapshots
    
    def calculate_performance_change(self, days: int = 7) -> Dict[str, Any]:
        """
        Calculate performance change over a time period.
        
        Args:
            days: Number of days to compare
            
        Returns:
            Dict with performance change metrics
        """
        if len(self.recent_errors) < 20:  # Need minimum samples
            return {"sufficient_data": False}
        
        now = datetime.utcnow()
        cutoff = now - timedelta(days=days)
        
        # Split errors into recent and older
        recent = [e for e in self.recent_errors if e["timestamp"] >= cutoff]
        older = [e for e in self.recent_errors if e["timestamp"] < cutoff]
        
        if not recent or not older:
            return {"sufficient_data": False}
        
        # Calculate metrics for both periods
        recent_mae = np.mean([e["abs_error"] for e in recent])
        older_mae = np.mean([e["abs_error"] for e in older])
        
        recent_mse = np.mean([e["squared_error"] for e in recent])
        older_mse = np.mean([e["squared_error"] for e in older])
        
        # Calculate changes
        mae_change = (recent_mae - older_mae) / older_mae if older_mae else 0
        mse_change = (recent_mse - older_mse) / older_mse if older_mse else 0
        
        # Determine improvement or degradation
        is_degrading = mae_change > 0.05  # 5% increase in error
        is_improving = mae_change < -0.05  # 5% decrease in error
        
        return {
            "sufficient_data": True,
            "recent_samples": len(recent),
            "older_samples": len(older),
            "recent_mae": recent_mae,
            "older_mae": older_mae,
            "mae_change_pct": mae_change * 100,
            "mse_change_pct": mse_change * 100,
            "is_degrading": is_degrading,
            "is_improving": is_improving,
            "status": "degrading" if is_degrading else "improving" if is_improving else "stable"
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.
        
        Returns:
            Dict: Current performance metrics
        """
        return self.metrics


class ModelTrainingFeedbackIntegrator:
    """
    The ModelTrainingFeedbackIntegrator connects trading outcomes with the model 
    training process for continuous improvement.
    
    Key capabilities:
    - Collect and process model prediction feedback
    - Track model performance across different conditions
    - Trigger model retraining when performance degrades
    - Provide feedback to model training processes
    """
    
    def __init__(
        self,
        ml_client=None,  # Client for ML model service
        config: Dict[str, Any] = None
    ):
        """
        Initialize the model training feedback integrator.
        
        Args:
            ml_client: Client for ML model service interactions
            config: Configuration parameters
        """
        self.ml_client = ml_client
        self.config = config or {}
        
        # Model configuration registry
        self.model_configs = {}
        
        # Performance trackers by model
        self.performance_trackers = {}
        
        # Feedback storage by model
        self.feedback_by_model = {}
        
        # Retraining status
        self.retraining_status = {}
        
        logger.info("ModelTrainingFeedbackIntegrator initialized")
    
    def register_model(
        self,
        model_id: str,
        config: Optional[Union[ModelFeedbackConfig, Dict[str, Any]]] = None
    ):
        """
        Register a model for feedback processing.
        
        Args:
            model_id: ID of the model to register
            config: Configuration for the model feedback
        """
        # Convert dict config to ModelFeedbackConfig if needed
        if isinstance(config, dict):
            config = ModelFeedbackConfig(model_id=model_id, **config)
        elif config is None:
            config = ModelFeedbackConfig(model_id=model_id)
        
        # Store model config
        self.model_configs[model_id] = config
        
        # Initialize performance tracker
        self.performance_trackers[model_id] = ModelPerformanceTracker(model_id)
        
        # Initialize feedback storage
        self.feedback_by_model[model_id] = []
        
        # Initialize retraining status
        self.retraining_status[model_id] = {
            "last_retrain_check": datetime.utcnow(),
            "last_retrain_time": None,
            "retraining_needed": False,
            "retraining_reason": None
        }
        
        logger.info(f"Registered model {model_id} for feedback processing")
    
    async def process_feedback(self, feedback: TradeFeedback) -> Dict[str, Any]:
        """
        Process model prediction feedback.
        
        Args:
            feedback: Feedback for model prediction
            
        Returns:
            Dict: Processing result
        """
        # Check if this is model prediction feedback
        if feedback.source != FeedbackSource.MODEL_PREDICTION or not feedback.model_id:
            return {"processed": False, "reason": "not_model_prediction"}
        
        model_id = feedback.model_id
        
        # Check if model is registered
        if model_id not in self.model_configs:
            # Auto-register with default config
            self.register_model(model_id)
        
        # Store feedback
        self.feedback_by_model[model_id].append(feedback)
        
        # Update performance metrics
        self.performance_trackers[model_id].update_metrics(feedback)
        
        # Check if retraining is needed
        await self._check_retraining_needed(model_id)
        
        logger.debug(f"Processed feedback for model {model_id}")
        
        return {"processed": True, "model_id": model_id}
    
    async def _check_retraining_needed(self, model_id: str):
        """
        Check if a model needs retraining based on feedback.
        
        Args:
            model_id: ID of the model to check
        """
        # Skip if we just checked recently (avoid checking too often)
        now = datetime.utcnow()
        last_check = self.retraining_status[model_id]["last_retrain_check"]
        min_check_interval = timedelta(hours=self.config.get("min_hours_between_checks", 4))
        
        if (now - last_check) < min_check_interval:
            return
        
        # Update last check time
        self.retraining_status[model_id]["last_retrain_check"] = now
        
        # Get model config
        config = self.model_configs[model_id]
        
        # Get recent feedback
        feedback_list = self.feedback_by_model[model_id]
        
        # Check if we have enough samples
        if len(feedback_list) < config.min_feedback_samples:
            logger.debug(f"Not enough samples for retraining check: {len(feedback_list)} < {config.min_feedback_samples}")
            return
        
        # Get performance change metrics
        performance_tracker = self.performance_trackers[model_id]
        perf_change = performance_tracker.calculate_performance_change(days=7)  # Last week vs previous
        
        needs_retraining = False
        reason = None
        
        if perf_change["sufficient_data"]:
            # Check for performance degradation
            if perf_change["is_degrading"]:
                degradation = perf_change["mae_change_pct"]
                threshold = config.retraining_threshold * 100
                
                if degradation > threshold:
                    needs_retraining = True
                    reason = f"Performance degradation: {degradation:.2f}% (threshold: {threshold:.2f}%)"
                    logger.info(f"Model {model_id} needs retraining: {reason}")
        
        # Update retraining status
        self.retraining_status[model_id].update({
            "retraining_needed": needs_retraining,
            "retraining_reason": reason
        })
        
        # Trigger retraining if needed and if we have a ML client
        if needs_retraining and self.ml_client and hasattr(self.ml_client, "trigger_model_retraining"):
            try:
                # Prepare data for retraining
                training_data = await self._prepare_training_data(model_id)
                
                # Trigger retraining
                result = await self.ml_client.trigger_model_retraining(
                    model_id=model_id,
                    training_data=training_data,
                    reason=reason
                )
                
                # Update retraining status
                if result.get("success", False):
                    self.retraining_status[model_id]["last_retrain_time"] = now
                    logger.info(f"Triggered retraining for model {model_id}")
                else:
                    logger.warning(f"Retraining request failed for model {model_id}: {result.get('error', 'unknown error')}")
                    
            except Exception as e:
                logger.error(f"Error triggering retraining for model {model_id}: {str(e)}", exc_info=True)
    
    async def _prepare_training_data(self, model_id: str) -> Dict[str, Any]:
        """
        Prepare training data for model retraining.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Dict: Training data package
        """
        # Get model config
        config = self.model_configs[model_id]
        
        # Get feedback list
        feedback_list = self.feedback_by_model[model_id]
        
        # Filter by age
        cutoff_date = datetime.utcnow() - timedelta(days=config.max_feedback_age_days)
        recent_feedback = [f for f in feedback_list if f.timestamp >= cutoff_date]
        
        # Extract features and labels
        features = []
        labels = []
        weights = []
        metadata = []
        
        for fb in recent_feedback:
            # Extract feature data from feedback
            # This is a simplified example - in practice, you'd extract the specific
            # features needed for your model from the feedback object
            feature_data = fb.metadata.get("feature_data", {})
            if not feature_data:
                continue
                
            # Extract label (actual value)
            actual_value = fb.outcome_metrics.get("actual_value")
            if actual_value is None:
                continue
                
            # Calculate sample weight based on category
            category = fb.category.value
            weight = config.importance_weights.get(category, 1.0)
            
            # Add to training data
            features.append(feature_data)
            labels.append(actual_value)
            weights.append(weight)
            
            # Add metadata
            meta = {
                "instrument": fb.instrument,
                "timeframe": fb.timeframe,
                "timestamp": fb.timestamp.isoformat(),
                "category": category
            }
            if fb.market_regime:
                meta["market_regime"] = fb.market_regime
                
            metadata.append(meta)
        
        # Prepare final training package
        return {
            "model_id": model_id,
            "features": features,
            "labels": labels,
            "sample_weights": weights,
            "metadata": metadata,
            "sample_count": len(features)
        }
    
    def get_model_performance(self, model_id: str) -> Dict[str, Any]:
        """
        Get performance metrics for a model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Dict: Performance metrics
        """
        if model_id in self.performance_trackers:
            tracker = self.performance_trackers[model_id]
            metrics = tracker.get_metrics()
            
            # Add retraining status
            metrics["retraining"] = self.retraining_status[model_id]
            
            return metrics
        
        return {"error": "model_not_found"}
    
    def get_all_model_statuses(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status summary for all registered models.
        
        Returns:
            Dict: Status summary by model ID
        """
        result = {}
        
        for model_id in self.model_configs:
            tracker = self.performance_trackers[model_id]
            metrics = tracker.get_metrics()
            
            # Simplified status summary
            result[model_id] = {
                "total_predictions": metrics["total_predictions"],
                "success_rate": metrics["success_rate"],
                "mean_absolute_error": metrics["mean_absolute_error"],
                "directional_accuracy": metrics["directional_accuracy"],
                "retraining_needed": self.retraining_status[model_id]["retraining_needed"],
                "last_retrain_time": self.retraining_status[model_id]["last_retrain_time"]
            }
        
        return result
