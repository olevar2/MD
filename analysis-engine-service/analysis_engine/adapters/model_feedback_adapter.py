"""
Model Feedback Adapter Module

This module provides adapters for model feedback functionality,
helping to break circular dependencies between services.
"""
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import uuid
import httpx
import asyncio
import json

from common_lib.ml.model_feedback_interfaces import (
    FeedbackSource,
    FeedbackCategory,
    FeedbackSeverity,
    ModelFeedback,
    IModelFeedbackProcessor,
    IModelTrainingFeedbackIntegrator
)
from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainingFeedbackAdapter(IModelTrainingFeedbackIntegrator):
    """
    Adapter for model training feedback integration that implements the common interface.

    This adapter can either use a direct API connection to the ML workbench service
    or provide standalone functionality to avoid circular dependencies.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the adapter.

        Args:
            config: Configuration parameters
        """
        self.config = config or {}

        # Get ML workbench service URL from config or environment
        import os
        ml_workbench_base_url = self.config.get(
            "ml_workbench_base_url",
            os.environ.get("ML_WORKBENCH_BASE_URL", "http://ml-workbench-service:8000")
        )

        # Set up the client with resolved URL
        self.client = httpx.AsyncClient(
            base_url=f"{ml_workbench_base_url.rstrip('/')}/api/v1",
            timeout=30.0
        )

        # Local cache for feedback data
        self.feedback_cache = {}
        self.cache_ttl = self.config.get("cache_ttl_minutes", 15)  # Cache TTL in minutes

    async def process_trading_feedback(
        self,
        feedback_list: List[ModelFeedback]
    ) -> Dict[str, Any]:
        """Process trading feedback for model training."""
        try:
            # Convert feedback to dict for API
            feedback_data = [self._feedback_to_dict(feedback) for feedback in feedback_list]

            # Send to ML workbench service
            response = await self.client.post("/feedback/trading", json={"feedback": feedback_data})
            response.raise_for_status()

            return response.json()

        except Exception as e:
            logger.error(f"Error processing trading feedback: {str(e)}")

            # Return fallback response
            return {
                "status": "error",
                "message": f"Failed to process trading feedback: {str(e)}",
                "processed_count": 0,
                "models_affected": []
            }

    async def prepare_training_data(
        self,
        model_id: str,
        feedback_list: List[ModelFeedback]
    ) -> Dict[str, Any]:
        """Prepare training data for a model based on feedback."""
        try:
            # Convert feedback to dict for API
            feedback_data = [self._feedback_to_dict(feedback) for feedback in feedback_list]

            # Send to ML workbench service
            response = await self.client.post(
                f"/models/{model_id}/prepare-training-data",
                json={"feedback": feedback_data}
            )
            response.raise_for_status()

            return response.json()

        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")

            # Return fallback response
            return {
                "status": "error",
                "message": f"Failed to prepare training data: {str(e)}",
                "model_id": model_id,
                "data_prepared": False
            }

    async def trigger_model_update(
        self,
        model_id: str,
        reason: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Trigger an update for a model."""
        try:
            # Prepare request data
            request_data = {
                "model_id": model_id,
                "reason": reason,
                "context": context or {},
                "timestamp": datetime.now().isoformat()
            }

            # Send to ML workbench service
            response = await self.client.post(f"/models/{model_id}/update", json=request_data)
            response.raise_for_status()

            return response.json()

        except Exception as e:
            logger.error(f"Error triggering model update: {str(e)}")

            # Return fallback response
            return {
                "status": "error",
                "message": f"Failed to trigger model update: {str(e)}",
                "model_id": model_id,
                "update_triggered": False
            }

    async def get_model_performance_metrics(
        self,
        model_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get performance metrics for a model."""
        try:
            # Check cache first
            cache_key = f"{model_id}_{start_date}_{end_date}"
            if cache_key in self.feedback_cache:
                cache_entry = self.feedback_cache[cache_key]
                cache_age = datetime.now() - cache_entry["timestamp"]
                if cache_age.total_seconds() < self.cache_ttl * 60:
                    return cache_entry["data"]

            # Prepare query parameters
            params = {"model_id": model_id}
            if start_date:
                params["start_date"] = start_date.isoformat()
            if end_date:
                params["end_date"] = end_date.isoformat()

            # Send to ML workbench service
            response = await self.client.get("/models/performance-metrics", params=params)
            response.raise_for_status()

            # Parse response
            performance_metrics = response.json()

            # Update cache
            self.feedback_cache[cache_key] = {
                "timestamp": datetime.now(),
                "data": performance_metrics
            }

            return performance_metrics

        except Exception as e:
            logger.error(f"Error getting model performance metrics: {str(e)}")

            # Return fallback metrics
            return {
                "model_id": model_id,
                "metrics": {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0
                },
                "is_fallback": True
            }

    def _feedback_to_dict(self, feedback: ModelFeedback) -> Dict[str, Any]:
        """Convert ModelFeedback to dictionary for API."""
        return {
            "model_id": feedback.model_id,
            "timestamp": feedback.timestamp.isoformat() if isinstance(feedback.timestamp, datetime) else feedback.timestamp,
            "source": feedback.source,
            "category": feedback.category,
            "severity": feedback.severity,
            "description": feedback.description,
            "metrics": feedback.metrics,
            "context": feedback.context,
            "feedback_id": feedback.feedback_id or str(uuid.uuid4())
        }
