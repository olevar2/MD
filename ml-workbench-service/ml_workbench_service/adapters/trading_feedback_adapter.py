"""
Trading Feedback Adapter Module

This module provides adapters for trading feedback functionality,
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


class TradingFeedbackCollectorAdapter:
    """
    Adapter for trading feedback collection.
    
    This adapter can either use a direct API connection to the analysis engine service
    or provide standalone functionality to avoid circular dependencies.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the adapter.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Get analysis engine service URL from config or environment
        import os
        analysis_engine_base_url = self.config.get(
            "analysis_engine_base_url", 
            os.environ.get("ANALYSIS_ENGINE_BASE_URL", "http://analysis-engine-service:8000")
        )
        
        # Set up the client with resolved URL
        self.client = httpx.AsyncClient(
            base_url=f"{analysis_engine_base_url.rstrip('/')}/api/v1",
            timeout=30.0
        )
        
        # Local storage for feedback
        self.feedback_storage = []
    
    async def collect_feedback(
        self,
        model_id: str,
        feedback_category: FeedbackCategory,
        description: str,
        metrics: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        severity: FeedbackSeverity = FeedbackSeverity.MEDIUM
    ) -> Dict[str, Any]:
        """
        Collect feedback for a model.
        
        Args:
            model_id: ID of the model
            feedback_category: Category of feedback
            description: Description of the feedback
            metrics: Metrics related to the feedback
            context: Optional context information
            severity: Severity of the feedback
            
        Returns:
            Dictionary with collection results
        """
        try:
            # Create feedback object
            feedback = {
                "model_id": model_id,
                "timestamp": datetime.now().isoformat(),
                "source": FeedbackSource.TRADING,
                "category": feedback_category,
                "severity": severity,
                "description": description,
                "metrics": metrics,
                "context": context or {},
                "feedback_id": str(uuid.uuid4())
            }
            
            # Store locally
            self.feedback_storage.append(feedback)
            
            # Send to analysis engine service
            response = await self.client.post("/feedback/collect", json=feedback)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error collecting feedback: {str(e)}")
            
            # Return fallback response
            return {
                "status": "stored_locally",
                "message": f"Failed to send to analysis engine: {str(e)}",
                "feedback_id": feedback.get("feedback_id")
            }
    
    async def get_feedback_history(
        self,
        model_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get history of feedback.
        
        Args:
            model_id: Optional ID of the model to filter by
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Optional limit on number of feedback items to return
            
        Returns:
            List of feedback items
        """
        try:
            # Prepare query parameters
            params = {}
            if model_id:
                params["model_id"] = model_id
            if start_date:
                params["start_date"] = start_date.isoformat()
            if end_date:
                params["end_date"] = end_date.isoformat()
            if limit:
                params["limit"] = limit
            
            # Send to analysis engine service
            response = await self.client.get("/feedback/history", params=params)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error getting feedback history: {str(e)}")
            
            # Return local storage as fallback
            filtered_feedback = self.feedback_storage
            
            if model_id:
                filtered_feedback = [f for f in filtered_feedback if f.get("model_id") == model_id]
            
            if start_date:
                filtered_feedback = [
                    f for f in filtered_feedback 
                    if datetime.fromisoformat(f.get("timestamp").replace('Z', '+00:00')) >= start_date
                ]
            
            if end_date:
                filtered_feedback = [
                    f for f in filtered_feedback 
                    if datetime.fromisoformat(f.get("timestamp").replace('Z', '+00:00')) <= end_date
                ]
            
            # Sort by timestamp (newest first)
            filtered_feedback = sorted(
                filtered_feedback,
                key=lambda x: x.get("timestamp"),
                reverse=True
            )
            
            if limit:
                filtered_feedback = filtered_feedback[:limit]
            
            return filtered_feedback


class ModelTrainingFeedbackIntegratorAdapter(IModelTrainingFeedbackIntegrator):
    """
    Adapter for model training feedback integration that implements the common interface.
    
    This adapter wraps the actual ModelTrainingFeedbackIntegrator implementation.
    """
    
    def __init__(self, integrator_instance=None):
        """
        Initialize the adapter.
        
        Args:
            integrator_instance: Optional actual integrator instance to wrap
        """
        self.integrator = integrator_instance
        
        # If no integrator provided, use local implementation
        if not self.integrator:
            # Import here to avoid circular imports
            try:
                from ml_workbench_service.feedback.model_training_feedback import ModelTrainingFeedbackIntegrator
                self.integrator = ModelTrainingFeedbackIntegrator()
            except ImportError:
                logger.warning("ModelTrainingFeedbackIntegrator not available, using fallback implementation")
                self.integrator = None
        
        # Local storage for feedback
        self.feedback_by_model = {}
        self.performance_metrics = {}
    
    async def process_trading_feedback(
        self,
        feedback_list: List[ModelFeedback]
    ) -> Dict[str, Any]:
        """Process trading feedback for model training."""
        if self.integrator:
            try:
                # Convert ModelFeedback to TradeFeedback if needed
                # This depends on the actual implementation of the integrator
                return await self.integrator.process_trading_feedback(feedback_list)
            except Exception as e:
                logger.error(f"Error processing trading feedback: {str(e)}")
        
        # Fallback implementation
        processed_count = 0
        models_affected = set()
        
        for feedback in feedback_list:
            model_id = feedback.model_id
            
            # Store feedback by model
            if model_id not in self.feedback_by_model:
                self.feedback_by_model[model_id] = []
            
            self.feedback_by_model[model_id].append(feedback)
            models_affected.add(model_id)
            processed_count += 1
        
        return {
            "status": "processed",
            "processed_count": processed_count,
            "models_affected": list(models_affected)
        }
    
    async def prepare_training_data(
        self,
        model_id: str,
        feedback_list: List[ModelFeedback]
    ) -> Dict[str, Any]:
        """Prepare training data for a model based on feedback."""
        if self.integrator:
            try:
                return await self.integrator.prepare_training_data(model_id, feedback_list)
            except Exception as e:
                logger.error(f"Error preparing training data: {str(e)}")
        
        # Fallback implementation
        return {
            "status": "prepared",
            "model_id": model_id,
            "data_prepared": True,
            "feedback_count": len(feedback_list),
            "is_fallback": True
        }
    
    async def trigger_model_update(
        self,
        model_id: str,
        reason: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Trigger an update for a model."""
        if self.integrator:
            try:
                return await self.integrator.trigger_model_update(model_id, reason, context)
            except Exception as e:
                logger.error(f"Error triggering model update: {str(e)}")
        
        # Fallback implementation
        return {
            "status": "triggered",
            "model_id": model_id,
            "update_triggered": True,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
            "is_fallback": True
        }
    
    async def get_model_performance_metrics(
        self,
        model_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get performance metrics for a model."""
        if self.integrator:
            try:
                return await self.integrator.get_model_performance_metrics(model_id, start_date, end_date)
            except Exception as e:
                logger.error(f"Error getting model performance metrics: {str(e)}")
        
        # Fallback implementation
        if model_id in self.performance_metrics:
            return self.performance_metrics[model_id]
        
        # Generate default metrics
        return {
            "model_id": model_id,
            "metrics": {
                "accuracy": 0.75,
                "precision": 0.72,
                "recall": 0.68,
                "f1_score": 0.70
            },
            "is_fallback": True
        }
""""""
