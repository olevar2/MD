"""
Model Feedback Interfaces Module

This module provides interfaces for model feedback functionality used across services,
helping to break circular dependencies between services.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field


class FeedbackSource(str, Enum):
    """Sources of model feedback."""
    TRADING = "trading"
    BACKTESTING = "backtesting"
    SIMULATION = "simulation"
    MANUAL = "manual"
    AUTOMATED = "automated"
    SYSTEM = "system"


class FeedbackCategory(str, Enum):
    """Categories of model feedback."""
    ACCURACY = "accuracy"
    TIMING = "timing"
    CONFIDENCE = "confidence"
    REGIME_CHANGE = "regime_change"
    CORRELATION = "correlation"
    FEATURE_IMPORTANCE = "feature_importance"
    OVERFITTING = "overfitting"
    UNDERFITTING = "underfitting"
    DRIFT = "drift"
    LATENCY = "latency"
    OTHER = "other"


class FeedbackSeverity(str, Enum):
    """Severity levels for model feedback."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ModelFeedback:
    """Model feedback data."""
    model_id: str
    timestamp: datetime
    source: FeedbackSource
    category: FeedbackCategory
    severity: FeedbackSeverity
    description: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    feedback_id: Optional[str] = None


class IModelFeedbackProcessor(ABC):
    """Interface for model feedback processing."""

    @abstractmethod
    async def process_feedback(
        self,
        feedback: ModelFeedback
    ) -> Dict[str, Any]:
        """
        Process model feedback.

        Args:
            feedback: Model feedback to process

        Returns:
            Dictionary with processing results
        """
        pass

    @abstractmethod
    async def get_feedback_summary(
        self,
        model_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        categories: Optional[List[FeedbackCategory]] = None
    ) -> Dict[str, Any]:
        """
        Get summary of feedback for a model.

        Args:
            model_id: ID of the model
            start_date: Optional start date filter
            end_date: Optional end date filter
            categories: Optional list of categories to filter by

        Returns:
            Dictionary with feedback summary
        """
        pass

    @abstractmethod
    async def get_feedback_history(
        self,
        model_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[ModelFeedback]:
        """
        Get history of feedback for a model.

        Args:
            model_id: ID of the model
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Optional limit on number of feedback items to return

        Returns:
            List of model feedback items
        """
        pass


class IModelTrainingFeedbackIntegrator(ABC):
    """Interface for model training feedback integration."""

    @abstractmethod
    async def process_trading_feedback(
        self,
        feedback_list: List[ModelFeedback]
    ) -> Dict[str, Any]:
        """
        Process trading feedback for model training.

        Args:
            feedback_list: List of model feedback from trading

        Returns:
            Dictionary with processing results
        """
        pass

    @abstractmethod
    async def prepare_training_data(
        self,
        model_id: str,
        feedback_list: List[ModelFeedback]
    ) -> Dict[str, Any]:
        """
        Prepare training data for a model based on feedback.

        Args:
            model_id: ID of the model to prepare data for
            feedback_list: List of model feedback

        Returns:
            Dictionary with prepared training data
        """
        pass

    @abstractmethod
    async def trigger_model_update(
        self,
        model_id: str,
        reason: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Trigger an update for a model.

        Args:
            model_id: ID of the model to update
            reason: Reason for the update
            context: Optional context information

        Returns:
            Dictionary with update status
        """
        pass

    @abstractmethod
    async def get_model_performance_metrics(
        self,
        model_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get performance metrics for a model.

        Args:
            model_id: ID of the model
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Dictionary with performance metrics
        """
        pass
