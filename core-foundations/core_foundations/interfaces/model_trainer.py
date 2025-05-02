"""
Interfaces for model training and retraining components.

This module defines the interfaces that model trainers must implement
to be compatible with the automated retraining system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import datetime


class IModelTrainer(ABC):
    """
    Interface for components that can train or retrain machine learning models
    using feedback data.
    """
    
    @abstractmethod
    def retrain_model(
        self, 
        model_id: str, 
        feedback_data: Dict[str, Any],
        hyperparameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Retrain a model with new feedback data.
        
        Args:
            model_id: Identifier for the model to be retrained
            feedback_data: Structured feedback data to incorporate into training
            hyperparameters: Optional hyperparameters to use for this retraining
            **kwargs: Additional retraining configuration options
            
        Returns:
            Dict with retraining results including:
                - status: "success" or "failure"
                - model_version: The new model version after retraining
                - metrics: Comparative performance metrics
                - timestamp: When retraining completed
        """
        pass
    
    @abstractmethod
    def evaluate_feedback_impact(
        self,
        model_id: str,
        feedback_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate the potential impact of feedback data before actual retraining.
        
        This method performs a dry-run analysis to estimate how incorporating
        the feedback would affect model performance.
        
        Args:
            model_id: Identifier for the model to evaluate against
            feedback_data: Structured feedback data to analyze
            
        Returns:
            Dict with impact analysis including:
                - estimated_improvement: Expected performance change
                - confidence: Confidence level in the estimate
                - recommendation: Whether retraining is recommended
        """
        pass
    
    @abstractmethod
    def get_retraining_history(
        self,
        model_id: str,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the retraining history for a specific model.
        
        Args:
            model_id: Identifier for the model
            start_date: Optional start date for filtering history
            end_date: Optional end date for filtering history
            
        Returns:
            List of retraining events with timestamps, feedback volumes,
            and performance changes.
        """
        pass


class IFeedbackRepository(ABC):
    """
    Interface for components that store and retrieve feedback data for model retraining.
    """
    
    @abstractmethod
    def get_prioritized_feedback_since(
        self,
        timestamp: datetime.datetime,
        model_id: Optional[str] = None,
        min_priority: str = "MEDIUM",
        limit: int = 100
    ) -> List[Any]:  # Should return List[ClassifiedFeedback] in implementations
        """
        Retrieve prioritized feedback items since a specific timestamp.
        
        Args:
            timestamp: Retrieve feedback items since this time
            model_id: Optional filter for specific model
            min_priority: Minimum priority level to include
            limit: Maximum number of items to retrieve
            
        Returns:
            List of ClassifiedFeedback objects matching the criteria
        """
        pass
    
    @abstractmethod
    def update_feedback_status(
        self,
        feedback_ids: List[str],
        status: str
    ) -> int:
        """
        Update the status of multiple feedback items.
        
        Args:
            feedback_ids: List of feedback IDs to update
            status: New status to set
            
        Returns:
            Number of feedback items successfully updated
        """
        pass
    
    @abstractmethod
    def create_feedback_batch(
        self,
        feedback_ids: List[str],
        batch_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new batch from existing feedback items.
        
        Args:
            feedback_ids: List of feedback IDs to include in the batch
            batch_metadata: Optional metadata for the batch
            
        Returns:
            ID of the created batch
        """
        pass
    
    @abstractmethod
    def get_feedback_batch(
        self,
        batch_id: str
    ) -> Any:  # Should return FeedbackBatch in implementations
        """
        Retrieve a feedback batch by ID.
        
        Args:
            batch_id: ID of the batch to retrieve
            
        Returns:
            FeedbackBatch object if found, None otherwise
        """
        pass
    
    @abstractmethod
    def mark_batch_processed(
        self,
        batch_id: str,
        processing_results: Dict[str, Any]
    ) -> bool:
        """
        Mark a feedback batch as processed with results.
        
        Args:
            batch_id: ID of the batch to update
            processing_results: Results of processing this batch
            
        Returns:
            True if successful, False otherwise
        """
        pass
