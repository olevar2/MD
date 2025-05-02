"""
Models for the feedback system that facilitate automated model retraining.

These models provide a structured representation of feedback data collected
from various sources throughout the trading platform, enabling effective
integration into model retraining workflows.
"""

from enum import Enum, auto
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid


class FeedbackPriority(str, Enum):
    """Priority levels for feedback items."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class FeedbackCategory(str, Enum):
    """Categories of feedback that can be received."""
    INCORRECT_PREDICTION = "INCORRECT_PREDICTION"
    MARKET_SHIFT = "MARKET_SHIFT"
    PARAMETER_ADJUSTMENT = "PARAMETER_ADJUSTMENT"
    FEATURE_IMPORTANCE = "FEATURE_IMPORTANCE"
    ANOMALY_DETECTION = "ANOMALY_DETECTION"
    STRATEGY_IMPROVEMENT = "STRATEGY_IMPROVEMENT"
    TIMEFRAME_ADJUSTMENT = "TIMEFRAME_ADJUSTMENT"
    OTHER = "OTHER"


class FeedbackSource(str, Enum):
    """Sources from which feedback might originate."""
    SYSTEM_AUTO = "SYSTEM_AUTO"
    USER_EXPLICIT = "USER_EXPLICIT"
    PERFORMANCE_METRICS = "PERFORMANCE_METRICS"
    BACKTESTING = "BACKTESTING"
    LIVE_TRADING = "LIVE_TRADING"
    EXTERNAL_DATA = "EXTERNAL_DATA"


class FeedbackStatus(str, Enum):
    """Status of a feedback item in the processing pipeline."""
    NEW = "NEW"
    VALIDATED = "VALIDATED"
    PRIORITIZED = "PRIORITIZED"
    QUEUED_FOR_RETRAINING = "QUEUED_FOR_RETRAINING"
    INCORPORATED = "INCORPORATED"
    REJECTED = "REJECTED"


class ClassifiedFeedback:
    """
    Represents a piece of validated and classified feedback ready for integration 
    into model retraining processes.
    """

    def __init__(
        self,
        feedback_id: str = None,
        category: FeedbackCategory = None,
        priority: FeedbackPriority = FeedbackPriority.MEDIUM,
        source: FeedbackSource = None,
        status: FeedbackStatus = FeedbackStatus.NEW,
        metadata: Dict[str, Any] = None,
        content: Dict[str, Any] = None,
        model_id: str = None,
        timestamp: datetime = None,
        statistical_significance: float = None,
        related_feedback_ids: List[str] = None,
    ):
        """
        Initialize a ClassifiedFeedback object.

        Args:
            feedback_id: Unique identifier for this feedback item
            category: The category of feedback
            priority: The priority/importance of this feedback
            source: Where this feedback originated from
            status: Current status in the processing pipeline
            metadata: Additional metadata associated with this feedback
            content: The actual feedback content (could be metrics, text, etc.)
            model_id: Identifier of the model this feedback applies to
            timestamp: When this feedback was created
            statistical_significance: Measure of statistical confidence in this feedback (0-1)
            related_feedback_ids: IDs of other feedback items correlated with this one
        """
        self.feedback_id = feedback_id or str(uuid.uuid4())
        self.category = category
        self.priority = priority
        self.source = source
        self.status = status
        self.metadata = metadata or {}
        self.content = content or {}
        self.model_id = model_id
        self.timestamp = timestamp or datetime.utcnow()
        self.statistical_significance = statistical_significance
        self.related_feedback_ids = related_feedback_ids or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert the feedback object to a dictionary."""
        return {
            "feedback_id": self.feedback_id,
            "category": self.category.value if self.category else None,
            "priority": self.priority.value if self.priority else None,
            "source": self.source.value if self.source else None,
            "status": self.status.value if self.status else None,
            "metadata": self.metadata,
            "content": self.content,
            "model_id": self.model_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "statistical_significance": self.statistical_significance,
            "related_feedback_ids": self.related_feedback_ids
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClassifiedFeedback':
        """Create a feedback object from a dictionary."""
        # Convert string enums back to enums
        if data.get("category"):
            data["category"] = FeedbackCategory(data["category"])
        if data.get("priority"):
            data["priority"] = FeedbackPriority(data["priority"])
        if data.get("source"):
            data["source"] = FeedbackSource(data["source"])
        if data.get("status"):
            data["status"] = FeedbackStatus(data["status"])
        if data.get("timestamp") and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        
        return cls(**data)


class TimeframeFeedback(ClassifiedFeedback):
    """
    Specialized feedback for multi-timeframe predictions, incorporating 
    temporal correlation analysis.
    """
    
    def __init__(
        self,
        timeframe: str,
        related_timeframes: List[str] = None,
        temporal_correlation_data: Dict[str, float] = None,
        **kwargs
    ):
        """
        Initialize a TimeframeFeedback object.
        
        Args:
            timeframe: The primary timeframe this feedback applies to (e.g., "1h", "4h", "1d")
            related_timeframes: Other timeframes correlated with this feedback
            temporal_correlation_data: Correlation coefficients between timeframes
            **kwargs: All other ClassifiedFeedback parameters
        """
        super().__init__(**kwargs)
        self.timeframe = timeframe
        self.related_timeframes = related_timeframes or []
        self.temporal_correlation_data = temporal_correlation_data or {}
        
        # Default to timeframe category if not specified
        if not self.category:
            self.category = FeedbackCategory.TIMEFRAME_ADJUSTMENT
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert the TimeframeFeedback object to a dictionary."""
        data = super().to_dict()
        data.update({
            "timeframe": self.timeframe,
            "related_timeframes": self.related_timeframes,
            "temporal_correlation_data": self.temporal_correlation_data,
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TimeframeFeedback':
        """Create a TimeframeFeedback object from a dictionary."""
        # Extract specialized fields
        timeframe = data.pop("timeframe", None)
        related_timeframes = data.pop("related_timeframes", None)
        temporal_correlation_data = data.pop("temporal_correlation_data", None)
        
        # Create using parent class method for base fields
        instance = super().from_dict(data)
        
        # Add specialized fields
        instance.timeframe = timeframe
        instance.related_timeframes = related_timeframes or []
        instance.temporal_correlation_data = temporal_correlation_data or {}
        
        return instance


class FeedbackBatch:
    """
    A collection of related feedback items that should be processed together
    during model retraining.
    """
    
    def __init__(
        self,
        batch_id: str = None,
        feedback_items: List[ClassifiedFeedback] = None,
        batch_priority: FeedbackPriority = FeedbackPriority.MEDIUM,
        created_at: datetime = None,
        processed_at: Optional[datetime] = None,
        status: str = "NEW",
        metadata: Dict[str, Any] = None,
    ):
        """
        Initialize a FeedbackBatch object.
        
        Args:
            batch_id: Unique identifier for this batch
            feedback_items: List of feedback items in this batch
            batch_priority: Overall priority for the batch
            created_at: When this batch was created
            processed_at: When this batch was processed (None if not yet processed)
            status: Status of the batch in the pipeline
            metadata: Additional metadata for the batch
        """
        self.batch_id = batch_id or str(uuid.uuid4())
        self.feedback_items = feedback_items or []
        self.batch_priority = batch_priority
        self.created_at = created_at or datetime.utcnow()
        self.processed_at = processed_at
        self.status = status
        self.metadata = metadata or {}
    
    def add_feedback(self, feedback: ClassifiedFeedback) -> None:
        """Add a feedback item to this batch."""
        self.feedback_items.append(feedback)
        
        # Optionally update batch priority based on highest individual priority
        if feedback.priority == FeedbackPriority.CRITICAL:
            self.batch_priority = FeedbackPriority.CRITICAL
        elif feedback.priority == FeedbackPriority.HIGH and self.batch_priority != FeedbackPriority.CRITICAL:
            self.batch_priority = FeedbackPriority.HIGH

    def to_dict(self) -> Dict[str, Any]:
        """Convert the batch to a dictionary."""
        return {
            "batch_id": self.batch_id,
            "feedback_items": [item.to_dict() for item in self.feedback_items],
            "batch_priority": self.batch_priority.value if self.batch_priority else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "status": self.status,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeedbackBatch':
        """Create a FeedbackBatch object from a dictionary."""
        # Process nested feedback items
        feedback_items = []
        for item_data in data.get("feedback_items", []):
            # Check for specialized feedback type
            if "timeframe" in item_data:
                feedback_items.append(TimeframeFeedback.from_dict(item_data))
            else:
                feedback_items.append(ClassifiedFeedback.from_dict(item_data))
                
        # Process dates
        if data.get("created_at") and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("processed_at") and isinstance(data["processed_at"], str):
            data["processed_at"] = datetime.fromisoformat(data["processed_at"])
            
        # Process enum
        if data.get("batch_priority"):
            data["batch_priority"] = FeedbackPriority(data["batch_priority"])
            
        # Create instance with processed data
        return cls(
            batch_id=data.get("batch_id"),
            feedback_items=feedback_items,
            batch_priority=data.get("batch_priority"),
            created_at=data.get("created_at"),
            processed_at=data.get("processed_at"),
            status=data.get("status", "NEW"),
            metadata=data.get("metadata", {})
        )
