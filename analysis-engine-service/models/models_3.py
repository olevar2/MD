"""
Database models for the analysis engine.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declared_attr

from analysis_engine.core.database import Base

class TimestampMixin:
    """Mixin for timestamp fields."""
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

class AnalysisResult(Base, TimestampMixin):
    """Model for storing analysis results."""
    
    __tablename__ = "analysis_results"
    
    id = Column(Integer, primary_key=True)
    analyzer_name = Column(String, nullable=False, index=True)
    symbol = Column(String, nullable=False, index=True)
    timeframe = Column(String, nullable=False)
    result = Column(JSON, nullable=False)
    metadata = Column(JSON)
    is_valid = Column(Boolean, default=True)
    error = Column(String)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "analyzer_name": self.analyzer_name,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "result": self.result,
            "metadata": self.metadata,
            "is_valid": self.is_valid,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

class FeedbackEntry(Base, TimestampMixin):
    """Model for storing feedback entries."""
    
    __tablename__ = "feedback_entries"
    
    id = Column(Integer, primary_key=True)
    analysis_result_id = Column(Integer, ForeignKey("analysis_results.id"), nullable=False)
    user_id = Column(String, nullable=False, index=True)
    rating = Column(Integer, nullable=False)
    comment = Column(String)
    metadata = Column(JSON)
    
    # Relationships
    analysis_result = relationship("AnalysisResult", backref="feedback")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "analysis_result_id": self.analysis_result_id,
            "user_id": self.user_id,
            "rating": self.rating,
            "comment": self.comment,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

class PerformanceMetrics(Base, TimestampMixin):
    """Model for storing performance metrics."""
    
    __tablename__ = "performance_metrics"
    
    id = Column(Integer, primary_key=True)
    analyzer_name = Column(String, nullable=False, index=True)
    metric_name = Column(String, nullable=False)
    value = Column(Float, nullable=False)
    metadata = Column(JSON)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "analyzer_name": self.analyzer_name,
            "metric_name": self.metric_name,
            "value": self.value,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

class SystemHealth(Base, TimestampMixin):
    """Model for storing system health metrics."""
    
    __tablename__ = "system_health"
    
    id = Column(Integer, primary_key=True)
    component = Column(String, nullable=False, index=True)
    status = Column(String, nullable=False)
    metrics = Column(JSON, nullable=False)
    error = Column(String)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "component": self.component,
            "status": self.status,
            "metrics": self.metrics,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        } 