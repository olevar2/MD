"""
Standardized Event Schema for Forex Trading Platform

This module defines the standardized event schema used across the Forex trading
platform. It provides a consistent structure for events, ensuring
they can be properly routed, processed, and stored across services.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Enum of all possible event types in the system."""
    
    # Market data events
    MARKET_DATA_UPDATE = "market_data.update"
    MARKET_DATA_SNAPSHOT = "market_data.snapshot"
    MARKET_DATA_ERROR = "market_data.error"
    
    # Trading events
    ORDER_CREATED = "order.created"
    ORDER_UPDATED = "order.updated"
    ORDER_FILLED = "order.filled"
    ORDER_CANCELLED = "order.cancelled"
    ORDER_REJECTED = "order.rejected"
    ORDER_ERROR = "order.error"
    
    # Position events
    POSITION_OPENED = "position.opened"
    POSITION_UPDATED = "position.updated"
    POSITION_CLOSED = "position.closed"
    
    # Account events
    ACCOUNT_BALANCE_UPDATED = "account.balance_updated"
    ACCOUNT_MARGIN_UPDATED = "account.margin_updated"
    ACCOUNT_SETTINGS_UPDATED = "account.settings_updated"
    
    # Strategy events
    STRATEGY_SIGNAL = "strategy.signal"
    STRATEGY_STARTED = "strategy.started"
    STRATEGY_STOPPED = "strategy.stopped"
    STRATEGY_ERROR = "strategy.error"
    STRATEGY_PARAMETER_UPDATED = "strategy.parameter_updated"
    
    # Risk management events
    RISK_LIMIT_BREACH = "risk.limit_breach" 
    RISK_ALERT = "risk.alert"
    RISK_SETTINGS_UPDATED = "risk.settings_updated"
    
    # Analysis events
    ANALYSIS_REGIME_CHANGE = "analysis.regime_change"
    ANALYSIS_INDICATOR_UPDATE = "analysis.indicator_update"
    ANALYSIS_PATTERN_DETECTED = "analysis.pattern_detected"
    
    # ML events
    MODEL_PREDICTION = "ml.prediction"
    MODEL_TRAINING_STARTED = "ml.training_started"
    MODEL_TRAINING_COMPLETED = "ml.training_completed"
    MODEL_TRAINING_FAILED = "ml.training_failed"
    MODEL_DEPLOYED = "ml.model_deployed"
    
    # System events
    SERVICE_HEALTH_CHANGED = "system.health_changed"
    SERVICE_STARTED = "system.service_started"
    SERVICE_STOPPED = "system.service_stopped"
    SERVICE_DEGRADED = "system.service_degraded"
    
    # User events
    USER_ACTION = "user.action"
    USER_SETTINGS_UPDATED = "user.settings_updated"
    USER_LOGIN = "user.login"
    USER_LOGOUT = "user.logout"
    
    # Feedback events
    TRADING_FEEDBACK = "feedback.trading"
    PARAMETER_FEEDBACK = "feedback.parameter"
    MODEL_FEEDBACK = "feedback.model"


class EventPriority(str, Enum):
    """Priority levels for events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Event(BaseModel):
    """
    Base event class that defines the standard event structure
    used across the Forex trading platform.
    """
    # Core event properties
    event_id: UUID = Field(default_factory=uuid4, description="Unique identifier for this event")
    event_type: EventType = Field(..., description="Type of event")
    event_version: str = Field(default="1.0", description="Schema version for this event type")
    event_time: datetime = Field(default_factory=datetime.utcnow, description="UTC timestamp when the event was created")
    
    # Source and destination
    source_service: str = Field(..., description="Name of the service that generated the event")
    target_services: Optional[List[str]] = Field(default=None, description="Specific target services, if any")
    
    # Routing information
    correlation_id: Optional[UUID] = Field(default=None, description="ID to correlate related events")
    causation_id: Optional[UUID] = Field(default=None, description="ID of the event that caused this event")
    priority: EventPriority = Field(default=EventPriority.MEDIUM, description="Event priority for processing")
    
    # Payload data
    data: Dict[str, Any] = Field(..., description="Event payload data")
    
    # Metadata - for additional information that might be useful for processing
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
            UUID: lambda uuid: str(uuid)
        }


# Specific Event Types
class MarketDataEvent(Event):
    """Events related to market data updates."""
    event_type: EventType = Field(..., description="Must be one of the MARKET_DATA_* event types")
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "symbol": "EUR/USD",
                "timestamp": "2025-04-17T14:32:15.123456Z",
                "bid": 1.0921,
                "ask": 1.0923,
                "volume": 1000000
            }
        }


class OrderEvent(Event):
    """Events related to order lifecycle."""
    event_type: EventType = Field(..., description="Must be one of the ORDER_* event types")
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "order_id": "abc123",
                "symbol": "EUR/USD",
                "side": "BUY",
                "type": "LIMIT",
                "quantity": 100000,
                "price": 1.0925,
                "status": "FILLED",
                "timestamp": "2025-04-17T14:35:22.123456Z"
            }
        }


class PositionEvent(Event):
    """Events related to position updates."""
    event_type: EventType = Field(..., description="Must be one of the POSITION_* event types")
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "position_id": "pos123",
                "symbol": "EUR/USD",
                "side": "LONG",
                "quantity": 100000,
                "entry_price": 1.0921,
                "current_price": 1.0950,
                "unrealized_pl": 290,
                "timestamp": "2025-04-17T15:12:05.123456Z"
            }
        }


class RiskEvent(Event):
    """Events related to risk management."""
    event_type: EventType = Field(..., description="Must be one of the RISK_* event types")
    priority: EventPriority = Field(default=EventPriority.HIGH, description="Risk events are high priority by default")
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "limit_type": "DAILY_DRAWDOWN",
                "current_value": -1200,
                "threshold": -1000,
                "account_id": "acc123",
                "timestamp": "2025-04-17T16:45:10.123456Z"
            }
        }


class StrategyEvent(Event):
    """Events related to trading strategies."""
    event_type: EventType = Field(..., description="Must be one of the STRATEGY_* event types")
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "strategy_id": "trend_follower_001",
                "symbol": "EUR/USD",
                "signal": "LONG",
                "confidence": 0.85,
                "timeframe": "H4",
                "timestamp": "2025-04-17T18:00:00.123456Z"
            }
        }


class MLEvent(Event):
    """Events related to machine learning operations."""
    event_type: EventType = Field(..., description="Must be one of the ML_* event types")
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "model_id": "regime_classifier_v2",
                "prediction": "TRENDING",
                "confidence": 0.92,
                "features": {"volatility": 0.12, "momentum": 0.78},
                "timestamp": "2025-04-17T18:05:22.123456Z"
            }
        }


class SystemEvent(Event):
    """Events related to system operations."""
    event_type: EventType = Field(..., description="Must be one of the SYSTEM_* event types")
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "service_name": "trading-gateway-service",
                "status": "DEGRADED",
                "details": "Broker API connection intermittent",
                "timestamp": "2025-04-17T19:22:15.123456Z"
            }
        }


class FeedbackEvent(Event):
    """Events related to trading feedback and adaptation."""
    event_type: EventType = Field(..., description="Must be one of the FEEDBACK_* event types")
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "feedback_type": "TRADE_OUTCOME",
                "strategy_id": "trend_follower_001", 
                "position_id": "pos123",
                "outcome": "PROFITABLE",
                "pl": 450,
                "analysis": {"entry_timing": 0.8, "exit_timing": 0.4},
                "timestamp": "2025-04-17T20:15:30.123456Z"
            }
        }


# Factory function to create events
def create_event(
    event_type: EventType,
    source_service: str,
    data: Dict[str, Any],
    correlation_id: Optional[UUID] = None,
    causation_id: Optional[UUID] = None,
    target_services: Optional[List[str]] = None,
    priority: Optional[EventPriority] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Event:
    """
    Factory function to create events with the appropriate type.
    
    Args:
        event_type: Type of the event
        source_service: Name of the service creating the event
        data: Event payload data
        correlation_id: Optional ID to correlate related events
        causation_id: Optional ID of the event that caused this event
        target_services: Optional list of specific target services
        priority: Optional event priority
        metadata: Optional additional metadata
        
    Returns:
        An Event instance of the appropriate type
    """
    # Set defaults
    if metadata is None:
        metadata = {}
        
    # Choose event class based on event type
    if event_type.startswith("market_data."):
        event_class = MarketDataEvent
    elif event_type.startswith("order."):
        event_class = OrderEvent
    elif event_type.startswith("position."):
        event_class = PositionEvent
    elif event_type.startswith("risk."):
        event_class = RiskEvent
        # Default to high priority for risk events if not specified
        priority = priority or EventPriority.HIGH
    elif event_type.startswith("strategy."):
        event_class = StrategyEvent
    elif event_type.startswith("ml."):
        event_class = MLEvent
    elif event_type.startswith("system."):
        event_class = SystemEvent
    elif event_type.startswith("feedback."):
        event_class = FeedbackEvent
    else:
        # Use base event class for other types
        event_class = Event
        
    # Create the event
    return event_class(
        event_type=event_type,
        source_service=source_service,
        data=data,
        correlation_id=correlation_id,
        causation_id=causation_id,
        target_services=target_services,
        priority=priority or EventPriority.MEDIUM,
        metadata=metadata
    )
