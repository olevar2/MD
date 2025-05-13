"""
Defines the schemas for events published and consumed by the Analysis Engine Service.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

class BaseEvent(BaseModel):
    """Base schema for all events."""
    event_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    event_type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_service: str = "analysis-engine-service"
    version: str = "1.0"
    payload: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

class AnalysisCompletionPayload(BaseModel):
    """Payload for the AnalysisCompletion event."""
    analysis_id: str
    symbol: str
    timeframe: str
    status: str # e.g., 'completed', 'failed'
    results_summary: Dict[str, Any] # Key indicators, patterns, signals
    error_message: Optional[str] = None

class AnalysisCompletionEvent(BaseEvent):
    """Event published when an analysis task is completed."""
    event_type: str = "AnalysisCompletion"
    payload: AnalysisCompletionPayload

class MarketRegimeChangePayload(BaseModel):
    """Payload for the MarketRegimeChange event."""
    symbol: str
    timeframe: str
    previous_regime: str
    current_regime: str
    confidence_score: float
    detection_timestamp: datetime

class MarketRegimeChangeEvent(BaseEvent):
    """Event published when a market regime change is detected."""
    event_type: str = "MarketRegimeChange"
    payload: MarketRegimeChangePayload

class SignalGeneratedPayload(BaseModel):
    """Payload for the SignalGenerated event."""
    signal_id: str
    symbol: str
    timeframe: str
    signal_type: str # e.g., 'buy', 'sell', 'hold'
    strategy_name: str
    confidence: float
    trigger_price: Optional[float] = None
    indicators: Dict[str, Any]

class SignalGeneratedEvent(BaseEvent):
    """Event published when a trading signal is generated."""
    event_type: str = "SignalGenerated"
    payload: SignalGeneratedPayload

# Add other event schemas as needed, e.g.:
# - ModelRetrainingTriggerEvent
# - FeatureUpdateEvent
# - AlertTriggeredEvent