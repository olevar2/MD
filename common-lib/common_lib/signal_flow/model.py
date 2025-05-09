"""
Core models for signal flow between analysis engine and strategy execution.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel

class SignalCategory(str, Enum):
    """Categories of trading signals"""
    TECHNICAL_ANALYSIS = "technical_analysis"
    MACHINE_LEARNING = "machine_learning"
    MARKET_SENTIMENT = "market_sentiment"
    ECONOMIC_INDICATORS = "economic_indicators"
    CORRELATION_SIGNALS = "correlation_signals"

class SignalSource(str, Enum):
    """Source types of trading signals"""
    INDICATOR = "indicator"
    PATTERN = "pattern"
    ML_MODEL = "ml_model"
    SENTIMENT = "sentiment"
    ECONOMIC = "economic"
    REGIME = "market_regime"

class SignalStrength(str, Enum):
    """Signal strength classifications"""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NEUTRAL = "neutral"

class SignalPriority(str, Enum):
    """Signal priority levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class SignalFlow(BaseModel):
    """
    Represents a unified trading signal flow from analysis to execution.
    This is the core data structure for signal communication between services.
    """
    signal_id: str
    generated_at: datetime
    symbol: str
    timeframe: str
    category: SignalCategory
    source: SignalSource
    direction: str
    strength: SignalStrength
    confidence: float
    priority: SignalPriority
    expiry: Optional[datetime] = None
    
    # Signal generation context
    market_context: Dict[str, Any]
    technical_context: Dict[str, Any]
    model_context: Optional[Dict[str, Any]] = None
    
    # Signal quality metrics
    quality_metrics: Dict[str, float]
    confluence_score: float
    
    # Risk parameters
    risk_parameters: Dict[str, Any]
    suggested_entry: float
    suggested_stop: Optional[float] = None
    suggested_target: Optional[float] = None
    position_size_factor: float = 1.0
    
    # Metadata
    metadata: Dict[str, Any] = {}

class SignalFlowState(str, Enum):
    """States of a signal in the flow pipeline"""
    GENERATED = "generated"
    VALIDATED = "validated"
    AGGREGATED = "aggregated"
    RISK_CHECKED = "risk_checked"
    EXECUTING = "executing"
    COMPLETED = "completed"
    REJECTED = "rejected"
    EXPIRED = "expired"

class SignalValidationResult(BaseModel):
    """Result of signal validation checks"""
    is_valid: bool
    validation_checks: Dict[str, bool]
    risk_metrics: Dict[str, float]
    notes: List[str]

class SignalAggregationResult(BaseModel):
    """Result of signal aggregation process"""
    aggregated_direction: str
    aggregated_confidence: float
    contributing_signals: List[str]
    weights_used: Dict[str, float]
    explanation: str
