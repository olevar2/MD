# filepath: d:\MD\forex_trading_platform\core-foundations\core_foundations\models\feedback_models.py
"""
Pydantic models for various types of feedback data used within the platform.
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Literal
from datetime import datetime
import uuid

class BaseFeedback(BaseModel):
    """
    BaseFeedback class that inherits from BaseModel.
    
    Attributes:
        Add attributes here
    """

    feedback_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: str # e.g., 'execution_engine', 'model_monitor', 'manual_input'
    feedback_type: Literal['trade', 'model_performance', 'parameter', 'user']

class TradeFeedbackData(BaseModel):
    """
    TradeFeedbackData class that inherits from BaseModel.
    
    Attributes:
        Add attributes here
    """

    trade_id: str
    strategy_id: str
    symbol: str
    outcome: Literal['profit', 'loss', 'breakeven']
    pnl: float
    slippage: Optional[float] = None
    execution_time_ms: Optional[float] = None
    market_conditions: Optional[Dict[str, Any]] = None # e.g., volatility, regime

class TradeFeedback(BaseFeedback):
    """
    TradeFeedback class that inherits from BaseFeedback.
    
    Attributes:
        Add attributes here
    """

    feedback_type: Literal['trade'] = 'trade'
    data: TradeFeedbackData

class ModelPerformanceData(BaseModel):
    """
    ModelPerformanceData class that inherits from BaseModel.
    
    Attributes:
        Add attributes here
    """

    model_id: str
    strategy_id: Optional[str] = None # Link to strategy if applicable
    metric: str # e.g., 'accuracy', 'precision', 'mae', 'prediction_drift'
    value: float
    evaluation_window_start: datetime
    evaluation_window_end: datetime
    dimensions: Optional[Dict[str, Any]] = None # e.g., {'symbol': 'EURUSD', 'timeframe': 'H1'}

class ModelPerformanceFeedback(BaseFeedback):
    """
    ModelPerformanceFeedback class that inherits from BaseFeedback.
    
    Attributes:
        Add attributes here
    """

    feedback_type: Literal['model_performance'] = 'model_performance'
    data: ModelPerformanceData

class ParameterFeedbackData(BaseModel):
    """
    ParameterFeedbackData class that inherits from BaseModel.
    
    Attributes:
        Add attributes here
    """

    strategy_id: str
    parameter_name: str
    suggested_change: Optional[Any] = None # e.g., new value, adjustment factor
    reasoning: Optional[str] = None # Why the change is suggested
    confidence: Optional[float] = None # Confidence in the suggestion (0-1)

class ParameterFeedback(BaseFeedback):
    """
    ParameterFeedback class that inherits from BaseFeedback.
    
    Attributes:
        Add attributes here
    """

    feedback_type: Literal['parameter'] = 'parameter'
    data: ParameterFeedbackData

# Add other feedback types as needed (e.g., UserFeedback)

