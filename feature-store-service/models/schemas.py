"""
Pydantic schemas for feature store service
"""
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum


class IndicatorType(str, Enum):
    """Available indicator types for configuration"""
    SMA = "sma"
    EMA = "ema"
    RSI = "rsi"
    MACD = "macd"
    BBANDS = "bollinger_bands"
    ATR = "atr"


class IndicatorConfig(BaseModel):
    """Configuration for a technical indicator"""
    type: IndicatorType
    name: Optional[str] = None
    window_size: Optional[int] = None
    price_type: Optional[str] = "close"
    
    # Specific parameters for different indicator types
    smoothing: Optional[float] = None  # For EMA
    fast_period: Optional[int] = None  # For MACD
    slow_period: Optional[int] = None  # For MACD
    signal_period: Optional[int] = None  # For MACD
    num_std: Optional[float] = None  # For Bollinger Bands
    
    @validator('name', always=True)
    def set_default_name(cls, v, values):
        """Set default name if not provided"""
        if not v:
            indicator_type = values.get('type')
            window_size = values.get('window_size')
            if indicator_type and window_size:
                return f"{indicator_type}_{window_size}"
            elif indicator_type:
                return indicator_type
        return v


class TickData(BaseModel):
    """Tick data for real-time processing"""
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: float
    volume: Optional[float] = None
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        
    @validator('timestamp', pre=True)
    def parse_timestamp(cls, v):
        """Convert string to datetime if needed"""
        if isinstance(v, str):
            return datetime.fromisoformat(v)
        return v


class IndicatorValue(BaseModel):
    """Value of a single indicator"""
    value: Union[float, Dict[str, float]]
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


class IndicatorResponse(BaseModel):
    """Response with indicator values"""
    symbol: str
    timestamp: datetime
    indicators: Dict[str, Any]
    
    class Config:
    """
    Config class.
    
    Attributes:
        Add attributes here
    """

        json_encoders = {
            datetime: lambda v: v.isoformat()
        }