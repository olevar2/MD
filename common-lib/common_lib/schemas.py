"""
Common schema definitions for use across the forex trading platform.

This module provides standardized data models for market data, trading instruments,
and related financial constructs.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator, field_validator


class TimeframeEnum(str, Enum):
    """Enum representing supported OHLCV timeframes."""
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1d"
    ONE_WEEK = "1w"


class AggregationMethodEnum(str, Enum):
    """Enum representing supported aggregation methods."""
    OHLCV = "ohlcv"  # Standard OHLCV aggregation
    VWAP = "vwap"    # Volume-weighted average price
    TWAP = "twap"    # Time-weighted average price


class OHLCVData(BaseModel):
    """Model for OHLCV (Open-High-Low-Close-Volume) data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    @classmethod
    def from_record(cls, record: Dict[str, Any]) -> "OHLCVData":
        """Create an OHLCVData instance from a database record."""
        return cls(
            timestamp=record["time"],
            open=record["open"],
            high=record["high"],
            low=record["low"],
            close=record["close"],
            volume=record["volume"]
        )


class TickData(BaseModel):
    """Tick data model."""
    
    symbol: str = Field(..., description="Instrument symbol (e.g., 'EUR/USD')")
    timestamp: datetime = Field(..., description="Tick timestamp in UTC")
    bid: float = Field(..., description="Bid price")
    ask: float = Field(..., description="Ask price")
    volume: Optional[float] = Field(None, description="Volume (if available)")
    
    @property
    def spread(self) -> float:
        """Calculate the spread in pips."""
        return self.ask - self.bid


class DataValidationError(Exception):
    """Exception raised for data validation errors."""
    
    def __init__(self, message: str, data: Any = None):
        self.message = message
        self.data = data
        super().__init__(self.message)