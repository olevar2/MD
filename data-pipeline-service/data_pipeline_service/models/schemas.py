"""
Pydantic schemas for data models used in the data pipeline service.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator


from data_pipeline_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class TimeframeEnum(str, Enum):
    """Enum representing supported OHLCV timeframes."""
    ONE_MINUTE = '1m'
    FIVE_MINUTES = '5m'
    FIFTEEN_MINUTES = '15m'
    THIRTY_MINUTES = '30m'
    ONE_HOUR = '1h'
    FOUR_HOURS = '4h'
    ONE_DAY = '1d'
    ONE_WEEK = '1w'


class AggregationMethodEnum(str, Enum):
    """Enum representing supported aggregation methods."""
    OHLCV = 'ohlcv'
    VWAP = 'vwap'
    TWAP = 'twap'


class ExportFormatEnum(str, Enum):
    """Enum representing supported export formats."""
    CSV = 'csv'
    JSON = 'json'
    PARQUET = 'parquet'


class InstrumentType(str, Enum):
    """Supported instrument types."""
    FOREX = 'forex'
    STOCK = 'stock'
    INDEX = 'index'
    COMMODITY = 'commodity'
    CRYPTO = 'crypto'


class OHLCVData(BaseModel):
    """Model for OHLCV data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @classmethod
    def from_record(cls, record: Dict[str, Any]) ->'OHLCVData':
        """Create an OHLCVData instance from a database record."""
        return cls(timestamp=record['time'], open=record['open'], high=
            record['high'], low=record['low'], close=record['close'],
            volume=record['volume'])


class OHLCVRequest(BaseModel):
    """Request model for OHLCV data queries."""
    symbol: str = Field(..., description="Instrument symbol (e.g., 'EUR/USD')")
    timeframe: str = Field(..., description='Timeframe of the candles to query'
        )
    from_time: Optional[datetime] = Field(None, description=
        'Start time for data (UTC)')
    to_time: Optional[datetime] = Field(None, description=
        'End time for data (UTC)')
    limit: Optional[int] = Field(1000, description=
        'Maximum number of candles to return')


class OHLCVResponse(BaseModel):
    """Model for OHLCV response."""
    instrument: str
    timeframe: TimeframeEnum
    start_time: datetime
    end_time: datetime
    candle_count: int
    data: List[OHLCVData]


class OHLCVBatchRequest(BaseModel):
    """Model for batch OHLCV requests."""
    instruments: List[str]
    start_time: datetime
    end_time: datetime
    timeframe: TimeframeEnum


class TickData(BaseModel):
    """Tick data model."""
    symbol: str = Field(..., description="Instrument symbol (e.g., 'EUR/USD')")
    timestamp: datetime = Field(..., description='Tick timestamp in UTC')
    bid: float = Field(..., description='Bid price')
    ask: float = Field(..., description='Ask price')
    volume: Optional[float] = Field(None, description='Volume (if available)')

    @property
    def spread(self) ->float:
        """Calculate the spread in pips."""
        return self.ask - self.bid


class TickDataRequest(BaseModel):
    """Request model for tick data queries."""
    symbol: str = Field(..., description="Instrument symbol (e.g., 'EUR/USD')")
    from_time: Optional[datetime] = Field(None, description=
        'Start time for data (UTC)')
    to_time: Optional[datetime] = Field(None, description=
        'End time for data (UTC)')
    limit: Optional[int] = Field(10000, description=
        'Maximum number of ticks to return')


class Instrument(BaseModel):
    """Trading instrument model."""
    symbol: str = Field(..., description="Instrument symbol (e.g., 'EUR/USD')")
    name: str = Field(..., description='Full name of the instrument')
    type: str = Field(..., description=
        'Type of instrument (forex, stock, etc.)')
    pip_size: float = Field(..., description='Size of one pip in decimal')
    min_lot_size: float = Field(..., description='Minimum trading lot size')
    max_lot_size: float = Field(..., description='Maximum trading lot size')
    lot_step: float = Field(..., description='Step size for lots')
    commission: Optional[float] = Field(None, description=
        'Commission per standard lot')
    swap_long: Optional[float] = Field(None, description=
        'Swap points for long positions')
    swap_short: Optional[float] = Field(None, description=
        'Swap points for short positions')
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @validator('type')
    @with_exception_handling
    def validate_type(cls, v):
        """Validate that the instrument type is supported."""
        try:
            return InstrumentType(v)
        except ValueError:
            raise ValueError(
                f"Type must be one of: {', '.join([t.value for t in InstrumentType])}"
                )


class TradingHours(BaseModel):
    """Trading hours model for instruments."""
    id: Optional[int] = None
    symbol: str = Field(..., description="Instrument symbol (e.g., 'EUR/USD')")
    day_of_week: int = Field(..., description=
        'Day of week (0=Monday, 6=Sunday)', ge=0, le=6)
    open_time: str = Field(..., description='Opening time in format HH:MM')
    close_time: str = Field(..., description='Closing time in format HH:MM')
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class PaginatedResponse(BaseModel):
    """Paginated response model."""
    total: int = Field(..., description='Total number of items')
    page: int = Field(..., description='Current page number')
    page_size: int = Field(..., description='Number of items per page')
    pages: int = Field(..., description='Total number of pages')
    data: List[Dict[str, Any]] = Field(..., description='Response data')


class ErrorResponse(BaseModel):
    """Error response model."""
    detail: str = Field(..., description='Error detail message')
    status_code: int = Field(..., description='HTTP status code')
    timestamp: datetime = Field(default_factory=datetime.utcnow,
        description='Error timestamp')


class ServiceAuth(BaseModel):
    """Model for service-to-service authentication."""
    service_name: str
    api_key: str
