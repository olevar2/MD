# Forex Trading Platform Coding Examples

This document provides concrete examples of both correct and incorrect implementations according to our coding standards. These examples are designed to help developers understand how to apply the standards in practice.

## Table of Contents

1. [Domain Model Examples](#domain-model-examples)
2. [Service Implementation Examples](#service-implementation-examples)
3. [API Design Examples](#api-design-examples)
4. [Error Handling Examples](#error-handling-examples)
5. [Testing Examples](#testing-examples)

## Domain Model Examples

### Correct Implementation

```python
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional, List, Literal

OrderSide = Literal['BUY', 'SELL']
OrderType = Literal['MARKET', 'LIMIT', 'STOP', 'STOP_LIMIT']
OrderStatus = Literal['PENDING', 'FILLED', 'PARTIALLY_FILLED', 'CANCELLED', 'REJECTED']

@dataclass
class Order:
    """Represents a trading order in the forex market."""
    
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: str = 'GTC'
    order_id: Optional[str] = None
    status: OrderStatus = 'PENDING'
    created_at: datetime = datetime.utcnow()
    updated_at: Optional[datetime] = None
    
    def is_valid(self) -> bool:
        """Check if the order is valid based on its type and parameters."""
        if self.order_type in ('LIMIT', 'STOP_LIMIT') and self.price is None:
            return False
        if self.order_type in ('STOP', 'STOP_LIMIT') and self.stop_price is None:
            return False
        return True
```

**Why this is correct:**
- Uses domain-specific terminology (`OrderSide`, `OrderType`, `OrderStatus`)
- Implements proper type annotations with `Literal` for enum-like types
- Uses `dataclass` for clean, concise model definition
- Includes comprehensive docstrings
- Implements domain validation logic
- Uses appropriate types (`Decimal` for financial values)

### Incorrect Implementation

```python
class Order:
    """Order class."""
    
    def __init__(self, data):
        """Initialize with data."""
        self.data = data
        
    def process(self):
        """Process the order."""
        # Implementation with no validation or domain logic
        pass
```

**Why this is incorrect:**
- Generic naming not aligned with domain language
- No type annotations
- Accepts generic `data` parameter without validation
- No domain-specific validation logic
- Minimal, unhelpful docstrings
- No clear representation of domain concepts

## Service Implementation Examples

### Correct Implementation

```python
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

import pandas as pd

from common_lib.errors import MarketDataError
from common_lib.models import Timeframe
from common_lib.adapters import DataProvider
from common_lib.caching import CacheManager

class MarketDataService:
    """Service for retrieving and processing market data."""
    
    def __init__(self, data_provider: DataProvider, cache_manager: CacheManager):
        """Initialize the market data service with dependencies."""
        self.data_provider = data_provider
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)
        
    async def get_ohlcv_data(
        self, 
        symbol: str, 
        timeframe: Timeframe, 
        start_time: datetime, 
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get OHLCV (Open, High, Low, Close, Volume) data for a symbol.
        
        Args:
            symbol: The trading symbol (e.g., 'EUR/USD')
            timeframe: The timeframe for the data
            start_time: The start time for the data
            end_time: The end time for the data (defaults to current time)
            
        Returns:
            DataFrame with OHLCV data
            
        Raises:
            MarketDataError: If data cannot be retrieved
        """
        # Check cache first
        cache_key = f"{symbol}_{timeframe}_{start_time.isoformat()}_{end_time.isoformat() if end_time else 'now'}"
        cached_data = self.cache_manager.get(cache_key)
        if cached_data is not None:
            self.logger.debug(f"Retrieved cached data for {cache_key}")
            return cached_data
            
        # Fetch from provider if not in cache
        try:
            data = await self.data_provider.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time
            )
            
            # Process data
            processed_data = self._process_ohlcv_data(data)
            
            # Cache the result
            self.cache_manager.set(cache_key, processed_data, ttl=3600)  # Cache for 1 hour
            
            return processed_data
        except Exception as e:
            self.logger.error(f"Failed to retrieve market data: {e}")
            raise MarketDataError(f"Failed to retrieve market data for {symbol}: {e}")
            
    def _process_ohlcv_data(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process raw OHLCV data into a pandas DataFrame."""
        # Implementation
        df = pd.DataFrame(data)
        # Process data
        return df
```

**Why this is correct:**
- Uses domain-specific terminology (`OHLCV`, `symbol`, `timeframe`)
- Implements proper dependency injection
- Uses type annotations throughout
- Includes comprehensive docstrings with Args, Returns, and Raises sections
- Implements proper error handling with domain-specific exceptions
- Uses caching for performance optimization
- Follows single responsibility principle with helper methods
- Includes proper logging

### Incorrect Implementation

```python
import requests

class MarketDataService:
    """Market data service."""
    
    def get_data(self, symbol, tf, start, end=None):
        """Get data."""
        # Direct HTTP call without abstraction
        data = requests.get(f"https://api.example.com/data?symbol={symbol}&tf={tf}&start={start}&end={end}")
        return data.json()
```

**Why this is incorrect:**
- Generic naming not aligned with domain language
- No type annotations
- No error handling
- Direct dependency on external HTTP library
- No caching strategy
- No proper parameter validation
- Minimal, unhelpful docstrings
- No logging

## API Design Examples

### Correct Implementation

```python
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, Query, Path, HTTPException
from pydantic import BaseModel, Field

from common_lib.models import Timeframe
from market_data_service.domain.services import MarketDataService
from market_data_service.api.dependencies import get_market_data_service

router = APIRouter(prefix="/market-data", tags=["Market Data"])

class OHLCVResponse(BaseModel):
    """OHLCV data response model."""
    
    symbol: str = Field(..., description="The trading symbol (e.g., 'EUR/USD')")
    timeframe: str = Field(..., description="The timeframe for the data")
    data: List[dict] = Field(..., description="OHLCV data points")
    start_time: datetime = Field(..., description="Start time of the data")
    end_time: datetime = Field(..., description="End time of the data")

@router.get(
    "/{symbol}/ohlcv",
    response_model=OHLCVResponse,
    summary="Get OHLCV data for a symbol",
    description="Retrieve Open, High, Low, Close, Volume data for a specific symbol and timeframe",
)
async def get_ohlcv_data(
    symbol: str = Path(..., description="The trading symbol (e.g., 'EUR/USD')"),
    timeframe: Timeframe = Query(..., description="The timeframe for the data"),
    start_time: datetime = Query(..., description="Start time for the data (ISO format)"),
    end_time: Optional[datetime] = Query(None, description="End time for the data (ISO format)"),
    market_data_service: MarketDataService = Depends(get_market_data_service),
):
    """
    Get OHLCV (Open, High, Low, Close, Volume) data for a symbol.
    
    This endpoint retrieves historical price data for the specified symbol and timeframe.
    """
    try:
        df = await market_data_service.get_ohlcv_data(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
        )
        
        return OHLCVResponse(
            symbol=symbol,
            timeframe=timeframe,
            data=df.to_dict(orient="records"),
            start_time=start_time,
            end_time=end_time or datetime.utcnow(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Why this is correct:**
- Uses domain-specific terminology
- Implements proper request and response models with Pydantic
- Uses dependency injection for services
- Includes comprehensive documentation with summary and description
- Uses proper parameter validation with Path and Query
- Implements proper error handling
- Returns structured response

### Incorrect Implementation

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/data")
def get_data():
    """Get data endpoint."""
    symbol = request.args.get("symbol")
    tf = request.args.get("tf")
    start = request.args.get("start")
    end = request.args.get("end")
    
    # Direct service call without abstraction
    service = MarketDataService()
    data = service.get_data(symbol, tf, start, end)
    
    return jsonify(data)
```

**Why this is incorrect:**
- Generic naming not aligned with domain language
- No parameter validation
- No error handling
- Direct service instantiation without dependency injection
- No response model definition
- Minimal, unhelpful docstrings
- No proper API versioning or structure

## Error Handling Examples

### Correct Implementation

```python
from common_lib.errors import MarketDataError, ServiceUnavailableError
from common_lib.logging import get_logger

logger = get_logger(__name__)

class MarketDataAdapter:
    """Adapter for retrieving market data from external providers."""
    
    def __init__(self, client, config):
        """Initialize the adapter with a client and configuration."""
        self.client = client
        self.config = config
        self.retry_count = config.get("retry_count", 3)
        self.retry_delay = config.get("retry_delay", 1.0)
        
    async def get_historical_data(self, symbol, timeframe, start_time, end_time=None):
        """
        Get historical market data from the provider.
        
        Args:
            symbol: The trading symbol
            timeframe: The timeframe for the data
            start_time: The start time for the data
            end_time: The end time for the data
            
        Returns:
            List of OHLCV data points
            
        Raises:
            MarketDataError: If data cannot be retrieved
            ServiceUnavailableError: If the service is unavailable
        """
        for attempt in range(self.retry_count):
            try:
                response = await self.client.get_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_time,
                    end_time=end_time,
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 404:
                    logger.warning(f"No data found for {symbol} from {start_time} to {end_time}")
                    return []
                elif response.status_code >= 500:
                    logger.error(f"Provider service error: {response.status_code} - {response.text}")
                    raise ServiceUnavailableError(f"Provider service error: {response.status_code}")
                else:
                    logger.error(f"Unexpected response: {response.status_code} - {response.text}")
                    raise MarketDataError(f"Failed to retrieve market data: {response.text}")
                    
            except ConnectionError as e:
                logger.warning(f"Connection error on attempt {attempt + 1}/{self.retry_count}: {e}")
                if attempt == self.retry_count - 1:
                    raise ServiceUnavailableError(f"Service unavailable after {self.retry_count} attempts: {e}")
                await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise MarketDataError(f"Failed to retrieve market data: {e}")
```

**Why this is correct:**
- Uses domain-specific exceptions
- Implements retry logic with exponential backoff
- Includes proper logging at appropriate levels
- Handles different error scenarios differently
- Provides clear error messages
- Preserves original exception context
- Follows the principle of failing fast when recovery is not possible

### Incorrect Implementation

```python
def get_historical_data(symbol, timeframe, start_time, end_time=None):
    """Get historical data."""
    try:
        response = requests.get(
            f"https://api.example.com/data?symbol={symbol}&tf={timeframe}&start={start_time}&end={end_time}"
        )
        return response.json()
    except:
        # Catch-all exception with no specific handling
        return []
```

**Why this is incorrect:**
- Uses generic try-except block that catches all exceptions
- Returns empty list on error instead of raising appropriate exceptions
- No logging of errors
- No retry logic for transient failures
- No differentiation between different types of errors
- No proper error messages
- Direct HTTP call without abstraction

## Testing Examples

### Correct Implementation

```python
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, patch

from common_lib.models import Timeframe
from market_data_service.domain.services import MarketDataService
from common_lib.errors import MarketDataError

@pytest.fixture
def mock_data_provider():
    """Fixture for mocked data provider."""
    provider = AsyncMock()
    provider.get_historical_data.return_value = [
        {"timestamp": "2023-01-01T00:00:00Z", "open": 1.1000, "high": 1.1100, "low": 1.0900, "close": 1.1050, "volume": 1000},
        {"timestamp": "2023-01-01T01:00:00Z", "open": 1.1050, "high": 1.1150, "low": 1.1000, "close": 1.1100, "volume": 1200},
    ]
    return provider

@pytest.fixture
def mock_cache_manager():
    """Fixture for mocked cache manager."""
    cache = AsyncMock()
    cache.get.return_value = None
    return cache

@pytest.mark.asyncio
async def test_get_ohlcv_data_success(mock_data_provider, mock_cache_manager):
    """Test successful retrieval of OHLCV data."""
    # Arrange
    service = MarketDataService(
        data_provider=mock_data_provider,
        cache_manager=mock_cache_manager,
    )
    symbol = "EUR/USD"
    timeframe = Timeframe.HOUR_1
    start_time = datetime(2023, 1, 1)
    end_time = start_time + timedelta(hours=2)
    
    # Act
    result = await service.get_ohlcv_data(
        symbol=symbol,
        timeframe=timeframe,
        start_time=start_time,
        end_time=end_time,
    )
    
    # Assert
    assert len(result) == 2
    assert result.iloc[0]["close"] == 1.1050
    assert result.iloc[1]["close"] == 1.1100
    mock_data_provider.get_historical_data.assert_called_once_with(
        symbol=symbol,
        timeframe=timeframe,
        start_time=start_time,
        end_time=end_time,
    )
    mock_cache_manager.set.assert_called_once()

@pytest.mark.asyncio
async def test_get_ohlcv_data_from_cache(mock_data_provider, mock_cache_manager):
    """Test retrieval of OHLCV data from cache."""
    # Arrange
    import pandas as pd
    cached_data = pd.DataFrame([
        {"timestamp": "2023-01-01T00:00:00Z", "open": 1.1000, "high": 1.1100, "low": 1.0900, "close": 1.1050, "volume": 1000},
    ])
    mock_cache_manager.get.return_value = cached_data
    
    service = MarketDataService(
        data_provider=mock_data_provider,
        cache_manager=mock_cache_manager,
    )
    symbol = "EUR/USD"
    timeframe = Timeframe.HOUR_1
    start_time = datetime(2023, 1, 1)
    end_time = start_time + timedelta(hours=1)
    
    # Act
    result = await service.get_ohlcv_data(
        symbol=symbol,
        timeframe=timeframe,
        start_time=start_time,
        end_time=end_time,
    )
    
    # Assert
    assert result is cached_data
    mock_data_provider.get_historical_data.assert_not_called()
    mock_cache_manager.set.assert_not_called()

@pytest.mark.asyncio
async def test_get_ohlcv_data_error(mock_data_provider, mock_cache_manager):
    """Test error handling when retrieving OHLCV data."""
    # Arrange
    mock_data_provider.get_historical_data.side_effect = Exception("Provider error")
    
    service = MarketDataService(
        data_provider=mock_data_provider,
        cache_manager=mock_cache_manager,
    )
    symbol = "EUR/USD"
    timeframe = Timeframe.HOUR_1
    start_time = datetime(2023, 1, 1)
    
    # Act & Assert
    with pytest.raises(MarketDataError) as excinfo:
        await service.get_ohlcv_data(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
        )
    
    assert "Failed to retrieve market data" in str(excinfo.value)
    mock_cache_manager.set.assert_not_called()
```

**Why this is correct:**
- Uses pytest fixtures for test setup
- Implements proper mocking of dependencies
- Tests both success and error scenarios
- Tests caching behavior
- Follows Arrange-Act-Assert pattern
- Includes assertions for both return values and side effects
- Uses descriptive test names
- Tests domain-specific error handling

### Incorrect Implementation

```python
def test_market_data():
    """Test market data."""
    service = MarketDataService()
    data = service.get_data("EURUSD", "1h", "2023-01-01")
    assert data is not None
```

**Why this is incorrect:**
- No proper test setup or teardown
- No mocking of dependencies
- Only tests the happy path
- Minimal assertions
- No testing of error scenarios
- No testing of side effects
- Generic test name
- Creates actual service instance instead of using mocks