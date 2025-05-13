"""
Market Data API Module

This module provides API endpoints for retrieving market data using the adapter pattern.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from common_lib.interfaces.market_data import IMarketDataProvider, IMarketDataCache
from common_lib.errors.base_exceptions import BaseError, ValidationError, DataError, ServiceError
from core.main_1 import get_market_data_provider, get_market_data_cache
from core_foundations.utils.logger import get_logger
logger = get_logger('data-pipeline-service.market-data-api')
market_data_router = APIRouter(prefix='/api/v1/market-data', tags=[
    'market-data'], responses={(404): {'description': 'Not found'}})


from data_pipeline_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class MarketDataResponse(BaseModel):
    """Response model for market data."""
    symbol: str
    timeframe: str
    data: List[Dict[str, Any]]
    columns: List[str]


class LatestPriceResponse(BaseModel):
    """Response model for latest price."""
    symbol: str
    bid: float
    ask: float
    timestamp: datetime


class SymbolsResponse(BaseModel):
    """Response model for available symbols."""
    symbols: List[str]


@market_data_router.get('/historical', response_model=MarketDataResponse)
@async_with_exception_handling
async def get_historical_data(symbol: str, timeframe: str, start_time:
    datetime, end_time: Optional[datetime]=None, market_data_provider:
    IMarketDataProvider=Depends(get_market_data_provider),
    market_data_cache: IMarketDataCache=Depends(get_market_data_cache)):
    """
    Get historical market data for a symbol.

    Args:
        symbol: The trading symbol (e.g., "EURUSD")
        timeframe: The timeframe (e.g., "1m", "5m", "1h", "1d")
        start_time: Start time for the data
        end_time: Optional end time for the data
        market_data_provider: The market data provider adapter
        market_data_cache: The market data cache adapter

    Returns:
        MarketDataResponse containing the historical data
    """
    try:
        cached_data = await market_data_cache.get_cached_data(symbol=symbol,
            timeframe=timeframe, start_time=start_time, end_time=end_time)
        if cached_data is not None:
            logger.info(f'Cache hit for {symbol} {timeframe}')
            result = []
            for timestamp, row in cached_data.iterrows():
                record = {'timestamp': timestamp.isoformat()}
                for col in cached_data.columns:
                    record[col] = float(row[col]) if not pd.isna(row[col]
                        ) else None
                result.append(record)
            return MarketDataResponse(symbol=symbol, timeframe=timeframe,
                data=result, columns=list(cached_data.columns))
        logger.info(
            f'Cache miss for {symbol} {timeframe}, fetching from provider')
        data = await market_data_provider.get_historical_data(symbol=symbol,
            timeframe=timeframe, start_time=start_time, end_time=end_time)
        await market_data_cache.cache_data(symbol=symbol, timeframe=
            timeframe, data=data)
        result = []
        for timestamp, row in data.iterrows():
            record = {'timestamp': timestamp.isoformat()}
            for col in data.columns:
                record[col] = float(row[col]) if not pd.isna(row[col]
                    ) else None
            result.append(record)
        return MarketDataResponse(symbol=symbol, timeframe=timeframe, data=
            result, columns=list(data.columns))
    except ValidationError as e:
        logger.warning(f'Validation error: {str(e)}')
        raise HTTPException(status_code=400, detail=str(e))
    except DataError as e:
        logger.warning(f'Data error: {str(e)}')
        raise HTTPException(status_code=404, detail=str(e))
    except ServiceError as e:
        logger.error(f'Service error: {str(e)}')
        raise HTTPException(status_code=503, detail=str(e))
    except BaseError as e:
        logger.error(f'Base error: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f'Unexpected error: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Unexpected error: {str(e)}')


@market_data_router.get('/latest-price', response_model=LatestPriceResponse)
@async_with_exception_handling
async def get_latest_price(symbol: str, market_data_provider:
    IMarketDataProvider=Depends(get_market_data_provider)):
    """
    Get the latest price for a symbol.

    Args:
        symbol: The trading symbol (e.g., "EURUSD")
        market_data_provider: The market data provider adapter

    Returns:
        LatestPriceResponse containing the latest price information
    """
    try:
        price_data = await market_data_provider.get_latest_price(symbol=symbol)
        return LatestPriceResponse(symbol=symbol, bid=price_data.get('bid',
            0.0), ask=price_data.get('ask', 0.0), timestamp=price_data.get(
            'timestamp', datetime.utcnow()))
    except ValidationError as e:
        logger.warning(f'Validation error: {str(e)}')
        raise HTTPException(status_code=400, detail=str(e))
    except DataError as e:
        logger.warning(f'Data error: {str(e)}')
        raise HTTPException(status_code=404, detail=str(e))
    except ServiceError as e:
        logger.error(f'Service error: {str(e)}')
        raise HTTPException(status_code=503, detail=str(e))
    except BaseError as e:
        logger.error(f'Base error: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f'Unexpected error: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Unexpected error: {str(e)}')


@market_data_router.get('/symbols', response_model=SymbolsResponse)
@async_with_exception_handling
async def get_symbols(market_data_provider: IMarketDataProvider=Depends(
    get_market_data_provider)):
    """
    Get available symbols.

    Args:
        market_data_provider: The market data provider adapter

    Returns:
        SymbolsResponse containing the list of available symbols
    """
    try:
        symbols = await market_data_provider.get_symbols()
        return SymbolsResponse(symbols=symbols)
    except ServiceError as e:
        logger.error(f'Service error: {str(e)}')
        raise HTTPException(status_code=503, detail=str(e))
    except BaseError as e:
        logger.error(f'Base error: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f'Unexpected error: {str(e)}')
        raise HTTPException(status_code=500, detail=
            f'Unexpected error: {str(e)}')
