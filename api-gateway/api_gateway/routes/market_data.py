"""
Market Data Routes

This module provides routes for market data.
"""

import logging
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, HTTPException, Depends, Query, Path
from pydantic import BaseModel, Field

from api_gateway.services.market_data_service import MarketDataService


# Create logger
logger = logging.getLogger(__name__)


# Create router
router = APIRouter()


# Create service
market_data_service = MarketDataService()


# Define models
class OHLCV(BaseModel):
    """OHLCV data."""
    
    timestamp: int = Field(..., description="Timestamp in milliseconds")
    open: float = Field(..., description="Open price")
    high: float = Field(..., description="High price")
    low: float = Field(..., description="Low price")
    close: float = Field(..., description="Close price")
    volume: float = Field(..., description="Volume")


class MarketData(BaseModel):
    """Market data."""
    
    symbol: str = Field(..., description="Symbol")
    timeframe: str = Field(..., description="Timeframe")
    data: List[OHLCV] = Field(..., description="OHLCV data")


class Symbol(BaseModel):
    """Symbol information."""
    
    symbol: str = Field(..., description="Symbol")
    name: str = Field(..., description="Name")
    type: str = Field(..., description="Type")
    exchange: str = Field(..., description="Exchange")
    is_tradable: bool = Field(..., description="Whether the symbol is tradable")


# Define routes
@router.get("/symbols", response_model=List[Symbol])
async def get_symbols():
    """
    Get available symbols.
    
    Returns:
        List of available symbols
    """
    try:
        return await market_data_service.get_symbols()
    except Exception as e:
        logger.error(f"Error getting symbols: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/symbols/{symbol}", response_model=Symbol)
async def get_symbol(symbol: str = Path(..., description="Symbol")):
    """
    Get symbol information.
    
    Args:
        symbol: Symbol
        
    Returns:
        Symbol information
    """
    try:
        return await market_data_service.get_symbol(symbol)
    except Exception as e:
        logger.error(f"Error getting symbol {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data/{symbol}/{timeframe}", response_model=MarketData)
async def get_market_data(
    symbol: str = Path(..., description="Symbol"),
    timeframe: str = Path(..., description="Timeframe"),
    start: Optional[int] = Query(None, description="Start timestamp in milliseconds"),
    end: Optional[int] = Query(None, description="End timestamp in milliseconds"),
    limit: Optional[int] = Query(None, description="Limit")
):
    """
    Get market data.
    
    Args:
        symbol: Symbol
        timeframe: Timeframe
        start: Start timestamp in milliseconds
        end: End timestamp in milliseconds
        limit: Limit
        
    Returns:
        Market data
    """
    try:
        return await market_data_service.get_market_data(symbol, timeframe, start, end, limit)
    except Exception as e:
        logger.error(f"Error getting market data for {symbol}/{timeframe}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/latest/{symbol}/{timeframe}", response_model=OHLCV)
async def get_latest_market_data(
    symbol: str = Path(..., description="Symbol"),
    timeframe: str = Path(..., description="Timeframe")
):
    """
    Get latest market data.
    
    Args:
        symbol: Symbol
        timeframe: Timeframe
        
    Returns:
        Latest market data
    """
    try:
        return await market_data_service.get_latest_market_data(symbol, timeframe)
    except Exception as e:
        logger.error(f"Error getting latest market data for {symbol}/{timeframe}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))