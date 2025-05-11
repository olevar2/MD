"""
Analysis Routes

This module provides routes for analysis.
"""

import logging
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, HTTPException, Depends, Query, Path
from pydantic import BaseModel, Field

from api_gateway.services.analysis_service import AnalysisService


# Create logger
logger = logging.getLogger(__name__)


# Create router
router = APIRouter()


# Create service
analysis_service = AnalysisService()


# Define models
class IndicatorValue(BaseModel):
    """Indicator value."""
    
    timestamp: int = Field(..., description="Timestamp in milliseconds")
    value: float = Field(..., description="Indicator value")


class IndicatorResult(BaseModel):
    """Indicator result."""
    
    indicator: str = Field(..., description="Indicator name")
    symbol: str = Field(..., description="Symbol")
    timeframe: str = Field(..., description="Timeframe")
    parameters: Dict[str, Any] = Field(..., description="Indicator parameters")
    values: List[IndicatorValue] = Field(..., description="Indicator values")


class IndicatorRequest(BaseModel):
    """Indicator request."""
    
    indicator: str = Field(..., description="Indicator name")
    symbol: str = Field(..., description="Symbol")
    timeframe: str = Field(..., description="Timeframe")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Indicator parameters")
    start: Optional[int] = Field(None, description="Start timestamp in milliseconds")
    end: Optional[int] = Field(None, description="End timestamp in milliseconds")
    limit: Optional[int] = Field(None, description="Limit")


class PatternResult(BaseModel):
    """Pattern result."""
    
    pattern: str = Field(..., description="Pattern name")
    symbol: str = Field(..., description="Symbol")
    timeframe: str = Field(..., description="Timeframe")
    occurrences: List[Dict[str, Any]] = Field(..., description="Pattern occurrences")


class PatternRequest(BaseModel):
    """Pattern request."""
    
    pattern: str = Field(..., description="Pattern name")
    symbol: str = Field(..., description="Symbol")
    timeframe: str = Field(..., description="Timeframe")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Pattern parameters")
    start: Optional[int] = Field(None, description="Start timestamp in milliseconds")
    end: Optional[int] = Field(None, description="End timestamp in milliseconds")
    limit: Optional[int] = Field(None, description="Limit")


# Define routes
@router.post("/indicators", response_model=IndicatorResult)
async def calculate_indicator(request: IndicatorRequest):
    """
    Calculate indicator.
    
    Args:
        request: Indicator request
        
    Returns:
        Indicator result
    """
    try:
        return await analysis_service.calculate_indicator(
            request.indicator,
            request.symbol,
            request.timeframe,
            request.parameters,
            request.start,
            request.end,
            request.limit
        )
    except Exception as e:
        logger.error(f"Error calculating indicator {request.indicator}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/indicators/{indicator}/{symbol}/{timeframe}", response_model=IndicatorResult)
async def get_indicator(
    indicator: str = Path(..., description="Indicator name"),
    symbol: str = Path(..., description="Symbol"),
    timeframe: str = Path(..., description="Timeframe"),
    start: Optional[int] = Query(None, description="Start timestamp in milliseconds"),
    end: Optional[int] = Query(None, description="End timestamp in milliseconds"),
    limit: Optional[int] = Query(None, description="Limit")
):
    """
    Get indicator.
    
    Args:
        indicator: Indicator name
        symbol: Symbol
        timeframe: Timeframe
        start: Start timestamp in milliseconds
        end: End timestamp in milliseconds
        limit: Limit
        
    Returns:
        Indicator result
    """
    try:
        return await analysis_service.get_indicator(
            indicator,
            symbol,
            timeframe,
            {},
            start,
            end,
            limit
        )
    except Exception as e:
        logger.error(f"Error getting indicator {indicator}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/patterns", response_model=PatternResult)
async def detect_pattern(request: PatternRequest):
    """
    Detect pattern.
    
    Args:
        request: Pattern request
        
    Returns:
        Pattern result
    """
    try:
        return await analysis_service.detect_pattern(
            request.pattern,
            request.symbol,
            request.timeframe,
            request.parameters,
            request.start,
            request.end,
            request.limit
        )
    except Exception as e:
        logger.error(f"Error detecting pattern {request.pattern}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns/{pattern}/{symbol}/{timeframe}", response_model=PatternResult)
async def get_pattern(
    pattern: str = Path(..., description="Pattern name"),
    symbol: str = Path(..., description="Symbol"),
    timeframe: str = Path(..., description="Timeframe"),
    start: Optional[int] = Query(None, description="Start timestamp in milliseconds"),
    end: Optional[int] = Query(None, description="End timestamp in milliseconds"),
    limit: Optional[int] = Query(None, description="Limit")
):
    """
    Get pattern.
    
    Args:
        pattern: Pattern name
        symbol: Symbol
        timeframe: Timeframe
        start: Start timestamp in milliseconds
        end: End timestamp in milliseconds
        limit: Limit
        
    Returns:
        Pattern result
    """
    try:
        return await analysis_service.get_pattern(
            pattern,
            symbol,
            timeframe,
            {},
            start,
            end,
            limit
        )
    except Exception as e:
        logger.error(f"Error getting pattern {pattern}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))