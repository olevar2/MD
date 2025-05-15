"""
Query models for the Market Analysis Service.

This module provides the query models for the Market Analysis Service.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from common_lib.cqrs.queries import Query
from market_analysis_service.models.market_analysis_models import (
    AnalysisType,
    PatternType,
    MarketRegimeType,
    SupportResistanceMethod
)


class GetAnalysisResultQuery(Query):
    """Query to get an analysis result by ID."""
    
    analysis_id: str = Field(..., description="ID of the analysis result to retrieve")


class ListAnalysisResultsQuery(Query):
    """Query to list analysis results."""
    
    symbol: Optional[str] = Field(None, description="Filter by symbol")
    timeframe: Optional[str] = Field(None, description="Filter by timeframe")
    analysis_type: Optional[AnalysisType] = Field(None, description="Filter by analysis type")
    start_date: Optional[datetime] = Field(None, description="Filter by start date")
    end_date: Optional[datetime] = Field(None, description="Filter by end date")
    limit: int = Field(10, description="Maximum number of results to return")
    offset: int = Field(0, description="Offset for pagination")


class GetPatternRecognitionResultQuery(Query):
    """Query to get a pattern recognition result by ID."""
    
    result_id: str = Field(..., description="ID of the pattern recognition result to retrieve")


class ListPatternRecognitionResultsQuery(Query):
    """Query to list pattern recognition results."""
    
    symbol: Optional[str] = Field(None, description="Filter by symbol")
    timeframe: Optional[str] = Field(None, description="Filter by timeframe")
    pattern_type: Optional[PatternType] = Field(None, description="Filter by pattern type")
    start_date: Optional[datetime] = Field(None, description="Filter by start date")
    end_date: Optional[datetime] = Field(None, description="Filter by end date")
    limit: int = Field(10, description="Maximum number of results to return")
    offset: int = Field(0, description="Offset for pagination")


class GetSupportResistanceResultQuery(Query):
    """Query to get a support/resistance result by ID."""
    
    result_id: str = Field(..., description="ID of the support/resistance result to retrieve")


class ListSupportResistanceResultsQuery(Query):
    """Query to list support/resistance results."""
    
    symbol: Optional[str] = Field(None, description="Filter by symbol")
    timeframe: Optional[str] = Field(None, description="Filter by timeframe")
    method: Optional[SupportResistanceMethod] = Field(None, description="Filter by detection method")
    start_date: Optional[datetime] = Field(None, description="Filter by start date")
    end_date: Optional[datetime] = Field(None, description="Filter by end date")
    limit: int = Field(10, description="Maximum number of results to return")
    offset: int = Field(0, description="Offset for pagination")


class GetMarketRegimeResultQuery(Query):
    """Query to get a market regime result by ID."""
    
    result_id: str = Field(..., description="ID of the market regime result to retrieve")


class ListMarketRegimeResultsQuery(Query):
    """Query to list market regime results."""
    
    symbol: Optional[str] = Field(None, description="Filter by symbol")
    timeframe: Optional[str] = Field(None, description="Filter by timeframe")
    regime_type: Optional[MarketRegimeType] = Field(None, description="Filter by regime type")
    start_date: Optional[datetime] = Field(None, description="Filter by start date")
    end_date: Optional[datetime] = Field(None, description="Filter by end date")
    limit: int = Field(10, description="Maximum number of results to return")
    offset: int = Field(0, description="Offset for pagination")


class GetCorrelationAnalysisResultQuery(Query):
    """Query to get a correlation analysis result by ID."""
    
    result_id: str = Field(..., description="ID of the correlation analysis result to retrieve")


class ListCorrelationAnalysisResultsQuery(Query):
    """Query to list correlation analysis results."""
    
    symbols: Optional[List[str]] = Field(None, description="Filter by symbols")
    timeframe: Optional[str] = Field(None, description="Filter by timeframe")
    method: Optional[str] = Field(None, description="Filter by correlation method")
    start_date: Optional[datetime] = Field(None, description="Filter by start date")
    end_date: Optional[datetime] = Field(None, description="Filter by end date")
    limit: int = Field(10, description="Maximum number of results to return")
    offset: int = Field(0, description="Offset for pagination")


class GetAvailableMethodsQuery(Query):
    """Query to get available analysis methods."""
    pass