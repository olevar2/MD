"""
Command models for the Market Analysis Service.

This module provides the command models for the Market Analysis Service.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from common_lib.cqrs.commands import Command
from market_analysis_service.models.market_analysis_models import (
    AnalysisType,
    PatternType,
    MarketRegimeType,
    SupportResistanceMethod
)


class AnalyzeMarketCommand(Command):
    """Command to perform comprehensive market analysis."""
    
    symbol: str = Field(..., description="Symbol to analyze")
    timeframe: str = Field(..., description="Timeframe for analysis (e.g., '1h', '1d')")
    start_date: datetime = Field(..., description="Start date for analysis")
    end_date: Optional[datetime] = Field(None, description="End date for analysis")
    analysis_types: List[AnalysisType] = Field(..., description="Types of analysis to perform")
    additional_parameters: Optional[Dict[str, Any]] = Field(None, description="Additional analysis parameters")


class RecognizePatternsCommand(Command):
    """Command to recognize chart patterns in market data."""
    
    symbol: str = Field(..., description="Symbol to analyze")
    timeframe: str = Field(..., description="Timeframe for analysis (e.g., '1h', '1d')")
    start_date: datetime = Field(..., description="Start date for analysis")
    end_date: Optional[datetime] = Field(None, description="End date for analysis")
    pattern_types: Optional[List[PatternType]] = Field(None, description="Types of patterns to recognize")
    min_confidence: float = Field(0.7, description="Minimum confidence level for pattern recognition")
    additional_parameters: Optional[Dict[str, Any]] = Field(None, description="Additional pattern recognition parameters")


class DetectSupportResistanceCommand(Command):
    """Command to detect support and resistance levels in market data."""
    
    symbol: str = Field(..., description="Symbol to analyze")
    timeframe: str = Field(..., description="Timeframe for analysis (e.g., '1h', '1d')")
    start_date: datetime = Field(..., description="Start date for analysis")
    end_date: Optional[datetime] = Field(None, description="End date for analysis")
    methods: Optional[List[SupportResistanceMethod]] = Field(None, description="Methods for support/resistance detection")
    additional_parameters: Optional[Dict[str, Any]] = Field(None, description="Additional detection parameters")


class DetectMarketRegimeCommand(Command):
    """Command to detect market regime in market data."""
    
    symbol: str = Field(..., description="Symbol to analyze")
    timeframe: str = Field(..., description="Timeframe for analysis (e.g., '1h', '1d')")
    start_date: datetime = Field(..., description="Start date for analysis")
    end_date: Optional[datetime] = Field(None, description="End date for analysis")
    window_size: int = Field(20, description="Window size for regime detection")
    additional_parameters: Optional[Dict[str, Any]] = Field(None, description="Additional detection parameters")


class AnalyzeCorrelationCommand(Command):
    """Command to analyze correlations between symbols."""
    
    symbols: List[str] = Field(..., description="Symbols to analyze")
    timeframe: str = Field(..., description="Timeframe for analysis (e.g., '1h', '1d')")
    start_date: datetime = Field(..., description="Start date for analysis")
    end_date: Optional[datetime] = Field(None, description="End date for analysis")
    window_size: int = Field(20, description="Window size for correlation analysis")
    method: str = Field("pearson", description="Correlation method")
    additional_parameters: Optional[Dict[str, Any]] = Field(None, description="Additional analysis parameters")


class AnalyzeVolatilityCommand(Command):
    """Command to analyze volatility in market data."""
    
    symbol: str = Field(..., description="Symbol to analyze")
    timeframe: str = Field(..., description="Timeframe for analysis (e.g., '1h', '1d')")
    start_date: datetime = Field(..., description="Start date for analysis")
    end_date: Optional[datetime] = Field(None, description="End date for analysis")
    window_size: int = Field(20, description="Window size for volatility analysis")
    method: str = Field("historical", description="Volatility calculation method")
    additional_parameters: Optional[Dict[str, Any]] = Field(None, description="Additional analysis parameters")


class AnalyzeSentimentCommand(Command):
    """Command to analyze market sentiment."""
    
    symbol: str = Field(..., description="Symbol to analyze")
    timeframe: str = Field(..., description="Timeframe for analysis (e.g., '1h', '1d')")
    start_date: datetime = Field(..., description="Start date for analysis")
    end_date: Optional[datetime] = Field(None, description="End date for analysis")
    sources: Optional[List[str]] = Field(None, description="Sentiment data sources")
    additional_parameters: Optional[Dict[str, Any]] = Field(None, description="Additional analysis parameters")