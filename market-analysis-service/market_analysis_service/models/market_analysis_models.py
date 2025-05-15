"""
Market Analysis Models for Market Analysis Service.

This module provides data models for the Market Analysis Service.
"""
from enum import Enum
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class AnalysisType(str, Enum):
    """
    Types of market analysis.
    """
    TECHNICAL = "technical"
    PATTERN = "pattern"
    SUPPORT_RESISTANCE = "support_resistance"
    MARKET_REGIME = "market_regime"
    CORRELATION = "correlation"
    VOLATILITY = "volatility"
    SENTIMENT = "sentiment"
    COMPREHENSIVE = "comprehensive"


class PatternType(str, Enum):
    """
    Types of chart patterns.
    """
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIPLE_TOP = "triple_top"
    TRIPLE_BOTTOM = "triple_bottom"
    ASCENDING_TRIANGLE = "ascending_triangle"
    DESCENDING_TRIANGLE = "descending_triangle"
    SYMMETRICAL_TRIANGLE = "symmetrical_triangle"
    FLAG = "flag"
    PENNANT = "pennant"
    WEDGE = "wedge"
    RECTANGLE = "rectangle"
    CUP_AND_HANDLE = "cup_and_handle"
    CUSTOM = "custom"


class MarketRegimeType(str, Enum):
    """
    Types of market regimes.
    """
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    CONSOLIDATION = "consolidation"
    CUSTOM = "custom"


class SupportResistanceMethod(str, Enum):
    """
    Methods for support and resistance detection.
    """
    PRICE_SWINGS = "price_swings"
    MOVING_AVERAGE = "moving_average"
    FIBONACCI = "fibonacci"
    PIVOT_POINTS = "pivot_points"
    VOLUME_PROFILE = "volume_profile"
    FRACTAL = "fractal"
    CUSTOM = "custom"


class MarketAnalysisRequest(BaseModel):
    """
    Request model for market analysis.
    """
    symbol: str
    timeframe: str
    start_date: str
    end_date: Optional[str] = None
    analysis_types: List[AnalysisType]
    additional_parameters: Optional[Dict[str, Any]] = None


class MarketAnalysisResponse(BaseModel):
    """
    Response model for market analysis.
    """
    request_id: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: Optional[str] = None
    analysis_results: List[Dict[str, Any]]
    execution_time_ms: int
    timestamp: datetime


class PatternRecognitionRequest(BaseModel):
    """
    Request model for pattern recognition.
    """
    symbol: str
    timeframe: str
    start_date: str
    end_date: Optional[str] = None
    pattern_types: Optional[List[str]] = None
    min_confidence: float = 0.7


class PatternRecognitionResponse(BaseModel):
    """
    Response model for pattern recognition.
    """
    request_id: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: Optional[str] = None
    patterns: List[Dict[str, Any]]
    execution_time_ms: int
    timestamp: datetime


class SupportResistanceRequest(BaseModel):
    """
    Request model for support and resistance detection.
    """
    symbol: str
    timeframe: str
    start_date: str
    end_date: Optional[str] = None
    methods: List[str]
    levels_count: int = 5
    additional_parameters: Optional[Dict[str, Any]] = None


class SupportResistanceResponse(BaseModel):
    """
    Response model for support and resistance detection.
    """
    request_id: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: Optional[str] = None
    levels: List[Dict[str, Any]]
    execution_time_ms: int
    timestamp: datetime


class MarketRegimeRequest(BaseModel):
    """
    Request model for market regime detection.
    """
    symbol: str
    timeframe: str
    start_date: str
    end_date: Optional[str] = None
    window_size: int = 20
    additional_parameters: Optional[Dict[str, Any]] = None


class MarketRegimeResponse(BaseModel):
    """
    Response model for market regime detection.
    """
    request_id: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: Optional[str] = None
    regimes: List[Dict[str, Any]]
    current_regime: str
    execution_time_ms: int
    timestamp: datetime


class CorrelationAnalysisRequest(BaseModel):
    """
    Request model for correlation analysis.
    """
    symbols: List[str]
    timeframe: str
    start_date: str
    end_date: Optional[str] = None
    window_size: int = 20
    method: str = "pearson"
    additional_parameters: Optional[Dict[str, Any]] = None


class CorrelationAnalysisResponse(BaseModel):
    """
    Response model for correlation analysis.
    """
    request_id: str
    symbols: List[str]
    timeframe: str
    start_date: str
    end_date: Optional[str] = None
    method: str
    correlation_matrix: Dict[str, Dict[str, float]]
    correlation_pairs: List[Dict[str, Any]]
    execution_time_ms: int
    timestamp: datetime