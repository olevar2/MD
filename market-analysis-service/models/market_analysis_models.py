from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from enum import Enum
from datetime import datetime

class AnalysisType(str, Enum):
    TECHNICAL = "technical"
    PATTERN = "pattern"
    SUPPORT_RESISTANCE = "support_resistance"
    MARKET_REGIME = "market_regime"
    CORRELATION = "correlation"
    VOLATILITY = "volatility"
    SENTIMENT = "sentiment"
    COMPREHENSIVE = "comprehensive"

class PatternType(str, Enum):
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
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    CONSOLIDATION = "consolidation"
    CUSTOM = "custom"

class SupportResistanceMethod(str, Enum):
    PRICE_SWINGS = "price_swings"
    MOVING_AVERAGE = "moving_average"
    FIBONACCI = "fibonacci"
    PIVOT_POINTS = "pivot_points"
    VOLUME_PROFILE = "volume_profile"
    FRACTAL = "fractal"
    CUSTOM = "custom"

class MarketAnalysisRequest(BaseModel):
    symbol: str = Field(..., description="Symbol to analyze")
    timeframe: str = Field(..., description="Timeframe for analysis (e.g., '1h', '1d')")
    start_date: str = Field(..., description="Start date for analysis (ISO format)")
    end_date: str = Field(..., description="End date for analysis (ISO format)")
    analysis_types: List[AnalysisType] = Field(..., description="Types of analysis to perform")
    additional_parameters: Optional[Dict[str, Any]] = Field(default=None, description="Additional analysis parameters")

class AnalysisResult(BaseModel):
    analysis_type: AnalysisType = Field(..., description="Type of analysis performed")
    result: Dict[str, Any] = Field(..., description="Analysis result")
    confidence: float = Field(..., description="Confidence level of the analysis (0-1)")
    execution_time_ms: int = Field(..., description="Execution time in milliseconds")

class MarketAnalysisResponse(BaseModel):
    request_id: str = Field(..., description="Unique ID for the analysis request")
    symbol: str = Field(..., description="Symbol analyzed")
    timeframe: str = Field(..., description="Timeframe used for analysis")
    start_date: str = Field(..., description="Start date of the analysis")
    end_date: str = Field(..., description="End date of the analysis")
    analysis_results: List[AnalysisResult] = Field(..., description="Results of the analyses")
    execution_time_ms: int = Field(..., description="Total execution time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the analysis")

class PatternRecognitionRequest(BaseModel):
    symbol: str = Field(..., description="Symbol to analyze")
    timeframe: str = Field(..., description="Timeframe for analysis (e.g., '1h', '1d')")
    start_date: str = Field(..., description="Start date for analysis (ISO format)")
    end_date: str = Field(..., description="End date for analysis (ISO format)")
    pattern_types: Optional[List[PatternType]] = Field(default=None, description="Types of patterns to recognize")
    min_confidence: float = Field(default=0.7, description="Minimum confidence level for pattern recognition (0-1)")
    additional_parameters: Optional[Dict[str, Any]] = Field(default=None, description="Additional recognition parameters")

class PatternInstance(BaseModel):
    pattern_type: PatternType = Field(..., description="Type of pattern recognized")
    start_index: int = Field(..., description="Start index of the pattern")
    end_index: int = Field(..., description="End index of the pattern")
    confidence: float = Field(..., description="Confidence level of the pattern recognition (0-1)")
    target_price: Optional[float] = Field(default=None, description="Target price based on the pattern")
    stop_loss: Optional[float] = Field(default=None, description="Suggested stop loss based on the pattern")
    risk_reward_ratio: Optional[float] = Field(default=None, description="Risk-reward ratio based on the pattern")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional pattern metadata")

class PatternRecognitionResponse(BaseModel):
    request_id: str = Field(..., description="Unique ID for the recognition request")
    symbol: str = Field(..., description="Symbol analyzed")
    timeframe: str = Field(..., description="Timeframe used for analysis")
    start_date: str = Field(..., description="Start date of the analysis")
    end_date: str = Field(..., description="End date of the analysis")
    patterns: List[PatternInstance] = Field(..., description="Recognized patterns")
    execution_time_ms: int = Field(..., description="Execution time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the analysis")

class SupportResistanceRequest(BaseModel):
    symbol: str = Field(..., description="Symbol to analyze")
    timeframe: str = Field(..., description="Timeframe for analysis (e.g., '1h', '1d')")
    start_date: str = Field(..., description="Start date for analysis (ISO format)")
    end_date: str = Field(..., description="End date for analysis (ISO format)")
    methods: List[SupportResistanceMethod] = Field(..., description="Methods to use for identification")
    levels_count: int = Field(default=5, description="Number of levels to identify")
    additional_parameters: Optional[Dict[str, Any]] = Field(default=None, description="Additional identification parameters")

class PriceLevel(BaseModel):
    price: float = Field(..., description="Price level")
    type: str = Field(..., description="Type of level (support or resistance)")
    strength: float = Field(..., description="Strength of the level (0-1)")
    method: SupportResistanceMethod = Field(..., description="Method used to identify the level")
    touches: int = Field(..., description="Number of times the price has touched this level")
    last_touch_date: Optional[str] = Field(default=None, description="Date of the last touch (ISO format)")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional level metadata")

class SupportResistanceResponse(BaseModel):
    request_id: str = Field(..., description="Unique ID for the identification request")
    symbol: str = Field(..., description="Symbol analyzed")
    timeframe: str = Field(..., description="Timeframe used for analysis")
    start_date: str = Field(..., description="Start date of the analysis")
    end_date: str = Field(..., description="End date of the analysis")
    levels: List[PriceLevel] = Field(..., description="Identified support and resistance levels")
    execution_time_ms: int = Field(..., description="Execution time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the analysis")

class MarketRegimeRequest(BaseModel):
    symbol: str = Field(..., description="Symbol to analyze")
    timeframe: str = Field(..., description="Timeframe for analysis (e.g., '1h', '1d')")
    start_date: str = Field(..., description="Start date for analysis (ISO format)")
    end_date: str = Field(..., description="End date for analysis (ISO format)")
    window_size: int = Field(default=20, description="Window size for regime detection")
    additional_parameters: Optional[Dict[str, Any]] = Field(default=None, description="Additional detection parameters")

class RegimeSegment(BaseModel):
    regime_type: MarketRegimeType = Field(..., description="Type of market regime")
    start_index: int = Field(..., description="Start index of the regime segment")
    end_index: int = Field(..., description="End index of the regime segment")
    start_date: str = Field(..., description="Start date of the regime segment (ISO format)")
    end_date: str = Field(..., description="End date of the regime segment (ISO format)")
    confidence: float = Field(..., description="Confidence level of the regime detection (0-1)")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional regime metadata")

class MarketRegimeResponse(BaseModel):
    request_id: str = Field(..., description="Unique ID for the detection request")
    symbol: str = Field(..., description="Symbol analyzed")
    timeframe: str = Field(..., description="Timeframe used for analysis")
    start_date: str = Field(..., description="Start date of the analysis")
    end_date: str = Field(..., description="End date of the analysis")
    regimes: List[RegimeSegment] = Field(..., description="Detected market regimes")
    current_regime: MarketRegimeType = Field(..., description="Current market regime")
    execution_time_ms: int = Field(..., description="Execution time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the analysis")

class CorrelationAnalysisRequest(BaseModel):
    symbols: List[str] = Field(..., description="Symbols to analyze")
    timeframe: str = Field(..., description="Timeframe for analysis (e.g., '1h', '1d')")
    start_date: str = Field(..., description="Start date for analysis (ISO format)")
    end_date: str = Field(..., description="End date for analysis (ISO format)")
    window_size: int = Field(default=20, description="Window size for rolling correlation")
    method: str = Field(default="pearson", description="Correlation method (pearson, spearman, kendall)")
    additional_parameters: Optional[Dict[str, Any]] = Field(default=None, description="Additional analysis parameters")

class CorrelationPair(BaseModel):
    symbol1: str = Field(..., description="First symbol in the pair")
    symbol2: str = Field(..., description="Second symbol in the pair")
    correlation: float = Field(..., description="Correlation coefficient")
    p_value: float = Field(..., description="P-value of the correlation")
    rolling_correlation: Optional[List[Dict[str, Any]]] = Field(default=None, description="Rolling correlation data points")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional correlation metadata")

class CorrelationAnalysisResponse(BaseModel):
    request_id: str = Field(..., description="Unique ID for the analysis request")
    symbols: List[str] = Field(..., description="Symbols analyzed")
    timeframe: str = Field(..., description="Timeframe used for analysis")
    start_date: str = Field(..., description="Start date of the analysis")
    end_date: str = Field(..., description="End date of the analysis")
    method: str = Field(..., description="Correlation method used")
    correlation_matrix: Dict[str, Dict[str, float]] = Field(..., description="Correlation matrix")
    correlation_pairs: List[CorrelationPair] = Field(..., description="Detailed correlation pairs")
    execution_time_ms: int = Field(..., description="Execution time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the analysis")